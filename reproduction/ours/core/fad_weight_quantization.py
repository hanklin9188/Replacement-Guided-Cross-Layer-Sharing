#!/usr/bin/env python3
"""Inference-only weight quantization helpers for the FAD Llama student.

Formats implemented here are deliberately explicit:

* fp8_e4m3: W8A8 E4M3 using ``torch._scaled_mm``.  This is native FP8
  Tensor-Core math on Hopper (H100).
* fp4_e2m1: packed bitsandbytes FP4 weights with BF16 compute.  H100 has no
  native FP4 Tensor Cores, so this path measures a real 4-bit stored model but
  must not be reported as native FP4 arithmetic.

The tied embedding/lm_head is kept in BF16 by default.  Quantizing lm_head
would break weight tying and can increase rather than decrease live storage.
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, Iterable, List, Tuple

import torch
import torch.nn as nn


FP8_MAX = 448.0


def _tensor_storage_bytes(tensor: torch.Tensor) -> int:
    try:
        return int(tensor.untyped_storage().nbytes())
    except Exception:
        return int(tensor.numel() * tensor.element_size())


def _unique_state_storage_bytes(module: nn.Module) -> int:
    seen = set()
    total = 0
    for tensor in module.state_dict().values():
        if not torch.is_tensor(tensor):
            continue
        try:
            storage = tensor.untyped_storage()
            key = (str(tensor.device), int(storage.data_ptr()), int(storage.nbytes()))
            nbytes = int(storage.nbytes())
        except Exception:
            key = (str(tensor.device), int(tensor.data_ptr()), int(tensor.numel()), str(tensor.dtype))
            nbytes = int(tensor.numel() * tensor.element_size())
        if key not in seen:
            seen.add(key)
            total += nbytes
    return int(total)


def _eligible_linear_modules(model: nn.Module, quantize_lm_head: bool) -> List[Tuple[str, nn.Linear]]:
    out: List[Tuple[str, nn.Linear]] = []
    for name, child in model.named_modules():
        if not isinstance(child, nn.Linear):
            continue
        if not quantize_lm_head and (name == "lm_head" or name.endswith(".lm_head")):
            continue
        if child.weight is None or child.weight.dim() != 2:
            continue
        if int(child.in_features) % 16 != 0 or int(child.out_features) % 16 != 0:
            continue
        out.append((str(name), child))
    return out


def _resolve_parent(model: nn.Module, qualified_name: str) -> Tuple[nn.Module, str]:
    if "." not in qualified_name:
        return model, qualified_name
    parent_name, leaf = qualified_name.rsplit(".", 1)
    return model.get_submodule(parent_name), leaf


class Float8E4M3Linear(nn.Module):
    """Tensorwise dynamically scaled W8A8 E4M3 linear for Hopper inference."""

    def __init__(self, source: nn.Linear):
        super().__init__()
        if source.weight.device.type != "cuda":
            raise ValueError("FP8 native linear must be constructed from a CUDA module")
        self.in_features = int(source.in_features)
        self.out_features = int(source.out_features)
        weight = source.weight.detach().float()
        scale = (weight.abs().amax() / FP8_MAX).clamp(min=1.0e-12)
        quantized = torch.clamp(weight / scale, min=-FP8_MAX, max=FP8_MAX).to(torch.float8_e4m3fn)
        self.register_buffer("weight", quantized.contiguous(), persistent=True)
        self.register_buffer("weight_scale", scale.reshape(1).to(dtype=torch.float32), persistent=True)
        if source.bias is None:
            self.register_buffer("bias", None, persistent=True)
        else:
            self.register_buffer("bias", source.bias.detach().clone(), persistent=True)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if input.device.type != "cuda":
            raise RuntimeError("FP8 E4M3 execution requires CUDA")
        original_shape = tuple(input.shape[:-1]) + (self.out_features,)
        flat = input.reshape(-1, self.in_features)
        input_scale = (flat.detach().abs().amax().float() / FP8_MAX).clamp(min=1.0e-12)
        flat_fp8 = torch.clamp(flat / input_scale, min=-FP8_MAX, max=FP8_MAX).to(torch.float8_e4m3fn)
        # A contiguous (N,K) weight transposes to the column-major (K,N)
        # layout expected by the Hopper scaled GEMM kernel.
        output = torch._scaled_mm(
            flat_fp8,
            self.weight.t(),
            scale_a=input_scale.reshape(1),
            scale_b=self.weight_scale,
            out_dtype=input.dtype,
            use_fast_accum=False,
        )
        if self.bias is not None:
            output = output + self.bias.to(dtype=output.dtype)
        return output.reshape(original_shape)

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, format=E4M3_W8A8"


def _replace_fp8(model: nn.Module, targets: Iterable[Tuple[str, nn.Linear]]) -> Dict[str, Any]:
    module_names: List[str] = []
    source_elements = 0
    stored_bytes = 0
    for name, source in targets:
        source_elements += int(source.weight.numel())
        replacement = Float8E4M3Linear(source)
        parent, leaf = _resolve_parent(model, name)
        setattr(parent, leaf, replacement)
        module_names.append(name)
        stored_bytes += _tensor_storage_bytes(replacement.weight) + _tensor_storage_bytes(replacement.weight_scale)
        if replacement.bias is not None:
            stored_bytes += _tensor_storage_bytes(replacement.bias)
    return {
        "module_names": module_names,
        "source_weight_elements": int(source_elements),
        "quantized_module_storage_bytes": int(stored_bytes),
        "execution_backend": "torch._scaled_mm",
        "weight_dtype": "torch.float8_e4m3fn",
        "activation_dtype": "dynamic torch.float8_e4m3fn",
        "compute": "native Hopper FP8 Tensor Core scaled GEMM",
    }


def _replace_fp4(model: nn.Module, targets: Iterable[Tuple[str, nn.Linear]]) -> Dict[str, Any]:
    try:
        import bitsandbytes as bnb
    except Exception as exc:  # pragma: no cover - exercised on compute nodes
        raise RuntimeError("fp4_e2m1 requires bitsandbytes") from exc

    module_names: List[str] = []
    source_elements = 0
    stored_bytes = 0
    for name, source in targets:
        source_elements += int(source.weight.numel())
        source_device = source.weight.device
        source_dtype = source.weight.dtype
        replacement = bnb.nn.Linear4bit(
            int(source.in_features),
            int(source.out_features),
            bias=source.bias is not None,
            compute_dtype=torch.bfloat16,
            compress_statistics=True,
            quant_type="fp4",
            quant_storage=torch.uint8,
        )
        # Params4bit performs the actual packing when moved from CPU to CUDA.
        replacement.weight = bnb.nn.Params4bit(
            source.weight.detach().to(device="cpu", dtype=source_dtype),
            requires_grad=False,
            compress_statistics=True,
            quant_type="fp4",
            quant_storage=torch.uint8,
            module=replacement,
        )
        if source.bias is not None:
            replacement.bias = nn.Parameter(source.bias.detach().to(device="cpu", dtype=source_dtype), requires_grad=False)
        replacement = replacement.to(source_device)
        if not bool(getattr(replacement.weight, "bnb_quantized", False)):
            raise RuntimeError(f"bitsandbytes did not pack FP4 weight for {name}")
        parent, leaf = _resolve_parent(model, name)
        setattr(parent, leaf, replacement)
        module_names.append(name)
        # State dict includes packed weights and the quantization state needed
        # for a reproducible reload.
        stored_bytes += _unique_state_storage_bytes(replacement)
    return {
        "module_names": module_names,
        "source_weight_elements": int(source_elements),
        "quantized_module_storage_bytes": int(stored_bytes),
        "execution_backend": f"bitsandbytes {getattr(bnb, '__version__', 'unknown')}",
        "weight_dtype": "packed FP4 E2M1 (uint8 holds two 4-bit values)",
        "activation_dtype": "torch.bfloat16",
        "compute": "FP4 weight dequantization plus BF16 compute; H100 has no native FP4 MMA",
    }


@torch.inference_mode()
def apply_weight_quantization(
    model: nn.Module,
    quantization: str,
    *,
    quantize_lm_head: bool = False,
) -> Dict[str, Any]:
    quantization = str(quantization).strip().lower()
    if quantization not in {"none", "fp8_e4m3", "fp4_e2m1"}:
        raise ValueError(f"Unsupported quantization format: {quantization}")
    before_bytes = _unique_state_storage_bytes(model)
    if quantization == "none":
        return {
            "requested": "none",
            "enabled": False,
            "model_state_storage_before_bytes": int(before_bytes),
            "model_state_storage_after_bytes": int(before_bytes),
        }

    targets = _eligible_linear_modules(model, bool(quantize_lm_head))
    if not targets:
        raise RuntimeError("No eligible nn.Linear modules found for weight quantization")
    if quantization == "fp8_e4m3":
        details = _replace_fp8(model, targets)
    else:
        details = _replace_fp4(model, targets)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    after_bytes = _unique_state_storage_bytes(model)
    source_weight_bytes_bf16 = int(details["source_weight_elements"] * 2)
    quantized_bytes = int(details["quantized_module_storage_bytes"])
    report: Dict[str, Any] = {
        "requested": quantization,
        "enabled": True,
        "quantize_lm_head": bool(quantize_lm_head),
        "excluded": [] if quantize_lm_head else ["tied embedding/lm_head (BF16)"],
        "module_count": int(len(details["module_names"])),
        "module_names": details.pop("module_names"),
        "source_weight_elements": int(details["source_weight_elements"]),
        "source_weight_bytes_bf16": source_weight_bytes_bf16,
        "quantized_module_storage_bytes": quantized_bytes,
        "eligible_weight_compression_ratio": (
            float(source_weight_bytes_bf16 / quantized_bytes) if quantized_bytes > 0 else None
        ),
        "model_state_storage_before_bytes": int(before_bytes),
        "model_state_storage_after_bytes": int(after_bytes),
        "model_state_compression_ratio": float(before_bytes / after_bytes) if after_bytes > 0 else None,
        **{k: v for k, v in details.items() if k not in {"source_weight_elements", "quantized_module_storage_bytes"}},
    }
    return report


def save_quantized_checkpoint(
    path: str,
    *,
    model: nn.Module,
    quantization_report: Dict[str, Any],
    source_deploy_bundle: str,
) -> str:
    path = os.path.abspath(str(path))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "format_version": 1,
        "artifact_type": "fad_25pct_adaptive_exit_quantized_state_dict",
        "source_deploy_bundle": os.path.abspath(str(source_deploy_bundle)),
        "quantization": dict(quantization_report),
        "state_dict": model.state_dict(),
    }
    torch.save(payload, path)
    manifest = {
        "checkpoint": path,
        "checkpoint_bytes": int(os.path.getsize(path)),
        "source_deploy_bundle": payload["source_deploy_bundle"],
        "quantization": quantization_report,
        "reload_order": [
            "reconstruct FAD shared model from source_deploy_bundle",
            "apply the same quantization format with fad_weight_quantization.apply_weight_quantization",
            "load this checkpoint state_dict with strict=True",
        ],
    }
    manifest_path = os.path.splitext(path)[0] + ".manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    return path
