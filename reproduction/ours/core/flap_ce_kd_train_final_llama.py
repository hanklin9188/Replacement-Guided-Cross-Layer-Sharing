#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from typing import Any, Dict, List, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from thesis_common_final_llama import str2bool  # noqa: E402
from thesis_geometry_redundancy_final_llama import (  # noqa: E402
    collate_tokenized_batch,
    prepare_records,
    tokenize_records,
)


def _select_dtype(dtype_name: str) -> torch.dtype:
    text = str(dtype_name or "auto").strip().lower()
    if text in {"float32", "fp32"}:
        return torch.float32
    if text in {"float16", "fp16", "half"}:
        return torch.float16
    if text in {"bfloat16", "bf16"}:
        return torch.bfloat16
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    if torch.cuda.is_available():
        return torch.float16
    return torch.float32


def _load_tokenizer(path: str, trust_remote_code: bool):
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=bool(trust_remote_code), use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
    if tokenizer.pad_token_id is None:
        raise ValueError("tokenizer has no pad/eos/unk token")
    return tokenizer


def _ce_shift_loss(logits: torch.Tensor, input_ids: torch.Tensor, pad_token_id: int) -> torch.Tensor:
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()
    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=int(pad_token_id),
    )


def _kd_shift_loss(
    *,
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    attention_mask: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    temp = max(1e-6, float(temperature))
    s_shift = student_logits[:, :-1, :].contiguous()
    t_shift = teacher_logits[:, :-1, :].contiguous()
    if int(s_shift.size(-1)) != int(t_shift.size(-1)):
        vocab = min(int(s_shift.size(-1)), int(t_shift.size(-1)))
        s_shift = s_shift[..., :vocab].contiguous()
        t_shift = t_shift[..., :vocab].contiguous()
    valid = attention_mask[:, 1:].to(dtype=s_shift.dtype)
    s_log_prob = F.log_softmax(s_shift / temp, dim=-1)
    t_prob = F.softmax(t_shift / temp, dim=-1)
    kl_per_token = F.kl_div(s_log_prob, t_prob, reduction="none").sum(dim=-1)
    denom = valid.sum().clamp(min=1.0)
    return (kl_per_token * valid).sum() / denom * (temp * temp)


def _collect_zero_masks(model: torch.nn.Module) -> List[Tuple[str, torch.nn.Parameter, torch.Tensor]]:
    masks: List[Tuple[str, torch.nn.Parameter, torch.Tensor]] = []
    for name, param in model.named_parameters():
        if not param.requires_grad or param.dim() < 2:
            continue
        zero_count = int(torch.count_nonzero(param.data == 0).item())
        if zero_count <= 0:
            continue
        masks.append((name, param, param.data.ne(0).detach().clone()))
    return masks


@torch.no_grad()
def _apply_zero_masks(masks: Sequence[Tuple[str, torch.nn.Parameter, torch.Tensor]]) -> None:
    for _, param, mask in masks:
        param.data.mul_(mask)


def _set_seed(seed: int) -> None:
    random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _save_report(path: str, payload: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Direct FLAP student recovery with CE or CE+KD.")
    parser.add_argument("--student_model", required=True)
    parser.add_argument("--teacher_ckpt", required=True)
    parser.add_argument("--tokenizer_name_or_path", default="")
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--trust_remote_code", type=str2bool, default=True)
    parser.add_argument("--dtype", default="auto", choices=["auto", "float32", "float16", "bfloat16"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--cutoff_len", type=int, default=512)
    parser.add_argument("--max_records", type=int, default=0)
    parser.add_argument("--shuffle_records", type=str2bool, default=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=12000)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_steps", type=int, default=0)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--lambda_ce", type=float, default=1.0)
    parser.add_argument("--lambda_kd", type=float, default=1.0)
    parser.add_argument("--kd_temperature", type=float, default=2.0)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument("--save_every", type=int, default=0)
    parser.add_argument("--gradient_checkpointing", type=str2bool, default=True)
    parser.add_argument("--preserve_zero_mask", type=str2bool, default=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    _set_seed(args.seed)
    use_kd = abs(float(args.lambda_kd)) > 0.0
    objective_name = "CE+KD" if use_kd else "CE"

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("device=cuda requested but CUDA is not available")
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    dtype = _select_dtype(args.dtype)
    tokenizer_path = args.tokenizer_name_or_path or args.teacher_ckpt or args.student_model
    tokenizer = _load_tokenizer(tokenizer_path, bool(args.trust_remote_code))

    print("[DirectTrain] loading FLAP student", flush=True)
    student = AutoModelForCausalLM.from_pretrained(
        args.student_model,
        trust_remote_code=bool(args.trust_remote_code),
        torch_dtype=dtype,
    ).to(device)
    teacher = None
    if use_kd:
        print("[DirectTrain] loading merged teacher", flush=True)
        teacher = AutoModelForCausalLM.from_pretrained(
            args.teacher_ckpt,
            trust_remote_code=bool(args.trust_remote_code),
            torch_dtype=dtype,
        ).to(device)
        teacher.eval()
        for param in teacher.parameters():
            param.requires_grad_(False)

    if bool(args.gradient_checkpointing):
        student.gradient_checkpointing_enable()
    if hasattr(student.config, "use_cache"):
        student.config.use_cache = False
    if teacher is not None and hasattr(teacher.config, "use_cache"):
        teacher.config.use_cache = False
    student.train()

    print("[DirectTrain] preparing data", flush=True)
    records = prepare_records(
        args.data_path,
        max_records=int(args.max_records),
        seed=int(args.seed),
        shuffle_records=bool(args.shuffle_records),
    )
    tokenized = tokenize_records(tokenizer, records, cutoff_len=int(args.cutoff_len))
    if not tokenized:
        raise ValueError(f"no tokenized records from {args.data_path}")
    loader = DataLoader(
        tokenized,
        batch_size=int(args.batch_size),
        shuffle=True,
        collate_fn=lambda batch: collate_tokenized_batch(batch, pad_token_id=int(tokenizer.pad_token_id)),
    )
    print(
        f"[DirectTrain] tokenized={len(tokenized)} batch_size={args.batch_size} "
        f"steps={args.steps} objective={objective_name} lambda_ce={args.lambda_ce} lambda_kd={args.lambda_kd} "
        f"kd_temperature={args.kd_temperature}",
        flush=True,
    )
    print("[DirectTrain] atlas disabled; core loss disabled", flush=True)

    zero_masks = _collect_zero_masks(student) if bool(args.preserve_zero_mask) else []
    if zero_masks:
        masked_params = len(zero_masks)
        masked_zeros = sum(int(torch.count_nonzero(~mask).item()) for _, _, mask in zero_masks)
        print(f"[DirectTrain] preserving FLAP zero mask params={masked_params} zeros={masked_zeros}", flush=True)

    optimizer = AdamW(student.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    warmup_steps = int(args.warmup_steps)
    if warmup_steps <= 0 and float(args.warmup_ratio) > 0.0:
        warmup_steps = int(math.ceil(int(args.steps) * float(args.warmup_ratio)))
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(0, warmup_steps),
        num_training_steps=max(1, int(args.steps)),
    )

    losses: List[Dict[str, float]] = []
    step = 0
    start_time = time.time()
    progress = tqdm(total=int(args.steps), desc="[DirectTrain] train", dynamic_ncols=True)
    while step < int(args.steps):
        for batch in loader:
            if step >= int(args.steps):
                break
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            if use_kd:
                assert teacher is not None
                with torch.no_grad():
                    teacher_logits = teacher(input_ids=input_ids, attention_mask=attention_mask).logits
            student_logits = student(input_ids=input_ids, attention_mask=attention_mask).logits
            ce_loss = _ce_shift_loss(student_logits, input_ids, int(tokenizer.pad_token_id))
            if use_kd:
                kd_loss = _kd_shift_loss(
                    student_logits=student_logits,
                    teacher_logits=teacher_logits,
                    attention_mask=attention_mask,
                    temperature=float(args.kd_temperature),
                )
            else:
                kd_loss = torch.zeros((), device=device, dtype=ce_loss.dtype)
            loss = float(args.lambda_ce) * ce_loss + float(args.lambda_kd) * kd_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if float(args.grad_clip) > 0.0:
                torch.nn.utils.clip_grad_norm_(student.parameters(), float(args.grad_clip))
            optimizer.step()
            scheduler.step()
            if zero_masks:
                _apply_zero_masks(zero_masks)

            step += 1
            progress.update(1)
            if step == 1 or step % int(args.log_every) == 0:
                item = {
                    "step": float(step),
                    "loss": float(loss.detach().cpu().item()),
                    "ce_loss": float(ce_loss.detach().cpu().item()),
                    "kd_loss": float(kd_loss.detach().cpu().item()),
                    "lr": float(scheduler.get_last_lr()[0]),
                }
                losses.append(item)
                print(
                    "[DirectTrain] "
                    f"step={step} loss={item['loss']:.6f} ce={item['ce_loss']:.6f} "
                    f"kd={item['kd_loss']:.6f} lr={item['lr']:.3e}",
                    flush=True,
                )
            if int(args.save_every) > 0 and step % int(args.save_every) == 0:
                ckpt_dir = os.path.join(args.output_dir, f"checkpoint-{step}")
                student.save_pretrained(ckpt_dir)
                tokenizer.save_pretrained(ckpt_dir)

    progress.close()
    if zero_masks:
        _apply_zero_masks(zero_masks)

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"[DirectTrain] saving model: {args.output_dir}", flush=True)
    student.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    report = {
        "student_model": os.path.abspath(args.student_model),
        "teacher_ckpt": os.path.abspath(args.teacher_ckpt),
        "data_path": os.path.abspath(args.data_path),
        "output_dir": os.path.abspath(args.output_dir),
        "objective": objective_name,
        "lambda_ce": float(args.lambda_ce),
        "lambda_kd": float(args.lambda_kd),
        "kd_temperature": float(args.kd_temperature),
        "steps": int(args.steps),
        "batch_size": int(args.batch_size),
        "cutoff_len": int(args.cutoff_len),
        "records": int(len(records)),
        "tokenized": int(len(tokenized)),
        "dtype": str(dtype).replace("torch.", ""),
        "preserve_zero_mask": bool(args.preserve_zero_mask),
        "elapsed_sec": float(time.time() - start_time),
        "log": losses,
    }
    _save_report(os.path.join(args.output_dir, "direct_train_report.json"), report)
    print("[DirectTrain] done", flush=True)


if __name__ == "__main__":
    main()
