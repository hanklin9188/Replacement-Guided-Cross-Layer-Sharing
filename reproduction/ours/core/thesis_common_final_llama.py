#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import random
import shlex
import subprocess
from datetime import datetime
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence


def str2bool(value) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"invalid bool value: {value!r}")


def now_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def ensure_parent(path: str) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)


def require_file(path: str, label: str) -> None:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"{label} not found: {path}")


def require_dir(path: str, label: str) -> None:
    if not os.path.isdir(path):
        raise FileNotFoundError(f"{label} not found: {path}")


def latest_match(pattern: str) -> Optional[str]:
    hits = [p for p in glob.glob(pattern) if os.path.exists(p)]
    if not hits:
        return None
    hits.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return hits[0]


def run_cmd(cmd: Sequence[str], env: Optional[dict] = None) -> None:
    pretty = " ".join(shlex.quote(x) for x in cmd)
    print(f"[Run] {pretty}", flush=True)
    subprocess.run(list(cmd), check=True, env=env)


def save_json(path: str, payload: dict) -> None:
    ensure_parent(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def parse_dataset_list(text: str) -> list[str]:
    flat = str(text).replace(",", " ")
    return [x for x in flat.split() if x]


def normalize_weight_map(weights: Dict[str, float]) -> Dict[str, float]:
    clean: Dict[str, float] = {}
    for key, value in weights.items():
        value_f = float(value)
        if value_f > 0.0:
            clean[str(key)] = value_f
    total = sum(clean.values())
    if total <= 0.0:
        raise ValueError("All source weights are non-positive.")
    return {key: value / total for key, value in clean.items()}


def _records_from_json_payload(payload: Any) -> List[Dict[str, Any]]:
    if isinstance(payload, list):
        out = []
        for item in payload:
            if isinstance(item, dict):
                out.append(item)
            else:
                out.append({"text": str(item)})
        return out
    if isinstance(payload, dict):
        if isinstance(payload.get("data"), list):
            return _records_from_json_payload(payload["data"])
        return [payload]
    return [{"text": str(payload)}]


def load_json_records(path: str) -> List[Dict[str, Any]]:
    if not path:
        raise ValueError("Empty path.")
    path = os.path.abspath(path)
    if os.path.isfile(path):
        lower = path.lower()
        if lower.endswith(".json"):
            with open(path, "r", encoding="utf-8") as f:
                return _records_from_json_payload(json.load(f))
        if lower.endswith(".jsonl"):
            rows: List[Dict[str, Any]] = []
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    obj = json.loads(line)
                    if isinstance(obj, dict):
                        rows.append(obj)
                    else:
                        rows.append({"text": str(obj)})
            return rows
        if lower.endswith(".txt"):
            rows = []
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    text = line.strip()
                    if text:
                        rows.append({"text": text})
            return rows
        raise ValueError(f"Unsupported file extension: {path}")

    if not os.path.isdir(path):
        raise FileNotFoundError(path)

    candidates = [
        os.path.join(path, "train.json"),
        os.path.join(path, "train.jsonl"),
        os.path.join(path, "data.json"),
        os.path.join(path, "data.jsonl"),
    ]
    for fp in candidates:
        if os.path.isfile(fp):
            return load_json_records(fp)
    raise FileNotFoundError(f"No supported train file found in directory: {path}")


def _split_plain_text_to_sft(text: str) -> Dict[str, str]:
    words = [w for w in str(text).strip().split() if w]
    if not words:
        return {"instruction": "", "input": "", "output": ""}
    if len(words) <= 8:
        joined = " ".join(words)
        return {"instruction": "Repeat and complete the short text.", "input": joined, "output": joined}
    cut = max(1, int(len(words) * 0.6))
    left = " ".join(words[:cut]).strip()
    right = " ".join(words[cut:]).strip()
    if not right:
        right = left
    return {
        "instruction": "Continue the following passage.",
        "input": left,
        "output": right,
    }


def normalize_record_for_sft(record: Dict[str, Any], source_name: str = "") -> Dict[str, str]:
    if not isinstance(record, dict):
        record = {"text": str(record)}

    output = record.get("output")
    if output is None:
        output = record.get("answer")
    if output is None:
        output = record.get("response")
    if output is None:
        output = record.get("completion")

    instruction = record.get("instruction")
    if instruction is None:
        instruction = record.get("question")
    if instruction is None:
        instruction = record.get("prompt")
    if instruction is None:
        instruction = record.get("query")

    input_text = record.get("input", "")

    if (instruction is None or str(instruction).strip() == "") and isinstance(record.get("messages"), list):
        msgs = [x for x in record["messages"] if isinstance(x, dict)]
        last_assistant = None
        user_turns: List[str] = []
        system_turns: List[str] = []
        for msg in msgs:
            role = str(msg.get("role", "")).strip().lower()
            content = str(msg.get("content", "")).strip()
            if not content:
                continue
            if role == "assistant":
                last_assistant = content
            elif role == "system":
                system_turns.append(content)
            elif role == "user":
                user_turns.append(content)
        if user_turns:
            instruction = user_turns[-1]
            if len(user_turns) > 1:
                input_text = "\n".join(user_turns[:-1]).strip()
        if system_turns:
            prefix = "\n".join(system_turns).strip()
            input_text = f"{prefix}\n{input_text}".strip() if input_text else prefix
        if output is None and last_assistant is not None:
            output = last_assistant

    if instruction is None and output is None and record.get("text") is not None:
        text_pack = _split_plain_text_to_sft(str(record.get("text", "")))
        instruction = text_pack["instruction"]
        input_text = text_pack["input"]
        output = text_pack["output"]

    instruction = str(instruction or "").strip()
    input_text = str(input_text or "").strip()
    output = str(output or "").strip()

    if not output:
        if input_text:
            output = input_text
        elif instruction:
            output = instruction

    return {
        "instruction": instruction,
        "input": input_text,
        "output": output,
        "source": str(source_name or record.get("source", "unknown")),
    }


def render_phase1_prompt_prefix(sample: Dict[str, Any]) -> str:
    instruction = str(sample.get("instruction", "")).strip()
    input_text = str(sample.get("input", "")).strip()
    if input_text:
        return (
            "### Instruction:\n"
            f"{instruction}\n\n"
            "### Input:\n"
            f"{input_text}\n\n"
            "### Response:\n"
        )
    return (
        "### Instruction:\n"
        f"{instruction}\n\n"
        "### Response:\n"
    )


def render_phase1_prompt(sample: Dict[str, Any]) -> str:
    output = str(sample.get("output", "")).strip()
    return f"{render_phase1_prompt_prefix(sample)}{output}"


def shuffled_copy(items: Iterable[Any], seed: int) -> List[Any]:
    values = list(items)
    rng = random.Random(int(seed))
    rng.shuffle(values)
    return values


def iter_chunks(items: Sequence[Any], chunk_size: int) -> Iterator[Sequence[Any]]:
    chunk = max(1, int(chunk_size))
    for idx in range(0, len(items), chunk):
        yield items[idx : idx + chunk]


def sha256_file(path: str, *, chunk_size: int = 1024 * 1024) -> str:
    fp = os.path.abspath(str(path))
    if not os.path.isfile(fp):
        raise FileNotFoundError(fp)
    h = hashlib.sha256()
    with open(fp, "rb") as f:
        while True:
            chunk = f.read(max(4096, int(chunk_size)))
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def fingerprint_dataset_path(path: str) -> str:
    target = os.path.abspath(str(path or ""))
    if not target:
        raise ValueError("Empty dataset path.")
    if os.path.isfile(target):
        return sha256_file(target)
    if os.path.isdir(target):
        files: List[str] = []
        for root, _, names in os.walk(target):
            for name in sorted(names):
                files.append(os.path.join(root, name))
        if not files:
            return hashlib.sha256(f"empty_dir::{target}".encode("utf-8")).hexdigest()
        h = hashlib.sha256()
        root_abs = os.path.abspath(target)
        for fp in files:
            rel = os.path.relpath(fp, root_abs).replace("\\", "/")
            h.update(rel.encode("utf-8"))
            h.update(b"\0")
            h.update(str(os.path.getsize(fp)).encode("utf-8"))
            h.update(b"\0")
            h.update(sha256_file(fp).encode("utf-8"))
            h.update(b"\n")
        return h.hexdigest()
    raise FileNotFoundError(target)


def load_json_if_exists(path: str) -> Dict[str, Any]:
    if not path:
        return {}
    fp = os.path.abspath(str(path))
    if not os.path.isfile(fp):
        return {}
    try:
        with open(fp, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return data
    except Exception:
        return {}
    return {}


def build_token_policy_id(
    *,
    use_last_pred: bool,
    cutoff_len: int,
    hidden_slice: str,
    window_size: int,
    window_sample_mode: str,
    window_random_pick_min: int,
    window_random_pick_max: int,
) -> str:
    token_rule = "last_pred" if bool(use_last_pred) else "last_content"
    return (
        f"{token_rule}@cut{int(cutoff_len)}"
        f"@slice={str(hidden_slice).strip()}"
        f"@win={int(window_size)}"
        f"@mode={str(window_sample_mode).strip().lower()}"
        f"@pick={int(window_random_pick_min)}-{int(window_random_pick_max)}"
    )


def load_q_meta_bundle(q_dir: str) -> Dict[str, Any]:
    """Load best-effort geometry metadata from a Phase-1 output directory."""
    root = os.path.abspath(str(q_dir or ""))
    if not root or not os.path.isdir(root):
        raise FileNotFoundError(f"q_dir not found: {q_dir}")

    out: Dict[str, Any] = {"q_dir": root}
    summary = load_json_if_exists(os.path.join(root, "summary.json"))
    final_meta = load_json_if_exists(os.path.join(root, "final_q_meta.json"))

    tau_meta: Dict[str, Any] = {}
    tau_path = os.path.join(root, "tau_state.pt")
    if os.path.isfile(tau_path):
        try:
            import torch  # local import to avoid hard dependency for non-torch scripts

            tau_obj = torch.load(tau_path, map_location="cpu")
            if isinstance(tau_obj, dict) and isinstance(tau_obj.get("meta"), dict):
                tau_meta = dict(tau_obj.get("meta"))
        except Exception:
            tau_meta = {}

    out["summary"] = summary
    out["final_q_meta"] = final_meta
    out["tau_meta"] = tau_meta
    merged: Dict[str, Any] = {}
    for src in (summary, tau_meta, final_meta):
        if isinstance(src, dict):
            merged.update(src)
    out["merged"] = merged
    return out
