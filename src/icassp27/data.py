from __future__ import annotations

import itertools
from pathlib import Path
from typing import Any

import torch
from datasets import load_dataset
from transformers import AutoTokenizer

from .config import model_config, run_dir
from .utils import atomic_json, base_manifest, stable_bucket, write_jsonl


DATASET_SPECS = {
    "piqa": ("ybisk/piqa", None, True),
    "social_i_qa": ("allenai/social_i_qa", None, True),
    "winogrande": ("allenai/winogrande", "winogrande_xl", True),
    "arc_challenge": ("allenai/ai2_arc", "ARC-Challenge", True),
    "arc_easy": ("allenai/ai2_arc", "ARC-Easy", True),
    "hellaswag": ("Rowan/hellaswag", None, True),
    "openbookqa": ("allenai/openbookqa", "main", True),
}


def _load(name: str, split: str):
    dataset, subset, trust_remote_code = DATASET_SPECS[name]
    kwargs = {"path": dataset, "split": split, "trust_remote_code": trust_remote_code}
    if subset:
        kwargs["name"] = subset
    return load_dataset(**kwargs)


def normalize_example(task: str, row: dict[str, Any], row_id: str) -> dict[str, Any]:
    if task == "piqa":
        prompt = f"Question: {row['goal']}\nChoose the most sensible solution.\nAnswer:"
        choices = [row["sol1"], row["sol2"]]
        label = int(row["label"])
    elif task == "social_i_qa":
        prompt = f"Context: {row['context']}\nQuestion: {row['question']}\nAnswer:"
        choices = [row["answerA"], row["answerB"], row["answerC"]]
        label = int(row["label"]) - 1
    elif task == "winogrande":
        prompt = f"Complete the sentence: {row['sentence']}\nAnswer:"
        choices = [row["option1"], row["option2"]]
        label = int(row["answer"]) - 1
    elif task in {"arc_challenge", "arc_easy", "openbookqa"}:
        question = row.get("question_stem", row.get("question"))
        if question is None:
            raise KeyError(f"No question field in {task}: {sorted(row)}")
        prompt = f"Question: {question}\nAnswer:"
        labels = [str(x) for x in row["choices"]["label"]]
        choices = [str(x) for x in row["choices"]["text"]]
        answer_key = str(row["answerKey"])
        label = labels.index(answer_key)
    elif task == "hellaswag":
        context = (str(row.get("ctx_a", "")) + " " + str(row.get("ctx_b", ""))).strip()
        if not context:
            context = str(row.get("ctx", ""))
        prompt = f"Complete the situation: {context}\nContinuation:"
        choices = [str(x) for x in row["endings"]]
        label = int(row["label"])
    else:
        raise KeyError(task)
    return {"id": row_id, "task": task, "prompt": prompt, "choices": choices, "label": label}


def _task_splits(task: str, seed: int, recovery_fraction: float, validation_fraction: float):
    train = _load(task, "train")
    try:
        final = _load(task, "validation")
    except ValueError:
        final = _load(task, "test")
    recovery, validation = [], []
    cutoff = int(recovery_fraction * 10_000)
    val_cutoff = min(10_000, cutoff + int(validation_fraction * 10_000))
    for idx, row in enumerate(train):
        item = normalize_example(task, row, f"{task}:train:{idx}")
        bucket = stable_bucket(item["id"], seed)
        if bucket < cutoff:
            recovery.append(item)
        elif bucket < val_cutoff:
            validation.append(item)
    final_rows = [normalize_example(task, row, f"{task}:final:{idx}") for idx, row in enumerate(final)]
    return recovery, validation, final_rows


def prepare_data(cfg: dict[str, Any], backbone: str) -> None:
    model = model_config(cfg, backbone)
    output = run_dir(cfg, backbone, "data")
    output.mkdir(parents=True, exist_ok=True)
    tokenizer_source = model.get("model_path", model["model_id"])
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_source, revision=model["tokenizer_revision"], use_fast=True,
        local_files_only=bool(model.get("model_path")),
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    cal_cfg = cfg["data"]["calibration"]
    raw = load_dataset(cal_cfg["dataset"], cal_cfg["subset"], split=cal_cfg["split"])
    block = int(cal_cfg["block_length"])
    max_tokens = max(int(v) for v in cal_cfg["token_counts"])
    token_ids: list[int] = []
    sample_ids: list[str] = []
    for idx, text in enumerate(raw["text"]):
        if not text.strip():
            continue
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        token_ids.extend(ids + [tokenizer.eos_token_id])
        sample_ids.append(f"{cal_cfg['dataset']}:{cal_cfg['split']}:{idx}")
        if len(token_ids) >= max_tokens + 1:
            break
    blocks = []
    for start in range(0, min(len(token_ids) - 1, max_tokens), block):
        segment = token_ids[start : start + block + 1]
        if len(segment) < 2:
            break
        blocks.append(torch.tensor(segment, dtype=torch.long))
    torch.save(blocks, output / "calibration_blocks.pt")

    held_cfg = cfg["data"]["heldout_nll"]
    held_raw = load_dataset(held_cfg["dataset"], held_cfg["subset"], split=held_cfg["split"])
    held_ids: list[int] = []
    held_samples: list[str] = []
    held_limit = int(held_cfg["valid_tokens"])
    for idx, text in enumerate(held_raw["text"]):
        if not text.strip():
            continue
        held_ids.extend(tokenizer(text, add_special_tokens=False)["input_ids"] + [tokenizer.eos_token_id])
        held_samples.append(f"{held_cfg['dataset']}:{held_cfg['split']}:{idx}")
        if len(held_ids) >= held_limit + 1:
            break
    held_blocks = []
    for start in range(0, min(len(held_ids) - 1, held_limit), block):
        segment = held_ids[start : start + block + 1]
        if len(segment) >= 2:
            held_blocks.append(torch.tensor(segment, dtype=torch.long))
    torch.save(held_blocks, output / "heldout_nll_blocks.pt")

    task_manifest = {}
    for task in cfg["data"]["tasks"]:
        recovery, validation, final_rows = _task_splits(
            task, int(cfg["project"]["seed"]),
            float(cfg["data"]["recovery_fraction"]),
            float(cfg["data"]["validation_fraction"]),
        )
        recovery = recovery[: int(cfg["data"]["max_recovery_examples_per_task"])]
        validation = validation[: int(cfg["data"]["max_validation_examples_per_task"])]
        counts = {
            "recovery": write_jsonl(output / f"{task}.recovery.jsonl", recovery),
            "validation": write_jsonl(output / f"{task}.validation.jsonl", validation),
            "final": write_jsonl(output / f"{task}.final.jsonl", final_rows),
        }
        task_manifest[task] = counts

    manifest = base_manifest(cfg, "data", backbone)
    manifest.update({
        "model_id": model["model_id"],
        "model_path": model.get("model_path"),
        "model_revision": model["revision"],
        "tokenizer_revision": model["tokenizer_revision"],
        "tokenizer_class": tokenizer.__class__.__name__,
        "calibration": {
            **cal_cfg,
            "source_sample_ids": sample_ids,
            "num_sequences": len(blocks),
            "stored_token_count": sum(len(x) - 1 for x in blocks),
            "file": "calibration_blocks.pt",
        },
        "task_splits": task_manifest,
        "heldout_nll": {**held_cfg, "sample_ids": held_samples,
                        "stored_token_count": sum(len(x) - 1 for x in held_blocks),
                        "file": "heldout_nll_blocks.pt"},
        "disjointness": "calibration=wikitext train; recovery/validation=hash-disjoint task train; final=official labelled validation/test",
    })
    atomic_json(output / "data_manifest.json", manifest)


def smoke_data(cfg: dict[str, Any], backbone: str) -> None:
    model = model_config(cfg, backbone)
    tokenizer_source = model.get("model_path", model["model_id"])
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_source, revision=model["tokenizer_revision"], use_fast=True,
        local_files_only=bool(model.get("model_path")),
    )
    checks = []
    for task in cfg["data"]["tasks"]:
        train = _load(task, "train")
        try:
            final = _load(task, "validation")
            final_split = "validation"
        except ValueError:
            final = _load(task, "test")
            final_split = "test"
        for split, dataset in [("train", train), (final_split, final)]:
            item = normalize_example(task, dataset[0], f"{task}:{split}:0")
            if not 0 <= int(item["label"]) < len(item["choices"]):
                raise ValueError(f"Invalid label for {task}/{split}: {item}")
            encoded = tokenizer(item["prompt"] + " " + item["choices"][item["label"]], add_special_tokens=True)
            if not encoded["input_ids"]:
                raise ValueError(f"Tokenizer returned no tokens for {task}/{split}")
            checks.append({"task": task, "split": split, "id": item["id"],
                           "choice_count": len(item["choices"]), "token_count": len(encoded["input_ids"])})
    cal_cfg = cfg["data"]["calibration"]
    calibration = load_dataset(cal_cfg["dataset"], cal_cfg["subset"], split=f"{cal_cfg['split']}[:2]")
    if len(calibration) != 2:
        raise RuntimeError("Calibration dataset smoke did not return two rows")
    output = run_dir(cfg, backbone, "data")
    output.mkdir(parents=True, exist_ok=True)
    manifest = base_manifest(cfg, "data_smoke", backbone)
    manifest.update({"passed": True, "checks": checks, "calibration_rows": len(calibration),
                     "model_path": model.get("model_path"), "tokenizer_revision": model["tokenizer_revision"]})
    atomic_json(output / "DATA_SMOKE_SUCCESS.json", manifest)


class RecoveryDataset(torch.utils.data.Dataset):
    def __init__(self, rows: list[dict[str, Any]], tokenizer, max_length: int):
        self.rows = rows
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        answer = row["choices"][int(row["label"])]
        prompt_ids = self.tokenizer(row["prompt"], add_special_tokens=True)["input_ids"]
        answer_ids = self.tokenizer(" " + answer, add_special_tokens=False)["input_ids"] + [self.tokenizer.eos_token_id]
        ids = (prompt_ids + answer_ids)[-self.max_length :]
        prompt_kept = max(0, len(ids) - len(answer_ids))
        labels = [-100] * prompt_kept + ids[prompt_kept:]
        return {"input_ids": ids, "labels": labels, "id": row["id"], "task": row["task"]}


def causal_collator(tokenizer):
    def collate(rows: list[dict[str, Any]]) -> dict[str, Any]:
        max_len = max(len(row["input_ids"]) for row in rows)
        input_ids, attention_mask, labels = [], [], []
        for row in rows:
            pad = max_len - len(row["input_ids"])
            input_ids.append([tokenizer.pad_token_id] * pad + row["input_ids"])
            attention_mask.append([0] * pad + [1] * len(row["input_ids"]))
            labels.append([-100] * pad + row["labels"])
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "ids": [row["id"] for row in rows],
            "tasks": [row["task"] for row in rows],
        }
    return collate


def load_task_rows(data_dir: str | Path, split: str, tasks: list[str]) -> list[dict[str, Any]]:
    import json
    rows = []
    for task in tasks:
        with (Path(data_dir) / f"{task}.{split}.jsonl").open(encoding="utf-8") as handle:
            rows.extend(json.loads(line) for line in handle if line.strip())
    return rows
