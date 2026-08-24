from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from .utils import atomic_json, base_manifest, read_jsonl


@torch.inference_mode()
def choice_score(model, tokenizer, prompt: str, choice: str, max_length: int) -> float:
    prompt_ids = tokenizer(prompt, add_special_tokens=True)["input_ids"]
    choice_ids = tokenizer(" " + choice, add_special_tokens=False)["input_ids"]
    room = max_length - len(choice_ids)
    prompt_ids = prompt_ids[-max(room, 1):]
    ids = prompt_ids + choice_ids
    input_ids = torch.tensor(ids, device="cuda", dtype=torch.long).unsqueeze(0)
    logits = model(input_ids=input_ids, use_cache=False).logits[0].float()
    start = len(prompt_ids) - 1
    prediction = logits[start : start + len(choice_ids)]
    targets = torch.tensor(choice_ids, device="cuda", dtype=torch.long)
    return float(-F.cross_entropy(prediction, targets, reduction="sum").item())


@torch.inference_mode()
def evaluate_rows(model, tokenizer, rows: list[dict[str, Any]], max_length: int, predictions_path: str | Path | None = None):
    model.eval()
    predictions = []
    by_task: dict[str, list[int]] = defaultdict(list)
    for row in rows:
        scores = [choice_score(model, tokenizer, row["prompt"], choice, max_length) for choice in row["choices"]]
        predicted = int(np.argmax(scores))
        correct = int(predicted == int(row["label"]))
        by_task[row["task"]].append(correct)
        predictions.append({"id": row["id"], "task": row["task"], "label": int(row["label"]),
                            "prediction": predicted, "correct": correct, "choice_scores": scores})
    per_task = {task: float(np.mean(values)) for task, values in by_task.items()}
    macro = float(np.mean(list(per_task.values()))) if per_task else float("nan")
    weighted = float(np.mean([row["correct"] for row in predictions])) if predictions else float("nan")
    metrics = {"macro_accuracy": macro, "weighted_accuracy": weighted,
               "per_task_accuracy": per_task, "num_examples": len(predictions)}
    if predictions_path:
        from .utils import write_jsonl
        write_jsonl(predictions_path, predictions)
    return metrics


@torch.inference_mode()
def heldout_nll(model, blocks: list[torch.Tensor], max_tokens: int) -> dict[str, float | int]:
    model.eval()
    nll_sum = 0.0
    count = 0
    for block in blocks:
        if count >= max_tokens:
            break
        take = min(len(block) - 1, max_tokens - count)
        input_ids = block[:take].unsqueeze(0).cuda()
        labels = block[1 : take + 1].cuda()
        logits = model(input_ids=input_ids, use_cache=False).logits[0].float()
        nll_sum += float(F.cross_entropy(logits, labels, reduction="sum").item())
        count += take
    nll = nll_sum / count
    return {"nll": nll, "perplexity": math.exp(min(nll, 50)), "valid_tokens": count}


def evaluate_split(cfg, backbone: str, role: str, model, tokenizer, data_dir: Path, output: Path, split: str):
    rows = []
    for task in cfg["data"]["tasks"]:
        rows.extend(read_jsonl(data_dir / f"{task}.{split}.jsonl"))
    output.mkdir(parents=True, exist_ok=True)
    metrics = evaluate_rows(model, tokenizer, rows, int(cfg["recovery"]["max_length"]), output / "predictions.jsonl")
    manifest = base_manifest(cfg, "evaluation", backbone)
    manifest.update({"role": role, "split": split, "metrics": metrics,
                     "sample_ids": [row["id"] for row in rows]})
    atomic_json(output / "metrics.json", manifest)
    return metrics
