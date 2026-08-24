from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterable

import torch


ANSWER_FORMAT = re.compile(r"Answer format:\s*([^\n]+)", re.IGNORECASE)


def _normalize_answer(value: Any) -> str:
    return re.sub(r"\s+", "", str(value).strip().lower())


def _normalize_row(row: dict[str, Any], index: int, namespace: str = "") -> dict[str, Any]:
    if {"prompt", "choices", "label"} <= row.keys():
        prompt = str(row["prompt"])
        choices = [str(value) for value in row["choices"]]
        label = int(row["label"])
    else:
        instruction = str(row.get("instruction", "")).strip()
        extra = str(row.get("input", "")).strip()
        match = ANSWER_FORMAT.search(instruction)
        if match is None:
            raise ValueError(f"row {index} has no choices and no 'Answer format:' declaration")
        choices = [value.strip() for value in match.group(1).split("/") if value.strip()]
        if len(choices) < 2:
            raise ValueError(f"row {index} exposes fewer than two choices: {choices}")
        answer = _normalize_answer(row.get("answer", row.get("output", "")))
        matches = [choice_index for choice_index, choice in enumerate(choices)
                   if _normalize_answer(choice) == answer or _normalize_answer(choice) in answer]
        if len(matches) != 1:
            raise ValueError(f"row {index} answer {answer!r} does not identify one choice {choices}")
        label = matches[0]
        prompt = instruction
        if extra:
            prompt += "\n\n" + extra
        prompt += "\nAnswer:"
    if not 0 <= label < len(choices):
        raise ValueError(f"row {index} label {label} is outside {len(choices)} choices")
    identifier = row.get("id")
    if identifier is None:
        fingerprint = hashlib.sha256(
            json.dumps([prompt, choices, label], ensure_ascii=False, separators=(",", ":")).encode()
        ).hexdigest()[:24]
        # The locked commonsense source contains a small number of duplicate
        # records and did not store IDs. Preserve every record with a stable,
        # split-local positional suffix instead of silently deduplicating it.
        identifier = f"{namespace}:{index:06d}:{fingerprint}"
    return {"id": str(identifier), "prompt": prompt, "choices": choices, "label": label,
            "task": str(row.get("task", "recovery")), "source_index": index}


def load_recovery_rows(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    value = json.loads(source.read_text())
    if isinstance(value, dict):
        for key in ("data", "rows", "examples", "train", "validation"):
            if isinstance(value.get(key), list):
                value = value[key]
                break
    if not isinstance(value, list):
        raise TypeError(f"recovery file {source} must contain a JSON list")
    rows = [_normalize_row(row, index, source.stem) for index, row in enumerate(value)]
    ids = [row["id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError(f"recovery file {source} contains duplicate sample IDs")
    return rows


def load_evaluation_rows(root: str | Path, tasks: Iterable[str], split: str = "final") -> list[dict[str, Any]]:
    rows = []
    for task in tasks:
        source = Path(root) / f"{task}.{split}.jsonl"
        if not source.is_file():
            raise FileNotFoundError(source)
        with source.open() as handle:
            for index, line in enumerate(handle):
                if line.strip():
                    row = _normalize_row(json.loads(line), index, f"{task}.{split}")
                    row["task"] = task
                    rows.append(row)
    ids = [row["id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("evaluation data contain duplicate IDs across tasks")
    return rows


def sample_id_hash(rows: list[dict[str, Any]]) -> str:
    return hashlib.sha256("\n".join(row["id"] for row in rows).encode()).hexdigest()


def deterministic_indices(length: int, start: int, count: int, seed: int) -> list[int]:
    if length <= 0:
        raise ValueError("cannot sample an empty recovery dataset")
    result = []
    position = int(start)
    while len(result) < count:
        epoch, offset = divmod(position, length)
        generator = torch.Generator().manual_seed(int(seed) + epoch)
        permutation = torch.randperm(length, generator=generator).tolist()
        take = min(count - len(result), length - offset)
        result.extend(permutation[offset:offset + take])
        position += take
    return result


def encode_multiple_choice(rows: list[dict[str, Any]], tokenizer, max_length: int, device: torch.device):
    sequences: list[list[int]] = []
    choice_starts: list[int] = []
    row_spans: list[tuple[int, int]] = []
    gold_flat_indices: list[int] = []
    for row in rows:
        start = len(sequences)
        prompt_ids = tokenizer(row["prompt"], add_special_tokens=True)["input_ids"]
        for choice in row["choices"]:
            choice_ids = tokenizer(" " + choice, add_special_tokens=False)["input_ids"]
            if not choice_ids:
                raise ValueError(f"choice tokenized to empty text: {choice!r}")
            choice_ids = choice_ids[:max(1, max_length - 1)]
            prefix = prompt_ids[-max(1, max_length - len(choice_ids)):]
            sequences.append(prefix + choice_ids)
            choice_starts.append(len(prefix))
        end = len(sequences)
        row_spans.append((start, end))
        gold_flat_indices.append(start + int(row["label"]))
    width = max(map(len, sequences))
    pad = int(tokenizer.pad_token_id)
    input_ids = torch.full((len(sequences), width), pad, dtype=torch.long, device=device)
    attention_mask = torch.zeros_like(input_ids)
    for index, sequence in enumerate(sequences):
        values = torch.tensor(sequence, dtype=torch.long, device=device)
        input_ids[index, :values.numel()] = values
        attention_mask[index, :values.numel()] = 1
    return {
        "input_ids": input_ids, "attention_mask": attention_mask,
        "choice_starts": choice_starts, "row_spans": row_spans,
        "gold_flat_indices": torch.tensor(gold_flat_indices, dtype=torch.long, device=device),
    }
