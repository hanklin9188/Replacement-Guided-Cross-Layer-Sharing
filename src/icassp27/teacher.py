from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from torch.utils.data import DataLoader

from .config import model_config, run_dir
from .data import RecoveryDataset, causal_collator, load_task_rows
from .evaluation import evaluate_rows
from .modeling import load_reference, load_tokenizer
from .utils import atomic_json, base_manifest, set_seed


def load_teacher(cfg: dict[str, Any], backbone: str):
    mcfg = model_config(cfg, backbone)
    base = load_reference(mcfg)
    adapter_dir = run_dir(cfg, backbone, "teacher", "adapter")
    teacher = PeftModel.from_pretrained(base, adapter_dir)
    return teacher.merge_and_unload().eval()


def train_teacher(cfg: dict[str, Any], backbone: str) -> None:
    seed = int(cfg["project"]["seed"])
    set_seed(seed)
    mcfg = model_config(cfg, backbone)
    tcfg = cfg["teacher"]
    output = run_dir(cfg, backbone, "teacher")
    output.mkdir(parents=True, exist_ok=True)
    tokenizer = load_tokenizer(mcfg)
    model = load_reference(mcfg, training=True)
    lora = LoraConfig(
        r=int(tcfg["rank"]), lora_alpha=int(tcfg["alpha"]),
        lora_dropout=float(tcfg["dropout"]), bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora)
    data_dir = run_dir(cfg, backbone, "data")
    train_rows = load_task_rows(data_dir, "recovery", cfg["data"]["tasks"])
    val_rows = load_task_rows(data_dir, "validation", cfg["data"]["tasks"])
    dataset = RecoveryDataset(train_rows, tokenizer, int(tcfg["max_length"]))
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(dataset, batch_size=int(tcfg["batch_size"]), shuffle=True,
                        generator=generator, collate_fn=causal_collator(tokenizer), num_workers=2)
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=float(tcfg["learning_rate"]))
    accumulation = int(tcfg["gradient_accumulation"])
    history = []
    best = (-float("inf"), float("inf"))
    best_epoch = -1
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(int(tcfg["epochs"])):
        model.train()
        loss_sum = 0.0
        updates = 0
        for step, batch in enumerate(loader):
            inputs = {key: value.cuda() for key, value in batch.items() if torch.is_tensor(value)}
            loss = model(**inputs, use_cache=False).loss / accumulation
            loss.backward()
            loss_sum += float(loss.item()) * accumulation
            if (step + 1) % accumulation == 0 or step + 1 == len(loader):
                torch.nn.utils.clip_grad_norm_(parameters, 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                updates += 1
        metrics = evaluate_rows(model, tokenizer, val_rows, int(tcfg["max_length"]))
        record = {"epoch": epoch, "train_loss": loss_sum / max(len(loader), 1), "updates": updates, **metrics}
        history.append(record)
        score = (metrics["macro_accuracy"], -record["train_loss"])
        if score > best:
            best = score
            best_epoch = epoch
            model.save_pretrained(output / "adapter")
            tokenizer.save_pretrained(output / "adapter")
    manifest = base_manifest(cfg, "teacher_training", backbone)
    manifest.update({"role": "M_T", "source_role": "M_P", "method": "LoRA",
                     "model_id": mcfg["model_id"], "model_revision": mcfg["revision"],
                     "train_sample_ids": [row["id"] for row in train_rows],
                     "validation_sample_ids": [row["id"] for row in val_rows],
                     "best_epoch": best_epoch, "history": history,
                     "checkpoint_rule": "highest validation macro accuracy; training loss tie-break",
                     "final_evaluation_used": False})
    atomic_json(output / "teacher_manifest.json", manifest)
