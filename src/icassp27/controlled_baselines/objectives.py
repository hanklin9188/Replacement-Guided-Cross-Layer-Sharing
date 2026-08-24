from __future__ import annotations

import torch
import torch.nn.functional as F


def candidate_scores(logits: torch.Tensor, encoded: dict) -> torch.Tensor:
    log_probabilities = F.log_softmax(logits.float(), dim=-1)
    values = []
    input_ids = encoded["input_ids"]
    attention = encoded["attention_mask"]
    for index, start in enumerate(encoded["choice_starts"]):
        end = int(attention[index].sum().item())
        targets = input_ids[index, start:end]
        positions = torch.arange(start - 1, end - 1, device=logits.device)
        values.append(log_probabilities[index, positions, targets].sum())
    return torch.stack(values)


def decision_cross_entropy(scores: torch.Tensor, rows: list[dict], row_spans: list[tuple[int, int]]) -> torch.Tensor:
    losses = []
    for row, (start, end) in zip(rows, row_spans):
        target = torch.tensor([int(row["label"])], device=scores.device)
        losses.append(F.cross_entropy(scores[start:end].unsqueeze(0), target))
    return torch.stack(losses).mean()


def full_vocabulary_kl(student_logits: torch.Tensor, teacher_logits: torch.Tensor,
                       gold_input_ids: torch.Tensor, gold_attention_mask: torch.Tensor,
                       *, temperature: float, eos_token_id: int | None, exclude_eos: bool) -> torch.Tensor:
    if student_logits.shape != teacher_logits.shape:
        raise ValueError(f"teacher/student logit shapes differ: {teacher_logits.shape} vs {student_logits.shape}")
    mask = gold_attention_mask[:, 1:].bool()
    if exclude_eos and eos_token_id is not None:
        mask &= gold_input_ids[:, 1:].ne(int(eos_token_id))
    if not bool(mask.any()):
        raise ValueError("KD mask contains no shifted non-padding tokens")
    temperature = float(temperature)
    student_log_probability = F.log_softmax(student_logits[:, :-1].float() / temperature, dim=-1)
    teacher_probability = F.softmax(teacher_logits[:, :-1].float() / temperature, dim=-1)
    per_token = F.kl_div(student_log_probability, teacher_probability, reduction="none").sum(dim=-1)
    return per_token.masked_select(mask).mean() * temperature * temperature


def controlled_loss(student, teacher, rows: list[dict], encoded: dict, *, objective: str,
                    temperature: float, lambda_ce: float, lambda_kd: float,
                    eos_token_id: int | None, exclude_eos: bool):
    student_output = student(input_ids=encoded["input_ids"], attention_mask=encoded["attention_mask"],
                             use_cache=False)
    scores = candidate_scores(student_output.logits, encoded)
    ce = decision_cross_entropy(scores, rows, encoded["row_spans"])
    kd = torch.zeros((), device=ce.device, dtype=ce.dtype)
    if objective == "ce":
        if teacher is not None:
            raise RuntimeError("CE-only invariant violated: teacher model was loaded")
    elif objective == "ce_kd":
        if teacher is None:
            raise RuntimeError("CE+KD requires a frozen teacher")
        gold = encoded["gold_flat_indices"]
        gold_ids = encoded["input_ids"].index_select(0, gold)
        gold_attention = encoded["attention_mask"].index_select(0, gold)
        with torch.inference_mode():
            teacher_logits = teacher(input_ids=gold_ids, attention_mask=gold_attention,
                                     use_cache=False).logits
        student_gold_logits = student_output.logits.index_select(0, gold)
        kd = full_vocabulary_kl(student_gold_logits, teacher_logits, gold_ids, gold_attention,
                                temperature=temperature, eos_token_id=eos_token_id,
                                exclude_eos=exclude_eos)
    else:
        raise ValueError(objective)
    return float(lambda_ce) * ce + (float(lambda_kd) * kd if objective == "ce_kd" else 0.0), ce, kd, scores
