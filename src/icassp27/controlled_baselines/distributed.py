from __future__ import annotations

import os
import random
from typing import Any

import numpy as np
import torch
import torch.distributed as dist


def initialize() -> tuple[int, int, torch.device]:
    world = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if not torch.cuda.is_available():
        raise RuntimeError("controlled baseline execution requires CUDA/H200")
    torch.cuda.set_device(local_rank)
    if world > 1 and not dist.is_initialized():
        dist.init_process_group("nccl", device_id=torch.device("cuda", local_rank))
    rank = dist.get_rank() if dist.is_initialized() else 0
    return rank, world, torch.device("cuda", local_rank)


def barrier() -> None:
    if dist.is_initialized():
        dist.barrier()


def is_main() -> bool:
    return not dist.is_initialized() or dist.get_rank() == 0


def all_gather_objects(value: Any) -> list[Any]:
    if not dist.is_initialized():
        return [value]
    values: list[Any] = [None] * dist.get_world_size()
    dist.all_gather_object(values, value)
    return values


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def close() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()
