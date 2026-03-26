from __future__ import annotations

import torch


def select_device(preferred_cuda_index: int = 0) -> torch.device:
    if torch.cuda.is_available():
        if torch.cuda.device_count() > preferred_cuda_index:
            return torch.device(f"cuda:{preferred_cuda_index}")
        return torch.device("cuda:0")
    return torch.device("cpu")
