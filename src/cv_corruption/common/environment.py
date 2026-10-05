import torch


def device(cpu=False, gpu_id=0):
    if cpu:
        return torch.device("cpu")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; use --cpu")
    return torch.device(f"cuda:{gpu_id}")
