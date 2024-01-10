import torch

stat = {
    "cuda": torch.cuda.is_available(),
    "version": torch.version.cuda,
    "device_count": torch.cuda.device_count(),
    "current": torch.cuda.current_device(),
    "name": [torch.cuda.get_device_name(name) for name in range(torch.cuda.device_count())]
}

print(stat)
