import torch


def get_gpu_status():
    return {
        "cuda": torch.cuda.is_available(),
        "version": torch.version.cuda,
        "device_count": torch.cuda.device_count(),
        "current": torch.cuda.current_device(),
        "name": [torch.cuda.get_device_name(name) for name in range(torch.cuda.device_count())]
    }


if __name__ == '__main__':
    stat = get_gpu_status()
    print(stat)
