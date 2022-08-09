import torch


def get_device_and_num_workers():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda":
        num_workers = 1
        pin_memory = True
    else:
        num_workers = 0
        pin_memory = False

    return device, num_workers, pin_memory

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
