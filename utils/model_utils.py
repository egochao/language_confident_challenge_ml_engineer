import torch


def get_loader_params():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda":
        num_workers = 2
        pin_memory = True
    else:
        num_workers = 0
        pin_memory = False

    return num_workers, pin_memory

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
