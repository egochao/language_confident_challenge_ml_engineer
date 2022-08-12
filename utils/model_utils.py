from torch import nn
import torch.nn.functional as F


def student_loss_fn(student_preds, labels):
    """Simple loss for student model."""
    return F.nll_loss(F.log_softmax(student_preds, dim=1), labels)


def kd_loss_fn(student_preds, teacher_preds, T):
    """Kullback-Leibler divergence loss between teacher and student predictions."""
    loss = nn.KLDivLoss(reduction="batchmean", log_target=True)

    return (
        loss(F.log_softmax(student_preds / T, dim=1), 
        F.log_softmax(teacher_preds.float() / T, dim=1)) * T * T
    )


def distillation_loss(student_preds, teacher_preds, labels, alpha=0.5, T=10):
    """Weighted sum of student and teacher losses."""
    student_loss = student_loss_fn(student_preds, labels)
    # return student_loss
    kd_loss = kd_loss_fn(student_preds, teacher_preds, T)

    return kd_loss * alpha + student_loss * (1 - alpha)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
