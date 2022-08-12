from torch import nn
import torch.nn.functional as F


def student_loss(student_preds, labels):
    return F.cross_entropy(student_preds, labels)


def kd_loss(student_preds, teacher_preds, T):
    loss = nn.KLDivLoss(reduction="batchmean", log_target=True)

    return (
        loss(
            F.log_softmax(student_preds / T, dim=1),
            F.log_softmax(teacher_preds / T, dim=1),
        )
        * T
        * T
    )


def distillation_loss(student_preds, teacher_preds, labels, alpha, T):
    student_loss = student_loss(student_preds, labels)
    kd_loss = kd_loss(student_preds, teacher_preds, T)

    return kd_loss * alpha + student_loss * (1 - alpha)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
