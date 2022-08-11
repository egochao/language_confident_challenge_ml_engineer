import torch

from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl

# your favorite machine learning tracking tool
from pytorch_lightning.loggers import WandbLogger

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import random_split, DataLoader

from torchmetrics import Accuracy

from torchvision import transforms
from torchvision.datasets import CIFAR10

import wandb


class ImagePredictionLogger(pl.callbacks.Callback):
    def __init__(self, val_samples, num_samples=32):
        super().__init__()
        self.num_samples = num_samples
        self.val_imgs, self.val_labels = val_samples

    def on_validation_epoch_end(self, trainer, pl_module):
        # Bring the tensors to CPU
        val_imgs = self.val_imgs.to(device=pl_module.device)
        val_labels = self.val_labels.to(device=pl_module.device)
        # Get model prediction
        logits = pl_module(val_imgs)
        preds = torch.argmax(logits, -1)
        # Log the images as wandb Image
        trainer.logger.experiment.log(
            {
                "examples": [
                    wandb.Image(x, caption=f"Pred:{pred}, Label:{y}")
                    for x, pred, y in zip(
                        val_imgs[: self.num_samples],
                        preds[: self.num_samples],
                        val_labels[: self.num_samples],
                    )
                ]
            }
        )


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
