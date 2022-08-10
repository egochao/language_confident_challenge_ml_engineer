import pytorch_lightning as pl
import torch
import torch.nn
import torchmetrics

from torch.nn import functional as F
import torch
from torch import nn

from argparse import ArgumentParser

import constants



class LitClassifier(pl.LightningModule):
    def __init__(self, backbone, learning_rate=constants.LEARNING_RATE):
        super().__init__()
        self.save_hyperparameters()
        self.backbone = backbone
        self.learning_rate = learning_rate
        self.val_acc = torchmetrics.Accuracy()
        self.test_acc = torchmetrics.Accuracy()

    def forward(self, x):
        embedding = self.backbone(x)
        return embedding

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.backbone(x)
        loss = F.cross_entropy(y_hat, y)
        self.log("train_loss", loss, on_epoch=True, on_step=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.backbone(x)
        loss = F.cross_entropy(y_hat, y)
        self.val_acc(y_hat.softmax(dim=-1), y)
        metrics = {"val_acc": self.val_acc, "val_loss": loss}
        self.log_dict(metrics, on_epoch=True, on_step=False)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.backbone(x)
        loss = F.cross_entropy(y_hat, y)
        self.test_acc(y_hat.softmax(dim=-1), y)
        self.log("test_acc", self.test_acc, on_epoch=True, on_step=False)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=0.0001)
        return optimizer

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--learning_rate", type=float, default=0.0001)
        return parser
