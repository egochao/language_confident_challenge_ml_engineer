import pytorch_lightning as pl
import torch
import torch.nn
from torchmetrics import Accuracy

import torch
import constants


class BaseTorchLightlingWrapper(pl.LightningModule):
    def __init__(self, core_model, loss_fn, label_converter=None, learning_rate=None):
        super().__init__()

        # log hyperparameters
        self.save_hyperparameters()
        self.learning_rate = learning_rate or constants.LEARNING_RATE
        self.core_model = core_model
        self.loss_fn = loss_fn
        self.acc = Accuracy()
        self.label_converter = label_converter

    # will be used during inference
    def forward(self, x):
        embedding = self.core_model(x)
        return embedding

    def _metric(self, preds, y):
        if self.label_converter is not None:
            y = self.label_converter(y)
        return self.acc(preds, y)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)

        # training metrics
        preds = torch.argmax(logits, dim=1)
        acc = self._metric(preds, y)
        self.log("train_loss", loss, on_step=True, on_epoch=True, logger=True)
        self.log("train_acc", acc, on_step=True, on_epoch=True, logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)

        # validation metrics
        preds = torch.argmax(logits, dim=1)
        acc = self._metric(preds, y)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)

        # validation metrics
        preds = torch.argmax(logits, dim=1)
        acc = self._metric(preds, y)
        self.log("test_loss", loss, prog_bar=True)
        self.log("test_acc", acc, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.learning_rate, weight_decay=0.0001
        )
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=10, gamma=0.1
        )  # reduce the learning after 20 epochs by a factor of 10
        return [optimizer], [scheduler]
