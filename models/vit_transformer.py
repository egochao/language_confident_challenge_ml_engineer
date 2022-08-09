# Installing libraries
# Importing
# Weights & Biases
import pytorch_lightning as pl
# Pytorch modules
import torch
import torch.nn
import torchmetrics

from torch import nn
from torch.nn import functional as F
from transformers import ViTConfig
from vit_pytorch import ViT
from vit_pytorch.vit import Transformer, pair
import torch
from torch import nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

from argparse import ArgumentParser

class VisionTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = ViT(
            image_size = 128,
            patch_size = 16,
            num_classes = 35,
            dim = 768,
            depth = 6,
            heads = 16,
            mlp_dim = 2048,
            channels=1,
            dropout = 0.0,
            emb_dropout = 0.0
        )
    
    @torch.no_grad()
    def init_weights(self):
        def _init(m):
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.normal_(m.bias, std=1e-6)
            
        self.apply(_init)
        nn.init.constant_(self.model.fc.weight, 0)
        nn.init.constant_(self.model.fc.bias, 0)
    
    def forward(self, x):
        return self.model(x)


class LitClassifier(pl.LightningModule):
    def __init__(self, backbone, learning_rate=1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.backbone = backbone
        self.val_acc = torchmetrics.Accuracy()
        self.test_acc = torchmetrics.Accuracy()

    def forward(self, x):
        # use forward for inference/predictions
        embedding = self.backbone(x)
        return embedding

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.backbone(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('train_loss', loss, on_epoch=True, on_step=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.backbone(x)
        loss = F.cross_entropy(y_hat, y)
        self.val_acc(y_hat.softmax(dim=-1), y)
        metrics = {'val_acc': self.val_acc, 'val_loss': loss}
        self.log_dict(metrics, on_epoch=True, on_step=False)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.backbone(x)
        loss = F.cross_entropy(y_hat, y)
        self.test_acc(y_hat.softmax(dim=-1), y)
        self.log('test_acc', self.test_acc, on_epoch=True, on_step=False)

    def configure_optimizers(self):
        # self.hparams available because we called self.save_hyperparameters()
        return torch.optim.Adam(self.parameters(), lr=1e-4)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--learning_rate', type=float, default=0.0001)
        return parser
