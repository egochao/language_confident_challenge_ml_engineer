import torch
import torch.nn

from torch import nn
from vit_pytorch import ViT
import torch
from torch import nn

class VisionTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = ViT(
            image_size=(48, 128),
            patch_size=16,
            num_classes=35,
            dim=256,
            depth=6,
            heads=4,
            mlp_dim=256,
            channels=1,
            dropout=0.0,
            emb_dropout=0.0,
        )

    @torch.no_grad()
    def init_weights(self):
        def _init(m):
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if hasattr(m, "bias") and m.bias is not None:
                    nn.init.normal_(m.bias, std=1e-6)

        self.apply(_init)
        nn.init.constant_(self.model.fc.weight, 0)
        nn.init.constant_(self.model.fc.bias, 0)

    def forward(self, x):
        return self.model(x)

