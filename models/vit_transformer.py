# Installing libraries
# Importing
# Weights & Biases
import pytorch_lightning as pl
# Pytorch modules
import torch
import torch.nn
import torchmetrics
import torchvision.models as models
import wandb
# Pytorch-Lightning
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from pytorch_lightning.loggers import WandbLogger
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader, random_split
from transformers import ViTConfig
from vit_pytorch import ViT
# Pytorch-Lightning
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
# Dataset
from torchvision.datasets import CIFAR10, CIFAR100



from argparse import ArgumentParser

class ViTConfigExtended(ViTConfig):
    def __init__(self, hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        is_encoder_decoder=False,
        image_size=224,
        patch_size=16,
        num_channels=3,
        num_classes: int = 1000):
        super().__init__()
        self.num_classes = num_classes
        
configuration = ViTConfigExtended()


class VisionTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.model = ViT(
            image_size = config.image_size,
            patch_size = config.patch_size,
            num_classes = config.num_classes,
            dim = config.hidden_size,
            depth = config.num_hidden_layers,
            heads = config.num_attention_heads,
            mlp_dim = config.intermediate_size,
            dropout = config.hidden_dropout_prob,
            emb_dropout = config.attention_probs_dropout_prob
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


class Backbone(torch.nn.Module):
    def __init__(self, model_type, config):
        super().__init__()
        if model_type == 'vit':
            self.model = VisionTransformer(config)
        
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


class CIFAR10DataModule(LightningDataModule):

    def __init__(self, data_dir='./data/', batch_size=256, image_size=32):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.image_size = image_size
        self.transform_train = transforms.Compose([
                    transforms.Resize((self.image_size+32, self.image_size+32)), 
                    transforms.RandomCrop((self.image_size, self.image_size)),
                    transforms.RandomHorizontalFlip(),
                    transforms.ColorJitter(brightness=0.1, 
                    contrast=0.1, saturation=0.1, hue=0.1),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                            std=[0.5, 0.5, 0.5])])
        self.transform_eval = transforms.Compose([
                    transforms.Resize((self.image_size, self.image_size)), 
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                    std=[0.5, 0.5, 0.5])])

    def prepare_data(self):
        '''called only once and on 1 GPU'''
        # download data
        CIFAR10(self.data_dir, train=True, download=True)
        CIFAR10(self.data_dir, train=False, download=True)

    def setup(self, stage=None):
        '''called on each GPU separately - stage defines if we are at fit or test step'''
        # we set up only relevant datasets when stage is specified (automatically set by Pytorch-Lightning)
        if stage == 'fit' or stage is None:
            dataset_train = CIFAR10(self.data_dir, train=True, transform=self.transform_train)
            no_train = int(len(dataset_train) * 0.9)
            no_val = len(dataset_train) - no_train
            self.dataset_train, self.dataset_val = random_split(dataset_train, [no_train, no_val])
            self.num_classes = len(dataset_train.classes)
        if stage == 'test' or stage is None:
            self.dataset_test = CIFAR10(self.data_dir, train=False, transform=self.transform_eval)
            self.num_classes = len(self.cifar_test.classes)

    def train_dataloader(self):
        '''returns training dataloader'''
        dataloader_train = DataLoader(self.dataset_train, batch_size=self.batch_size, num_workers=4)
        return dataloader_train

    def val_dataloader(self):
        '''returns validation dataloader'''
        dataloader_val = DataLoader(self.dataset_val, batch_size=self.batch_size, num_workers=4)
        return dataloader_val

    def test_dataloader(self):
        '''returns test dataloader'''
        dataloader_test = DataLoader(self.dataset_test, batch_size=self.batch_size, num_workers=4)
        return dataloader_test

class CIFAR100DataModule(CIFAR10DataModule):

    def __init__(self):
        super().__init__()

    def prepare_data(self):
        '''called only once and on 1 GPU'''
        # download data
        CIFAR100(self.data_dir, train=True, download=True)
        CIFAR100(self.data_dir, train=False, download=True)

    def setup(self, stage=None):
        '''called on each GPU separately - stage defines if we are at fit or test step'''
        # we set up only relevant datasets when stage is specified (automatically set by Pytorch-Lightning)
        if stage == 'fit' or stage is None:
            dataset_train = CIFAR10(self.data_dir, train=True, transform=self.transform_train)
            no_train = int(len(dataset_train) * 0.9)
            no_val = len(dataset_train) - no_train
            self.dataset_train, self.dataset_val = random_split(dataset_train, [no_train, no_val])
            self.num_classes = len(dataset_train.classes)
        if stage == 'test' or stage is None:
            self.dataset_test = CIFAR10(self.data_dir, train=False, transform=self.transform_eval)
            self.num_classes = len(self.cifar_test.classes)


pl.seed_everything(0)
wandb_logger = WandbLogger(project='ViT_experiments')


# setup data
dm = CIFAR10DataModule(batch_size=32, image_size=configuration.image_size)


dm.prepare_data()
dm.setup('fit')


# setup model and trainer 
backbone = Backbone(model_type='vit', config=configuration)
model = LitClassifier(backbone)
if torch.cuda.is_available():
    trainer = pl.Trainer(gpus=1, max_epochs=10, logger=wandb_logger)
else:
    trainer = pl.Trainer(max_epochs=10, logger=wandb_logger)

# train, validate
trainer.fit(model, dm)


trainer.test()
wandb.finish()
