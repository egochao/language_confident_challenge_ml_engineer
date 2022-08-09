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
from pytorch_lightning import LightningDataModule
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




if __name__ == '__main__':

    pl.seed_everything(0)
    wandb_logger = WandbLogger(project='ViT_experiments')


    # setup data
    dm = CIFAR10DataModule(batch_size=32, image_size=configuration.image_size)


    dm.prepare_data()
    dm.setup('fit')

