import sys

sys.path.append("..")
from lightling_wrapper.train_module import BaseTorchLightlingWrapper
from models.simple_conv import SimpleConv
import torch
from lightling_wrapper.data_module import SpeechCommandDataModule
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from datasets.base_dataset import AudioArrayDataSet
from models.simple_conv import simconv_collate_fn
from models.mobile_vit import spec_collate_fn
from models.mobile_vit import MobileViTModelCustom
from datasets.mel_spec_dataset import MelSpecDataSet, mel_collate_fn


if __name__ == "__main__":
    data_module = SpeechCommandDataModule(
        MelSpecDataSet, mel_collate_fn, batch_size=256
    )

    for idx, data in enumerate(data_module.train_dataloader()):
        print(idx, data[0].shape)
        if idx > 100:
            break
