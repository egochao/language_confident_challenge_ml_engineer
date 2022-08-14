import argparse

import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import WandbLogger
from torch.nn import functional as F

import constants
from lightling_wrapper import BaseTorchLightlingWrapper, SpeechCommandDataModule
from models.bc_resnet.bc_resnet_model import BcResNetModel
from models.bc_resnet.mel_spec_dataset import MelSpecDataSet, mel_collate_fn
from models.simple_conv.base_dataset import AudioArrayDataSet, simconv_collate_fn
from models.simple_conv.simple_conv_model import SimpleConv


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="conv")
    parser.add_argument("--pretrain", type=str)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.model == "conv":
        core_model = SimpleConv()
        loss_fn = F.nll_loss
        collate_fn = simconv_collate_fn
        dataset_fn = AudioArrayDataSet
    elif args.model == "bc_resnet":
        core_model = BcResNetModel(scale=constants.SCALE_BC_RESNET)
        loss_fn = F.nll_loss
        collate_fn = mel_collate_fn
        dataset_fn = MelSpecDataSet
    else:
        raise ValueError("Invalid model name")

    model = BaseTorchLightlingWrapper(
        core_model=core_model,
        loss_fn=loss_fn,
        learning_rate=constants.LEARNING_RATE,
    )

    data_module = SpeechCommandDataModule(
        dataset_fn, collate_fn, batch_size=constants.BATCH_SIZE
    )

    if torch.cuda.is_available():
        trainer = pl.Trainer(
            accelerator="gpu", devices=1
        )
    else:
        trainer = pl.Trainer()

    trainer.test(model, data_module)
