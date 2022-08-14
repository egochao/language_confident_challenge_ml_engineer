import argparse

import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import WandbLogger
from pathlib import Path

import constants
from lightling_wrapper import (
    DistillSpeechCommandDataModule,
    DistillModelTorchLightlingWrapper,
)
from models.bc_resnet.bc_resnet_model import BcResNetModelNoSoftMax
from models.bc_resnet.mel_spec_dataset import (
    mel_collate_logit_fn,
    MelSpecWithLogitDataset,
)
from models.simple_conv.base_dataset import (
    AudioArrayWithLogitDataset,
    simconv_collate_logit_fn,
)
from models.simple_conv.simple_conv_model import SimpleConvNoSoftMax
from utils.model_utils import distillation_loss


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str)
    parser.add_argument("--logit", type=str, default=str(constants.LOGITS_PATH))
    parser.add_argument("--batch_size", type=int, default=constants.BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=constants.LEARNING_RATE)
    parser.add_argument("--epochs", type=int, default=constants.EPOCHS)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.model == "conv" or args.model is None:
        core_model = SimpleConvNoSoftMax()
        loss_fn = distillation_loss
        collate_fn = simconv_collate_logit_fn
        dataset_fn = AudioArrayWithLogitDataset
    elif args.model == "bc_resnet":
        core_model = BcResNetModelNoSoftMax(scale=constants.SCALE_BC_RESNET)
        loss_fn = distillation_loss
        collate_fn = mel_collate_logit_fn
        dataset_fn = MelSpecWithLogitDataset

    pl.seed_everything(10)
    wandb_logger = WandbLogger(project="ViT_experiments")
    model = DistillModelTorchLightlingWrapper(
        core_model=core_model,
        loss_fn=loss_fn,
        learning_rate=args.lr,
    )

    data_module = DistillSpeechCommandDataModule(
        logit_path=Path(args.logit),
        dataset=dataset_fn,
        collate_fn=collate_fn,
        batch_size=args.batch_size,
    )

    if torch.cuda.is_available():
        trainer = pl.Trainer(
            accelerator="gpu", devices=1, max_epochs=args.epochs, logger=wandb_logger
        )
    else:
        trainer = pl.Trainer(max_epochs=args.epochs, logger=wandb_logger)

    trainer.fit(model, data_module)
