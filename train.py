from models.torch_lightling_train_module import BaseTorchLightlingWrapper
from models.simple_conv import SimpleConv
import torch
from datasets.torch_lightling_datamodule import SpeechCommandDataModule
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from datasets.base_dataset import AudioArrayDataSet
from models.simple_conv import simconv_collate_fn
from torch.nn import functional as F
import argparse
import constants
from models.mobile_vit import MobileViTModelCustom, spec_collate_fn, one_hot_to_index
from models.bc_resnet import BcResNetModel
from datasets.mel_spec_dataset import MelSpecDataSet, mel_collate_fn

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str)
    parser.add_argument("--batch_size", type=int, default=constants.BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=constants.LEARNING_RATE)
    parser.add_argument("--epochs", type=int, default=constants.EPOCHS)

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()


    if args.model == "vit":
        core_model = MobileViTModelCustom()
        loss_fn = torch.nn.BCEWithLogitsLoss()
        label_converter = one_hot_to_index
        collate_fn = spec_collate_fn
        dataset_fn = AudioArrayDataSet
    elif args.model == "conv" or args.model is None:
        core_model = SimpleConv()
        loss_fn = F.nll_loss
        label_converter = None
        collate_fn = simconv_collate_fn
        dataset_fn = AudioArrayDataSet
    elif args.model == "bc_resnet":
        core_model = BcResNetModel()
        loss_fn = F.nll_loss
        label_converter = None
        collate_fn = mel_collate_fn
        dataset_fn = MelSpecDataSet


    pl.seed_everything(0)
    wandb_logger = WandbLogger(project="ViT_experiments")
    model = BaseTorchLightlingWrapper(
        core_model = core_model, 
        loss_fn = loss_fn, 
        label_converter=label_converter,
        learning_rate=args.lr
        )

    data_module = SpeechCommandDataModule(dataset_fn, collate_fn, batch_size=args.batch_size)

    if torch.cuda.is_available():
        trainer = pl.Trainer(accelerator='gpu', devices=1, max_epochs=args.epochs, logger=wandb_logger)
    else:
        trainer = pl.Trainer(max_epochs=args.epochs, logger=wandb_logger)

    trainer.fit(model, data_module)
