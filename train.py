from models.torch_lightling_train_module import BaseTorchLightlingWrapper
from models.simple_conv import SimpleConv
import torch
from datasets.torch_lightling_datamodule import SpeechCommandDataModule
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from datasets.base_dataset import AudioArrayDataSet
from models.simple_conv import simconv_collate_fn
from torch.nn import functional as F


if __name__ == "__main__":
    core_model = SimpleConv()

    pl.seed_everything(0)
    wandb_logger = WandbLogger(project="ViT_experiments")
    loss_fn = F.nll_loss
    model = BaseTorchLightlingWrapper(core_model, loss_fn)

    data_module = SpeechCommandDataModule(AudioArrayDataSet, simconv_collate_fn)
    data_module.prepare_data()
    data_module.setup()

    if torch.cuda.is_available():
        trainer = pl.Trainer(gpus=1, max_epochs=80, logger=wandb_logger)
    else:
        trainer = pl.Trainer(max_epochs=80, logger=wandb_logger)

    trainer.fit(model, data_module)
