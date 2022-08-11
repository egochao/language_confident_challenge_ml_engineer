from models.torch_lightling_train_module import BaseTorchLightlingWrapper
import torch
from datasets.torch_lightling_datamodule import SpeechCommandDataModule
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from datasets.base_dataset import AudioArrayDataSet
from models.mobile_vit import MobileViTModelCustom, spec_collate_fn, one_hot_to_index


if __name__ == "__main__":
    core_model = MobileViTModelCustom()

    pl.seed_everything(0)
    wandb_logger = WandbLogger(project="ViT_experiments")
    loss_fn = torch.nn.BCEWithLogitsLoss()
    model = BaseTorchLightlingWrapper(
        core_model=core_model, loss_fn=loss_fn, label_converter=one_hot_to_index
    )

    data_module = SpeechCommandDataModule(
        AudioArrayDataSet, spec_collate_fn, batch_size=64
    )
    data_module.prepare_data()
    data_module.setup()

    if torch.cuda.is_available():
        trainer = pl.Trainer(gpus=1, max_epochs=50, logger=wandb_logger)
    else:
        trainer = pl.Trainer(max_epochs=50, logger=wandb_logger)

    trainer.fit(model, data_module)
