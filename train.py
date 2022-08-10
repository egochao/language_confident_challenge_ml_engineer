from models.vit_transformer import VisionTransformer, LitClassifier
import torch
from datasets.sc_dataset import SpeechCommandDataModule
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger


if __name__ == '__main__':
    pl.seed_everything(0)
    wandb_logger = WandbLogger(project='ViT_experiments')
    core_model = VisionTransformer()
    model = LitClassifier(core_model)

    data_dir = './data/'
    data_module = SpeechCommandDataModule(data_dir)
    data_module.prepare_data()
    data_module.setup()

    if torch.cuda.is_available():
        trainer = pl.Trainer(gpus=1, max_epochs=50, logger=wandb_logger)
    else:
        trainer = pl.Trainer(max_epochs=50, logger=wandb_logger)

    # train, validate
    trainer.fit(model, data_module)
