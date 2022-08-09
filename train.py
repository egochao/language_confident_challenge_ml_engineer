import torch
import torch.nn.functional as F

import torch
from models.vit_transformer import Backbone, LitClassifier, ViTConfigExtended
import pytorch_lightning as pl
import wandb
from pytorch_lightning.loggers import WandbLogger




def main():
    pl.seed_everything(0)
    wandb_logger = WandbLogger(project='ViT_experiments')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from datasets import SpeechCommandDataModule
    dm = SpeechCommandDataModule()
    dm.prepare_data()
    dm.setup()


    # setup model and trainer 
    configuration = ViTConfigExtended()

    backbone = Backbone(model_type='vit', config=configuration)
    model = LitClassifier(backbone)
    if torch.cuda.is_available():
        trainer = pl.Trainer(gpus=1, max_epochs=10, logger=wandb_logger)
    else:
        trainer = pl.Trainer(max_epochs=10, logger=wandb_logger)

    # train, validate
    trainer.fit(model, dm)


if __name__ == '__main__':
    main()
