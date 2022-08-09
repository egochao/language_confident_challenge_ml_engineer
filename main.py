import torch
from dataset2 import CIFAR10DataModule
from models.vit_transformer import ViTConfigExtended, Backbone, LitClassifier
from pytorch_lightning.loggers import WandbLogger
import pytorch_lightning as pl
import wandb

pl.seed_everything(0)
wandb_logger = WandbLogger(project='ViT_experiments')

configuration = ViTConfigExtended()

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
