from models.vit_transformer import VisionTransformer, LitClassifier
import torch
from datasets.sc_dataset import SpeechCommandDataModule
from utils.model_utils import get_loader_params
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger


if __name__ == '__main__':
    pl.seed_everything(0)
    wandb_logger = WandbLogger(project='ViT_experiments')

    data_dir = './data/'
    batch_size = 256
    num_workers, pin_memory = get_loader_params()
    data_module = SpeechCommandDataModule(data_dir, batch_size)
    data_module.prepare_data()
    data_module.setup()

    core_model = VisionTransformer()
    model = LitClassifier(core_model)

    if torch.cuda.is_available():
        trainer = pl.Trainer(gpus=1, max_epochs=10, logger=wandb_logger)
    else:
        trainer = pl.Trainer(max_epochs=10, logger=wandb_logger)

    # train, validate
    trainer.fit(model, data_module)
