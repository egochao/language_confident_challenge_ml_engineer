import torch
from torchaudio.datasets import SPEECHCOMMANDS
from pytorch_lightning import LightningDataModule
from pytorch_lightning import LightningDataModule
import constants
from utils.model_utils import get_loader_params

from datasets.simple_dataloader import AudioDataset

class SpeechCommandDataModule(LightningDataModule):
    def __init__(self, dataset: AudioDataset, data_dir=constants.DATA_DIR, batch_size=constants.BATCH_SIZE):
        super().__init__()
        self.dataset_obj = dataset
        self.data_dir = data_dir
        self.batch_size = batch_size
        num_workers, pin_memory = get_loader_params()
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.data_dir = data_dir

    def prepare_data(self):
        """called only once and on 1 GPU"""
        # download data
        SPEECHCOMMANDS(self.data_dir, download=True)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.dataset_obj(self.data_dir, "train"),
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.dataset_obj(self.data_dir, "validation"),
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.dataset_obj(self.data_dir, "testing"),
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
