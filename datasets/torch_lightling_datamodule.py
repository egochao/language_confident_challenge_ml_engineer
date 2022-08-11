import torch
from torchaudio.datasets import SPEECHCOMMANDS
from pytorch_lightning import LightningDataModule
from pytorch_lightning import LightningDataModule
import constants
from constants import NUM_WORKERS, PIN_MEMORY
from datasets.prebuild_dataset import AudioArrayDataSet
from typing import Callable, Optional

class SpeechCommandDataModule(LightningDataModule):
    def __init__(self, dataset: AudioArrayDataSet, collate_fn: Optional[Callable], data_dir= None, batch_size=None):
        super().__init__()
        self.dataset_obj = dataset
        self.collate_fn = collate_fn
        self.data_dir = data_dir or constants.DATA_DIR
        self.batch_size = batch_size or constants.BATCH_SIZE
        self.num_workers = NUM_WORKERS
        self.pin_memory = PIN_MEMORY

    def prepare_data(self):
        """called only once and on 1 GPU"""
        # download data
        SPEECHCOMMANDS(self.data_dir, download=True)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.dataset_obj("train", self.data_dir),
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self.collate_fn,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.dataset_obj("validation", self.data_dir),
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.collate_fn,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.dataset_obj("testing", self.data_dir),
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.collate_fn,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
