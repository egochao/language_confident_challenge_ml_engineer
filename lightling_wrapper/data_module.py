from typing import Callable, Optional

import torch
from pytorch_lightning import LightningDataModule
from torchaudio.datasets import SPEECHCOMMANDS
from pathlib import Path
import constants


class SpeechCommandDataModule(LightningDataModule):
    def __init__(
        self,
        dataset,
        collate_fn: Optional[Callable],
        data_dir=None,
        batch_size=None,
    ):
        super().__init__()
        self.dataset_obj = dataset
        self.collate_fn = collate_fn
        self.data_dir = data_dir or constants.DATA_DIR
        self.batch_size = batch_size or constants.BATCH_SIZE
        self.num_workers = constants.NUM_WORKERS
        self.pin_memory = constants.PIN_MEMORY

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


class DistillSpeechCommandDataModule(SpeechCommandDataModule):
    def __init__(self, logit_path: Path, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.logit_path = logit_path

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.dataset_obj(
                logit_path=self.logit_path, subset="train", data_dir=self.data_dir
            ),
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self.collate_fn,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.dataset_obj(
                logit_path=self.logit_path, subset="train", data_dir=self.data_dir
            ),
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.collate_fn,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.dataset_obj(
                logit_path=self.logit_path, subset="train", data_dir=self.data_dir
            ),
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.collate_fn,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
