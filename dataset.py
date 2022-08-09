import torch
from torchaudio.datasets import SPEECHCOMMANDS
import os
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from pytorch_lightning import LightningDataModule
import constants
from utils.model_utils import get_loader_params
from torch.utils.data import DataLoader
from pathlib import Path
from datasets.distill_dataloader import AudioDistillDataset


class SpeechCommandDataModule(LightningDataModule):

    def __init__(self, data_dir=constants.DATA_DIR, batch_size=constants.BATCH_SIZE):
        super().__init__()
        self.batch_size = batch_size
        self.data_dir = data_dir
        num_workers, pin_memory = get_loader_params()
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.dataset_path = Path(data_dir).joinpath(constants.SUB_DATASET_PATH)

    def prepare_data(self):
        '''called only once and on 1 GPU'''
        # download data
        SPEECHCOMMANDS(self.data_dir, download=True)


    def train_dataloader(self):
        return torch.utils.data.DataLoader(
        AudioDistillDataset(self.dataset_path, "train"),
        batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=self.pin_memory)
    
    def val_dataloader(self):
        return torch.utils.data.DataLoader(
        AudioDistillDataset(self.dataset_path, "validation"),
        batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=self.pin_memory)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
        AudioDistillDataset(self.dataset_path, "testing"),
        batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=self.pin_memory)        
        
if __name__ == '__main__':
    data_dir = './data/'
    batch_size = 256
    num_workers, pin_memory = get_loader_params()
    data_module = SpeechCommandDataModule(data_dir, batch_size)
    data_module.prepare_data()
    data_module.setup()
    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()
    test_loader = data_module.test_dataloader()
    for idx, batch in enumerate(train_loader):
        print(idx)
        print(batch['student_input'][0].shape)
        if idx > 1000:
            break
    # for idx, batch in enumerate(val_loader):
    #     print(idx)
    #     print(batch['student_input'][0].shape)
    #     if idx > 1000:
    #         break
    # for idx, batch in enumerate(test_loader):
    #     print(idx)
    #     print(batch['student_input'][0].shape)
    #     if idx > 1000:
            # break