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
            collate_fn=collate_fn,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.dataset_obj(self.data_dir, "validation"),
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.dataset_obj(self.data_dir, "testing"),
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )


def pad_sequence(batch):
    # Make all tensor in a batch the same length by padding with zeros
    batch = [item.t() for item in batch]
    batch = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=0.)
    return batch.permute(0, 2, 1)

def label_to_index(word):
    # Return the position of the word in labels
    return torch.tensor(constants.LABELS.index(word))


def collate_fn(batch):

    # A data tuple has the form:
    # waveform, sample_rate, label, speaker_id, utterance_number

    tensors, targets = [], []

    # Gather in lists, and encode labels as indices
    for waveform, _, label, *_ in batch:
        tensors += [waveform]
        targets += [label_to_index(label)]

    # Group the list of tensors into a batched tensor
    tensors = pad_sequence(tensors)
    targets = torch.stack(targets)

    return tensors, targets
