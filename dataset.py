import torch
from torchaudio.datasets import SPEECHCOMMANDS
import os
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from pytorch_lightning import LightningDataModule
from constants import BATCH_SIZE, ORIGINAL_SAMPLE_RATE, NEW_SAMPLE_RATE
import constants
from utils.model_utils import get_loader_params
from torch.utils.data import DataLoader
import torchaudio

def get_dataloader(name, batch_size=BATCH_SIZE, shuffle=True, drop_last=True):
    num_workers, pin_memory = get_loader_params()

    dataset = SubsetSC(name)
    dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=shuffle,
    drop_last=drop_last,
    collate_fn=collate_fn,
    num_workers=num_workers,
    pin_memory=pin_memory,
    )      

    labels = sorted(list(set(datapoint[2] for datapoint in dataset)))
    if name == "training":
        constants.LABELS = labels
    return dataloader

class SpeechCommandDataModule(LightningDataModule):

    def __init__(self, data_dir='./data/', batch_size=256):
        super().__init__()
        self.batch_size = batch_size
        self.data_dir = data_dir
        num_workers, pin_memory = get_loader_params()
        self.num_workers = num_workers
        self.pin_memory = pin_memory

    def prepare_data(self):
        '''called only once and on 1 GPU'''
        # download data
        SPEECHCOMMANDS(self.data_dir, download=True)


    def setup(self):
        '''called every time a new epoch begins'''
        self.train_dataset = SubsetSC("training")
        self.val_dataset = SubsetSC("validation")
        self.test_dataset = SubsetSC("testing")

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
            collate_fn=collate_fn,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            )      
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
            collate_fn=collate_fn,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
            collate_fn=collate_fn,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            )
        

class SubsetSC(SPEECHCOMMANDS):
    def __init__(self, subset: str = None):
        super().__init__("./data", download=True, url='speech_commands_v0.02')

        def load_list(filename):
            filepath = os.path.join(self._path, filename)
            with open(filepath) as fileobj:
                return [os.path.normpath(os.path.join(self._path, line.strip())) for line in fileobj]

        if subset == "validation":
            self._walker = load_list("validation_list.txt")
        elif subset == "testing":
            self._walker = load_list("testing_list.txt")
        elif subset == "training":
            excludes = load_list("validation_list.txt") + load_list("testing_list.txt")
            excludes = set(excludes)
            self._walker = [w for w in self._walker if w not in excludes]


def label_to_index(word):
    # Return the position of the word in labels
    return torch.tensor(constants.LABELS.index(word))


def index_to_label(index):
    # Return the word corresponding to the index in LABELS
    # This is the inverse of label_to_index
    return constants.LABELS[index]

def pad_sequence(batch):
    # Make all tensor in a batch the same length by padding with zeros
    batch = [item.t() for item in batch]
    batch = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=0.)
    return batch.permute(0, 2, 1)

transform = torchaudio.transforms.Resample(orig_freq=ORIGINAL_SAMPLE_RATE, new_freq=NEW_SAMPLE_RATE)
transform = transform.to("cuda")

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
    tensors = transform(tensors)
    targets = torch.stack(targets)

    return tensors, targets

if __name__ == '__main__':
    dm = SpeechCommandDataModule()
    dm.prepare_data()
    dm.setup(stage="fit")

    for idx, data in enumerate(dm.train_dataloader()):
        print(data)
        if idx > 2:
            break