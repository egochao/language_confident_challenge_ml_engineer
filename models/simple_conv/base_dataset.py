from torchaudio.datasets import SPEECHCOMMANDS
from pathlib import Path
import torchaudio
import torch
from tqdm import tqdm

import constants
import torch
from typing import Tuple


class AudioArrayDataSet(SPEECHCOMMANDS):
    def __init__(self, subset: str = None, data_dir: Path = constants.DATA_DIR):
        super().__init__(data_dir, download=True)
        
        audio_path = Path(self._path)
        create_train_subset_file(audio_path)
        self.file_list = load_list(audio_path / f'{subset}_list.txt')

        self._walker = [audio_path.joinpath(each) for each in self.file_list]

    def __getitem__(self, index: int) -> Tuple:
        waveform, _, label, _, _ = super().__getitem__(index)
        return waveform, label


class AudioArrayWithLogitDataset(AudioArrayDataSet):
    def __init__(self, logit_path: Path, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.logit_walker = [logit_path.joinpath(each.replace('wav', 'pt')) for each in self.file_list]
        self.logit_cache = [torch.load(each) for each in tqdm(self.logit_walker)]

    def __getitem__(self, index: int) -> Tuple:
        waveform, label = super().__getitem__(index)
        logit = self.logit_cache[index]
        return waveform, logit, label


transform = torchaudio.transforms.Resample(
    orig_freq=constants.ORIGINAL_SAMPLE_RATE, new_freq=constants.NEW_SAMPLE_RATE
)


def pad_sequence(batch):
    """Make all tensor in a batch the same length by padding with zeros"""
    batch = [item.t() for item in batch]
    batch = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=0.0)
    batch = transform(batch.squeeze())
    return batch.unsqueeze(1)
    
def label_to_index(word):
    """Return the position of the word in labels"""
    return torch.tensor(constants.LABELS.index(word))


def simconv_collate_fn(batch):
    """Collate function for dataset"""
    tensors, targets = [], []

    for waveform, label in batch:
        tensors += [waveform]
        targets += [label_to_index(label)]

    tensors = pad_sequence(tensors)
    targets = torch.stack(targets)

    return tensors, targets

def simconv_collate_logit_fn(batch):
    """Collate function for logit dataset"""
    tensor_wav, tensor_logit, targets = [], [], []

    for waveform, logit, label in batch:
        tensor_wav += [waveform]
        tensor_logit += [logit]
        targets += [label_to_index(label)]

    tensor_wav = pad_sequence(tensor_wav)
    targets = torch.stack(targets)
    tensor_logit = torch.stack(tensor_logit)

    return tensor_wav, tensor_logit, targets


def create_train_subset_file(base_path, replace_existing=False):
    """Create training file exclude validation and test"""
    train_filepath = base_path / 'train_list.txt'

    if not replace_existing and train_filepath.exists():
        return

    with open(base_path / 'validation_list.txt', 'r') as f:
        val_list = f.readlines()
    with open(base_path / 'testing_list.txt', 'r') as f:
        test_list = f.readlines()
    val_test_list = set(test_list+val_list)

    all_list = []
    for path in base_path.glob('*/'):
        if path.stem in constants.LABELS:
            audio_files = list(path.glob('*.wav'))
            file_list = [f"{f.parent.stem}/{f.name}" for f in audio_files]
            all_list += file_list

    training_list = [x for x in all_list if x not in val_test_list]
    with open(train_filepath, 'w') as f:
        for line in training_list:
            f.write(f"{line}\n")


def load_list(filepath):
    """Load text and clean"""
    with open(filepath) as fileobj:
        return [line.strip() for line in fileobj]
