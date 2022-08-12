from torchaudio.datasets import SPEECHCOMMANDS
import os
import constants
from pathlib import Path
import torchaudio
import constants
import torch

import constants


class AudioArrayDataSet(SPEECHCOMMANDS):
    def __init__(self, subset: str = None, data_dir: Path = constants.DATA_DIR):
        super().__init__(data_dir)

        def load_list(filepath,):
            with open(filepath) as fileobj:
                return [
                    line.strip()
                    for line in fileobj
                ]
        audio_path = Path(self._path)
        if subset == "validation":
            self.file_list = load_list(audio_path / "validation_list.txt")
        elif subset == "testing":
            self.file_list = load_list(audio_path / "testing_list.txt")
        elif subset == "train":
            excludes = load_list(audio_path / "validation_list.txt") + load_list(audio_path / "testing_list.txt")
            excludes = set(excludes)
            self.file_list = [w for w in self._walker if w not in excludes]

        self._walker = [audio_path.joinpath(each) for each in self.file_list]


transform = torchaudio.transforms.Resample(
    orig_freq=constants.ORIGINAL_SAMPLE_RATE, new_freq=constants.NEW_SAMPLE_RATE
)


def pad_sequence(batch):
    # Make all tensor in a batch the same length by padding with zeros
    batch = [item.t() for item in batch]
    batch = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=0.0)
    batch = transform(batch.squeeze())
    return batch.unsqueeze(1)
    # return batch.permute(0, 2, 1)


def label_to_index(word):
    # Return the position of the word in labels
    return torch.tensor(constants.LABELS.index(word))


def simconv_collate_fn(batch):

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
