import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torch
from constants import LABELS

from transformers import MobileViTConfig, MobileViTForImageClassification


class MobileViTModelCustom(nn.Module):
    def __init__(self, num_labels=35, image_size=(257, 63), num_channels=1):
        super().__init__()
        self.configuration = MobileViTConfig(
            num_labels=35, num_channels=num_channels, image_size=image_size
        )
        self.model = MobileViTForImageClassification(self.configuration)

    def forward(self, x):
        return self.model(x).logits


transform = torchaudio.transforms.Spectrogram(n_fft=512, hop_length=256)


def pad_sequence(batch):
    batch = [item.t() for item in batch]
    batch = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=0.0)
    batch = transform(batch.squeeze())
    return batch.unsqueeze(1)


def label_to_index(word):
    # Return the position of the word in labels
    return torch.tensor(LABELS.index(word))


def spec_collate_fn(batch):
    tensors, targets = [], []

    for waveform, _, label, *_ in batch:
        tensors += [waveform]
        targets += [label_to_index(label)]

    tensors = pad_sequence(tensors)
    targets = torch.stack(targets)

    return tensors, targets
