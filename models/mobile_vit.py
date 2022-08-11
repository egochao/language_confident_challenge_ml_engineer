import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torch
from constants import LABELS

from transformers import MobileViTConfig, MobileViTForImageClassification


class MobileViTModelCustom(nn.Module):
    def __init__(self, num_labels=35, image_size=(513, 32), num_channels=1):
        super().__init__()
        self.configuration = MobileViTConfig(
            num_labels=35, num_channels=num_channels, image_size=image_size
        )
        self.model = MobileViTForImageClassification(self.configuration)

    def forward(self, x):
        return self.model(x).logits


transform = torchaudio.transforms.Spectrogram(n_fft=1024, hop_length=512)


def pad_sequence(batch):
    batch = [item.t() for item in batch]
    batch = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=0.0)
    batch = transform(batch.squeeze())
    return batch.unsqueeze(1)


def label_to_index(word):
    # Return the position of the word in labels
    return LABELS.index(word)

def one_hot_to_index(one_hot_labels):
    index_labels = torch.argmax(one_hot_labels, dim=1)
    return index_labels

import numpy as np
def spec_collate_fn(batch):
    tensors_input, labels = [], []

    for waveform, _, label, *_ in batch:
        tensors_input += [waveform]
        labels += [label_to_index(label)]

    tensors_input = pad_sequence(tensors_input)

    one_hot_targets = np.eye(len(LABELS))[labels]
    tensor_labels = torch.tensor(one_hot_targets, dtype=torch.float)

    return tensors_input, tensor_labels
