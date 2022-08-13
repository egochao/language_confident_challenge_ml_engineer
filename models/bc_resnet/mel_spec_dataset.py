import os
import math
import random
import torch
import torchaudio
from torchaudio import transforms
from torchaudio.datasets import SPEECHCOMMANDS
from pathlib import Path
from typing import Tuple
from tqdm import tqdm

import constants


class MelSpecDataSet(SPEECHCOMMANDS):
    def __init__(self, subset: str, data_dir: Path = constants.DATA_DIR):
        super().__init__(data_dir, download=True)
        self.to_mel = transforms.MelSpectrogram(
            sample_rate=constants.ORIGINAL_SAMPLE_RATE,
            n_fft=1024,
            f_max=8000,
            n_mels=40,
        )
        self.subset = subset

        self._noise = []

        audio_path = Path(self._path)
        create_train_subset_file(audio_path)
        self.file_list = load_list(audio_path / f'{subset}_list.txt')

        self._walker = [audio_path.joinpath(each) for each in self.file_list]
        
        if subset == "train":
            noise_paths = [
                w
                for w in os.listdir(os.path.join(self._path, "_background_noise_"))
                if w.endswith(".wav")
            ]
            for item in noise_paths:
                noise_path = os.path.join(self._path, "_background_noise_", item)
                noise_waveform, noise_sr = torchaudio.sox_effects.apply_effects_file(
                    noise_path, effects=[]
                )
                noise_waveform = transforms.Resample(
                    orig_freq=noise_sr, new_freq=constants.ORIGINAL_SAMPLE_RATE
                )(noise_waveform)
                self._noise.append(noise_waveform)
        
    def _noise_augment(self, waveform):
        noise_waveform = random.choice(self._noise)

        noise_sample_start = 0
        if noise_waveform.shape[1] - waveform.shape[1] > 0:
            noise_sample_start = random.randint(
                0, noise_waveform.shape[1] - waveform.shape[1]
            )
        noise_waveform = noise_waveform[
            :, noise_sample_start : noise_sample_start + waveform.shape[1]
        ]

        signal_power = waveform.norm(p=2)
        noise_power = noise_waveform.norm(p=2)

        snr_dbs = [20, 10, 3]
        snr = random.choice(snr_dbs)

        snr = math.exp(snr / 10)
        scale = snr * noise_power / signal_power
        noisy_signal = (scale * waveform + noise_waveform) / 2
        return noisy_signal

    def _shift_augment(self, waveform):
        shift = random.randint(-1600, 1600)
        waveform = torch.roll(waveform, shift)
        if shift > 0:
            waveform[0][:shift] = 0
        elif shift < 0:
            waveform[0][shift:] = 0
        return waveform

    def _augment(self, waveform):
        if random.random() < 0.8:
            waveform = self._noise_augment(waveform)

        waveform = self._shift_augment(waveform)

        return waveform

    def __getitem__(self, n):
        waveform, sample_rate, label, _, _ = super().__getitem__(n)
        if sample_rate != constants.ORIGINAL_SAMPLE_RATE:
            resampler = transforms.Resample(
                orig_freq=sample_rate, new_freq=constants.ORIGINAL_SAMPLE_RATE
            )
            waveform = resampler(waveform)
        if self.subset == "training":
            waveform = self._augment(waveform)
        log_mel = (self.to_mel(waveform) + constants.EPS).log2()

        return log_mel, label



class MelSpecWithLogitDataset(MelSpecDataSet):
    def __init__(self, logit_path: Path, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.logit_walker = [logit_path.joinpath(each.replace('wav', 'pt')) for each in self.file_list]
        self.logit_cache = [torch.load(each) for each in tqdm(self.logit_walker)]

    def __getitem__(self, index: int) -> Tuple:
        log_mel, label = super().__getitem__(index)
        logit = self.logit_cache[index]
        return log_mel, logit, label


_label_to_idx = {label: i for i, label in enumerate(constants.LABELS)}
_idx_to_label = {i: label for label, i in _label_to_idx.items()}


def label_to_idx(label):
    return _label_to_idx[label]


def idx_to_label(idx):
    return _idx_to_label[idx]


def pad_sequence(batch):
    batch = [item.permute(2, 1, 0) for item in batch]
    batch = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True)
    return batch.permute(0, 3, 2, 1)


def mel_collate_fn(batch):
    tensors, targets = [], []
    for log_mel, label in batch:
        tensors.append(log_mel)
        targets.append(label_to_idx(label))

    tensors = pad_sequence(tensors)
    targets = torch.LongTensor(targets)

    return tensors, targets


def mel_collate_logit_fn(batch):
    tensor_wav, tensor_logit, targets = [], [], []

    for log_mel, logit, label in batch:
        tensor_wav.append(log_mel)
        tensor_logit += [logit]
        targets.append(label_to_idx(label))

    tensor_wav = pad_sequence(tensor_wav)
    targets = torch.LongTensor(targets)
    tensor_logit = torch.stack(tensor_logit)


    return tensor_wav, tensor_logit, targets


def prepare_wav(waveform, sample_rate):
    if sample_rate != constants.ORIGINAL_SAMPLE_RATE:
        resampler = transforms.Resample(
            orig_freq=constants.NEW_SAMPLE_RATE, new_freq=constants.ORIGINAL_SAMPLE_RATE
        )
        waveform = resampler(waveform)
    to_mel = transforms.MelSpectrogram(
        sample_rate=constants.ORIGINAL_SAMPLE_RATE, n_fft=1024, f_max=8000, n_mels=40
    )
    log_mel = (to_mel(waveform) + constants.EPS).log2()
    return log_mel



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
