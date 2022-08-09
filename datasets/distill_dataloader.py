
from pathlib import Path
import torchaudio
import torch
import torch.nn.functional
from torch.utils.data import Dataset
import constants
from typing import Optional

def _create_train_subset_file(base_path):
    with open(base_path / 'validation_list.txt', 'r') as f:
        val_list = f.readlines()
    with open(base_path / 'testing_list.txt', 'r') as f:
        test_list = f.readlines()
    val_test_list = set(test_list+val_list)

    all_list = []
    for path in base_path.glob('**/*.wav'):
        file = f"{path.parent.stem}/{path.name}"
        all_list.append(file)

    training_list = [x for x in all_list if x not in val_test_list]
    with open(base_path / 'train_list.txt', 'w') as f:
        for line in training_list:
            f.write(f"{line}\n")


class AudioDistillDataset(Dataset):
    def __init__(self, audios_path: Path, subset: str, logits_path: Optional[Path] = None):
        """Loads speech commands dataset with optional teacher model last layer logits.

        Args:
            audios_path (Path): path to speech command dataset
            logits_path (Path): path to teacher logit
            subset (str): 'train' or 'validation' or 'testing'
        """
        self.logits_path = logits_path
        subset_file = audios_path / f'{subset}_list.txt'
        if subset == "train":
            _create_train_subset_file(audios_path)

        with open(subset_file, 'r') as f:
            file_list = f.readlines()

        self.audio_path_list = [audios_path / sub_path.strip() for sub_path in file_list]
        if self.logits_path:
            self.logit_path_list = [logits_path / sub_path.strip().replace('wav', 'pt') for sub_path in file_list]

        label_list = [filepath.parent.stem for filepath in self.audio_path_list]
        self.label_list = [constants.LABELS.index(label) for label in label_list if label in constants.LABELS]

    def _load_logits(self, filepath:Path) -> torch.Tensor:
        return torch.load(filepath)

    def _load_audio_input(self, filepath: Path) -> torch.Tensor:
        # modify this function for audio preprocessing
        # STFT for example
        waveform, sr = torchaudio.load(filepath)
        waveform = waveform - waveform.mean()
        if waveform.shape[1] < constants.INPUT_AUDIO_LENGTH:
            print(waveform.shape)
            waveform = torch.cat([waveform, torch.zeros((1 ,constants.INPUT_AUDIO_LENGTH - waveform.shape[1]))], dim=1)
            print(f"{filepath} is padded with {constants.INPUT_AUDIO_LENGTH - waveform.shape[1]} zeros")
        fbank = torchaudio.compliance.kaldi.fbank(waveform, htk_compat=True, sample_frequency=sr, use_energy=False,
                                                  window_type='hanning', num_mel_bins=128, dither=0.0, frame_shift=10)
        return fbank, sr

    def __len__(self):
        return len(self.label_list)

    def __getitem__(self, index):
        student_input = self._load_audio_input(self.audio_path_list[index])
        label = self.label_list[index]
        if self.logits_path:
            teacher_logits = self._load_logits(self.logit_path_list[index])
            output =  {'student_input': student_input, 'teacher_logits': teacher_logits, 'label': self.label_list[index]}
        else:
            output = {'student_input': student_input, 'label': self.label_list[index]}
        return output
