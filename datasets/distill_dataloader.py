
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
        self.transform = torchaudio.transforms.Resample(orig_freq=constants.ORIGINAL_SAMPLE_RATE, new_freq=constants.NEW_SAMPLE_RATE)
        
    def _load_logits(self, filepath:Path) -> torch.Tensor:
        return torch.load(filepath)

    def _load_audio_input(self, filepath: Path) -> torch.Tensor:
        # modify this function for audio preprocessing
        # STFT for example
        cached_spec_path = str(filepath).replace('.wav', '.pt')
        if Path(cached_spec_path).exists():
            return torch.load(cached_spec_path)
        else:
            waveform, sr = torchaudio.load(filepath)
            waveform = self.transform(waveform)

            waveform = waveform - waveform.mean()
            fbank = torchaudio.compliance.kaldi.fbank(waveform, htk_compat=True, sample_frequency=sr, use_energy=False,
                                                    window_type='hanning', num_mel_bins=128, dither=0.0, frame_shift=10)

            fbank = self._pad_spec(fbank)
            fbank = torch.unsqueeze(fbank, dim=0)
            torch.save(fbank, cached_spec_path)
            return fbank

    def _pad_spec(self, spec):
        target_length = constants.PADDED_SPEC_HEIGHTS
        n_frames = spec.shape[0]
        miss_match_dimension = target_length - n_frames
        # cut and pad
        if miss_match_dimension > 0:
            pad = torch.nn.ZeroPad2d((0, 0, 0, miss_match_dimension))
            spec = pad(spec)
        elif miss_match_dimension < 0:
            spec = spec[0:target_length, :]
        return spec

    def __len__(self):
        return len(self.label_list)

    def __getitem__(self, index):
        student_input = self._load_audio_input(self.audio_path_list[index])
        label = self.label_list[index]
        if self.logits_path:
            teacher_logits = self._load_logits(self.logit_path_list[index])
            output =  student_input, teacher_logits, label
        else:
            output = student_input, label
        return output
