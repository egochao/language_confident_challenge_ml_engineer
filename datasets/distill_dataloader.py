
from pathlib import Path
import torchaudio
import torch
import torch.nn.functional
from torch.utils.data import Dataset
import constants

LABEL_LIST =  ["backward", "follow", "five", "bed", "zero", "on", "learn", "two", "house", "tree", "dog", "stop", "seven", "eight", "down", "six", "forward", "cat", "right", "visual", "four",
    "wow", "no", "nine", "off", "three", "left", "marvin", "yes", "up", "sheila", "happy", "bird", "go", "one"
]

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
        f.writelines(training_list)

class AudioDistillDataset(Dataset):
    def __init__(self, audios_path: Path, logits_path: Path, subset: str):
        """_summary_

        Args:
            audios_path (Path): path to speech command dataset
            logits_path (Path): path to teacher logit
            subset (str): 'train' or 'validation' or 'testing'
        """
        subset_file = audios_path / f'{subset}_list.txt'
        if subset == "train" and not subset_file.exists():
            _create_train_subset_file(audios_path)

        with open(subset_file, 'r') as f:
            file_list = f.readlines()

        self.audio_path_list = [audios_path / sub_path.strip() for sub_path in file_list]
        self.logit_path_list = [logits_path / sub_path.strip().replace('wav', 'pt') for sub_path in file_list]

        label_list = [filepath.parent.stem for filepath in self.audio_path_list]
        self.label_list = [LABEL_LIST.index(label) for label in label_list]

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
        return waveform, sr

    def __len__(self):
        return len(self.label_list)

    def __getitem__(self, index):
        student_input = self._load_audio_input(self.audio_path_list[index])
        teacher_logits = self._load_logits(self.logit_path_list[index])
        label = self.label_list[index]

        return {
            "student_input": student_input,
            "teacher_logits": teacher_logits,
            "label": label
        }
