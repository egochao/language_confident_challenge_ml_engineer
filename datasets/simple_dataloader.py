from pathlib import Path
import torchaudio
from torch.utils.data import Dataset
from data_objects import DataSample
from typing import List

import constants

class AudioDataset(Dataset):
    def __init__(
        self,
        audio_folder: Path,
        subset: str,
    ):
        """Loads speech commands dataset with optional teacher model last layer logits.

        Args:
            audios_folder (Path): path to speech command dataset
            logits_path (Path): path to teacher logit
            subset (str): 'train' or 'validation' or 'testing'
        """
        audio_folder = audio_folder.joinpath(constants.SUB_DATASET_PATH)

        self.transform = torchaudio.transforms.Resample(
            orig_freq=constants.ORIGINAL_SAMPLE_RATE, new_freq=constants.NEW_SAMPLE_RATE
        )

        subset_file = audio_folder / f"{subset}_list.txt"
        if subset == "train":
            _create_train_subset_file(audio_folder)

        with open(subset_file, "r") as f:
            file_list = f.readlines()

        self.list_data_label_mapping = self._get_data_label_mapping(
            file_list, audio_folder
        )

    def _get_data_label_mapping(
        self, file_list: List[str], audios_folder: Path) -> List[DataSample]:
        audio_path_list = [audios_folder / sub_path.strip() for sub_path in file_list]
        label_list = [filepath.parent.stem for filepath in audio_path_list]
        label_list = [constants.LABELS.index(label) for label in label_list]

        list_data_label_mapping = []
        for audio_path, label in zip(
            audio_path_list, label_list
        ):
            data_label_map = DataSample(
                audio_path=audio_path,
                label=label
            )
            list_data_label_mapping.append(data_label_map)
        return list_data_label_mapping

    def load_train_input(self, data_sample: DataSample):
        waveform, sr = torchaudio.load(data_sample.audio_path)
        waveform = self.transform(waveform)
        return waveform

    def __len__(self):
        return len(self.list_data_label_mapping)

    def __getitem__(self, index):
        data_sample = self.list_data_label_mapping[index]
        
        train_input = self.load_train_input(data_sample)
        
        return train_input, data_sample.label
        # return torch.rand((1, 48, 128)), label


def _create_train_subset_file(base_path: Path, replace_existing=True):
    train_filepath = base_path / "train_list.txt"

    if not replace_existing and train_filepath.exists():
        return

    with open(base_path / "validation_list.txt", "r") as f:
        val_list = f.readlines()
    with open(base_path / "testing_list.txt", "r") as f:
        test_list = f.readlines()
    val_test_list = set(test_list + val_list)

    all_list = []
    for path in base_path.glob("*/"):
        if path.stem in constants.LABELS:
            audio_files = list(path.glob("*.wav"))
            file_list = [f"{f.parent.stem}/{f.name}" for f in audio_files]
            all_list += file_list

    training_list = [x for x in all_list if x not in val_test_list]
    if isinstance(constants.NUM_TRAIN_SAMPLE, int):
        training_list = training_list[:constants.NUM_TRAIN_SAMPLE]
    with open(train_filepath, "w") as f:
        for line in training_list:
            f.write(f"{line}\n")


