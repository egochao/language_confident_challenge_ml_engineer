from pathlib import Path
import torchaudio
import torch
import torch.nn.functional
from torch.utils.data import Dataset
import constants
from typing import Optional
from data_objects import DataSample
from typing import List
from tqdm import tqdm
import logging


class AudioDistillDataset(Dataset):
    def __init__(
        self,
        audios_path: Path,
        subset: str,
        logits_path: Optional[Path] = None,
        cache_spec: bool = True,
        keep_dataset_in_ram: bool = constants.KEEP_DATASET_IN_RAM,
    ):
        """Loads speech commands dataset with optional teacher model last layer logits.

        Args:
            audios_path (Path): path to speech command dataset
            logits_path (Path): path to teacher logit
            subset (str): 'train' or 'validation' or 'testing'
        """
        logging.info("Loading {} dataset".format(subset))
        logging.info("Cache spec: {}".format(cache_spec))
        logging.info("Keep dataset in RAM: {}".format(keep_dataset_in_ram))
        logging.info("Use distill: {}".format(bool(logits_path)))

        self.cache_spec = cache_spec
        self.logits_path = logits_path
        self.keep_dataset_in_ram = keep_dataset_in_ram
        self.transform = torchaudio.transforms.Resample(
            orig_freq=constants.ORIGINAL_SAMPLE_RATE, new_freq=constants.NEW_SAMPLE_RATE
        )

        subset_file = audios_path / f"{subset}_list.txt"
        if subset == "train":
            _create_train_subset_file(audios_path)

        with open(subset_file, "r") as f:
            file_list = f.readlines()

        self.list_data_label_mapping = self._get_data_label_mapping(
            file_list, audios_path, logits_path
        )

    def _get_data_label_mapping(
        self, file_list: List[str], audios_path: Path, logits_path: Path
    ) -> List[DataSample]:
        audio_path_list = [audios_path / sub_path.strip() for sub_path in file_list]
        label_list = [filepath.parent.stem for filepath in audio_path_list]
        label_list = [constants.LABELS.index(label) for label in label_list]

        if logits_path:
            logit_path_list = [
                logits_path / sub_path.strip().replace("wav", "pt")
                for sub_path in file_list
            ]
        else:
            logit_path_list = [None] * len(audio_path_list)

        if self.keep_dataset_in_ram:
            cache_spec_list = [self._load_audio_input(audio_path) for audio_path in tqdm(audio_path_list)]
        else:
            cache_spec_list = [None] * len(audio_path_list)
            
        list_data_label_mapping = []
        for audio_path, label, logit_path, cache_spec in zip(
            audio_path_list, label_list, logit_path_list, cache_spec_list
        ):
            data_label_map = DataSample(
                audio_path=audio_path,
                label=label,
                logit_path=logit_path,
                cache_spec=cache_spec,
            )
            list_data_label_mapping.append(data_label_map)
        return list_data_label_mapping

    def _load_audio_input(self, audio_path: Path) -> torch.Tensor:
        cached_spec_path = str(audio_path).replace(".wav", ".pt")
        if Path(cached_spec_path).exists() and self.cache_spec:
            spec = torch.load(cached_spec_path)
        else:
            spec = _load_audio_to_spec(audio_path, self.transform)
            torch.save(spec, cached_spec_path)

        return spec

    def __len__(self):
        return len(self.list_data_label_mapping)

    def __getitem__(self, index):
        data_sample = self.list_data_label_mapping[index]
        label = data_sample.label

        if self.keep_dataset_in_ram and data_sample.cache_spec is not None:
            student_input = data_sample.cache_spec 
        else:
            student_input = self._load_audio_input(data_sample.audio_path)
        if self.logits_path:
            teacher_logits = torch.load(data_sample.logit_path)
            output = student_input, teacher_logits, label
        else:
            output = student_input, label
        return output
        # return torch.rand((1, 48, 128)), label

def _load_audio_to_spec(filepath: Path, transform=None) -> torch.Tensor:
    waveform, sr = torchaudio.load(filepath)
    if transform:
        waveform = transform(waveform)

    waveform = waveform - waveform.mean()
    fbank = torchaudio.compliance.kaldi.fbank(
        waveform,
        htk_compat=True,
        sample_frequency=sr,
        use_energy=False,
        window_type="hanning",
        num_mel_bins=128,
        dither=0.0,
        frame_shift=10,
    )

    fbank = _pad_spec(fbank)
    fbank = torch.unsqueeze(fbank, dim=0)
    return fbank


def _pad_spec(spec: torch.Tensor) -> torch.Tensor:
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
