from pathlib import Path
import torchaudio
import torch
import torch.nn.functional
from torch.utils.data import Dataset
import constants
from typing import Optional
from datasets.distill_dataloader import AudioDistillDataset

data_path = Path("data/SpeechCommands/speech_commands_v0.02/")
all_wav = data_path.glob("**/*.wav")
print(len(list(all_wav)))

dataset = AudioDistillDataset(data_path, "train", cache_spec=True)

for wav in all_wav:
    print(wav)
    dataset._load_audio_input(wav)
