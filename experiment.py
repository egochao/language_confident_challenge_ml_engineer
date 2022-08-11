import torch

from tqdm import tqdm
from constants import BATCH_SIZE, NUM_WORKERS, PIN_MEMORY

from datasets.prebuild_dataset import AudioArrayDataSet
from datasets.sc_dataset import SpeechCommandDataModule
from models.simple_conv import simconv_collate_fn

dm = SpeechCommandDataModule(AudioArrayDataSet, simconv_collate_fn)

for idx, da in tqdm(enumerate(dm.train_dataloader())):
    if idx == 200:
        # print(da)
        break



