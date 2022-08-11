import torch

from tqdm import tqdm
from constants import BATCH_SIZE, NUM_WORKERS, PIN_MEMORY
from datasets.sc_dataset import collate_fn

from datasets.prebuild_dataset import SubsetSC
from datasets.sc_dataset import SpeechCommandDataModule


dm = SpeechCommandDataModule(SubsetSC)

for idx, da in tqdm(enumerate(dm.train_dataloader())):
    if idx == 200:
        # print(da)
        break



