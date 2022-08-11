import torch

from tqdm import tqdm
from constants import BATCH_SIZE, NUM_WORKERS, PIN_MEMORY
from datasets.sc_dataset import collate_fn as new_collate_fn

from datasets.prebuild_dataset import SubsetSC

train_set = SubsetSC(subset = "train")

train_loader = torch.utils.data.DataLoader(
    train_set,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=new_collate_fn,
    num_workers=NUM_WORKERS,
    pin_memory=PIN_MEMORY,
)
for idx, da in tqdm(enumerate(train_loader)):
    # print(idx)
    if idx > 200:
        break
