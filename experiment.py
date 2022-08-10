from pathlib import Path
import torch
from datasets.distill_dataloader import AudioDistillDataset
import time

audios_path = Path('data/SpeechCommands/speech_commands_v0.02/')
logits_path = Path('data/teacher_logits')

subsets = ['train', 'validation', 'testing']

batch_size = 256
num_workers = 2

eval_loader = torch.utils.data.DataLoader(
    AudioDistillDataset(audios_path, subsets[0]),
    batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
)

st = time.time()
for idx, batch in enumerate(eval_loader):
    print(idx)

    print(batch[0].shape)
    if idx > 1000:
        break
print(time.time() - st)