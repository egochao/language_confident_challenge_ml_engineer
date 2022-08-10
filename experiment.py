from pathlib import Path
import torch
from datasets.distill_dataloader import AudioDistillDataset
import time
from datasets.sc_dataset import SpeechCommandDataModule
from utils.model_utils import get_loader_params

audios_path = Path('data/SpeechCommands/speech_commands_v0.02/')
logits_path = Path('data/teacher_logits')

data_dir = './data/'
batch_size = 1028
num_workers, pin_memory = get_loader_params()
data_module = SpeechCommandDataModule(data_dir, batch_size)
data_module.prepare_data()
data_module.setup()


st = time.time()
for idx, batch in enumerate(data_module.train_dataloader()):
    print(idx)

    print(batch[0].shape)
    if idx > 100:
        break
print(time.time() - st)


# from pathlib import Path
# import torchaudio
# import torch
# import torch.nn.functional
# from torch.utils.data import Dataset
# import constants
# from typing import Optional
# from datasets.distill_dataloader import AudioDistillDataset

# data_path = Path('data/SpeechCommands/speech_commands_v0.02/')
# all_wav = data_path.glob('**/*.pt')
# # print(len(list(all_wav)))

# list_data = []
# for idx, wav in enumerate(all_wav):
#     print(idx, wav)
#     list_data.append(torch.load(wav))