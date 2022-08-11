import torch
import torchaudio
from tqdm import tqdm
from constants import BATCH_SIZE, NUM_WORKERS, PIN_MEMORY, LABELS

from datasets.prebuild_dataset import AudioArrayDataSet
from datasets.sc_dataset import SpeechCommandDataModule
from models.simple_conv import simconv_collate_fn
import torch.nn.functional as F

from models.simple_conv import SimpleConv
# dm = SpeechCommandDataModule(AudioArrayDataSet, simconv_collate_fn)

# for idx, da in tqdm(enumerate(dm.train_dataloader())):
#     if idx == 200:
#         # print(da)
#         break



train_set = AudioArrayDataSet("train")
test_set = AudioArrayDataSet("testing")

train_loader = torch.utils.data.DataLoader(
    train_set,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=simconv_collate_fn,
    num_workers=NUM_WORKERS,
    pin_memory=PIN_MEMORY,
)
test_loader = torch.utils.data.DataLoader(
    test_set,
    batch_size=BATCH_SIZE,
    shuffle=False,
    drop_last=False,
    collate_fn=simconv_collate_fn,
    num_workers=NUM_WORKERS,
    pin_memory=PIN_MEMORY,
)
# for idx, da in tqdm(enumerate(train_loader)):
#     if idx == 200:
#         # print(da)
#         break

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleConv()
model.to(device)
# print(model)

transform = torchaudio.transforms.Resample(orig_freq=16000, new_freq=8000)
transform = transform.to(device)


optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0.0001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)  # reduce the learning after 20 epochs by a factor of 10



def train(model, epoch, log_interval):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):

        data = data.to(device)
        target = target.to(device)

        # apply transform and model on whole batch directly on device
        data = transform(data)
        output = model(data)

        # negative log-likelihood for a tensor of size (batch x 1 x n_output)
        loss = F.nll_loss(output.squeeze(), target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # print training stats
        if batch_idx % log_interval == 0:
            print(f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}")

        # update progress bar
        pbar.update(pbar_update)
        # record loss
        losses.append(loss.item())



def number_of_correct(pred, target):
    # count number of correct predictions
    return pred.squeeze().eq(target).sum().item()


def get_likely_index(tensor):
    # find most likely label index for each element in the batch
    return tensor.argmax(dim=-1)


def test(model, epoch):
    model.eval()
    correct = 0
    for data, target in test_loader:

        data = data.to(device)
        target = target.to(device)

        # apply transform and model on whole batch directly on device
        data = transform(data)
        output = model(data)

        pred = get_likely_index(output)
        correct += number_of_correct(pred, target)

        # update progress bar
        pbar.update(pbar_update)

    print(f"\nTest Epoch: {epoch}\tAccuracy: {correct}/{len(test_loader.dataset)} ({100. * correct / len(test_loader.dataset):.0f}%)\n")


log_interval = 20
n_epoch = 2

pbar_update = 1 / (len(train_loader) + len(test_loader))
losses = []

# The transform needs to live on the same device as the model and the data.
transform = transform.to(device)
with tqdm(total=n_epoch) as pbar:
    for epoch in range(1, n_epoch + 1):
        train(model, epoch, log_interval)
        test(model, epoch)
        scheduler.step()
