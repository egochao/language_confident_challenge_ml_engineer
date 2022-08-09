import torch
import torch.nn.functional as F
import torch.optim as optim
import torchaudio


from tqdm import tqdm

from dataset import get_dataloader
from models.simple_conv import SimpleConv

from constants import BATCH_SIZE, LABELS, ORIGINAL_SAMPLE_RATE, NEW_SAMPLE_RATE

from utils.model_utils import count_parameters

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_loader = get_dataloader("training", BATCH_SIZE, shuffle=True, drop_last=True)
test_loader = get_dataloader("testing", BATCH_SIZE, shuffle=False, drop_last=False)
val_loader = get_dataloader("validation", BATCH_SIZE, shuffle=False, drop_last=False)

transform = torchaudio.transforms.Resample(orig_freq=ORIGINAL_SAMPLE_RATE, new_freq=NEW_SAMPLE_RATE)

model = SimpleConv(n_input=1, n_output=len(LABELS))
model.to(device)
print(model)

n = count_parameters(model)
print("Number of parameters: %s" % n)

optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=0.0001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)  # reduce the learning after 20 epochs by a factor of 10
losses = []

def train(model, epoch, log_interval, pbar):
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

from test import test

log_interval = 20
n_epoch = 40

pbar_update = 1 / (len(train_loader) + len(test_loader))
losses = []

# The transform needs to live on the same device as the model and the data.
transform = transform.to(device)
with tqdm(total=n_epoch) as pbar:
    for epoch in range(1, n_epoch + 1):
        train(model, epoch, log_interval, pbar)
        test(model, epoch, test_loader, transform, device)
        scheduler.step()

# Let's plot the training loss versus the number of iteration.
# plt.plot(losses);
# plt.title("training loss");