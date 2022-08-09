import torch
import torch.nn.functional as F
import torch.optim as optim
import torchaudio

from dataset import get_dataloader
from models.simple_conv import SimpleConv

from constants import BATCH_SIZE, ORIGINAL_SAMPLE_RATE, NEW_SAMPLE_RATE
import constants

from test import test

def train(model, loader, optimizer, device, epoch=40, log_interval=40):
    model.train()
    losses = []
    for batch_idx, (data, target) in enumerate(loader):

        data = data.to(device)
        target = target.to(device)

        # apply transform and model on whole batch directly on device
        output = model(data)

        # negative log-likelihood for a tensor of size (batch x 1 x n_output)
        loss = F.nll_loss(output.squeeze(), target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # print training stats
        if batch_idx % log_interval == 0:
            print(f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(loader.dataset)} ({100. * batch_idx / len(loader):.0f}%)]\tLoss: {loss.item():.6f}")

        losses.append(loss.item())

    return losses


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from dataset import SpeechCommandDataModule
    dm = SpeechCommandDataModule()
    dm.prepare_data()
    dm.setup()


    train_loader = dm.train_dataloader()
    test_loader = dm.test_dataloader()
    val_loader = dm.val_dataloader()


    model = SimpleConv(n_input=1, n_output=len(constants.LABELS))
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=0.0001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)  # reduce the learning after 20 epochs by a factor of 10

    log_interval = 20
    epoch = 40

    # The transform needs to live on the same device as the model and the data.
    for epoch in range(1, epoch + 1):
        train(model, train_loader, optimizer, device, epoch, log_interval)
        test(model, epoch, test_loader, device)
        scheduler.step()


if __name__ == '__main__':
    main()
