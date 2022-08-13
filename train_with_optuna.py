import pytorch_lightning as pl
from torch.nn import functional as F
import optuna
from optuna.integration import PyTorchLightningPruningCallback

import constants
from lightling_wrapper import BaseTorchLightlingWrapper, SpeechCommandDataModule
from models.bc_resnet.bc_resnet_model import BcResNetModel
from models.bc_resnet.mel_spec_dataset import MelSpecDataSet, mel_collate_fn
from models.simple_conv.base_dataset import AudioArrayDataSet, simconv_collate_fn
from models.simple_conv.simple_conv_model import SimpleConv


def sim_conv_param_search(trial: optuna.trial.Trial) -> float:
    loss_fn = F.nll_loss
    collate_fn = simconv_collate_fn
    dataset_fn = AudioArrayDataSet

    # stride = trial.suggest_int("stride", 10, 20)
    n_channel = trial.suggest_int("n_channel", 24, 38)
    kernel_size_l1 = trial.suggest_int("kernel_size_l1", 60, 100)

    model = BaseTorchLightlingWrapper(
        core_model=SimpleConv(
            n_channel=n_channel, kernel_size_l1=kernel_size_l1
        ),
        loss_fn=loss_fn,
        learning_rate=constants.LEARNING_RATE,
    )

    data_module = SpeechCommandDataModule(
        dataset_fn, collate_fn, batch_size=constants.BATCH_SIZE
    )

    trainer = pl.Trainer(
        logger=True,
        enable_checkpointing=False,
        max_epochs=constants.EPOCHS,
        accelerator="gpu",
        devices=1,
        callbacks=[PyTorchLightningPruningCallback(trial, monitor="val_acc")],
    )

    hyperparameters = dict(
        n_channel=n_channel, kernel_size_l1=kernel_size_l1
    )
    trainer.logger.log_hyperparams(hyperparameters)
    trainer.fit(model, datamodule=data_module)

    return trainer.callback_metrics["val_acc"].item()


if __name__ == "__main__":
    pruner: optuna.pruners.BasePruner = optuna.pruners.MedianPruner()

    study = optuna.create_study(direction="maximize", pruner=pruner)
    study.optimize(sim_conv_param_search, n_trials=100, timeout=1000)

    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
