import pytorch_lightning as pl
from torch.nn import functional as F
import optuna
from optuna.integration import PyTorchLightningPruningCallback
import logging
import argparse

import constants
from lightling_wrapper import BaseTorchLightlingWrapper, SpeechCommandDataModule
from lightling_wrapper.data_module import DistillSpeechCommandDataModule
from lightling_wrapper.train_module import DistillModelTorchLightlingWrapper
from models.bc_resnet.bc_resnet_model import BcResNetModel, BcResNetModelNoSoftMax
from models.bc_resnet.mel_spec_dataset import MelSpecDataSet, MelSpecWithLogitDataset, mel_collate_fn, mel_collate_logit_fn
from models.simple_conv.base_dataset import AudioArrayDataSet, AudioArrayWithLogitDataset, simconv_collate_fn, simconv_collate_logit_fn
from models.simple_conv.simple_conv_model import SimpleConv, SimpleConvNoSoftMax
from utils.model_utils import distillation_loss


logging.getLogger().setLevel(logging.INFO)
logging.getLogger().addHandler(logging.FileHandler('foo_distill.log'))


def sim_conv_param_search(trial: optuna.trial.Trial) -> float:
    loss_fn = distillation_loss
    collate_fn = simconv_collate_logit_fn
    dataset = AudioArrayWithLogitDataset

    # stride = trial.suggest_int("stride", 10, 20)
    alpha = trial.suggest_float("alpha", 0.1, 0.9)
    temperature = trial.suggest_int("temperature", 2, 20)

    model = DistillModelTorchLightlingWrapper(
        core_model=SimpleConvNoSoftMax(),
        loss_fn=loss_fn,
        learning_rate=constants.LEARNING_RATE,
        alpha=alpha, temperature=temperature
    )

    data_module = DistillSpeechCommandDataModule(
        dataset=dataset, collate_fn=collate_fn, batch_size=constants.BATCH_SIZE
    )

    trainer = pl.Trainer(
        logger=True,
        enable_checkpointing=False,
        max_epochs=constants.EPOCHS,
        accelerator="gpu",
        devices=1,
        callbacks=[PyTorchLightningPruningCallback(trial, monitor="val_acc")],
    )

    hyperparameters = dict(alpha=alpha, temperature=temperature)
    trainer.logger.log_hyperparams(hyperparameters)
    trainer.fit(model, datamodule=data_module)

    return trainer.callback_metrics["val_acc"].item()


def bc_resnet_param_search(trial: optuna.trial.Trial) -> float:
    loss_fn = distillation_loss
    collate_fn = mel_collate_logit_fn
    dataset = MelSpecWithLogitDataset

    alpha = trial.suggest_float("alpha", 0.1, 0.9)
    temperature = trial.suggest_int("temperature", 2, 20)

    model = DistillModelTorchLightlingWrapper(
        core_model=BcResNetModelNoSoftMax(scale=constants.SCALE_BC_RESNET),
        loss_fn=loss_fn,
        learning_rate=constants.LEARNING_RATE,
        alpha=alpha, temperature=temperature
    )

    data_module = DistillSpeechCommandDataModule(
        dataset=dataset, collate_fn=collate_fn, batch_size=constants.BATCH_SIZE
    )

    trainer = pl.Trainer(
        logger=True,
        enable_checkpointing=False,
        max_epochs=constants.EPOCHS,
        accelerator="gpu",
        devices=1,
        callbacks=[PyTorchLightningPruningCallback(trial, monitor="val_acc")],
    )

    hyperparameters = dict(alpha=alpha, temperature=temperature)
    trainer.logger.log_hyperparams(hyperparameters)
    trainer.fit(model, datamodule=data_module)

    return trainer.callback_metrics["val_acc"].item()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str)
    args = parser.parse_args()

    pruner: optuna.pruners.BasePruner = optuna.pruners.MedianPruner()

    optuna.logging.enable_propagation()  # enable optuna logging
    optuna.logging.disable_default_handler()  # Stop showing logs in stderr.

    study = optuna.create_study(direction="maximize", pruner=pruner)
    logging.getLogger().info("Start optimization.")


    if args.model == "sim_conv" or args.model is None:
        study.optimize(sim_conv_param_search, n_trials=200, timeout=None)
    elif args.model == "bc_resnet":
        study.optimize(bc_resnet_param_search, n_trials=1000, timeout=None)

    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
