from pathlib import Path
from models.simple_conv.base_dataset import AudioArrayWithLogitDataset
import torch

import constants

logit_path = Path("data/teacher_logits")


def test_logit_dataset():
    ds = AudioArrayWithLogitDataset(logit_path=logit_path, subset="train")
    for (log, au) in zip(ds.logit_walker, ds._walker):
        assert log.stem == au.stem

    expected_logit_shape = len(constants.LABELS)

    logit_shape = len(ds.__getitem__(0)[1])

    assert logit_shape == expected_logit_shape
