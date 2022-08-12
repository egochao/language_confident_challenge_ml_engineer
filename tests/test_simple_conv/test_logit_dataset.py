from pathlib import Path
from models.simple_conv.logit_dataset import AudioArrayWithLogitDataset

logit_path=Path("data/teacher_logits")
logit_path=Path("data/SpeechCommands/speech_commands_v0.02")


def test_logit_dataset():
    ds = AudioArrayWithLogitDataset(logit_path=logit_path, subset='train')
    for idx, (log, au) in enumerate(zip(ds.logit_walker, ds._walker)):
        assert log.stem == au.stem
