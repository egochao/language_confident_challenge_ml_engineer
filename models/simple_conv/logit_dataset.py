from pathlib import Path
from models.simple_conv.base_dataset import AudioArrayDataSet


class AudioArrayWithLogitDataset(AudioArrayDataSet):
    def __init__(self, logit_path: Path, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.logit_walker = [logit_path.joinpath(each.replace('wav', 'pt')) for each in self.file_list]
