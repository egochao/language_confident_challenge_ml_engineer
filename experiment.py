from pathlib import Path
from datasets.simple_dataloader import AudioDataset

if __name__ == "__main__":
    ds = AudioDataset(Path("./data/"), "train")

    print(ds[1])

