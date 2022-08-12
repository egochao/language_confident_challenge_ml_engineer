from pathlib import Path

BATCH_SIZE = 256
ORIGINAL_SAMPLE_RATE = 16000
NEW_SAMPLE_RATE = 8000

DATA_DIR = Path("./data/")
SUB_DATASET_PATH = "SpeechCommands/speech_commands_v0.02/"
LEARNING_RATE = 1e-2
EPOCHS = 50

NUM_WORKERS = 4
PIN_MEMORY = True

LABELS = [
    "backward",
    "follow",
    "five",
    "bed",
    "zero",
    "on",
    "learn",
    "two",
    "house",
    "tree",
    "dog",
    "stop",
    "seven",
    "eight",
    "down",
    "six",
    "forward",
    "cat",
    "right",
    "visual",
    "four",
    "wow",
    "no",
    "nine",
    "off",
    "three",
    "left",
    "marvin",
    "yes",
    "up",
    "sheila",
    "happy",
    "bird",
    "go",
    "one",
]
