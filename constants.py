from pathlib import Path

BATCH_SIZE = 256
ORIGINAL_SAMPLE_RATE = 16000
PADDED_SPEC_HEIGHTS = 48
NEW_SAMPLE_RATE = 8000

DATA_DIR = Path("./data/")
SUB_DATASET_PATH = "SpeechCommands/speech_commands_v0.02/"
KEEP_DATASET_IN_RAM = True
LEARNING_RATE = 1e-3

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
