from pathlib import Path

BATCH_SIZE = 256
ORIGINAL_SAMPLE_RATE = 16000
NEW_SAMPLE_RATE = 8000
EPS = 1e-9
SCALE_BC_RESNET = 1  # Smallest model in bc_resnet farmily


DATA_DIR = Path("./data/")
SUB_DATASET_PATH = "SpeechCommands/speech_commands_v0.02/"
LEARNING_RATE = 1e-2
EPOCHS = 50
DROPOUT = 0.1

# distill loss
ALPHA = 0.1
TEMPERATURE = 10

LOGITS_PATH = Path("./data/teacher_logits")

NUM_WORKERS = 2
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
