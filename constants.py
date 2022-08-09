from pathlib import Path

BATCH_SIZE = 128
ORIGINAL_SAMPLE_RATE = 16000
INPUT_AUDIO_LENGTH = 16000
NEW_SAMPLE_RATE = 8000

DATA_DIR = Path('./data/')
SUB_DATASET_PATH = 'SpeechCommands/speech_commands_v0.02/'



LABELS = ['backward',
 'follow',
 'five',
 'bed',
 'zero',
 'on',
 'learn',
 'two',
 'house',
 'tree',
 'dog',
 'stop',
 'seven',
 'eight',
 'down',
 'six',
 'forward',
 'cat',
 'right',
 'visual',
 'four',
 'wow',
 'no',
 'nine',
 'off',
 'three',
 'left',
 'marvin',
 'yes',
 'up',
 'sheila',
 'happy',
 'bird',
 'go',
 'one']

