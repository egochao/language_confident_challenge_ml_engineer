from pathlib import Path
from pydoc import pathdirs
from typing import Any, Optional

from pydantic import BaseModel


class DataSample(BaseModel):
    audio_path: Path
    label: int
