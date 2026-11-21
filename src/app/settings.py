"""Load server-side settings before creating visitor-facing components."""

import os
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_environment():
    load_dotenv(PROJECT_ROOT / ".env", override=False)
    external_file = os.getenv("AFTERLIFE_ENV_FILE")
    if external_file:
        path = Path(external_file).expanduser()
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        if not path.is_file():
            raise FileNotFoundError("AFTERLIFE_ENV_FILE points to a missing settings file.")
        load_dotenv(path, override=False)
