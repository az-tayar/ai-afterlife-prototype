#!/bin/zsh
set -eu
cd "${0:A:h}"

if [[ -s .visitor-python && -x "$(<.visitor-python)" ]]; then
    python_bin="$(<.visitor-python)"
else
    [[ -x .venv/bin/python ]] || python3 -m venv .venv
    python_bin=".venv/bin/python"
    if [[ ! -f .venv/.visitor-ready || requirements.txt -nt .venv/.visitor-ready ]]; then
        "$python_bin" -m pip install -r requirements.txt
        touch .venv/.visitor-ready
    fi
fi

export ANCESTOR_INBROWSER=1
export GRADIO_ANALYTICS_ENABLED=False
if "$python_bin" - <<'PY'
import json
import os
import sys
import webbrowser
from urllib.request import urlopen
from src.app.settings import load_environment

load_environment()
url = "http://127.0.0.1:" + os.getenv("GRADIO_SERVER_PORT", "7860")
try:
    with urlopen(url + "/config", timeout=1) as response:
        running = json.load(response).get("title") == "AI Ancestor Interaction"
except (OSError, ValueError):
    running = False
if running:
    print(f"The visitor app is running at {url}", flush=True)
    webbrowser.open(url)
sys.exit(0 if running else 1)
PY
then
    exit 0
fi
exec "$python_bin" -m src.app.gradio_app
