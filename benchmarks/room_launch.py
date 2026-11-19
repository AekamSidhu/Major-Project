"""Starts the interview room for the benchmarks, without reading ~/.openmeet/room.env.

    python benchmarks/room_launch.py --port 8463 --followups local --whisper-model base.en

Same app as local-ai/room_server.py (`create_room_app`), on plain http on
127.0.0.1, but the follow-up settings are built here from the command line
instead of Settings.load(), so no API key file is ever read. Online (Gemini)
mode is refused.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "local-ai"))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8463)
    ap.add_argument("--whisper-model", default="base.en")
    ap.add_argument("--followups", choices=("local", "rules"), default="local")
    ap.add_argument("--ollama-model", default="llama3.2")
    ap.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    for noisy in ("httpx", "httpx2", "httpcore", "huggingface_hub", "urllib3", "faster_whisper"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    import uvicorn
    from room.followups import FollowupWriter, Settings
    from room_server import create_room_app

    settings = Settings(mode=a.followups, ollama_model=a.ollama_model, ollama_url=a.ollama_url)
    writer = FollowupWriter(mode=a.followups, settings=settings)
    app = create_room_app(a.port, "http", a.whisper_model, followup_writer=writer)
    uvicorn.run(app, host="127.0.0.1", port=a.port, log_level="warning", access_log=False)


if __name__ == "__main__":
    main()
