"""The benchmark's own inputs: answers written by llama3.2, then speech for every question and answer.

    python benchmarks/make_inputs.py llm-answers          # data/llm_answers.json (already committed)
    python benchmarks/make_inputs.py speech --out M:/openmeet-data/bench-speech

`llm-answers` asks llama3.2 (Ollama) to answer two of the interview
questions the way the room's reference answers are written (the recipe in
local-ai/integrity/signals.py) but with other seeds (1000 + i), so the room
benchmark has answers read out from an AI next to written-by-hand ones.
`speech` runs make_speech.ps1 (Windows text-to-speech, 16 kHz mono WAV).
"""

from __future__ import annotations

import argparse
import importlib.util
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import benchlib  # noqa: E402

DATA = benchlib.HERE / "data"
LLM_QUESTIONS = ("q01", "q06")


def signals_module():
    spec = importlib.util.spec_from_file_location("om_signals", benchlib.LOCAL_AI / "integrity" / "signals.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def llm_answers(model: str) -> None:
    signals = signals_module()
    pairs = {p["id"]: p for p in benchlib.read_json(DATA / "interview_pairs.json")["pairs"]}
    out = []
    for i, qid in enumerate(LLM_QUESTIONS):
        body = {"model": model, "messages": signals.reference_messages(pairs[qid]["question"]), "stream": False,
                "options": {"temperature": signals.REFERENCE_TEMPERATURE, "num_predict": signals.REFERENCE_MAX_TOKENS,
                            "seed": 1000 + i}}
        text = benchlib.post_json(f"{benchlib.OLLAMA_URL}/api/chat", body)["message"]["content"].strip()
        out.append({"id": qid, "question": pairs[qid]["question"], "answer": text, "seed": 1000 + i})
    benchlib.write_json(DATA / "llm_answers.json", {
        "about": f"Answers written by {model} (Ollama) with the reference recipe {signals.REFERENCE_RECIPE} of "
                 "local-ai/integrity/signals.py and seeds 1000+i; read out by text-to-speech in the room benchmark",
        "model": model, "answers": out})


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("llm-answers")
    s.add_argument("--model", default="llama3.2")
    s = sub.add_parser("speech")
    s.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    if a.cmd == "llm-answers":
        llm_answers(a.model)
    else:
        subprocess.run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(benchlib.HERE / "make_speech.ps1"),
                        "-Out", str(a.out)], check=True)


if __name__ == "__main__":
    main()
