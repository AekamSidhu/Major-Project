"""The desktop engine's pipeline latency: its built-in benchmark, and a live run with local models.

    python benchmarks/bench_engine.py offline --runs 20 --out benchmarks/results/engine_offline.json
    python benchmarks/bench_engine.py live --speech M:/openmeet-data/bench-speech --device cpu \
        --out benchmarks/results/engine_live_cpu.json

Builds benchmarks/engine-bench (a separate Cargo workspace that only calls
the engine's public API; set CARGO_TARGET_DIR to keep its build elsewhere)
and runs it:

- offline: Engine::run_benchmark() --runs times (the Diagnostics benchmark,
  synthetic input and a mock LLM: question detection, VAD, hashing
  embeddings, retrieval over 1,500 items, prompt assembly, pipeline overhead).
- live: the real pipeline (VAD -> sidecar Whisper on port 8795 -> question
  detection -> llama3.2 through Ollama), audio pushed in real time: 12 spoken
  questions in candidate mode, 12 spoken answers in interviewer mode, then
  the meeting summary. Timings only.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import benchlib  # noqa: E402

CRATE = benchlib.HERE / "engine-bench"
LIVE_FIELDS = ["speech_end_to_transcript_ms", "speech_end_to_question_ms", "question_to_first_token_ms",
               "speech_end_to_first_token_ms", "first_token_to_complete_ms", "speech_end_to_complete_ms"]
FOLLOWUP_FIELDS = ["speech_end_to_first_followup_ms", "speech_end_to_followups_done_ms"]


def build(jobs: int) -> Path:
    target = Path(os.environ.get("CARGO_TARGET_DIR") or CRATE / "target")
    subprocess.run(["cargo", "build", "--release", "--quiet", "-j", str(jobs), "--manifest-path", str(CRATE / "Cargo.toml")],
                   check=True)
    exe = target / "release" / ("openmeet-engine-bench.exe" if sys.platform == "win32" else "openmeet-engine-bench")
    if not exe.exists():
        raise FileNotFoundError(exe)
    return exe


def rust_version() -> str | None:
    try:
        return subprocess.run(["rustc", "--version"], capture_output=True, text=True, timeout=30).stdout.strip() or None
    except OSError:
        return None


def summarise_live(raw: dict[str, Any]) -> dict[str, Any]:
    qs, ans = raw["questions"], raw["answers"]
    return {
        "candidate_mode": {"questions": len(qs), "detected": sum(bool(q.get("question_detected")) for q in qs),
                           "single_transcript": sum(q.get("transcripts") == 1 for q in qs),
                           **{f: benchlib.summary([q.get(f) for q in qs]) for f in LIVE_FIELDS},
                           "answer_chars": benchlib.summary([q.get("answer_chars") for q in qs], 0)},
        "interviewer_mode": {"answers": len(ans), "with_followups": sum(bool(a.get("followups")) for a in ans),
                             **{f: benchlib.summary([a.get(f) for a in ans]) for f in FOLLOWUP_FIELDS},
                             "followups": benchlib.summary([a.get("followups") for a in ans], 0)},
        "summary_ms": raw.get("summary_ms"), "summary_ok": raw.get("summary_ok"),
        "engine_latency_stats": raw.get("engine_latency_stats"),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=("offline", "live"))
    ap.add_argument("--runs", type=int, default=20)
    ap.add_argument("--speech", type=Path)
    ap.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    ap.add_argument("--cuda-libs", type=Path, default=None)
    ap.add_argument("--model-dir", type=Path, default=None)
    ap.add_argument("--whisper-model", default="base.en")
    ap.add_argument("--binary", type=Path, default=None, help="an engine-bench binary already built")
    ap.add_argument("--jobs", type=int, default=2, help="cargo build jobs (keeps memory low)")
    ap.add_argument("--port", type=int, default=benchlib.SIDECAR_PORT)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    exe = a.binary or build(a.jobs)
    raw_path = benchlib.tmp_dir() / f"engine-{a.mode}-raw.json"
    env_info = {**benchlib.environment(["faster-whisper", "ctranslate2"] if a.mode == "live" else []), "rustc": rust_version(),
                "build": "cargo build --release (opt-level 3, no LTO)"}
    if a.mode == "offline":
        subprocess.run([str(exe), "offline", "--runs", str(a.runs), "--out", str(raw_path)], check=True)
        raw = benchlib.read_json(raw_path)
        items = [{"name": it["name"], "unit": it["unit"], "note": it["note"], **benchlib.summary(it["values"], 4)}
                 for it in raw["items"]]
        benchlib.write_json(a.out, {"what": "Engine::run_benchmark (Diagnostics benchmark), repeated",
                                    "environment": env_info, "runs": raw["runs"], "items": items})
    else:
        if a.speech is None:
            ap.error("live needs --speech")
        base = f"http://127.0.0.1:{a.port}"
        env = benchlib.sidecar_env(device=a.device, cuda_libs=a.cuda_libs, model_dir=a.model_dir)
        log = benchlib.tmp_dir() / f"engine-live-sidecar-{a.device}.log"
        with benchlib.server(["server.py", "--port", str(a.port), "--whisper-model", a.whisper_model], f"{base}/health", log, env=env):
            wav = benchlib.load_wav(next(a.speech.glob("*_question.wav")))
            body, ctype = benchlib.multipart([("model", a.whisper_model), ("language", "en")], [("file", "w.wav", wav, "audio/wav")])
            benchlib.post_bytes(f"{base}/v1/audio/transcriptions", body, ctype)  # load Whisper before timing
            benchlib.post_json(f"{benchlib.OLLAMA_URL}/api/chat", {"model": "llama3.2", "stream": False,
                                                                  "messages": [{"role": "user", "content": "Say ok."}]})
            subprocess.run([str(exe), "live", "--speech", str(a.speech), "--local-ai", base, "--ollama", benchlib.OLLAMA_URL,
                            "--whisper-model", a.whisper_model, "--out", str(raw_path)], check=True)
        raw = benchlib.read_json(raw_path)
        from bench_asr import LOADED

        loaded = [{"model": m, "device": d, "load_s": float(s)} for m, d, s in LOADED.findall(log.read_text(encoding="utf-8", errors="replace"))]
        benchlib.write_json(a.out, {"what": "desktop engine, live local pipeline (VAD, sidecar Whisper, question detection, Ollama)",
                                    "environment": {**env_info, "ollama": benchlib.ollama_version()},
                                    "config": {"whisper_model": a.whisper_model, "device": a.device, "llm": raw["llm"],
                                               "settings": raw["settings"], "speech": "Windows text-to-speech (make_inputs.py)"},
                                    "device_log": loaded, **summarise_live(raw), "raw": raw})
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()
