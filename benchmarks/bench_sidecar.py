"""Latency of the sidecar's embedding and answer-integrity endpoints.

    python benchmarks/bench_sidecar.py --face M:/openmeet-data/faces/astronaut.png --out benchmarks/results/sidecar_endpoints.json

One sidecar (local-ai/server.py, port 8795) on this machine; client-side
wall time of each request, after a first (cold) request that loads the model:

- POST /v1/embeddings: one answer, and three texts at once (an answer and
  two reference answers, as the room's echo-similarity check sends them);
- POST /v1/integrity/echo: the AI-written-text model on an answer;
- POST /v1/integrity/gaze/frame: one JPEG video frame (a public-domain
  portrait, at most 640 px, quality 0.8, as the room's host page sends them),
  10 a second for --gaze-seconds, never two in flight;
- POST /v1/integrity/reading: the reading network over the last 10 s and the
  last 60 s of that gaze.

Inputs are the benchmark's own answers (data/interview_pairs.json and
data/llm_answers.json); nothing else is sent.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Callable

sys.path.insert(0, str(Path(__file__).resolve().parent))
import benchlib  # noqa: E402

DATA = benchlib.HERE / "data"


def answers() -> list[str]:
    own = [p["answer"] for p in benchlib.read_json(DATA / "interview_pairs.json")["pairs"]]
    llm = [a["answer"] for a in benchlib.read_json(DATA / "llm_answers.json")["answers"]]
    return own + llm


def timed(call: Callable[[], Any]) -> tuple[Any, float]:
    started = time.perf_counter()
    out = call()
    return out, (time.perf_counter() - started) * 1000.0


def repeat(n: int, call: Callable[[int], Any]) -> tuple[list[float], list[Any]]:
    times, outs = [], []
    for i in range(n):
        out, ms = timed(lambda: call(i))
        times.append(ms)
        outs.append(out)
    return times, outs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--face", type=Path, required=True, help="a face photo for the gaze tracker")
    ap.add_argument("--model-dir", type=Path, default=None, help="OPENMEET_MODEL_DIR for the sidecar")
    ap.add_argument("--n", type=int, default=50, help="requests per endpoint")
    ap.add_argument("--gaze-seconds", type=float, default=70.0)
    ap.add_argument("--port", type=int, default=benchlib.SIDECAR_PORT)
    ap.add_argument("--out", type=Path, default=benchlib.RESULTS / "sidecar_endpoints.json")
    a = ap.parse_args()

    base = f"http://127.0.0.1:{a.port}"
    texts = answers()
    words = [len(t.split()) for t in texts]
    frame = benchlib.jpeg_frame(a.face)
    result: dict[str, Any] = {
        "what": "client-side latency of the sidecar's embedding and integrity endpoints (one request at a time)",
        "environment": benchlib.environment(["fastapi", "uvicorn", "fastembed", "onnxruntime", "mediapipe", "opencv-contrib-python", "numpy"]),
        "inputs": {"answers": len(texts), "words_median": sorted(words)[len(words) // 2], "words_max": max(words),
                   "frame_bytes": len(frame), "frame": "public-domain NASA portrait (scikit-image astronaut.png), JPEG q80"},
        "endpoints": {},
    }
    log = benchlib.tmp_dir() / "sidecar-endpoints.log"
    env = benchlib.sidecar_env(device="cpu", model_dir=a.model_dir)
    with benchlib.server(["server.py", "--port", str(a.port)], f"{base}/health", log, env=env):
        print(f"free RAM {benchlib.free_ram_gb()} GB", flush=True)
        ep = result["endpoints"]
        emb = f"{base}/v1/embeddings"
        _, cold = timed(lambda: benchlib.post_json(emb, {"input": texts[0]}))
        single, _ = repeat(a.n, lambda i: benchlib.post_json(emb, {"input": texts[i % len(texts)]}))
        triple, _ = repeat(a.n, lambda i: benchlib.post_json(emb, {"input": [texts[i % 12], texts[12], texts[13]]}))
        ep["embeddings_1_text"] = {"cold_first_ms": round(cold, 1), "latency_ms": benchlib.summary(single)}
        ep["embeddings_3_texts"] = {"latency_ms": benchlib.summary(triple), "note": "an answer and two reference answers"}
        print("embeddings", ep["embeddings_1_text"]["latency_ms"], ep["embeddings_3_texts"]["latency_ms"], flush=True)

        echo = f"{base}/v1/integrity/echo"
        _, cold = timed(lambda: benchlib.post_json(echo, {"text": texts[0]}))
        lat, outs = repeat(a.n, lambda i: benchlib.post_json(echo, {"text": texts[i % len(texts)]}))
        ep["integrity_echo"] = {"cold_first_ms": round(cold, 1), "latency_ms": benchlib.summary(lat, 2),
                                "levels": sorted({o["level"] for o in outs})}
        print("echo", ep["integrity_echo"]["latency_ms"], flush=True)

        frames = f"{base}/v1/integrity/gaze/frame"
        _, cold = timed(lambda: benchlib.post_bytes(frames, frame, "image/jpeg"))
        benchlib.post_json(f"{base}/v1/integrity/gaze/reset", {})
        with benchlib.FrameFeed(frames, frame) as feed:
            time.sleep(a.gaze_seconds)
        sent = len(feed.latencies_ms)
        ep["integrity_gaze_frame"] = {"cold_first_ms": round(cold, 1), "latency_ms": benchlib.summary(feed.latencies_ms),
                                      "frames": sent, "face_found": feed.face_found, "errors": feed.errors,
                                      "achieved_fps": round(sent / a.gaze_seconds, 2)}
        print("gaze frame", ep["integrity_gaze_frame"], flush=True)

        reading = f"{base}/v1/integrity/reading"
        for span in (10, 60):
            lat, outs = repeat(a.n, lambda i: benchlib.post_json(reading, {"seconds": span}))
            ep[f"integrity_reading_{span}s"] = {"latency_ms": benchlib.summary(lat, 2), "levels": sorted({o["level"] for o in outs}),
                                                "windows": len(outs[-1].get("window_probabilities") or [])}
            print(f"reading {span}s", ep[f"integrity_reading_{span}s"], flush=True)
    benchlib.write_json(a.out, result)
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()
