"""Transcription accuracy and speed of the local AI sidecar on LibriSpeech.

    python benchmarks/bench_asr.py --root M:/openmeet-data/librispeech --models tiny.en base.en small.en \
        --device cpu --n 300 --out benchmarks/results/asr_cpu.json
    python benchmarks/bench_asr.py ... --device cuda --cuda-libs M:/openmeet-work/cuda-libs --out benchmarks/results/asr_cuda.json

For each Whisper model a fresh sidecar (local-ai/server.py, port 8795) is
started, so only one model is in memory and its cold start is measured.
Every utterance of a seeded subset of test-clean and test-other is sent
through the real endpoint, POST /v1/audio/transcriptions (16 kHz 16-bit WAV,
language "en", no prompt, response_format json; with --word-timestamps,
verbose_json with word timings as the room asks), one request at a time.

WER: OpenAI Whisper's English text normaliser (the `whisper_normalizer`
package, a copy of whisper/normalizers/english.py) on reference and
hypothesis, then word errors summed over the subset (jiwer). Latency: wall
time of the HTTP request (client side) and the sidecar's own inference time
(from its log line); real-time factor = latency / audio length.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import benchlib  # noqa: E402
import librispeech  # noqa: E402

LOADED = re.compile(r"loaded whisper model (\S+) on (\S+) in ([\d.]+)s")
TRANSCRIBED = re.compile(r"transcribed ([\d.]+)s audio in (\d+) ms")
GPU_FALLBACK = re.compile(r"GPU (unavailable|transcription failed)")


def normaliser():
    from whisper_normalizer.english import EnglishTextNormalizer

    return EnglishTextNormalizer()


def word_errors(ref: str, hyp: str) -> dict[str, int]:
    """Substitutions, deletions, insertions and reference words of one normalised pair."""
    import jiwer

    ref_words = len(ref.split())
    if not hyp.strip():
        return {"ref_words": ref_words, "sub": 0, "del": ref_words, "ins": 0}
    if not ref.strip():
        return {"ref_words": 0, "sub": 0, "del": 0, "ins": len(hyp.split())}
    out = jiwer.process_words(ref, hyp)
    return {"ref_words": ref_words, "sub": out.substitutions, "del": out.deletions, "ins": out.insertions}


def wer_of(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = sum(r["ref_words"] for r in rows)
    s, d, i = (sum(r[k] for r in rows) for k in ("sub", "del", "ins"))
    return {"wer": (s + d + i) / n if n else None, "ref_words": n, "substitutions": s, "deletions": d, "insertions": i}


def transcribe(url: str, wav: bytes, model: str, word_timestamps: bool) -> tuple[str, float]:
    fields = [("model", model), ("language", "en")]
    if word_timestamps:
        fields += [("response_format", "verbose_json"), ("timestamp_granularities[]", "word")]
    else:
        fields += [("response_format", "json")]
    body, ctype = benchlib.multipart(fields, [("file", "utterance.wav", wav, "audio/wav")])
    started = time.perf_counter()
    result = benchlib.post_bytes(url, body, ctype, timeout=600)
    return result["text"], (time.perf_counter() - started) * 1000.0


def audio_wav(path: Path) -> bytes:
    from faster_whisper.audio import decode_audio  # PyAV; LibriSpeech is 16 kHz 16-bit FLAC

    return benchlib.wav_bytes(decode_audio(str(path), sampling_rate=16000))


def run_model(model: str, items: list[librispeech.Utterance], device: str, a: argparse.Namespace, norm) -> tuple[dict, list[dict]]:
    log_path = benchlib.tmp_dir() / f"asr-sidecar-{model}-{device}.log"
    env = benchlib.sidecar_env(device=device, cuda_libs=a.cuda_libs, model_dir=a.model_dir)
    base = f"http://127.0.0.1:{a.port}"
    rows: list[dict[str, Any]] = []
    with benchlib.server(["server.py", "--port", str(a.port), "--whisper-model", model], f"{base}/health", log_path, env=env):
        warm = audio_wav(items[0].path)
        started = time.perf_counter()
        transcribe(f"{base}/v1/audio/transcriptions", warm, model, a.word_timestamps)
        cold_ms = (time.perf_counter() - started) * 1000.0
        print(f"  {model} on {device}: first request (loads the model) {cold_ms / 1000:.1f} s", flush=True)
        for k, utt in enumerate(items, 1):
            wav = audio_wav(utt.path)
            hyp, ms = transcribe(f"{base}/v1/audio/transcriptions", wav, model, a.word_timestamps)
            duration = benchlib.wav_duration(wav)
            rows.append({"model": model, "device": device, "split": utt.split, "id": utt.id, "duration_s": round(duration, 3),
                         "latency_ms": round(ms, 1), **word_errors(norm(utt.text), norm(hyp))})
            if k % 100 == 0 or k == len(items):
                so_far = wer_of(rows)["wer"]
                print(f"    {k}/{len(items)} utterances, WER so far {100 * so_far:.2f}%", flush=True)
    log = log_path.read_text(encoding="utf-8", errors="replace")
    loaded = LOADED.findall(log)
    server_ms = [int(m[1]) for m in TRANSCRIBED.findall(log)][1:]  # the first is the warm-up request
    if len(server_ms) == len(rows):
        for r, ms in zip(rows, server_ms):
            r["server_ms"] = ms
    info = {"cold_first_request_ms": round(cold_ms, 1),
            "device_used": loaded[-1][1] if loaded else None,
            "model_load_s": float(loaded[-1][2]) if loaded else None,
            "gpu_fallback_in_log": bool(GPU_FALLBACK.search(log))}
    return info, rows


def summarise(rows: list[dict[str, Any]]) -> dict[str, Any]:
    lat = [r["latency_ms"] for r in rows]
    rtf = [r["latency_ms"] / 1000.0 / r["duration_s"] for r in rows]
    out = {"utterances": len(rows), "speakers": len({r["id"].split("-")[0] for r in rows}),
           "audio_s": round(sum(r["duration_s"] for r in rows), 1), **wer_of(rows),
           "latency_ms": benchlib.summary(lat), "rtf": benchlib.summary(rtf, 3),
           "aggregate_rtf": round(sum(lat) / 1000.0 / sum(r["duration_s"] for r in rows), 4)}
    if all("server_ms" in r for r in rows):
        out["server_inference_ms"] = benchlib.summary([r["server_ms"] for r in rows])
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, required=True, help="folder with LibriSpeech/ (librispeech.py downloads it)")
    ap.add_argument("--models", nargs="+", default=["tiny.en", "base.en", "small.en"])
    ap.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    ap.add_argument("--cuda-libs", type=Path, default=None, help="folder with nvidia/*/bin (cuBLAS, cuDNN 9), put on the sidecar's PATH")
    ap.add_argument("--model-dir", type=Path, default=None, help="OPENMEET_MODEL_DIR for the sidecar (where Whisper models are kept)")
    ap.add_argument("--splits", nargs="+", default=list(librispeech.SPLITS))
    ap.add_argument("--n", type=int, default=300, help="utterances per split")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--word-timestamps", action="store_true", help="ask for word timings, as the room does")
    ap.add_argument("--port", type=int, default=benchlib.SIDECAR_PORT)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    items = []
    for split in a.splits:
        items += librispeech.subset(librispeech.utterances(a.root, split), a.n, a.seed)
    if not items:
        raise SystemExit(f"no utterances under {a.root}/LibriSpeech; run librispeech.py first")
    norm = normaliser()
    result: dict[str, Any] = {
        "what": "sidecar transcription (faster-whisper) on LibriSpeech through POST /v1/audio/transcriptions",
        "environment": benchlib.environment(["faster-whisper", "ctranslate2", "fastapi", "uvicorn", "whisper_normalizer", "jiwer"]),
        "config": {"device": a.device, "compute_type": "int8 (sidecar default)", "beam_size": "1 (sidecar default)",
                   "cpu_threads": "faster-whisper default (4)", "language": "en", "prompt": None,
                   "word_timestamps": a.word_timestamps, "per_split": a.n, "seed": a.seed,
                   "subset": "random.Random(seed).sample over the sorted utterance ids, per split",
                   "normaliser": "whisper_normalizer.english.EnglishTextNormalizer (OpenAI Whisper's)",
                   "cuda_libs": "nvidia-cublas-cu12 + nvidia-cudnn-cu12 9 wheels on PATH" if a.cuda_libs else None},
        "models": {},
    }
    all_rows: list[dict[str, Any]] = []
    for model in a.models:
        print(f"free RAM {benchlib.free_ram_gb()} GB; {model} on {a.device}, {len(items)} utterances", flush=True)
        info, rows = run_model(model, items, a.device, a, norm)
        all_rows += rows
        result["models"][model] = {**info, "all": summarise(rows),
                                   **{split: summarise([r for r in rows if r["split"] == split]) for split in a.splits}}
        m = result["models"][model]
        print(f"  {model}: " + ", ".join(f"{s} WER {100 * m[s]['wer']:.2f}%" for s in a.splits)
              + f"; median latency {m['all']['latency_ms']['median']} ms, aggregate RTF {m['all']['aggregate_rtf']}"
              + f"; device {info['device_used']}", flush=True)
        benchlib.write_json(a.out, result)  # after every model, so a stopped run keeps what it measured
    with a.out.with_name(a.out.stem + "_utterances.csv").open("w", newline="", encoding="utf-8") as f:
        fields = ["model", "device", "split", "id", "duration_s", "latency_ms", "server_ms", "ref_words", "sub", "del", "ins"]
        w = csv.DictWriter(f, fieldnames=fields, lineterminator="\n", extrasaction="ignore")
        w.writeheader()
        w.writerows(all_rows)
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()
