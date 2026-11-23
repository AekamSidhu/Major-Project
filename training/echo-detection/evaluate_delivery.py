"""Evaluate the spec's delivery index on read vs spontaneous speech (Phase 2 of the echo plan).

    # SpiCE (Johnson et al. 2021, CC BY 4.0): the same 34 speakers read sentences and are interviewed
    python evaluate_delivery.py spice --textgrids M:/openmeet-data/spice/english --out results/delivery_spice.json
    python evaluate_delivery.py spice --textgrids ... --whisper --audio-cache M:/openmeet-data/spice/audio \
        --files M:/openmeet-data/spice/english_files.json --out results/delivery_spice_whisper.json

    # Any folder of labelled recordings: DIR/<speaker>/<read|spontaneous>/<answer>.wav (or .json)
    python evaluate_delivery.py folder DIR --out results/delivery_folder.json

Delivery is defined in spec/integrity/README.md (features, baseline and
index; implemented in local-ai/integrity/signals.py, which this script
loads). Nothing here changes the spec; fitted weights are reported only.

Protocol, fixed before any result was read:
- An "answer" is a stretch of one speaker's speech. In SpiCE, answers are
  cut from the hand-aligned word tier: 30 consecutive words within one task,
  ended early (and dropped) at a gap over 2 s, so read and spontaneous
  answers have the same length in words. Read = the sentence-reading task;
  spontaneous = the conversational interview; the picture-story narration is
  reported on its own.
- Baseline, per speaker: their first 3 spontaneous answers (as the room
  uses the candidate's first answers). Every other answer is scored against
  it with the spec's weights and thresholds.
- Up to 15 further spontaneous answers per speaker, evenly spread over the
  rest of the interview, plus every read answer.
- Main number: ROC-AUC of the delivery index, read (positive) against
  spontaneous, with a 95% interval from resampling speakers. Also the share
  of each class at the spec's `possible` and `high` levels.
- Fitted weights (for information only): logistic regression on the same
  z-scores, trained on 70% of speakers and tested on the other 30% (split by
  speaker, seed 2026), against the spec index on the same test speakers.
- Latency cannot be measured on this data (no question before a read
  sentence), so the index uses speech rate, pauses and fillers only.

Timings: `textgrid` uses the corpus's hand transcripts with forced-aligned
word times (fillers transcribed); `--whisper` instead transcribes each
answer's audio with faster-whisper base.en and word timestamps, the way the
room does (which drops many fillers).
"""

from __future__ import annotations

import argparse
import importlib.util
import io
import json
import re
import struct
import sys
import time
import urllib.request
import wave
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
SEED = 2026
CHUNK_WORDS = 30
MAX_GAP_S = 2.0
UTTERANCE_GAP_S = 0.8  # the room's browser cuts speech after 0.8 s of silence
BASELINE = 3
SPONTANEOUS_PER_SPEAKER = 15
TASKS = {"sentences": "read", "interview": "spontaneous", "storyboard": "narration"}


def load_signals():
    spec = importlib.util.spec_from_file_location("openmeet_signals", REPO / "local-ai" / "integrity" / "signals.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


SIGNALS = load_signals()


# --- answers from words ------------------------------------------------------------


def utterances_from_words(words: list[dict], gap: float = UTTERANCE_GAP_S) -> list[dict]:
    """Group timed words into utterances at silences of at least `gap` seconds."""
    out: list[dict] = []
    for w in sorted(words, key=lambda w: w["start"]):
        if out and w["start"] - out[-1]["end"] < gap:
            out[-1]["words"].append(w)
            out[-1]["end"] = max(out[-1]["end"], w["end"])
        else:
            out.append({"start": w["start"], "end": w["end"], "words": [w]})
    for u in out:
        u["text"] = " ".join(w["word"] for w in u["words"])
    return out


# --- SpiCE ------------------------------------------------------------------------------


def read_textgrid(path: Path) -> dict[str, list[tuple[float, float, str]]]:
    """Interval tiers of a Praat TextGrid (long format, UTF-16 or UTF-8)."""
    raw = path.read_bytes()
    text = raw.decode("utf-16") if raw[:2] in (b"\xff\xfe", b"\xfe\xff") else raw.decode("utf-8-sig")
    tiers: dict[str, list[tuple[float, float, str]]] = {}
    for block in re.split(r"\n\s*item \[\d+\]:", text)[1:]:
        name = re.search(r'name = "(.*?)"', block)
        if not name:
            continue
        tiers[name.group(1)] = [(float(a), float(b), t.strip()) for a, b, t in
                                re.findall(r'xmin = ([\d.]+)\s+xmax = ([\d.]+)\s+text = "(.*?)"', block, re.S)]
    return tiers


def spice_answers(path: Path, speaker: str | None = None) -> list[dict]:
    """30-word answers of one SpiCE session, labelled by task; utterances from the utterance tier."""
    tiers = read_textgrid(path)
    speaker = speaker or path.name.split("_")[0]
    utts = [(a, b) for a, b, t in tiers["utterance"] if t]
    words = [(a, b, t) for a, b, t in tiers["word"] if t and t not in ("sil", "sp", "<unk>")]
    out: list[dict] = []
    for t0, t1, task in tiers["task"]:
        label = TASKS.get(task.strip().lower())
        if label is None:
            continue
        chunk: list[tuple[float, float, str]] = []

        def close() -> None:
            if len(chunk) == CHUNK_WORDS:
                out.append({"speaker": speaker, "label": label, "order": len(out), "question_end": None,
                            "utterances": group_by_utterance(chunk, utts), "source": f"{path.name}@{chunk[0][0]:.2f}"})

        for w in (w for w in words if t0 <= w[0] < t1):
            if chunk and w[0] - chunk[-1][1] > MAX_GAP_S:
                close()
                chunk = []
            chunk.append(w)
            if len(chunk) == CHUNK_WORDS:
                close()
                chunk = []
    return out


def group_by_utterance(chunk: list[tuple[float, float, str]], utts: list[tuple[float, float]]) -> list[dict]:
    """The chunk's words, grouped by the transcript utterance they fall in (spec utterances)."""
    groups: dict[int, list[dict]] = {}
    for a, b, t in chunk:
        mid = (a + b) / 2
        idx = next((i for i, (u0, u1) in enumerate(utts) if u0 <= mid <= u1), -1)
        groups.setdefault(idx if idx >= 0 else -int(a * 1000) - 1, []).append({"word": t, "start": a, "end": b})
    out = []
    for ws in groups.values():
        out.append({"start": ws[0]["start"], "end": ws[-1]["end"], "text": " ".join(w["word"] for w in ws), "words": ws})
    return sorted(out, key=lambda u: u["start"])


def select(answers: list[dict]) -> list[dict]:
    """Per speaker: the first BASELINE spontaneous answers (baseline), up to SPONTANEOUS_PER_SPEAKER
    more spread evenly over the rest, and every read and narration answer."""
    out: list[dict] = []
    for speaker in sorted({a["speaker"] for a in answers}):
        mine = sorted((a for a in answers if a["speaker"] == speaker), key=lambda a: a["order"])
        spont = [a for a in mine if a["label"] == "spontaneous"]
        base, rest = spont[:BASELINE], spont[BASELINE:]
        if len(rest) > SPONTANEOUS_PER_SPEAKER:
            rest = [rest[int(i)] for i in np.linspace(0, len(rest) - 1, SPONTANEOUS_PER_SPEAKER).round().astype(int)]
        for a in base:
            a["role"] = "baseline"
        out += base + rest + [a for a in mine if a["label"] != "spontaneous"]
    return out


# --- audio (Whisper mode) ---------------------------------------------------------------


class RemoteWav:
    """Reads stretches of one channel of a remote PCM WAV with HTTP range requests (no full download)."""

    def __init__(self, url: str) -> None:
        self.url = url
        head = self._get(0, 65535)
        if head[:4] != b"RIFF" or head[8:12] != b"WAVE":
            raise ValueError("not a WAV file")
        pos = 12
        while pos + 8 <= len(head):
            cid, size = head[pos : pos + 4], struct.unpack("<I", head[pos + 4 : pos + 8])[0]
            if cid == b"fmt ":
                _, self.channels, self.rate, _, self.block, self.bits = struct.unpack("<HHIIHH", head[pos + 8 : pos + 24])
            elif cid == b"data":
                self.data = pos + 8
                break
            pos += 8 + size + (size & 1)
        else:
            raise ValueError("no data chunk in the first 64 KB")

    def _get(self, start: int, end: int) -> bytes:
        request = urllib.request.Request(self.url, headers={"Range": f"bytes={start}-{end}", "User-Agent": "openmeet-training"})
        with urllib.request.urlopen(request, timeout=120) as response:
            return response.read()

    def read(self, t0: float, t1: float, channel: int = 0) -> np.ndarray:
        """Samples of `channel` between t0 and t1 seconds, as float32 at 16 kHz."""
        f0, f1 = int(max(0.0, t0) * self.rate), int(t1 * self.rate)
        raw = self._get(self.data + f0 * self.block, self.data + f1 * self.block - 1)
        x = np.frombuffer(raw[: len(raw) // self.block * self.block], dtype="<i2").reshape(-1, self.channels)[:, channel]
        x = x.astype(np.float32) / 32768.0
        n = int(round(len(x) / self.rate * 16000))
        return np.interp(np.arange(n) * self.rate / 16000, np.arange(len(x)), x).astype(np.float32)


def whisper_words(model_name: str = "base.en", cache: Path | None = None) -> Callable[[np.ndarray], list[dict]]:
    """faster-whisper with the sidecar's settings and word timestamps (first window only for short clips)."""
    from faster_whisper import WhisperModel

    model = WhisperModel(model_name, device="cpu", compute_type="int8", download_root=str(cache) if cache else None)

    def run(audio: np.ndarray) -> list[dict]:
        segments, info = model.transcribe(audio, language="en", beam_size=1, vad_filter=False, condition_on_previous_text=False,
                                          without_timestamps=True, word_timestamps=True)
        words: list[dict] = []
        first_seek = None
        for seg in segments:
            if info.duration <= 30.0 and first_seek is not None and seg.seek != first_seek:
                break  # the re-decoded tail (see local-ai/server.py)
            first_seek = seg.seek if first_seek is None else first_seek
            words += [{"word": w.word.strip(), "start": float(w.start), "end": float(w.end)} for w in seg.words or [] if w.word.strip()]
        return words

    return run


def retime_with_whisper(answers: list[dict], files: dict[str, int], transcribe: Callable[[np.ndarray], list[dict]],
                        cache: Path, log: Callable[[str], None] = print) -> list[dict]:
    """Replace each SpiCE answer's words by Whisper's, from the participant's channel of its audio."""
    wavs: dict[str, RemoteWav] = {}
    cache.mkdir(parents=True, exist_ok=True)
    out = []
    started = time.perf_counter()
    for n, a in enumerate(answers, 1):
        name = a["source"].split("@")[0].replace(".TextGrid", ".wav")
        t0, t1 = a["utterances"][0]["start"] - 0.3, a["utterances"][-1]["end"] + 0.3
        if t1 - t0 > 29.0:
            continue
        snippet = cache / f"{name[:-4]}_{t0:.2f}_{t1:.2f}.npy"
        if snippet.exists():
            audio = np.load(snippet)
        else:
            if name not in wavs:
                wavs[name] = RemoteWav(f"https://borealisdata.ca/api/access/datafile/{files[name]}")
            audio = wavs[name].read(t0, t1, channel=0)  # left: the participant
            np.save(snippet, audio)
        words = [{**w, "start": w["start"] + t0, "end": w["end"] + t0} for w in transcribe(audio)]
        if words:
            out.append({**a, "utterances": utterances_from_words(words), "timing": "whisper"})
        if n % 50 == 0:
            log(f"  whisper: {n}/{len(answers)} answers [{time.perf_counter() - started:.0f} s]")
    return out


# --- a folder of labelled recordings ------------------------------------------------------------


def folder_answers(root: Path, transcribe: Callable[[np.ndarray], list[dict]] | None = None) -> list[dict]:
    """DIR/<speaker>/<read|spontaneous>/<answer>.(wav|json), in file-name order per speaker.

    A .json file holds the answer's timed words: OpenAI verbose_json ({"words": [...]}) or
    spec utterances ({"utterances": [...]}). A .wav is transcribed with `transcribe`.
    """
    out: list[dict] = []
    for speaker_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        files = sorted((f for label in ("read", "spontaneous") for f in (speaker_dir / label).glob("*")
                        if f.suffix.lower() in (".wav", ".json")), key=lambda f: f.name)
        for order, f in enumerate(files):
            label = f.parent.name
            if f.suffix.lower() == ".json":
                data = json.loads(f.read_text(encoding="utf-8"))
                utts = data.get("utterances") or utterances_from_words(data["words"])
            else:
                if transcribe is None:
                    raise ValueError(f"{f} needs Whisper (run without --no-whisper)")
                utts = utterances_from_words(transcribe(load_wav(f)))
            out.append({"speaker": speaker_dir.name, "label": label, "order": order, "question_end": data.get("question_end")
                        if f.suffix.lower() == ".json" else None, "utterances": utts, "source": str(f.relative_to(root))})
    return out


def load_wav(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as w:
        rate, ch, width, frames = w.getframerate(), w.getnchannels(), w.getsampwidth(), w.readframes(w.getnframes())
    if width != 2:
        raise ValueError("16-bit PCM WAV only")
    x = np.frombuffer(frames, dtype="<i2").reshape(-1, ch).mean(axis=1).astype(np.float32) / 32768.0
    n = int(round(len(x) / rate * 16000))
    return np.interp(np.arange(n) * rate / 16000, np.arange(len(x)), x).astype(np.float32)


def baseline_first(answers: list[dict]) -> list[dict]:
    """Folder mode: each speaker's first BASELINE spontaneous answers form the baseline."""
    for speaker in {a["speaker"] for a in answers}:
        spont = sorted((a for a in answers if a["speaker"] == speaker and a["label"] == "spontaneous"), key=lambda a: a["order"])
        for a in spont[:BASELINE]:
            a["role"] = "baseline"
    return answers


# --- scoring and metrics ------------------------------------------------------------------------


def score(answers: list[dict]) -> list[dict]:
    """Spec delivery for every non-baseline answer, against its speaker's baseline answers."""
    rows = []
    for speaker in sorted({a["speaker"] for a in answers}):
        mine = [a for a in answers if a["speaker"] == speaker]
        base = [a for a in mine if a.get("role") == "baseline"]
        scorer = SIGNALS.DeliveryBaseline()
        for a in base:
            scorer.add(a["question_end"], a["utterances"])
        if scorer.baseline is None:
            continue  # fewer than BASELINE usable spontaneous answers
        for a in mine:
            if a.get("role") == "baseline":
                continue
            r = scorer.add(a["question_end"], a["utterances"])
            if r["score"]["index"] is None:
                continue
            rows.append({"speaker": speaker, "label": a["label"], "index": r["score"]["index"], "level": r["score"]["level"],
                         "z": r["score"]["z"], "features": r["features"]})
    return rows


def auc(pos: Iterable[float], neg: Iterable[float]) -> float | None:
    pos, neg = np.asarray(list(pos), float), np.asarray(list(neg), float)
    if not len(pos) or not len(neg):
        return None
    order = np.concatenate([pos, neg]).argsort(kind="mergesort")
    ranks = np.empty(len(order))
    allv = np.concatenate([pos, neg])[order]
    # average ranks for ties
    i = 0
    while i < len(allv):
        j = i
        while j + 1 < len(allv) and allv[j + 1] == allv[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return float((ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def speaker_bootstrap(rows: list[dict], n: int = 1000, seed: int = SEED) -> list[float] | None:
    speakers = sorted({r["speaker"] for r in rows})
    rng = np.random.default_rng(seed)
    by = {s: [r for r in rows if r["speaker"] == s] for s in speakers}
    vals = []
    for _ in range(n):
        pick = [r for s in rng.choice(speakers, len(speakers)) for r in by[s]]
        v = auc([r["index"] for r in pick if r["label"] == "read"], [r["index"] for r in pick if r["label"] == "spontaneous"])
        if v is not None:
            vals.append(v)
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))] if vals else None


def summarise(rows: list[dict]) -> dict[str, Any]:
    read = [r for r in rows if r["label"] == "read"]
    spont = [r for r in rows if r["label"] == "spontaneous"]
    out: dict[str, Any] = {
        "speakers": len({r["speaker"] for r in rows}), "answers": {lbl: sum(r["label"] == lbl for r in rows) for lbl in
                                                                    sorted({r["label"] for r in rows})},
        "roc_auc": auc([r["index"] for r in read], [r["index"] for r in spont]),
        "roc_auc_95ci_by_speaker": speaker_bootstrap(rows),
        "levels": {lbl: {"possible_or_high": float(np.mean([r["level"] in ("possible", "high") for r in part])) if part else None,
                         "high": float(np.mean([r["level"] == "high" for r in part])) if part else None,
                         "mean_index": float(np.mean([r["index"] for r in part])) if part else None}
                   for lbl, part in (("read", read), ("spontaneous", spont), ("narration", [r for r in rows if r["label"] == "narration"]))},
        "feature_auc": {}, "feature_means": {},
    }
    for name, weight in SIGNALS.WEIGHTS.items():
        rp = [np.sign(weight) * r["z"][name] for r in read if name in r["z"]]
        sp = [np.sign(weight) * r["z"][name] for r in spont if name in r["z"]]
        out["feature_auc"][name] = auc(rp, sp)  # in the direction the spec weights it
        for lbl, part in (("read", read), ("spontaneous", spont)):
            vals = [r["features"][name] for r in part if r["features"].get(name) is not None]
            out["feature_means"].setdefault(name, {})[lbl] = float(np.mean(vals)) if vals else None
    return out


def fitted_weights(rows: list[dict]) -> dict[str, Any] | None:
    """For information: logistic regression on the z-scores, split by speaker (70/30)."""
    from sklearn.linear_model import LogisticRegression

    rows = [r for r in rows if r["label"] in ("read", "spontaneous")]
    speakers = np.array(sorted({r["speaker"] for r in rows}))
    if len(speakers) < 4:
        return None
    test = set(speakers[np.random.default_rng(SEED).permutation(len(speakers))[: max(1, round(0.3 * len(speakers)))]].tolist())
    names = list(SIGNALS.WEIGHTS)
    X = np.array([[r["z"].get(n, 0.0) for n in names] for r in rows])
    y = np.array([r["label"] == "read" for r in rows])
    is_test = np.array([r["speaker"] in test for r in rows])
    if len(set(y[~is_test])) < 2 or len(set(y[is_test])) < 2:
        return None
    model = LogisticRegression(class_weight="balanced", max_iter=1000).fit(X[~is_test], y[~is_test])
    p = model.decision_function(X[is_test])
    spec_index = np.array([r["index"] for r in rows])[is_test]
    return {"train_speakers": int(len(speakers) - len(test)), "test_speakers": len(test),
            "coefficients": dict(zip(names, model.coef_[0].round(3).tolist())), "intercept": round(float(model.intercept_[0]), 3),
            "test_roc_auc_fitted": auc(p[y[is_test]], p[~y[is_test]]),
            "test_roc_auc_spec_index": auc(spec_index[y[is_test]], spec_index[~y[is_test]])}


def evaluate(answers: list[dict]) -> dict[str, Any]:
    rows = score(answers)
    return {"summary": summarise(rows), "fitted_weights_for_information": fitted_weights(rows),
            "thresholds": {k: SIGNALS.thresholds()["delivery"][k] for k in ("possible_index", "high_index", "calibrated")}}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="mode", required=True)
    sp = sub.add_parser("spice")
    sp.add_argument("--textgrids", type=Path, required=True)
    sp.add_argument("--whisper", action="store_true", help="time the words with faster-whisper on the audio instead")
    sp.add_argument("--files", type=Path, help="english_files.json (Borealis file ids), for --whisper")
    sp.add_argument("--audio-cache", type=Path, help="where audio snippets and the Whisper model are kept, for --whisper")
    fp = sub.add_parser("folder")
    fp.add_argument("root", type=Path)
    fp.add_argument("--no-whisper", action="store_true", help="only .json answers")
    for p in (sp, fp):
        p.add_argument("--out", type=Path, required=True)
        p.add_argument("--whisper-model", default="base.en")
    a = ap.parse_args()
    started = time.time()
    log = lambda m: print(m, flush=True)  # noqa: E731

    if a.mode == "spice":
        answers = select([x for f in sorted(a.textgrids.glob("*_English_*.TextGrid")) for x in spice_answers(f)])
        timing = "SpiCE hand transcripts, forced-aligned word times (Montreal Forced Aligner)"
        if a.whisper:
            files = {r["filename"]: r["id"] for r in json.loads(a.files.read_text(encoding="utf-8"))["files"]}
            transcribe = whisper_words(a.whisper_model, a.audio_cache / "whisper")
            answers = retime_with_whisper(answers, files, transcribe, a.audio_cache, log)
            timing = f"faster-whisper {a.whisper_model} word timestamps on the participant's channel (as in the room)"
        data = "SpiCE English sessions (Johnson, Babel, Fong & Yiu; CC BY 4.0; doi:10.5683/SP2/MJOXP3)"
    else:
        transcribe = None if a.no_whisper else whisper_words(a.whisper_model)
        answers = baseline_first(folder_answers(a.root, transcribe))
        timing, data = "words from .json files or faster-whisper", str(a.root.name)
    log(f"{len(answers)} answers from {len({x['speaker'] for x in answers})} speakers")
    result = {"what": "spec delivery index (spec/integrity, v1): read vs spontaneous answers of the same speakers",
              "data": data, "timing": timing,
              "protocol": {"chunk_words": CHUNK_WORDS, "max_gap_s": MAX_GAP_S, "baseline_answers": BASELINE,
                           "spontaneous_per_speaker": SPONTANEOUS_PER_SPEAKER, "latency": "not measurable (no questions)"},
              **evaluate(answers), "seconds": round(time.time() - started)}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    s = result["summary"]
    log(f"ROC-AUC read vs spontaneous {s['roc_auc']:.3f} (95% CI {s['roc_auc_95ci_by_speaker']}); "
        f"flagged possible+: read {s['levels']['read']['possible_or_high']:.2f}, spontaneous {s['levels']['spontaneous']['possible_or_high']:.2f} -> {a.out}")


if __name__ == "__main__":
    main()
