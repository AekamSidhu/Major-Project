"""The interview room end to end: per-answer integrity checks and follow-ups, in real time.

    python benchmarks/bench_room.py --speech M:/openmeet-data/bench-speech --face M:/openmeet-data/faces/astronaut.png \
        --out benchmarks/results/room_e2e.json

Starts the room (benchmarks/room_launch.py: local-ai/room_server.py's app,
follow-ups and reference answers from llama3.2, port 8463) and plays a
scripted 8-answer interview against its HTTP API following the host page's
own rules (local-ai/room/host.html), on a real-time timeline:

- a video frame (a public-domain portrait) goes to /v1/integrity/gaze/frame
  10 times a second, one at a time, the whole session;
- each side's speech (text-to-speech, make_inputs.py) is cut into pieces by
  the host page's segmenter (split_speech: 0.256 s blocks, a piece ends after
  more than 0.8 s of silence or at 20 s) and each piece goes to
  /api/rooms/<id>/transcribe when the segmenter would send it;
- the question is the interviewer's transcripts that began before the
  candidate did: read when the answer closes (`--question-at close`, the host
  page after the fix), or only the ones back when the candidate starts
  (`--question-at start`, the host page before the fix);
- the answer closes when the interviewer starts the next question (or "End
  answer now" after the last): the host waits for the candidate pieces sent
  so far, posts /answers (the card: AI text, delivery, reading and the fused
  flag), then asks /answers/<i>/similarity and /followups at the same time.

Turns 1-7 leave the candidate 2.5 s to start and the interviewer 1.5 s
before the next question. Turn 8 is quick turn-taking (0.8 s and 0.6 s), to
see what the host page's timing rules do then. Answers 1-3 set the delivery
baseline; answers 5 and 7 are llama3.2's own answers read out
(data/llm_answers.json), the rest are written by hand. This measures latency;
the levels are a sanity check on 8 answers, not an accuracy result.
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import benchlib  # noqa: E402

# (question id, answer kind, candidate starts after the question ends [s], interviewer speaks after the answer ends [s])
TURNS = [("q02", "own", 2.5, 1.5), ("q03", "own", 2.5, 1.5), ("q04", "own", 2.5, 1.5), ("q05", "own", 2.5, 1.5),
         ("q01", "llm", 2.5, 1.5), ("q07", "own", 2.5, 1.5), ("q06", "llm", 2.5, 1.5), ("q08", "own", 0.8, 0.6)]
DETECT_S = 4096 / 16000  # the host page notices speech at the end of its first voiced block


def split_speech(samples: np.ndarray, rate: int = 16000, block: int = 4096, tail_s: float = 2.0) -> list[tuple[float, float, float]]:
    """The host page's speech segmenter (local-ai/room/host.html `startSegmenter`), in Python.

    Blocks of 4096 samples; a block is voiced when its RMS is above
    max(0.012, 3 x the noise floor), and the floor follows unvoiced blocks.
    A piece starts with two blocks of pre-roll and ends after more than 0.8 s
    of silence or at 20 s; it is sent if it holds more than 0.6 s of sound.
    `tail_s` of silence is added after the audio (the speaker has stopped).
    Returns (start_s, end_s, sent_at_s) per piece, from the audio's start.
    """
    dur = block / rate
    audio = np.concatenate([np.asarray(samples, dtype=np.float32), np.zeros(int(tail_s * rate), dtype=np.float32)])
    floor, speaking, silent, n_chunks, started, preroll = 0.005, False, 0.0, 0, 0.0, 0
    pieces: list[tuple[float, float, float]] = []
    for b in range(len(audio) // block):
        x = audio[b * block:(b + 1) * block]
        rms = float(np.sqrt(np.mean(x * x)))
        now = (b + 1) * dur
        voiced = rms > max(0.012, floor * 3)
        if not voiced:
            floor = 0.95 * floor + 0.05 * rms
        if voiced and not speaking:
            speaking, n_chunks, started = True, preroll, now - dur * (preroll + 1)
        if speaking:
            n_chunks += 1
            silent = 0.0 if voiced else silent + dur
            length = n_chunks * dur
            if silent > 0.8 or length > 20:
                speaking = False
                if length - silent > 0.6:
                    pieces.append((max(0.0, started), now - silent, now))
        preroll = min(preroll + 1, 2)
    return pieces


def timeline(turns: list[tuple[str, str, float, float]], questions: dict[str, np.ndarray], replies: dict[str, np.ndarray],
             lead_s: float = 3.0) -> list[tuple[float, str, dict[str, Any]]]:
    """Every event of the session as (seconds from the session start, kind, details), sorted."""
    events: list[tuple[float, str, dict[str, Any]]] = []
    t = lead_s
    for turn, (qid, kind, answer_delay, close_delay) in enumerate(turns, 1):
        q, a = questions[qid], replies[qid]
        if turn > 1:
            events.append((t, "close", {"turn": turn - 1}))
        for s, e, sent in split_speech(q):
            events.append((t + sent, "piece", {"turn": turn, "speaker": "interviewer", "audio": q[int(s * 16000):int(e * 16000)],
                                               "start": t + s, "end": t + e}))
        q_end = t + len(q) / 16000
        a_start = q_end + answer_delay
        pieces = split_speech(a)
        events.append((a_start + DETECT_S, "answer_start", {"turn": turn, "id": qid, "kind": kind, "q_end": q_end,
                                                            "start": a_start + (pieces[0][0] if pieces else 0.0),
                                                            "answer_delay": answer_delay, "close_delay": close_delay}))
        for s, e, sent in pieces:
            events.append((a_start + sent, "piece", {"turn": turn, "speaker": "candidate", "audio": a[int(s * 16000):int(e * 16000)],
                                                     "start": a_start + s, "end": a_start + e}))
        t = a_start + len(a) / 16000 + close_delay
    events.append((t, "close", {"turn": len(turns)}))
    return sorted(events, key=lambda ev: (ev[0], ev[1] != "close"))


class Session:
    def __init__(self, base: str, room: str, t0: float, question_at: str = "close") -> None:
        self.base, self.room, self.t0, self.question_at = base, room, t0, question_at
        self.pool = ThreadPoolExecutor(max_workers=8)
        self.rows: list[dict[str, Any]] = []
        self.segment_ms: list[float] = []
        self.jobs: list[Future] = []
        self.lock = threading.Lock()

    def transcribe(self, speaker: str, audio: np.ndarray, start: float, end: float) -> Future:
        def run() -> dict[str, Any]:
            t = time.perf_counter()
            out = benchlib.post_bytes(f"{self.base}/api/rooms/{self.room}/transcribe?speaker={speaker}&start={start}&end={end}",
                                      benchlib.wav_bytes(audio), "audio/wav")
            with self.lock:
                self.segment_ms.append((time.perf_counter() - t) * 1000.0)
            return {**out, "done_at": time.time()}

        return self.pool.submit(run)

    def close(self, answer: dict[str, Any]) -> None:
        """The host page's closeAnswer, then (renderAnswer) similarity and follow-ups together."""
        t_close = time.perf_counter()
        texts = [f.result(timeout=600)["text"] for f in answer["pieces"]]
        if self.question_at == "close":  # the host page: the question is read now, after its transcripts settle
            question = " ".join(t for t in (f.result(timeout=600)["text"] for f in answer["question_jobs"]) if t)
        else:  # the host page before the fix: only the transcripts back when the candidate started
            question = answer["question"]
        t_post = time.perf_counter()
        end = max([answer["start"]] + answer["ends"])
        row = {"turn": answer["turn"], "id": answer["id"], "kind": answer["kind"], "answer_delay_s": answer["answer_delay"],
               "close_delay_s": answer["close_delay"], "question_transcribed": bool(question),
               "question_ready_after_s": answer["question_ready_after_s"], "pieces_in_answer": len(answer["pieces"]),
               "pieces_sent_after_close": answer["late"], "wait_for_transcripts_ms": (t_post - t_close) * 1000.0}
        self.rows.append(row)
        if not any(texts) and max(end, answer["start"] + 1) - answer["start"] < 5:
            row["dropped_by_host"] = True  # the host page's "a cough or noise" rule
            return
        body = {"start": answer["start"], "end": max(end, answer["start"] + 1), "text": " ".join(t for t in texts if t),
                "question": question}
        card = benchlib.post_json(f"{self.base}/api/rooms/{self.room}/answers", body)
        t_card = time.perf_counter()
        row.update({"answers_request_ms": (t_card - t_post) * 1000.0, "close_to_card_ms": (t_card - t_close) * 1000.0,
                    "answer_words": len(body["text"].split()), "card": card})

        def similarity() -> None:
            row["final"] = benchlib.post_json(f"{self.base}/api/rooms/{self.room}/answers/{card['index']}/similarity", {})
            row["card_to_similarity_ms"] = (time.perf_counter() - t_card) * 1000.0

        def followups() -> None:
            out = benchlib.post_json(f"{self.base}/api/rooms/{self.room}/followups", {"answer": card["index"]})
            row["card_to_followups_ms"] = (time.perf_counter() - t_card) * 1000.0
            row.update({"followup_source": out["source"], "followups_kept": len(out["items"]),
                        "followups_dropped": out.get("dropped", [])})

        self.jobs += [self.pool.submit(similarity), self.pool.submit(followups)]

    def run(self, events: list[tuple[float, str, dict[str, Any]]]) -> None:
        questions: dict[int, list[Future]] = {}
        answers: dict[int, dict[str, Any]] = {}
        closes: list[Future] = []
        for at, kind, ev in events:
            time.sleep(max(0.0, self.t0 + at - time.time()))
            turn = ev["turn"]
            if kind == "piece":
                job = self.transcribe(ev["speaker"], ev["audio"], self.t0 + ev["start"], self.t0 + ev["end"])
                if ev["speaker"] == "interviewer":
                    questions.setdefault(turn, []).append(job)
                elif not answers[turn].get("closed"):
                    answers[turn]["pieces"].append(job)
                    answers[turn]["ends"].append(self.t0 + ev["end"])
                else:
                    answers[turn]["late"] += 1
            elif kind == "answer_start":
                # As the host page: only the interviewer transcripts back by now make the question.
                arrived = [f.result()["text"] for f in questions.get(turn, []) if f.done()]
                question = " ".join(t for t in arrived if t)
                answers[turn] = {**ev, "start": self.t0 + ev["start"], "question": question, "pieces": [], "ends": [], "late": 0,
                                 "question_ready_after_s": None, "_ready_jobs": questions.get(turn, []),
                                 "question_jobs": questions.get(turn, [])}
                if question and self.question_at == "start":
                    self.pool.submit(benchlib.post_json, f"{self.base}/api/rooms/{self.room}/references", {"question": question})
            elif kind == "close":
                answer = answers[turn]
                answer["closed"] = True
                closes.append(self.pool.submit(self.close, answer))
        for job in closes:
            job.result(timeout=900)
        for job in list(self.jobs):
            job.result(timeout=900)
        for turn, answer in answers.items():  # when each question's full text came back, after the interviewer stopped
            jobs = answer.pop("_ready_jobs")
            row = next(r for r in self.rows if r["turn"] == turn)
            if jobs:
                row["question_ready_after_s"] = round(max(f.result()["done_at"] for f in jobs) - (self.t0 + answer["q_end"]), 3)


def levels(row: dict[str, Any]) -> dict[str, Any] | None:
    final = row.get("final") or row.get("card")
    if final is None:
        return None
    flag = final["flag"]
    by_name = {s["name"]: s for s in flag["signals"]}
    sim = final.get("similarity") or {}
    echo = final.get("echo") or {}
    reading = final.get("reading") or {}
    delivery = (final.get("delivery") or {}).get("score") or {}
    return {"flag": flag["level"], "elevated": flag["elevated"],
            "ai_text": by_name["ai_text"]["level"], "ai_probability": echo.get("ai_probability"),
            "echo_similarity": by_name["echo_similarity"]["level"], "similarity": sim.get("similarity"),
            "similarity_detail": sim.get("detail"),
            "delivery": by_name["delivery"]["level"], "delivery_index": delivery.get("index"),
            "reading": by_name["reading"]["level"], "reading_probability": reading.get("reading_probability")}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--speech", type=Path, required=True, help="make_inputs.py speech output")
    ap.add_argument("--face", type=Path, required=True)
    ap.add_argument("--whisper-model", default="base.en")
    ap.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    ap.add_argument("--cuda-libs", type=Path, default=None)
    ap.add_argument("--model-dir", type=Path, default=None)
    ap.add_argument("--port", type=int, default=benchlib.ROOM_PORT)
    ap.add_argument("--out", type=Path, default=benchlib.RESULTS / "room_e2e.json")
    ap.add_argument("--question-at", choices=("start", "close"), default="close",
                    help="close: the question is the interviewer's transcripts read at close (the host page after the fix); "
                         "start: the transcripts back when the candidate starts (the committed run before the fix)")
    a = ap.parse_args()

    samples_of_wav = benchlib.wav_samples
    base = f"http://127.0.0.1:{a.port}"
    env = benchlib.sidecar_env(device=a.device, cuda_libs=a.cuda_libs, model_dir=a.model_dir)
    log = benchlib.tmp_dir() / f"room-{a.device}.log"
    frame = benchlib.jpeg_frame(a.face)
    questions = {q: samples_of_wav(a.speech / f"{q}_question.wav") for q, *_ in TURNS}
    replies = {q: samples_of_wav(a.speech / (f"{q}_answer.wav" if kind == "own" else f"{q}_llm_answer.wav"))
               for q, kind, *_ in TURNS}
    launcher = str(benchlib.HERE / "room_launch.py")
    with benchlib.server([launcher, "--port", str(a.port), "--whisper-model", a.whisper_model, "--followups", "local"],
                         f"{base}/api/status", log, env=env):
        print(f"room up; free RAM {benchlib.free_ram_gb()} GB", flush=True)
        # Warm up as an interviewer testing the microphone would: load Whisper and the face model.
        t = time.perf_counter()
        room = benchlib.post_json(f"{base}/api/rooms", {})["room"]
        Session(base, room, time.time()).transcribe("interviewer", questions["q02"], time.time() - 5, time.time() - 1).result(600)
        benchlib.post_bytes(f"{base}/v1/integrity/gaze/frame", frame, "image/jpeg")
        warm_ms = (time.perf_counter() - t) * 1000.0
        room = benchlib.post_json(f"{base}/api/rooms", {})["room"]  # a fresh room (resets the gaze)
        events = timeline(TURNS, questions, replies)
        session = Session(base, room, time.time(), a.question_at)
        with benchlib.FrameFeed(f"{base}/v1/integrity/gaze/frame", frame) as feed:
            session.run(events)
        report = benchlib.get_json(f"{base}/api/rooms/{room}/report")
    rows = sorted(session.rows, key=lambda r: r["turn"])
    for r in rows:
        r["levels"] = levels(r)
        r.pop("card", None)
        r.pop("final", None)
    normal = [r for r in rows if r["answer_delay_s"] >= 2.0]
    result = {
        "what": "interview room end to end: per-answer integrity checks and follow-ups (8 answers, real time)",
        "environment": {**benchlib.environment(["faster-whisper", "ctranslate2", "fastembed", "onnxruntime", "mediapipe",
                                                "langchain-ollama", "fastapi", "uvicorn"]), "ollama": benchlib.ollama_version()},
        "config": {"whisper_model": a.whisper_model, "device": a.device, "followups": "local (llama3.2)",
                   "turns": [{"id": q, "answer": k, "answer_delay_s": d, "close_delay_s": c} for q, k, d, c in TURNS],
                   "frames_fps": 10, "session_s": round(events[-1][0], 1),
                   "question_at": a.question_at},
        "warm_up_ms": round(warm_ms, 1),
        "gaze_feed": {"frames": len(feed.latencies_ms), "face_found": feed.face_found, "errors": feed.errors,
                      "latency_ms": benchlib.summary(feed.latencies_ms)},
        "segment_transcription_ms": benchlib.summary(session.segment_ms),
        "turns_1_to_7": {
            "close_to_card_ms": benchlib.summary([r.get("close_to_card_ms") for r in normal]),
            "wait_for_transcripts_ms": benchlib.summary([r["wait_for_transcripts_ms"] for r in normal]),
            "answers_request_ms": benchlib.summary([r.get("answers_request_ms") for r in normal]),
            "card_to_similarity_ms": benchlib.summary([r.get("card_to_similarity_ms") for r in normal]),
            "card_to_followups_ms": benchlib.summary([r.get("card_to_followups_ms") for r in normal]),
            "question_ready_after_s": benchlib.summary([r["question_ready_after_s"] for r in normal], 2),
        },
        "transcript_segments_in_report": len(report["transcript"]),
        "answers": rows,
    }
    benchlib.write_json(a.out, result)
    for r in rows:
        lv = r["levels"] or {}
        print(f"{r['turn']} {r['id']} {r['kind']:3s} q? {r['question_transcribed']} (ready +{r['question_ready_after_s']} s) "
              f"card {r.get('close_to_card_ms', 0):.0f} ms, sim +{r.get('card_to_similarity_ms', 0):.0f}, fu +{r.get('card_to_followups_ms', 0):.0f}"
              f" | late pieces {r['pieces_sent_after_close']} | flag {lv.get('flag')} {lv.get('elevated')} ai {lv.get('ai_text')} "
              f"echo {lv.get('echo_similarity')} ({lv.get('similarity')}) delivery {lv.get('delivery')} reading {lv.get('reading')}")
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()
