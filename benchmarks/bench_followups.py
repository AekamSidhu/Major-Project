"""Follow-up questions in the room: local LLM (llama3.2) against rules, and the echo reference answers.

    python benchmarks/bench_followups.py --out benchmarks/results/followups.json

In process, with the room's own code (local-ai/room/followups.py, guardrails.py,
followup_rules.py, similarity.py), on the benchmark's 12 written interview
answers (data/interview_pairs.json), each answer on its own (no earlier
questions), --repeats times:

- local: FollowupWriter("local").write(), the LangChain chain with input and
  output guardrails, structured output from llama3.2 through Ollama, falling
  back to rules when nothing passes. Wall time per call; how many
  suggestions the model gave, how many passed the guardrails and why the
  others were dropped; how often it fell back to rules.
- local, streamed: the same prompt and model streamed as plain text, for the
  time to the first token (the room itself waits for the whole structured
  answer; the desktop app streams).
- rules: rule_followups(), timed, and its suggestions put through the same
  output guardrails (the room does not, but it shows how they compare).
- references: ReferenceWriter("local").write(question, 2), the two answers
  echo similarity compares against.

The room's settings are built here (never loaded from ~/.openmeet/room.env).
`--mode online` runs the same on Gemini (the room's online mode; the key is
read from the OPENMEET_GEMINI_API_KEY environment variable only); without a
key it stops. The committed results are local only: no key was used.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import benchlib  # noqa: E402

sys.path.insert(0, str(benchlib.LOCAL_AI))


def transcript_of(pair: dict[str, str]) -> list[dict[str, Any]]:
    return [{"speaker": "interviewer", "text": pair["question"], "start": 0.0, "end": 4.0},
            {"speaker": "candidate", "text": pair["answer"], "start": 5.0, "end": 25.0}]


class Counting:
    """Wraps followups.guard_output to see the model's suggestions before the output guardrails."""

    def __init__(self, inner):
        self.inner, self.raw = inner, []

    def __call__(self, data):
        self.raw.append(len(data.get("items") or []))
        return self.inner(data)


def summarise_quality(rows: list[dict[str, Any]], key_kept: str = "kept") -> dict[str, Any]:
    reasons: dict[str, int] = {}
    for r in rows:
        for why in r.get("dropped", []):
            reasons[why] = reasons.get(why, 0) + 1
    n = len(rows)
    return {
        "calls": n,
        "suggestions_before_guardrails": sum(r.get("raw", 0) for r in rows),
        "kept_total": sum(r[key_kept] for r in rows),
        "kept_per_call_mean": round(sum(r[key_kept] for r in rows) / n, 2) if n else None,
        "calls_with_at_least_one_kept": sum(r[key_kept] > 0 for r in rows),
        "dropped_by_reason": dict(sorted(reasons.items(), key=lambda kv: -kv[1])),
        "kept_with_grounded_trigger": sum(r.get("grounded", 0) for r in rows),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--mode", choices=("local", "online"), default="local")
    ap.add_argument("--model", default=None, help="default llama3.2 (local) or gemini-3.5-flash (online)")
    ap.add_argument("--out", type=Path, default=benchlib.RESULTS / "followups.json")
    a = ap.parse_args()

    from langchain_ollama import ChatOllama

    from room import followups, guardrails
    from room.followup_rules import MAX_FOLLOWUPS, rule_followups
    from room.similarity import ReferenceWriter

    pairs = benchlib.read_json(benchlib.HERE / "data" / "interview_pairs.json")["pairs"]
    if a.mode == "online":
        key = os.environ.get("OPENMEET_GEMINI_API_KEY", "")
        if not key:
            raise SystemExit("--mode online needs OPENMEET_GEMINI_API_KEY in the environment")
        a.model = a.model or "gemini-3.5-flash"
        settings = followups.Settings(mode="online", gemini_model=a.model, gemini_key=key)
    else:
        a.model = a.model or "llama3.2"
        settings = followups.Settings(mode="local", ollama_model=a.model, ollama_url=benchlib.OLLAMA_URL)
    counting = Counting(followups.guard_output)
    followups.guard_output = counting  # build_chain picks it up when the writer builds its chain
    writer = followups.FollowupWriter(mode=a.mode, settings=settings)
    result: dict[str, Any] = {
        "what": "room follow-up questions: llama3.2 (local) vs rules; time to first token; reference answers",
        "environment": {**benchlib.environment(["langchain-core", "langchain-ollama", "ollama"]),
                        "ollama": benchlib.ollama_version()},
        "config": {"mode": a.mode, "model": a.model, "temperature": 0.3, "num_predict": 600, "repeats": a.repeats, "answers": len(pairs),
                   "max_followups": MAX_FOLLOWUPS},
    }

    started = time.perf_counter()
    writer.write(pairs[0]["question"], pairs[0]["answer"], transcript_of(pairs[0]), [])
    result["local_cold_first_call_ms"] = round((time.perf_counter() - started) * 1000.0, 1)
    try:
        ps = benchlib.get_json(f"{benchlib.OLLAMA_URL}/api/ps")["models"]
        result["ollama_loaded"] = [{"name": m["name"], "size_vram_gb": round(m.get("size_vram", 0) / 2**30, 2),
                                    "size_gb": round(m.get("size", 0) / 2**30, 2)} for m in ps]
    except Exception:  # noqa: BLE001
        result["ollama_loaded"] = None
    print(f"cold first call {result['local_cold_first_call_ms']} ms; loaded {result['ollama_loaded']}", flush=True)

    local_rows, rule_rows, stream_rows, ref_rows = [], [], [], []
    stream_llm = ChatOllama(model=a.model, base_url=benchlib.OLLAMA_URL, temperature=0.3, num_predict=600) if a.mode == "local" else None
    references = ReferenceWriter(a.mode, settings)
    for rep in range(a.repeats):
        for pair in pairs:
            transcript = transcript_of(pair)
            before = len(counting.raw)
            t = time.perf_counter()
            out = writer.write(pair["question"], pair["answer"], transcript, [])
            ms = (time.perf_counter() - t) * 1000.0
            raw = counting.raw[before] if len(counting.raw) > before else 0
            kept = out["items"] if out["source"] == "ai" else []
            local_rows.append({"id": pair["id"], "ms": ms, "source": out["source"], "raw": raw, "kept": len(kept),
                               "dropped": out.get("dropped", []), "grounded": sum(bool(k.get("trigger")) for k in kept)})

            t = time.perf_counter()
            items = rule_followups(pair["question"], pair["answer"], [])
            rms = (time.perf_counter() - t) * 1000.0
            checked = guardrails.check_followups(items, pair["question"], pair["answer"], [], MAX_FOLLOWUPS)
            rule_rows.append({"id": pair["id"], "ms": rms, "raw": len(items), "kept": len(checked.kept),
                              "dropped": [why for _, why in checked.dropped],
                              "grounded": sum(bool(k.get("trigger")) for k in checked.kept)})

            first = None
            if stream_llm is not None:
                messages = followups.PROMPT.format_messages(**followups.guard_input(
                    {"question": pair["question"], "answer": pair["answer"], "transcript": transcript, "previous": []}))
                t = time.perf_counter()
                chars = 0
                for chunk in stream_llm.stream(messages):
                    text = chunk.content if isinstance(chunk.content, str) else ""
                    if text and first is None:
                        first = (time.perf_counter() - t) * 1000.0
                    chars += len(text)
                stream_rows.append({"id": pair["id"], "first_token_ms": first, "total_ms": (time.perf_counter() - t) * 1000.0,
                                    "chars": chars})

            if rep == 0:
                t = time.perf_counter()
                refs = references.write(pair["question"], 2)
                ref_rows.append({"id": pair["id"], "ms": (time.perf_counter() - t) * 1000.0, "references": len(refs),
                                 "words": [len(r.split()) for r in refs]})
            print(f"rep {rep + 1} {pair['id']}: {a.mode} {ms:.0f} ms ({out['source']}, kept {len(kept)}/{raw}); "
                  f"stream first token {first if first is None else round(first)} ms; rules {rms:.2f} ms", flush=True)

    result["local"] = {"latency_ms": benchlib.summary([r["ms"] for r in local_rows]),
                       "fell_back_to_rules": sum(r["source"] == "rules" for r in local_rows),
                       **summarise_quality(local_rows)}
    result["local_streamed"] = {"first_token_ms": benchlib.summary([r["first_token_ms"] for r in stream_rows]),
                                "total_ms": benchlib.summary([r["total_ms"] for r in stream_rows]),
                                "chars": benchlib.summary([r["chars"] for r in stream_rows], 0)}
    result["rules"] = {"latency_ms": benchlib.summary([r["ms"] for r in rule_rows], 3), **summarise_quality(rule_rows)}
    ref_words = [w for r in ref_rows for w in r["words"]]
    result["references"] = {"two_references_ms": benchlib.summary([r["ms"] for r in ref_rows]),
                            "words_per_reference": benchlib.summary(ref_words, 0), "questions": len(ref_rows)}
    result["per_answer"] = {"local": local_rows, "rules": rule_rows, "streamed": stream_rows}
    benchlib.write_json(a.out, result)
    print(f"local {result['local']['latency_ms']}, kept {result['local']['kept_total']}, fallback "
          f"{result['local']['fell_back_to_rules']}; streamed first token {result['local_streamed']['first_token_ms']}; "
          f"references {result['references']['two_references_ms']} -> {a.out}")


if __name__ == "__main__":
    main()
