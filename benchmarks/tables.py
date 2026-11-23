"""Markdown tables for docs/BENCHMARKS.md, printed from the result files (so the doc matches them).

    python benchmarks/tables.py                                   # print every table
    python benchmarks/tables.py --doc docs/BENCHMARKS.md          # rewrite the doc's table blocks
    python benchmarks/tables.py --doc docs/BENCHMARKS.md --check  # fail if they are out of date

Every number in the doc's tables comes from here; a section whose result
file is missing prints nothing.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import benchlib  # noqa: E402

R = benchlib.RESULTS
TRAIN = benchlib.REPO / "training"


def load(path: Path):
    return benchlib.read_json(path) if path.exists() else None


def pct(x, digits: int = 1) -> str:
    return "–" if x is None else f"{100 * x:.{digits}f}%"


def ms(s: dict | None, key: str = "median", digits: int = 0) -> str:
    if not s or not s.get("n"):
        return "–"
    return f"{s[key]:,.{digits}f}"


def lat(s: dict | None, digits: int = 0) -> str:
    """median / p90 / p95."""
    if not s or not s.get("n"):
        return "–"
    return f"{s['median']:,.{digits}f} / {s['p90']:,.{digits}f} / {s['p95']:,.{digits}f}"


def env_line(res: dict) -> str:
    e = res.get("environment", {})
    pk = ", ".join(f"{k} {v}" for k, v in (e.get("packages") or {}).items() if v)
    return (f"*Run {e.get('date', '?')[:16].replace('T', ' ')}, commit {e.get('repo_commit')}; {e.get('cpu')}, "
            f"{e.get('ram_gb')} GB RAM, {e.get('gpu') or 'no GPU'}; {e.get('os')}; Python {e.get('python')}"
            + (f"; {pk}" if pk else "") + (f"; Ollama {e['ollama']}" if e.get("ollama") else "")
            + (f"; {e['rustc']}" if e.get("rustc") else "") + ".*")


def wer_ci(csv_path: Path, model: str, split: str, n_boot: int = 2000, seed: int = 0) -> tuple[float, float] | None:
    """95% bootstrap interval of a WER, resampling utterances (from the run's per-utterance CSV)."""
    import csv

    import numpy as np

    if not csv_path.exists():
        return None
    with csv_path.open(encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["model"] == model and r["split"] == split]
    if not rows:
        return None
    errors = np.array([int(r["sub"]) + int(r["del"]) + int(r["ins"]) for r in rows], dtype=float)
    words = np.array([int(r["ref_words"]) for r in rows], dtype=float)
    idx = np.random.default_rng(seed).integers(0, len(rows), size=(n_boot, len(rows)))
    boot = errors[idx].sum(axis=1) / words[idx].sum(axis=1)
    return float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def wer_cell(res_path: Path, m: dict, model: str, split: str) -> str:
    ci = wer_ci(res_path.with_name(res_path.stem + "_utterances.csv"), model, split)
    return pct(m[split]["wer"], 2) + (f" ({100 * ci[0]:.1f}–{100 * ci[1]:.1f})" if ci else "")


def asr() -> None:
    runs = [(d, load(R / f"asr_{d}.json")) for d in ("cpu", "cuda")]
    runs = [(d, r) for d, r in runs if r]
    if not runs:
        return
    print("### Transcription: WER and speed\n")
    print("WER with a 95% bootstrap interval over utterances in brackets. First request: a fresh sidecar's first "
          "transcription, which loads the model (models already on disk).\n")
    print("| Model | Device | WER test-clean | WER test-other | Latency per utterance, ms (median / p90 / p95) | "
          "Real-time factor (median) | Aggregate RTF | First request, s |")
    print("|---|---|---|---|---|---|---|---|")
    cold = load(R / "asr_cpu_coldstart.json")
    for device, res in runs:
        for model, m in res["models"].items():
            path = R / f"asr_{device}.json"
            first = m["cold_first_request_ms"]
            if device == "cpu" and cold and model in cold["models"]:  # the main CPU run downloaded base.en and small.en first
                first = cold["models"][model]["cold_first_request_ms"]
            print(f"| {model} | {m.get('device_used') or device} | {wer_cell(path, m, model, 'test-clean')} | "
                  f"{wer_cell(path, m, model, 'test-other')} | "
                  f"{lat(m['all']['latency_ms'])} | {m['all']['rtf']['median']:.3f} | {m['all']['aggregate_rtf']:.4f} | "
                  f"{first / 1000:.1f} |")
    first = runs[0][1]
    some = next(iter(first["models"].values()))
    print(f"\n{some['test-clean']['utterances']} + {some['test-other']['utterances']} utterances "
          f"({some['test-clean']['speakers']} + {some['test-other']['speakers']} speakers, "
          f"{some['all']['audio_s'] / 60:.1f} min of audio), seed {first['config']['seed']}.\n")
    for _, res in runs:
        print(env_line(res) + "\n")
    words = load(R / "asr_cpu_words.json")
    if words:
        print("Word timestamps (verbose_json, as the room asks), CPU:\n")
        print("| Model | Word timings | WER test-clean | WER test-other | Latency, ms (median / p90 / p95) | Aggregate RTF |")
        print("|---|---|---|---|---|---|")
        for model, m in words["models"].items():
            path = R / "asr_cpu_words.json"
            print(f"| {model} | yes | {wer_cell(path, m, model, 'test-clean')} | {wer_cell(path, m, model, 'test-other')} | "
                  f"{lat(m['all']['latency_ms'])} | "
                  f"{m['all']['aggregate_rtf']:.4f} |")
        print(f"\n({words['config']['per_split']} utterances per split, the first of the same seeded order.) " + env_line(words) + "\n")


def held_out() -> None:
    reading = load(TRAIN / "gaze-detection" / "results" / "reading_detector_results.json")
    ritw = load(TRAIN / "gaze-detection" / "results" / "ritw_10s.json")
    echo = load(TRAIN / "echo-detection" / "results" / "echo_similarity.json")
    delivery = load(TRAIN / "echo-detection" / "results" / "delivery_spice.json")
    ai = load(R / "ai_text_v1_vs_v2.json")
    print("### Detection rate and false flags: held-out results already measured\n")
    print("| Signal | Test data (held out) | ROC-AUC | Detection rate | False-flag rate | Source |")
    print("|---|---|---|---|---|---|")
    if reading:
        n = reading["test"]["network"]["gazebase_nominal"]
        h = reading["test"]["network"]["gazebase_small_hard"]
        print(f"| Reading (shipped network) | GazeBase, 64 unseen people, 10 s windows, webcam-quality noise | {n['window_roc_auc']:.3f} | "
              f"{pct(n['window_recall_at_low_fpr'])} at `possible` | {pct(n['window_fpr_at_low_fpr'])} at `possible` "
              f"(balanced accuracy {pct(n['window_balanced_accuracy'])} at `high`) | `training/gaze-detection/results/reading_detector_results.json` |")
        print(f"| Reading, hardest condition | same people, small face + heavy noise | {h['window_roc_auc']:.3f} | "
              f"{pct(h['window_recall_at_low_fpr'])} at `possible` | {pct(h['window_fpr_at_low_fpr'])} at `possible` | same file |")
    if ritw:
        a = ritw["test"]["average"]["eye_tracker"]["window"]["balanced"]
        print(f"| Reading, real life (a model trained on RitW, **not** the shipped one) | Reading in the Wild, 17 unseen people | "
              f"{a['roc_auc']:.3f} | {pct(a['recall'])} | {pct(a['false_positive_rate'])} | `training/gaze-detection/results/ritw_10s.json` |")
    if ai:
        m = ai["models"]["v2"]
        n, h = m["at_high"]["of_ai"], m["at_high"]["of_human"]
        print(f"| AI-written text (v2, shipped) | HC3, all test questions ({n:,} ChatGPT + {h:,} human answers) | {m['roc_auc']:.3f} | "
              f"{pct(m['at_possible']['detection_rate'])} / {pct(m['at_high']['detection_rate'])} | "
              f"{pct(m['at_possible']['false_flag_rate'])} / {pct(m['at_high']['false_flag_rate'])} | "
              f"`benchmarks/results/ai_text_v1_vs_v2.json` |")
    if echo:
        t = echo["test"]
        print(f"| Echo similarity | HC3, 500 test questions (100 per source) | {t['roc_auc']:.3f} | {pct(t['at_possible']['recall'])} / "
              f"{pct(t['at_high']['recall'])} | {pct(t['at_possible']['false_positive_rate'])} / {pct(t['at_high']['false_positive_rate'])} | "
              f"`training/echo-detection/results/echo_similarity.json` |")
    if delivery:
        s = delivery["summary"]
        print(f"| Delivery (uncalibrated) | SpiCE, 34 speakers: read vs spontaneous 30-word stretches | {s['roc_auc']:.3f} | "
              f"{pct(s['levels']['read']['possible_or_high'])} / {pct(s['levels']['read']['high'])} of read | "
              f"{pct(s['levels']['spontaneous']['possible_or_high'])} / {pct(s['levels']['spontaneous']['high'])} of spontaneous | "
              f"`training/echo-detection/results/delivery_spice.json` |")
    print("\nA / B = at `possible` / at `high`.\n")


def ai_text() -> None:
    res = load(R / "ai_text_v1_vs_v2.json")
    if not res:
        return
    v1, v2 = res["models"]["v1"], res["models"]["v2"]
    d = res["data"]
    print("### AI-written text model: v1 against v2 (HC3 test questions, the same split)\n")
    print("| Model | ROC-AUC | Balanced accuracy at `high` | Detected at `possible` | False flags at `possible` | Detected at `high` | False flags at `high` |")
    print("|---|---|---|---|---|---|---|")
    for name, m in (("v1 (commit 0836f1e, HC3 only)", v1), ("v2 (shipped; HC3 + interview Q&A)", v2)):
        p_, h_ = m["at_possible"], m["at_high"]
        print(f"| {name} | {m['roc_auc']:.4f} | {pct(h_['balanced_accuracy'])} | {pct(p_['detection_rate'])} | "
              f"{pct(p_['false_flag_rate'])} | {pct(h_['detection_rate'])} | {pct(h_['false_flag_rate'])} |")
    print(f"\nThresholds (the sidecar's: `possible` is the lower of the two stored thresholds, `high` the higher): "
          f"v1 {v1['possible']:.4f} / {v1['high']:.4f}; v2 {v2['possible']:.4f} / {v2['high']:.4f}. "
          f"{d['test_answers']:,} answers ({d['chatgpt_answers']:,} ChatGPT, {d['human_answers']:,} human) to "
          f"{d['test_questions']:,} test questions.\n")
    print("By HC3 source (ROC-AUC within the source; detection and false flags at `high`):\n")
    print("| Source | ChatGPT / human | v1 AUC | v2 AUC | v1 detected / false flags | v2 detected / false flags |")
    print("|---|---|---|---|---|---|")
    for src, row in res["by_source"].items():
        a, b = row["v1"], row["v2"]
        print(f"| {src} | {row['chatgpt_answers']:,} / {row['human_answers']:,} | {a['roc_auc']:.3f} | {b['roc_auc']:.3f} | "
              f"{pct(a['at_high']['detection_rate'])} / {pct(a['at_high']['false_flag_rate'])} | "
              f"{pct(b['at_high']['detection_rate'])} / {pct(b['at_high']['false_flag_rate'])} |")
    fv1, fv2 = load(R / "fusion_hc3_v1.json"), load(R / "fusion_hc3.json")
    if fv1 and fv2:
        print("\nIn the fused flag (HC3 sample of 1,411 answers, two references; see section 3):\n")
        print("| Configuration | v1 detected / false flags at `possible` | v2 detected / false flags at `possible` | "
              "v1 detected / false flags at `high` | v2 detected / false flags at `high` |")
        print("|---|---|---|---|---|")
        for name in ("ai_text alone", "fused (ai_text + echo)"):
            a, b = fv1["configs"][name], fv2["configs"][name]
            print(f"| {name} | {pct(a['at_possible']['detection_rate'])} / {pct(a['at_possible']['false_flag_rate'])} | "
                  f"{pct(b['at_possible']['detection_rate'])} / {pct(b['at_possible']['false_flag_rate'])} | "
                  f"{pct(a['at_high']['detection_rate'])} / {pct(a['at_high']['false_flag_rate'])} | "
                  f"{pct(b['at_high']['detection_rate'])} / {pct(b['at_high']['false_flag_rate'])} |")
    print("\nStored test metrics in each model file (`balanced` level, written by its training script), against the recomputed ones:\n")
    print("| File | Stored balanced accuracy | Recomputed | Stored recall | Recomputed | Stored false-positive rate | Recomputed |")
    print("|---|---|---|---|---|---|---|")
    for name, m in (("v1", v1), ("v2", v2)):
        st = res["stored_test_metrics"][name]
        st = st.get("balanced", st) if st else None
        if st:
            print(f"| {name} | {pct(st['balanced_accuracy'], 2)} | {pct(m['at_high']['balanced_accuracy'], 2)} | "
                  f"{pct(st['recall'], 2)} | {pct(m['at_high']['detection_rate'], 2)} | "
                  f"{pct(st['false_positive_rate'], 2)} | {pct(m['at_high']['false_flag_rate'], 2)} |")
    print("\n" + env_line(res) + "\n")


def fusion() -> None:
    res = load(R / "fusion_hc3.json")
    if not res:
        return
    print("### The combined flag on HC3 test questions (new)\n")
    print("| Configuration | ROC-AUC | Detected at `possible` | False flags at `possible` | Detected at `high` | False flags at `high` |")
    print("|---|---|---|---|---|---|")
    auc = res["roc_auc"]
    aucs = {"ai_text alone": auc["ai_text"], "echo similarity alone": auc["echo similarity (2 references)"],
            "echo similarity alone, 1 reference": auc["echo similarity (1 reference)"]}
    for name, c in res["configs"].items():
        p, h = c["at_possible"], c["at_high"]
        a = f"{aucs[name]:.3f}" if name in aucs else "–"
        ci = lambda v: f" ({100 * v[0]:.1f}–{100 * v[1]:.1f})" if v else ""  # noqa: E731
        print(f"| {name} | {a} | {pct(p['detection_rate'])} | {pct(p['false_flag_rate'])}{ci(p['false_flag_95ci'])} | "
              f"{pct(h['detection_rate'])} | {pct(h['false_flag_rate'])}{ci(h['false_flag_95ci'])} |")
    c0 = next(iter(res["configs"].values()))
    print(f"\n{res['answers']} answers ({c0['chatgpt_answers']} ChatGPT, {c0['human_answers']} human) to "
          f"{res['protocol']['questions']} test questions; 95% Wilson intervals in brackets.\n")
    print("By source (fused flag, 2 references; detected / false flags):\n")
    print("| Source | ChatGPT / human answers | At `possible` | At `high` | AI text alone at `high` | Echo alone at `high` |")
    print("|---|---|---|---|---|---|")
    for src, v in res["by_source"].items():
        f, t, e = v["fused (ai_text + echo)"], v["ai_text alone"], v["echo similarity alone"]
        print(f"| {src} | {f['chatgpt_answers']} / {f['human_answers']} | {pct(f['at_possible']['detection_rate'], 0)} / "
              f"{pct(f['at_possible']['false_flag_rate'], 0)} | {pct(f['at_high']['detection_rate'], 0)} / {pct(f['at_high']['false_flag_rate'], 0)} | "
              f"{pct(t['at_high']['detection_rate'], 0)} / {pct(t['at_high']['false_flag_rate'], 0)} | "
              f"{pct(e['at_high']['detection_rate'], 0)} / {pct(e['at_high']['false_flag_rate'], 0)} |")
    print("\nHuman answers flagged by the fused rule, by cause:\n")
    for k, v in res["false_flags_by_cause"].items():
        print(f"- {k}: {v}")
    print("\n" + env_line(res) + "\n")


def sidecar() -> None:
    res = load(R / "sidecar_endpoints.json")
    if not res:
        return
    print("### Sidecar endpoints\n")
    print("| Endpoint | What is sent | Latency, ms (median / p90 / p95) | First request, ms | n |")
    print("|---|---|---|---|---|")
    ep = res["endpoints"]
    rows = [("POST /v1/embeddings", "one answer", "embeddings_1_text", 1),
            ("POST /v1/embeddings", "answer + 2 reference answers", "embeddings_3_texts", 1),
            ("POST /v1/integrity/echo", "one answer (AI-text model)", "integrity_echo", 2),
            ("POST /v1/integrity/gaze/frame", "one JPEG frame, 10 a second", "integrity_gaze_frame", 1),
            ("POST /v1/integrity/reading", "last 10 s of gaze", "integrity_reading_10s", 2),
            ("POST /v1/integrity/reading", "last 60 s of gaze", "integrity_reading_60s", 2)]
    for name, what, key, digits in rows:
        e = ep.get(key)
        if e:
            cold = f"{e['cold_first_ms']:,.0f}" if e.get("cold_first_ms") is not None else "–"
            print(f"| `{name}` | {what} | {lat(e['latency_ms'], digits)} | {cold} | {e['latency_ms']['n']} |")
    g = ep.get("integrity_gaze_frame", {})
    if g:
        print(f"\nGaze feed: {g['frames']} frames in {res['inputs'].get('frame', '')}; face found in {g['face_found']}; "
              f"{g['achieved_fps']} frames a second achieved; {g['errors']} errors.")
    print("\n" + env_line(res) + "\n")


def followups() -> None:
    res = load(R / "followups.json")
    if not res:
        return
    loc, rul, st, ref = res["local"], res["rules"], res["local_streamed"], res["references"]
    print("### Follow-up questions: local LLM vs rules\n")
    print("| Mode | Latency per answer, ms (median / p90 / p95) | Suggestions before guardrails | Kept (shown) | Answers with at least one kept | Fell back to rules |")
    print("|---|---|---|---|---|---|")
    print(f"| local (llama3.2, structured output) | {lat(loc['latency_ms'])} | {loc['suggestions_before_guardrails']} | {loc['kept_total']} "
          f"({loc['kept_per_call_mean']} per answer) | {loc['calls_with_at_least_one_kept']} / {loc['calls']} | {loc['fell_back_to_rules']} |")
    print(f"| rules | {lat(rul['latency_ms'], 3)} | {rul['suggestions_before_guardrails']} | {rul['kept_total']} ({rul['kept_per_call_mean']} per answer) | "
          f"{rul['calls_with_at_least_one_kept']} / {rul['calls']} | – |")
    print(f"\nStreamed (same prompt and model, plain text): first token {lat(st['first_token_ms'])} ms, whole answer "
          f"{lat(st['total_ms'])} ms (median / p90 / p95). First local call after loading: {res['local_cold_first_call_ms']:,.0f} ms.\n")
    print(f"Echo reference answers (2 per question, llama3.2): {lat(ref['two_references_ms'])} ms; "
          f"{ref['words_per_reference']['median']:.0f} words each (median).\n")
    print(f"Dropped by the guardrails (local): {loc['dropped_by_reason'] or 'none'}; rules output under the same guardrails: "
          f"{rul['dropped_by_reason'] or 'none dropped'}. Kept with a trigger the candidate actually said: local "
          f"{loc['kept_with_grounded_trigger']} of {loc['kept_total']}, rules {rul['kept_with_grounded_trigger']} of {rul['kept_total']}.\n")
    print(env_line(res) + "\n")


def engine() -> None:
    off = load(R / "engine_offline.json")
    if off:
        print("### Desktop engine: built-in benchmark (synthetic input, mock LLM)\n")
        print("| Item | Median | p90 | p95 | Unit |")
        print("|---|---|---|---|---|")
        for it in off["items"]:
            print(f"| {it['name']} | {it['median']:,.4g} | {it['p90']:,.4g} | {it['p95']:,.4g} | {it['unit']} |")
        print(f"\n{off['runs']} runs. " + env_line(off) + "\n")
    for device in ("cpu", "cuda"):
        live = load(R / f"engine_live_{device}.json")
        if not live:
            continue
        cm, im = live["candidate_mode"], live["interviewer_mode"]
        print(f"### Desktop engine: live pipeline, Whisper {live['config']['whisper_model']} on {device.upper()}\n")
        print("| Stage (from the end of the speech) | ms (median / p90 / p95) | n |")
        print("|---|---|---|")
        for key, label in (("speech_end_to_transcript_ms", "transcript final (VAD hangover + Whisper)"),
                           ("speech_end_to_question_ms", "question detected"),
                           ("question_to_first_token_ms", "question detected -> first answer token"),
                           ("speech_end_to_first_token_ms", "first answer token"),
                           ("first_token_to_complete_ms", "first token -> answer complete"),
                           ("speech_end_to_complete_ms", "answer complete")):
            print(f"| {label} | {lat(cm[key])} | {cm[key].get('n', 0)} |")
        for key, label in (("speech_end_to_first_followup_ms", "interviewer mode: first follow-up"),
                           ("speech_end_to_followups_done_ms", "interviewer mode: follow-ups done")):
            print(f"| {label} | {lat(im[key])} | {im[key].get('n', 0)} |")
        print(f"\nQuestions detected: {cm['detected']} of {cm['questions']}; one transcript per question: {cm['single_transcript']}; "
              f"answers with follow-ups: {im['with_followups']} of {im['answers']}; summary after stop: "
              f"{live['summary_ms'] / 1000 if live['summary_ms'] else float('nan'):.1f} s ({'ok' if live['summary_ok'] else 'failed'}).\n")
        print("Engine's own latency tracker: " + "; ".join(
            f"{s['stage']} p50 {s['p50_ms']:,.0f} ms, p95 {s['p95_ms']:,.0f} ms (n {s['count']})" for s in live["engine_latency_stats"]) + "\n")
        print(env_line(live) + "\n")


def room() -> None:
    res = load(R / "room_e2e.json")
    if not res:
        return
    t = res["turns_1_to_7"]
    print("### Interview room, end to end\n")
    print("| Step (turns 1-7) | ms (median / p90 / p95) | n |")
    print("|---|---|---|")
    for key, label in (("wait_for_transcripts_ms", "answer closed -> its last transcripts back"),
                       ("answers_request_ms", "POST /answers (AI text, delivery, reading, flag)"),
                       ("close_to_card_ms", "answer closed -> card shown"),
                       ("card_to_similarity_ms", "card -> echo similarity filled in"),
                       ("card_to_followups_ms", "card -> follow-ups shown")):
        print(f"| {label} | {lat(t[key])} | {t[key].get('n', 0)} |")
    print(f"| every transcription request (both speakers, all turns) | {lat(res['segment_transcription_ms'])} | {res['segment_transcription_ms']['n']} |")
    print(f"| every gaze frame | {lat(res['gaze_feed']['latency_ms'])} | {res['gaze_feed']['latency_ms']['n']} |")
    q = t["question_ready_after_s"]
    print(f"\nThe question's text was back {q['median']:.2f} s (median; max {q['max']:.2f} s) after the interviewer stopped.\n")
    print("| Turn | Answer | Candidate starts after | Question known | Pieces (late) | Flag | AI text | Echo similarity | Delivery | Reading | Follow-ups |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in res["answers"]:
        lv = r.get("levels") or {}
        sim = lv.get("similarity")
        print(f"| {r['turn']} | {r['id']} ({'llama3.2, read out' if r['kind'] == 'llm' else 'written'}) | {r['answer_delay_s']} s | "
              f"{'yes' if r['question_transcribed'] else '**no**'} | {r['pieces_in_answer']} ({r['pieces_sent_after_close']}) | "
              f"{lv.get('flag', 'dropped')} | {lv.get('ai_text', '–')} ({pct(lv.get('ai_probability'), 0)}) | {lv.get('echo_similarity', '–')}"
              f"{f' ({sim:.3f})' if sim is not None else ''} | {lv.get('delivery', '–')} | {lv.get('reading', '–')} | "
              f"{r.get('followups_kept', '–')} ({r.get('followup_source', '–')}) |")
    print("\n" + env_line(res) + "\n")


SECTIONS = {"asr": asr, "held_out": held_out, "ai_text": ai_text, "fusion": fusion, "sidecar": sidecar, "followups": followups,
            "engine": engine, "room": room}
MARK = re.compile(r"(<!-- tables:(\w+) -->\n)(.*?)(<!-- /tables -->)", re.S)


def render(name: str) -> str:
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        SECTIONS[name]()
    return out.getvalue().strip() + "\n"


def fill(doc: str) -> str:
    """The doc with every `<!-- tables:NAME -->...<!-- /tables -->` block regenerated."""
    return MARK.sub(lambda m: m.group(1) + render(m.group(2)) + m.group(4), doc)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--doc", type=Path, help="rewrite the marked table blocks of this Markdown file")
    ap.add_argument("--check", action="store_true", help="with --doc: only check that the blocks are up to date")
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    if a.doc is None:
        for name in SECTIONS:
            print(render(name))
        return
    doc = a.doc.read_text(encoding="utf-8")
    new = fill(doc)
    if a.check:
        if new != doc:
            raise SystemExit(f"{a.doc}: tables differ from benchmarks/results; run tables.py --doc {a.doc}")
        print(f"{a.doc}: tables match the result files")
        return
    a.doc.write_text(new, encoding="utf-8", newline="\n")
    print(f"{a.doc}: {len(MARK.findall(doc))} table blocks written")


if __name__ == "__main__":
    main()
