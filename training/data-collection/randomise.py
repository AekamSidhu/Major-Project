"""Each participant's questions and condition order, as a pre-filled label sheet.

    python randomise.py --participants P01 P02 P03 --out sheets/
    python randomise.py --count 20 --out sheets/          # P01 ... P20

Per participant: the 3 warm-up questions first (honest; they form the
delivery baseline), then 12 study questions. Each condition gets one
technical, one open and one behavioural question, so condition and
question type are balanced within every person. Questions are drawn from
question_bank.csv, the order is shuffled (never the same condition three
times in a row), and half of the read or paraphrased answers put the
material below the camera. The plan depends only on the study seed and the
participant id, so a sheet can be regenerated.

Writes sheets/<participant>.csv and sheets/all.csv, with the columns of
label_sheet_template.csv plus the question text.
"""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

HERE = Path(__file__).resolve().parent
CONDITIONS = ("honest", "read_script", "read_ai", "paraphrase_ai")  # as in local-ai/room/study.py
TYPES = ("technical", "open", "behavioural")
WITH_MATERIAL = ("read_script", "read_ai", "paraphrase_ai")
SEED = 2026
COLUMNS = ["participant", "session", "answer_no", "question_id", "question_type", "question", "condition", "screen_position",
           "material_id", "ai_model", "english", "glasses", "camera", "notes", "exclude", "exclude_reason"]


def load_bank(path: Path = HERE / "question_bank.csv") -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def no_triples(order: list[str]) -> bool:
    return all(not (a == b == c) for a, b, c in zip(order, order[1:], order[2:]))


def plan(participant: str, bank: list[dict[str, str]], seed: int = SEED) -> list[dict[str, str]]:
    rng = random.Random(f"{seed}:{participant}")
    rows = [{"question": q, "condition": "honest"} for q in bank if q["type"] == "warmup"]
    study = []
    for qtype in TYPES:
        pool = [q for q in bank if q["type"] == qtype]
        if len(pool) < len(CONDITIONS):
            raise ValueError(f"the question bank needs at least {len(CONDITIONS)} {qtype} questions")
        conditions = list(CONDITIONS)
        rng.shuffle(conditions)
        study += [{"question": q, "condition": c} for q, c in zip(rng.sample(pool, len(CONDITIONS)), conditions)]
    while True:
        rng.shuffle(study)
        if no_triples([s["condition"] for s in study]):
            break
    material = [s for s in study if s["condition"] in WITH_MATERIAL]
    positions = ["side", "below_camera"] * (len(material) // 2) + [rng.choice(["side", "below_camera"])] * (len(material) % 2)
    rng.shuffle(positions)
    for s, pos in zip(material, positions):
        s["screen_position"] = pos
    out = []
    for n, item in enumerate(rows + study, 1):
        q = item["question"]
        out.append({**{c: "" for c in COLUMNS}, "participant": participant, "answer_no": str(n), "question_id": q["id"],
                    "question_type": q["type"], "question": q["question"], "condition": item["condition"],
                    "screen_position": item.get("screen_position", ""),
                    "material_id": f"{participant}-{q['id']}" if item["condition"] in WITH_MATERIAL else "", "exclude": "no"})
    return out


def write(rows: list[dict[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    who = ap.add_mutually_exclusive_group(required=True)
    who.add_argument("--participants", nargs="+")
    who.add_argument("--count", type=int)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--bank", type=Path, default=HERE / "question_bank.csv")
    ap.add_argument("--seed", type=int, default=SEED)
    a = ap.parse_args()
    ids = a.participants or [f"P{i:02d}" for i in range(1, a.count + 1)]
    bank = load_bank(a.bank)
    everyone = []
    for pid in ids:
        rows = plan(pid, bank, a.seed)
        write(rows, a.out / f"{pid}.csv")
        everyone += rows
    write(everyone, a.out / "all.csv")
    print(f"wrote {len(ids)} sheets and all.csv ({len(everyone)} answers) to {a.out}")


if __name__ == "__main__":
    main()
