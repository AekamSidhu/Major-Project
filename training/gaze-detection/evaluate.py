"""Test a trained model on data it has never seen, at its saved threshold.

    python evaluate.py --model results/model.json --features data/features/own_recordings.csv

Use this for the honest numbers: our own recordings, or OEP as an external
test set. The feature table must come from prepare.py-style processing
(same window settings as training).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from readgaze.evaluation import flag_rate_by_group, summarize
from readgaze.features import FEATURE_NAMES
from readgaze.model import score_from_json
from readgaze.training import recording_scores


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--out", type=Path, help="write the report here as JSON")
    args = parser.parse_args(argv)

    model = json.loads(args.model.read_text())
    table = pd.read_csv(args.features)
    table["model_score"] = [score_from_json(model, row) for row in table[list(FEATURE_NAMES)].to_dict("records")]

    recordings = recording_scores(table, "model_score")
    threshold = model["threshold"]
    report = {
        "model": str(args.model),
        "features": str(args.features),
        "recordings": summarize(recordings["label"], recordings["model_score"], threshold),
        "recording_flag_rate_by_task": flag_rate_by_group(recordings["task"].to_numpy(), recordings["model_score"], threshold),
    }
    text = json.dumps(report, indent=2)
    print(text)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text)


if __name__ == "__main__":
    main()
