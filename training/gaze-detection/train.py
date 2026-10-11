"""Train and cross-validate the reading detector on a feature table.

    python train.py                                      # data/features/gazebase_10fps.csv
    python train.py --features data/features/synthetic_10fps.csv --out results/smoke

Writes to the output folder (default: results/):
  cv_report.json   person-level cross-validation metrics for the learned
                   model and the rule-based detector
  model.json       the learned model fitted on everything, with the
                   answer-level threshold; this is what OpenMeet will load
Out-of-fold window scores are saved next to the feature table (in data/),
because they are per-person data.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from readgaze.training import cross_validate, fit_final

HERE = Path(__file__).parent


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--features", type=Path, default=HERE / "data" / "features" / "gazebase_10fps.csv")
    parser.add_argument("--out", type=Path, default=HERE / "results")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--target-fpr", type=float, default=0.05, help="allowed false-positive rate (default 5%%)")
    parser.add_argument("--c", type=float, default=1.0, help="inverse regularisation strength")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    if not args.features.exists():
        raise SystemExit(f"{args.features} not found. Run prepare.py first.")
    table = pd.read_csv(args.features)
    settings_path = args.features.with_suffix(".json")
    data_settings = json.loads(settings_path.read_text()) if settings_path.exists() else {}

    scored, report = cross_validate(table, n_splits=args.folds, target_fpr=args.target_fpr, c=args.c, seed=args.seed)
    report["data"] = data_settings
    report["c"] = args.c
    report["seed"] = args.seed

    threshold = report["model_score"]["recordings"]["threshold"]
    model = fit_final(
        table,
        threshold,
        c=args.c,
        metadata={"trained_on": str(args.features.name), "target_false_positive_rate": args.target_fpr, "data": data_settings},
    )

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "cv_report.json").write_text(json.dumps(report, indent=2))
    (args.out / "model.json").write_text(json.dumps(model, indent=2))
    scored.to_csv(args.features.with_name(f"{args.features.stem}_oof.csv"), index=False)

    print(f"{report['participants']} participants, {report['recordings']} recordings, {report['windows']} windows")
    for name in ("model_score", "rule_score"):
        rec = report[name]["recordings"]
        print(
            f"{name:12s} per answer: ROC-AUC {rec['roc_auc']:.3f}  "
            f"recall {rec['recall']:.2f} at false-positive rate {rec['false_positive_rate']:.2f}"
        )
    print(f"saved {args.out / 'cv_report.json'} and {args.out / 'model.json'}")


if __name__ == "__main__":
    main()
