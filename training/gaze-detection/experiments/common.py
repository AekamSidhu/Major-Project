"""Shared helpers for the exploration scripts."""
import os, sys, json, time
from pathlib import Path
import numpy as np
import pandas as pd

SRC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC))
from sklearn.model_selection import StratifiedGroupKFold  # noqa: E402
from readgaze import protocol  # noqa: E402
from readgaze.detector import aggregate  # noqa: E402
from readgaze.features import FEATURE_NAMES  # noqa: E402
from readgaze.features_ext import ALL_FEATURE_NAMES, EXT_FEATURE_NAMES  # noqa: E402
from readgaze.features_fold import FOLD_FEATURE_NAMES  # noqa: E402
FULL_FEATURE_NAMES = tuple(ALL_FEATURE_NAMES) + tuple(FOLD_FEATURE_NAMES)

DATA = Path(os.environ.get("OPENMEET_DATA", r"C:/Users/karaa/openmeet-data"))
TABLES = DATA / "tables"


def load(name, raw=False):
    table = pd.read_pickle(TABLES / f"{name}.pkl")
    fold_path = TABLES / f"{name}_fold.npy"
    if fold_path.exists():
        fold = np.load(fold_path)
        for j, col in enumerate(FOLD_FEATURE_NAMES):
            table[col] = fold[:, j]
    if raw:
        return table, np.load(TABLES / f"{name}_raw.npy")
    return table


def dev_test(table):
    dev_people, test_people = protocol.split_people(table["participant"])
    is_test = table["participant"].isin(test_people).to_numpy()
    return ~is_test, is_test


def folds(table, n_splits=5, seed=0):
    y = table["label"].to_numpy()
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(cv.split(np.zeros(len(table)), y, groups=table["participant"].to_numpy()))


def recording_level(table, scores):
    frame = pd.DataFrame({"recording": table["recording"].to_numpy(), "label": table["label"].to_numpy(), "task": table["task"].to_numpy(), "score": scores})
    return frame.groupby("recording").agg(label=("label", "first"), task=("task", "first"), score=("score", aggregate)).reset_index()


def summarise(name, table, scores, cuts=None, verbose=True):
    """Window- and recording-level metrics. If `cuts` is None, thresholds are picked on these scores (dev)."""
    y = table["label"].to_numpy()
    rec = recording_level(table, scores)
    if cuts is None:
        cuts = {"window": protocol.thresholds(y, scores), "recording": protocol.thresholds(rec["label"].to_numpy(), rec["score"].to_numpy())}
    win = protocol.report(y, scores, cuts["window"])
    recm = protocol.report(rec["label"].to_numpy(), rec["score"].to_numpy(), cuts["recording"])
    out = {"window": win, "recording": recm, "cuts": cuts,
           "window_flag_rate_by_task": protocol.flag_rates(table["task"].to_numpy(), scores, cuts["window"]["balanced"]),
           "recording_flag_rate_by_task": protocol.flag_rates(rec["task"].to_numpy(), rec["score"].to_numpy(), cuts["recording"]["balanced"])}
    if verbose:
        w, r = win["balanced"], recm["balanced"]
        wl, rl = win["low_fpr"], recm["low_fpr"]
        print(f"{name:28s} WIN auc {w['roc_auc']:.4f} bal-acc {w['balanced_accuracy']:.4f} (rec {w['recall']:.3f} spec {w['specificity']:.3f}) | @fpr5 rec {wl['recall']:.3f} fpr {wl['false_positive_rate']:.3f} "
              f"|| REC auc {r['roc_auc']:.4f} bal-acc {r['balanced_accuracy']:.4f} (rec {r['recall']:.3f} spec {r['specificity']:.3f}) | @fpr5 rec {rl['recall']:.3f} fpr {rl['false_positive_rate']:.3f}", flush=True)
    return out
