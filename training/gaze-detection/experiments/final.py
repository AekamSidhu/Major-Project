"""Pick the detector on development people, then score the locked test people once.

    python final.py --cnn-tag cnn_gpu --out results/reading_detector_results.json

Needs, in the tables folder:
- the boosted model's out-of-fold scores (`robust.py full "" nominal+mixed`),
- the network's out-of-fold and final scores (`cnn.py --final`).

Candidates are the boosted model (58 features), the network, and the mean of
the two. The rule, fixed before any test score is read: highest mean window
balanced accuracy over the four development conditions, at thresholds chosen
on development nominal scores. The winner's thresholds are then applied
unchanged to the test people and to the outside datasets.
"""

import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).parent))
import argparse, json, time
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from common import *  # noqa

ap = argparse.ArgumentParser()
ap.add_argument("--cnn-tag", default="cnn_gpu")
ap.add_argument("--conditions", default="gazebase_nominal,gazebase_hard,gazebase_small,gazebase_small_hard")
ap.add_argument("--test-only", default="gazebase_slow", help="extra conditions scored on test people only")
ap.add_argument("--outside", default="gazebasevr_nominal,desktopactivity_raw")
ap.add_argument("--out", default=str(TABLES / "reading_detector_results.json"))
ap.add_argument("--dry-run", action="store_true", help="plumbing check: small final fit, test metrics computed but never shown")
a = ap.parse_args()
conditions = a.conditions.split(",")
test_only = [n for n in a.test_only.split(",") if n and (TABLES / f"{n}.pkl").exists()]
outside = [n for n in a.outside.split(",") if n]
cols = list(FULL_FEATURE_NAMES)
t0 = time.time()

nominal = load("gazebase_nominal")
dev_people, test_people = protocol.split_people(nominal["participant"])
dev_people, test_people = set(dev_people), set(test_people)

# --- development: out-of-fold scores of every candidate -------------------
dev_scores, dev_tables = {}, {}
for n in conditions:
    t = load(n)
    t = t[t["participant"].isin(dev_people)].reset_index(drop=True)
    dev_tables[n] = t
    boost = np.load(TABLES / f"{n}_oof_hgb_nominal_mixed_full.npy")
    cnn = np.load(TABLES / f"{n}_oof_{a.cnn_tag}.npy")
    assert len(boost) == len(cnn) == len(t), n
    dev_scores[n] = {"boosting": boost, "network": cnn, "average": (boost + cnn) / 2}

CANDIDATES = ["boosting", "network", "average"]
selection = {}
for c in CANDIDATES:
    cuts = summarise(c, dev_tables["gazebase_nominal"], dev_scores["gazebase_nominal"][c], verbose=False)["cuts"]
    per = {n: summarise(f"dev {c} {n}", dev_tables[n], dev_scores[n][c], cuts=cuts) for n in conditions}
    selection[c] = {
        "cuts": cuts,
        "mean_window_balanced_accuracy": float(np.mean([per[n]["window"]["balanced"]["balanced_accuracy"] for n in conditions])),
        "per_condition": per,
    }
chosen = max(CANDIDATES, key=lambda c: selection[c]["mean_window_balanced_accuracy"])
cuts = selection[chosen]["cuts"]
print(f"\nchosen on development people: {chosen} "
      + ", ".join(f"{c} {100 * selection[c]['mean_window_balanced_accuracy']:.2f}%" for c in CANDIDATES), flush=True)

# --- final boosted model on every development person -----------------------
train = [load(n) for n in ("gazebase_nominal", "gazebase_mixed")]
train = [t[t["participant"].isin(dev_people)] for t in train]
booster = HistGradientBoostingClassifier(class_weight="balanced", max_iter=400, learning_rate=0.06, random_state=0)
if a.dry_run:
    train = [t.sample(n=min(len(t), 20000), random_state=0) for t in train]
booster.fit(np.concatenate([t[cols].to_numpy(dtype=float) for t in train]), np.concatenate([t["label"].to_numpy() for t in train]))


def final_scores(name):
    t = load(name)
    boost = booster.predict_proba(t[cols].to_numpy(dtype=float))[:, 1]
    cnn = np.load(TABLES / f"{name}_final_{a.cnn_tag}.npy")
    assert len(cnn) == len(t), name
    return t, {"boosting": boost, "network": cnn, "average": (boost + cnn) / 2}


# --- locked test people, scored once ----------------------------------------
test, outside_res = {}, {}
for n in conditions + test_only:
    t, s = final_scores(n)
    keep = t["participant"].isin(test_people).to_numpy()
    assert not set(t.loc[keep, "participant"]) & dev_people
    test[n] = {c: summarise(f"TEST {c:9s} {n}", t[keep], s[c][keep], cuts=selection[c]["cuts"], verbose=(c == chosen and not a.dry_run)) for c in CANDIDATES}
for n in outside:
    t, s = final_scores(n)
    outside_res[n] = {c: summarise(f"OUTSIDE {c:9s} {n}", t, s[c], cuts=selection[c]["cuts"], verbose=(c == chosen and not a.dry_run)) for c in CANDIDATES}


def headline(res):
    w, r = res["window"], res["recording"]
    return {
        "window_balanced_accuracy": w["balanced"]["balanced_accuracy"],
        "window_accuracy": w["balanced"]["accuracy"],
        "window_roc_auc": w["balanced"]["roc_auc"],
        "window_recall_at_low_fpr": w["low_fpr"]["recall"],
        "window_fpr_at_low_fpr": w["low_fpr"]["false_positive_rate"],
        "recording_balanced_accuracy": r["balanced"]["balanced_accuracy"],
        "recording_roc_auc": r["balanced"]["roc_auc"],
        "recording_recall_at_low_fpr": r["low_fpr"]["recall"],
        "recording_fpr_at_low_fpr": r["low_fpr"]["false_positive_rate"],
    }


summary = {
    "about": "OpenMeet reading detector: is this person reading? 10 s gaze windows at 10 fps, GazeBase degraded to webcam quality.",
    "protocol": {
        "split": f"{len(dev_people)} development / {len(test_people)} test people (protocol.split_people, seed {protocol.SPLIT_SEED}); test people scored once",
        "selection": "highest mean development window balanced accuracy over the four conditions; thresholds from development nominal scores",
        "thresholds": "balanced = best development balanced accuracy; low_fpr = at most 5% of development non-reading flagged",
        "recording": "recording score = mean of its top half of window scores",
    },
    "chosen": chosen,
    "development_cv": {c: {"mean_window_balanced_accuracy": selection[c]["mean_window_balanced_accuracy"],
                           **{n: headline(selection[c]["per_condition"][n]) for n in conditions}} for c in CANDIDATES},
    "test": {c: {n: headline(test[n][c]) for n in test} for c in CANDIDATES},
    "outside": {c: {n: headline(outside_res[n][c]) for n in outside_res} for c in CANDIDATES},
    "flag_rate_by_task": {
        "test": {n: test[n][chosen]["window_flag_rate_by_task"] for n in test},
        "outside": {n: outside_res[n][chosen]["window_flag_rate_by_task"] for n in outside_res},
        "outside_recordings": {n: outside_res[n][chosen]["recording_flag_rate_by_task"] for n in outside_res},
    },
    "thresholds": cuts,
    "details": {"selection": selection, "test": test, "outside": outside_res},
}
pathlib.Path(a.out).parent.mkdir(parents=True, exist_ok=True)
json.dump(summary, open(a.out, "w"), indent=1, default=float)
if a.dry_run:
    print(f"dry run ok: {len(test)} test conditions, {len(outside_res)} outside datasets scored (not shown) [{time.time() - t0:.0f} s]")
    sys.exit(0)
h = summary["test"][chosen]["gazebase_nominal"]
print(f"\nHEADLINE ({chosen}, test people, nominal): window balanced accuracy {100 * h['window_balanced_accuracy']:.2f}% "
      f"(AUC {h['window_roc_auc']:.4f}); per answer {100 * h['recording_balanced_accuracy']:.2f}%  [{time.time() - t0:.0f} s] -> {a.out}")
