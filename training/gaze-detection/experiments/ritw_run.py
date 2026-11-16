"""Reading in the Wild: train and test the detectors on real-world reading.

    python ritw_run.py --length 10 --out results/ritw_10s.json
    python ritw_run.py --length 5  --out results/ritw_5s.json  [--gazebase-weights DIR]

Tables needed (make_table.py on the `ritw` cache), for L = --length:
  ritw_L         the glasses' gaze at 10 fps, nothing added   ("eye tracker")
  ritw_L_webcam  the same with webcam noise, drift, blinks     ("webcam")
  ritw_L_mixed   noise drawn per recording (boosting training)
  ritw_L_clean   clean windows (network training)

Protocol, fixed before any test score is read:
- the authors' split: train + val people are development, test people are
  locked. A person in both is dropped from development.
- 5-fold person-level cross-validation on development picks the detector
  (boosting, network, or their average) by mean window balanced accuracy
  over the two conditions; thresholds come from development scores of each
  condition.
- the chosen detectors are refitted on all development people and the test
  people are scored once. Plain accuracy is reported as well, since that is
  what the paper reports.
With --gazebase-weights, the network trained only on GazeBase (Kaggle run 1)
is also scored on the test people, untouched: a zero-shot transfer check.
"""

import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).parent))
import argparse, json, time
from dataclasses import replace
import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from common import *  # noqa
from readgaze import seqmodel

ap = argparse.ArgumentParser()
ap.add_argument("--length", default="10")
ap.add_argument("--seeds", default="0,1")
ap.add_argument("--epochs", type=int, default=15)
ap.add_argument("--gazebase-weights", default="")
ap.add_argument("--out", default=str(TABLES / "ritw_results.json"))
a = ap.parse_args()
L = a.length
CONDITIONS = {"eye_tracker": f"ritw_{L}", "webcam": f"ritw_{L}_webcam"}
cols = list(FULL_FEATURE_NAMES)
SEEDS = [int(s) for s in a.seeds.split(",")]
t0 = time.time()

tables = {c: load(n, raw=True) for c, n in CONDITIONS.items()}
mixed = load(f"ritw_{L}_mixed")
clean, clean_raw = load(f"ritw_{L}_clean", raw=True)

ref = tables["eye_tracker"][0]
test_people = set(ref.loc[ref["split"] == "test", "participant"])
both = set(ref.loc[ref["split"] != "test", "participant"]) & test_people
dev_people = set(ref["participant"]) - test_people
print(f"RitW {L} s windows: {len(dev_people)} development people, {len(test_people)} test people; "
      f"{len(both)} people appear in both official splits and are kept out of development", flush=True)


def is_dev(t):
    return t["participant"].isin(dev_people).to_numpy()


def is_test(t):
    return (t["split"] == "test").to_numpy()


d0 = ref[is_dev(ref)].reset_index(drop=True)
fold_of = {}
for k, (_, te) in enumerate(folds(d0)):
    for p in set(d0["participant"].to_numpy()[te]):
        fold_of[p] = k


def fold(t):
    return t["participant"].map(fold_of).to_numpy()


# --- boosting -----------------------------------------------------------------
def booster():
    return HistGradientBoostingClassifier(class_weight="balanced", max_iter=400, learning_rate=0.06, random_state=0)


train_parts = [tables["eye_tracker"][0], mixed]
dev_scores = {c: {} for c in CONDITIONS}
test_scores = {c: {} for c in CONDITIONS}
for c, (t, _) in tables.items():
    dev_scores[c]["boosting"] = np.full(is_dev(t).sum(), np.nan)
for k in range(5):
    X = np.concatenate([t.loc[is_dev(t) & (fold(t) != k), cols].to_numpy(float) for t in train_parts])
    y = np.concatenate([t.loc[is_dev(t) & (fold(t) != k), "label"].to_numpy() for t in train_parts])
    m = booster().fit(X, y)
    for c, (t, _) in tables.items():
        dt = t[is_dev(t)]
        sel = fold(dt) == k
        dev_scores[c]["boosting"][sel] = m.predict_proba(dt.loc[sel, cols].to_numpy(float))[:, 1]
m = booster().fit(np.concatenate([t.loc[is_dev(t), cols].to_numpy(float) for t in train_parts]),
                  np.concatenate([t.loc[is_dev(t), "label"].to_numpy() for t in train_parts]))
for c, (t, _) in tables.items():
    test_scores[c]["boosting"] = m.predict_proba(t.loc[is_test(t), cols].to_numpy(float))[:, 1]
print(f"boosting done [{time.time() - t0:.0f} s]", flush=True)

# --- network --------------------------------------------------------------------
cfg = seqmodel.TrainConfig(epochs=a.epochs, threads=4)
aug = seqmodel.Augment()
cdev = is_dev(clean)
cfold = fold(clean)
cy = clean["label"].to_numpy()


def fit(rows):
    return [seqmodel.train(np.asarray(clean_raw[rows]), cy[rows], 10.0, replace(cfg, seed=s), aug) for s in SEEDS]


def score(models, raw):
    return np.mean([seqmodel.predict(m, raw) for m in models], axis=0)


for c, (t, _) in tables.items():
    dev_scores[c]["network"] = np.full(is_dev(t).sum(), np.nan)
for k in range(5):
    models = fit(np.flatnonzero(cdev & (cfold != k)))
    for c, (t, raw) in tables.items():
        dmask = is_dev(t)
        sel = fold(t[dmask]) == k
        dev_scores[c]["network"][sel] = score(models, raw[dmask][sel])
    print(f"network fold {k} [{time.time() - t0:.0f} s]", flush=True)
models = fit(np.flatnonzero(cdev))
out_dir = pathlib.Path(a.out).parent
out_dir.mkdir(parents=True, exist_ok=True)
for s, mdl in zip(SEEDS, models):
    torch.save(mdl.state_dict(), out_dir / f"ritw_{L}s_network_seed{s}.pt")
for c, (t, raw) in tables.items():
    test_scores[c]["network"] = score(models, raw[is_test(t)])

CANDIDATES = ["boosting", "network", "average"]
for c in CONDITIONS:
    dev_scores[c]["average"] = (dev_scores[c]["boosting"] + dev_scores[c]["network"]) / 2
    test_scores[c]["average"] = (test_scores[c]["boosting"] + test_scores[c]["network"]) / 2

# Zero-shot: the GazeBase-only network, never shown RitW.
if a.gazebase_weights:
    zs = []
    for f in sorted(pathlib.Path(a.gazebase_weights).glob("model_cnn_gpu_seed*.pt")):
        net = seqmodel.ReadingNet()
        net.load_state_dict(torch.load(f, map_location="cpu"))
        zs.append(net.to(seqmodel._device("auto")).eval())
    if zs and L == "10":  # trained on 10 s windows
        CANDIDATES.append("gazebase_zero_shot")
        for c, (t, raw) in tables.items():
            dev_scores[c]["gazebase_zero_shot"] = score(zs, raw[is_dev(t)])
            test_scores[c]["gazebase_zero_shot"] = score(zs, raw[is_test(t)])

# --- choose on development, report the test people once ----------------------------
result = {"windows_seconds": float(L), "fps": 10, "development_people": len(dev_people), "test_people": len(test_people),
          "people_in_both_official_splits": len(both), "development": {}, "test": {}}
mean_dev = {}
for cand in CANDIDATES:
    per = {}
    for c, (t, _) in tables.items():
        dt = t[is_dev(t)]
        per[c] = summarise(f"DEV  {cand:18s} {c:11s}", dt, dev_scores[c][cand])
    result["development"][cand] = per
    mean_dev[cand] = float(np.mean([per[c]["window"]["balanced"]["balanced_accuracy"] for c in CONDITIONS]))
eligible = [c for c in CANDIDATES if c != "gazebase_zero_shot"]
chosen = max(eligible, key=mean_dev.get)
result["chosen"] = chosen
result["mean_development_window_balanced_accuracy"] = mean_dev
print(f"\nchosen on development: {chosen}  " + ", ".join(f"{c} {100 * v:.2f}%" for c, v in mean_dev.items()), flush=True)
for cand in CANDIDATES:
    result["test"][cand] = {}
    for c, (t, _) in tables.items():
        tt = t[is_test(t)]
        cuts = result["development"][cand][c]["cuts"]
        result["test"][cand][c] = summarise(f"TEST {cand:18s} {c:11s}", tt, test_scores[c][cand], cuts=cuts, verbose=(cand in (chosen, "gazebase_zero_shot")))
        # Also at a plain 0.5 cut, the way the paper reports accuracy.
        result["test"][cand][c]["at_0.5"] = protocol.metrics_at(tt["label"].to_numpy(), test_scores[c][cand], 0.5)
json.dump(result, open(a.out, "w"), indent=1, default=float)
w = result["test"][chosen]
print(f"\nHEADLINE RitW {L} s ({chosen}, test people): eye tracker window accuracy {100 * w['eye_tracker']['window']['balanced']['accuracy']:.2f}% "
      f"(balanced {100 * w['eye_tracker']['window']['balanced']['balanced_accuracy']:.2f}%), "
      f"webcam {100 * w['webcam']['window']['balanced']['accuracy']:.2f}% (balanced {100 * w['webcam']['window']['balanced']['balanced_accuracy']:.2f}%)  "
      f"[{time.time() - t0:.0f} s] -> {a.out}", flush=True)
