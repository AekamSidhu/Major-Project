"""Boosted detector across conditions: train at one condition (or the mix), test at each."""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).parent))
import json, time
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from common import *  # noqa

EVAL = ["gazebase_nominal", "gazebase_hard", "gazebase_small", "gazebase_small_hard"]
tag = sys.argv[1] if len(sys.argv) > 1 else "all"
SETS = {"all": ALL_FEATURE_NAMES, "full": FULL_FEATURE_NAMES, "six": FEATURE_NAMES, "fold": FOLD_FEATURE_NAMES}
cols = list(SETS[tag.split("-")[0]])
drop = [c for c in sys.argv[2].split(",")] if len(sys.argv) > 2 and sys.argv[2] else []
cols = [c for c in cols if c not in drop]
only = sys.argv[3].split(",") if len(sys.argv) > 3 else None

nominal = load("gazebase_nominal")
dev, _ = dev_test(nominal)
d = nominal[dev].reset_index(drop=True)
fold_of = {}
for k, (tr, te) in enumerate(folds(d)):
    for p in set(d["participant"].to_numpy()[te]):
        fold_of[p] = k

def dev_part(name):
    t = load(name)
    t = t[t["participant"].isin(fold_of.keys())].reset_index(drop=True)
    return t, t["participant"].map(fold_of).to_numpy()

evals = {n: dev_part(n) for n in EVAL}
trains = {"nominal": [dev_part("gazebase_nominal")], "mixed": [dev_part("gazebase_mixed")]}
trains["nominal+mixed"] = trains["nominal"] + trains["mixed"]
if only:
    trains = {k: v for k, v in trains.items() if k in only}

out = {}
for train_name, parts in trains.items():
    scores = {n: np.full(len(evals[n][0]), np.nan) for n in EVAL}
    t0 = time.time()
    for k in range(5):
        X = np.concatenate([t.loc[f != k, cols].to_numpy(dtype=float) for t, f in parts])
        y = np.concatenate([t.loc[f != k, "label"].to_numpy() for t, f in parts])
        m = HistGradientBoostingClassifier(class_weight="balanced", max_iter=400, learning_rate=0.06, random_state=0).fit(X, y)
        for n in EVAL:
            t, f = evals[n]
            scores[n][f == k] = m.predict_proba(t.loc[f == k, cols].to_numpy(dtype=float))[:, 1]
    print(f"== trained on {train_name} ({tag}, {len(cols)} features), {time.time()-t0:.0f} s")
    cuts = None
    out[train_name] = {}
    for n in EVAL:
        t, _ = evals[n]
        own = summarise(f"  {n} (own cut)", t, scores[n], verbose=False)
        if n == "gazebase_nominal":
            cuts = own["cuts"]
        fixed = summarise(f"  {n}", t, scores[n], cuts=cuts, verbose=False)
        w, r = fixed["window"]["balanced"], fixed["recording"]["balanced"]
        print(f"  {n:22s} WIN auc {own['window']['balanced']['roc_auc']:.4f} bal-acc(own cut) {own['window']['balanced']['balanced_accuracy']:.4f} bal-acc(nominal cut) {w['balanced_accuracy']:.4f} (rec {w['recall']:.3f} spec {w['specificity']:.3f}) "
              f"|| REC auc {own['recording']['balanced']['roc_auc']:.4f} bal-acc(nominal cut) {r['balanced_accuracy']:.4f} (rec {r['recall']:.3f} spec {r['specificity']:.3f})", flush=True)
        out[train_name][n] = {"own": own, "fixed": fixed}
        np.save(TABLES / f"{n}_oof_hgb_{train_name.replace('+','_')}_{tag}.npy", scores[n])
json.dump(out, open(TABLES / f"robust_hgb_{tag}.json", "w"), indent=1)
