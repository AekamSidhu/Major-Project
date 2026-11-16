"""Leave one kind of non-reading out of training: is a behaviour the detector
never saw still told apart from reading? (Our real negative, a person talking
in an interview, is not in any dataset, so this is the closest stand-in.)"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).parent))
import json, time
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from common import *  # noqa

train_names = sys.argv[1].split(",") if len(sys.argv) > 1 else ["gazebase_nominal", "gazebase_mixed"]
eval_name = sys.argv[2] if len(sys.argv) > 2 else "gazebase_nominal"
cols = list(ALL_FEATURE_NAMES)
GROUPS = {"fixation (FXS)": ["FXS"], "horizontal saccades (HSS)": ["HSS"], "random saccades (RAN)": ["RAN"], "video (VD1+VD2)": ["VD1", "VD2"], "game (BLG)": ["BLG"]}

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

parts = [dev_part(n) for n in train_names]
ev, ev_fold = dev_part(eval_name)
task = ev["task"].to_numpy()
y = ev["label"].to_numpy()
out = {}
for group, left_out in {"nothing (reference)": [], **GROUPS}.items():
    scores = np.full(len(ev), np.nan)
    t0 = time.time()
    for k in range(5):
        X = np.concatenate([t.loc[(f != k) & ~t["task"].isin(left_out), cols].to_numpy(dtype=float) for t, f in parts])
        yy = np.concatenate([t.loc[(f != k) & ~t["task"].isin(left_out), "label"].to_numpy() for t, f in parts])
        m = HistGradientBoostingClassifier(class_weight="balanced", max_iter=400, learning_rate=0.06, random_state=0).fit(X, yy)
        scores[ev_fold == k] = m.predict_proba(ev.loc[ev_fold == k, cols].to_numpy(dtype=float))[:, 1]
    seen = ~np.isin(task, left_out)
    cut = protocol.best_balanced_threshold(y[seen], scores[seen])  # picked without the left-out task
    row = {"threshold": cut, "seen_balanced_accuracy": protocol.metrics_at(y[seen], scores[seen], cut)["balanced_accuracy"]}
    for g, tasks in GROUPS.items():
        sel = np.isin(task, tasks)
        both = sel | (y == 1)
        row[g] = {"flag_rate": float((scores[sel] >= cut).mean()), "auc_vs_reading": float(roc_auc_score(y[both], scores[both]))}
    row["reading_recall"] = float((scores[y == 1] >= cut).mean())
    out[group] = row
    if left_out:
        r = row[group]
        ref = out["nothing (reference)"][group]
        print(f"left out {group:28s}: flagged {100*r['flag_rate']:.1f}% of its windows (seen in training: {100*ref['flag_rate']:.1f}%), AUC vs reading {r['auc_vs_reading']:.4f} (seen: {ref['auc_vs_reading']:.4f}); reading recall {row['reading_recall']:.3f}  [{time.time()-t0:.0f} s]", flush=True)
    else:
        print(f"reference: balanced accuracy {row['seen_balanced_accuracy']:.4f}, reading recall {row['reading_recall']:.3f}", flush=True)
json.dump(out, open(TABLES / f"loto_hgb_{eval_name}.json", "w"), indent=1)
