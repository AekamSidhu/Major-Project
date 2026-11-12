"""Zero-shot on other datasets: fit on GazeBase development people, score GazeBaseVR and DesktopActivity."""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).parent))
import json
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from common import *  # noqa

train_names = (sys.argv[1] if len(sys.argv) > 1 else "gazebase_nominal,gazebase_mixed").split(",")
oof_tag = sys.argv[2] if len(sys.argv) > 2 else "nominal_mixed_all"
drop = sys.argv[3].split(",") if len(sys.argv) > 3 else []
cols = [c for c in ALL_FEATURE_NAMES if c not in drop]

nominal = load("gazebase_nominal")
dev, _ = dev_test(nominal)
dev_people = set(nominal.loc[dev, "participant"])
parts = [load(n) for n in train_names]
parts = [t[t["participant"].isin(dev_people)] for t in parts]
X = np.concatenate([t[cols].to_numpy(dtype=float) for t in parts])
y = np.concatenate([t["label"].to_numpy() for t in parts])
model = HistGradientBoostingClassifier(class_weight="balanced", max_iter=400, learning_rate=0.06, random_state=0).fit(X, y)

# Thresholds come from GazeBase out-of-fold scores (robust.py), never from the outside data.
dnom = nominal[dev].reset_index(drop=True)
oof = np.load(TABLES / f"gazebase_nominal_oof_hgb_{oof_tag}.npy")
cuts = summarise("gazebase dev (reference)", dnom, oof)["cuts"]

out = {}
for name in ["gazebasevr_nominal", "desktopactivity_raw"]:
    t = load(name)
    s = model.predict_proba(t[cols].to_numpy(dtype=float))[:, 1]
    res = summarise(f"{name} @GazeBase cuts", t, s, cuts=cuts)
    own = summarise(f"{name} @own best cut", t, s, verbose=False)
    print(f"   own-cut window bal-acc {own['window']['balanced']['balanced_accuracy']:.4f}; window flag rate by task @GazeBase cut: " + ", ".join(f"{k} {v:.3f}" for k, v in res["window_flag_rate_by_task"].items()))
    print("   recording flag rate by task: " + ", ".join(f"{k} {v:.3f}" for k, v in res["recording_flag_rate_by_task"].items()))
    out[name] = {"gazebase_cuts": res, "own_cuts": own}
    np.save(TABLES / f"{name}_scores_hgb.npy", s)
json.dump(out, open(TABLES / "external_hgb.json", "w"), indent=1)
