"""Person-level CV on the development people: rule, logistic (6), logistic (all), boosting (all)."""
import sys, time, json
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).parent))
from common import *  # noqa

name = sys.argv[1] if len(sys.argv) > 1 else "gazebase_nominal"
table = load(name)
dev, test = dev_test(table)
d = table[dev].reset_index(drop=True)
print(f"{name}: dev {d['participant'].nunique()} people, {len(d)} windows, reading share {d['label'].mean():.3f}; test people held out: {table[test]['participant'].nunique()}")
y = d["label"].to_numpy()
fs = folds(d)

def oof(make, cols):
    X = d[list(cols)].to_numpy(dtype=float)
    out = np.full(len(d), np.nan)
    for tr, te in fs:
        m = make().fit(X[tr], y[tr])
        out[te] = m.predict_proba(X[te])[:, 1]
    return out

results = {}
results["rule"] = summarise("rule-based", d, d["rule_score"].to_numpy())
t0 = time.time()
results["lr6"] = summarise("logistic, 6 features", d, oof(lambda: make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=1000)), FEATURE_NAMES))
if "snr_x" in d.columns:
    results["lr_all"] = summarise("logistic, 42 features", d, oof(lambda: make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=2000)), ALL_FEATURE_NAMES))
    results["hgb6"] = summarise("boosting, 6 features", d, oof(lambda: HistGradientBoostingClassifier(class_weight="balanced", max_iter=300, learning_rate=0.08, random_state=0), FEATURE_NAMES))
    s = oof(lambda: HistGradientBoostingClassifier(class_weight="balanced", max_iter=400, learning_rate=0.06, random_state=0), ALL_FEATURE_NAMES)
    results["hgb_all"] = summarise("boosting, 42 features", d, s)
    np.save(TABLES / f"{name}_oof_hgb_all.npy", s)
    print("window flag rate by task (hgb_all, balanced cut):", {k: round(v, 3) for k, v in results["hgb_all"]["window_flag_rate_by_task"].items()})
    print("recording flag rate by task:", {k: round(v, 3) for k, v in results["hgb_all"]["recording_flag_rate_by_task"].items()})
print(f"{time.time()-t0:.0f} s")
json.dump(results, open(TABLES / f"{name}_baselines.json", "w"), indent=1)
