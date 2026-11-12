"""Cache the folding features for every window of the given tables."""
import os, sys, pathlib, time
import numpy as np
SRC = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC))
from readgaze.features_fold import FOLD_FEATURE_NAMES, fold_features  # noqa: E402
TABLES = pathlib.Path(os.environ.get("OPENMEET_DATA", r"C:/Users/karaa/openmeet-data")) / "tables"
fps = float(sys.argv[1])
for name in sys.argv[2:]:
    t0 = time.time()
    raw = np.load(TABLES / f"{name}_raw.npy", mmap_mode="r")
    out = np.empty((len(raw), len(FOLD_FEATURE_NAMES)), dtype=np.float32)
    for a in range(0, len(raw), 4096):
        f = fold_features(np.asarray(raw[a : a + 4096]), fps)
        out[a : a + 4096] = np.stack([f[k] for k in FOLD_FEATURE_NAMES], axis=1)
    np.save(TABLES / f"{name}_fold.npy", out)
    print(f"{name}: {out.shape} in {time.time()-t0:.0f} s", flush=True)
