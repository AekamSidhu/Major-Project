"""Sequence model: person-level CV on the development people, then a final fit.

Trains on clean windows (fresh noise each step). Each fold scores the fixed
noisy windows of its held-out people in every --eval table, so the
out-of-fold scores line up row by row with the feature models'.
--final also fits on all development people and scores every row of the
--eval and --extra tables (test people and outside datasets included; those
scores are only read by the final report).
"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).parent))
import argparse, json, time
import numpy as np
import pandas as pd
import torch
from common import *  # noqa
from readgaze import seqmodel

ap = argparse.ArgumentParser()
ap.add_argument("--eval", default="gazebase_nominal,gazebase_hard,gazebase_small,gazebase_small_hard")
ap.add_argument("--extra", default="")
ap.add_argument("--clean", default="gazebase_clean")
ap.add_argument("--tag", default="cnn")
ap.add_argument("--epochs", type=int, default=10)
ap.add_argument("--widths", default="32,48,64,96")
ap.add_argument("--kernel", type=int, default=5)
ap.add_argument("--threads", type=int, default=4)
ap.add_argument("--batch", type=int, default=256)
ap.add_argument("--lr", type=float, default=2e-3)
ap.add_argument("--dropout", type=float, default=0.1)
ap.add_argument("--noise", default="0.3,3.5")
ap.add_argument("--amp", default="0.4,1.3")
ap.add_argument("--folds", type=int, default=5)
ap.add_argument("--skip-cv", action="store_true")
ap.add_argument("--final", action="store_true")
ap.add_argument("--fps", type=float, default=10.0)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--seeds", default="", help="comma-separated seeds; their scores are averaged (overrides --seed)")
a = ap.parse_args()
KEEP = ["participant", "recording", "task", "label"]

def slim(name):
    t = pd.read_pickle(TABLES / f"{name}.pkl")[KEEP].reset_index(drop=True)
    return t, np.load(TABLES / f"{name}_raw.npy", mmap_mode="r")

evals = {n: slim(n) for n in a.eval.split(",")}
first = next(iter(evals.values()))[0]
dev_people, test_people = protocol.split_people(first["participant"])
d = first[first["participant"].isin(dev_people)].reset_index(drop=True)
fold_of = {}
for k, (tr, te) in enumerate(folds(d, a.folds)):
    for p in set(d["participant"].to_numpy()[te]):
        fold_of[p] = k

ctable, craw = slim(a.clean)
cdev = ctable["participant"].isin(dev_people).to_numpy()
cy = ctable["label"].to_numpy()
cfold = ctable["participant"].map(fold_of).to_numpy()  # NaN for test people
cfg = seqmodel.TrainConfig(epochs=a.epochs, widths=tuple(int(v) for v in a.widths.split(",")), kernel=a.kernel, lr=a.lr, dropout=a.dropout, seed=a.seed, threads=a.threads, batch=a.batch)
aug = seqmodel.Augment(noise=tuple(float(v) for v in a.noise.split(",")), amplitude=tuple(float(v) for v in a.amp.split(",")))
print(f"{a.tag}: clean dev windows {cdev.sum()}; seeds {a.seeds or a.seed}; device {seqmodel._device(cfg.device)}; {cfg}; {aug}", flush=True)

SEEDS = [int(v) for v in a.seeds.split(",")] if a.seeds else [a.seed]


def fit(rows):
    """One network per seed, trained on the clean windows `rows`."""
    from dataclasses import replace
    return [seqmodel.train(np.asarray(craw[rows]), cy[rows], a.fps, replace(cfg, seed=s), aug, log=lambda m: None) for s in SEEDS]


def score(models, raw, rows):
    if not len(rows):
        return np.zeros(0)
    window = np.asarray(raw[rows])
    return np.mean([seqmodel.predict(m, window) for m in models], axis=0)

t0 = time.time()
if not a.skip_cv:
    oof = {n: np.full(len(t), np.nan) for n, (t, _) in evals.items()}
    for k in range(a.folds):
        rows = np.flatnonzero(cdev & (cfold != k))
        model = fit(rows)
        for n, (t, raw) in evals.items():
            te = np.flatnonzero((t["participant"].map(fold_of) == k).to_numpy())
            oof[n][te] = score(model, raw, te)
        t, _ = evals[a.eval.split(",")[0]]
        te = np.flatnonzero((t["participant"].map(fold_of) == k).to_numpy())
        summarise(f"{a.tag} fold {k} [{time.time()-t0:.0f} s]", t.iloc[te], oof[a.eval.split(',')[0]][te])
    res, cuts = {}, None
    for n, (t, _) in evals.items():
        isdev = t["participant"].isin(dev_people).to_numpy()
        own = summarise(f"{a.tag} {n} (own cut)", t[isdev], oof[n][isdev], verbose=False)
        if cuts is None:
            cuts = own["cuts"]
        fixed = summarise(f"{a.tag} {n} @nominal cut", t[isdev], oof[n][isdev], cuts=cuts)
        res[n] = {"own": own, "fixed": fixed}
        np.save(TABLES / f"{n}_oof_{a.tag}.npy", oof[n][isdev])
    json.dump(res, open(TABLES / f"cv_{a.tag}.json", "w"), indent=1)

if a.final:
    rows = np.flatnonzero(cdev)
    model = fit(rows)
    for s, m in zip(SEEDS, model):
        torch.save(m.state_dict(), TABLES / f"model_{a.tag}_seed{s}.pt")
    names = list(evals) + [n for n in a.extra.split(",") if n]
    for n in names:
        t, raw = evals[n] if n in evals else slim(n)
        np.save(TABLES / f"{n}_final_{a.tag}.npy", score(model, raw, np.arange(len(t))))
    print(f"final model saved, scored {names} [{time.time()-t0:.0f} s]", flush=True)
