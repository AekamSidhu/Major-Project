# Reading in the Wild checkpoints

The reading network (`readgaze/seqmodel.py`) trained on Reading in the Wild
(Meta, Seattle subset) by `experiments/ritw_run.py`, from the Kaggle run
"OpenMeet reading detector RitW": 10 s and 5 s windows, two seeds each.
Their test numbers are in `../../results/ritw_10s.json` and `ritw_5s.json`.

These are research checkpoints. The sidecar ships the GazeBase network in
`../reading/` (exported to `local-ai/integrity/models/reading_network.onnx`).
They were trained on Reading in the Wild data, so use them only as that
dataset's license allows.
