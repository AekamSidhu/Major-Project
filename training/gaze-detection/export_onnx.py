"""Export the trained reading network for the OpenMeet sidecar.

    python export_onnx.py model_cnn_gpu_seed0.pt model_cnn_gpu_seed1.pt model_cnn_gpu_seed2.pt \
        --results results/reading_detector_results.json --out ../../local-ai/integrity/models

Writes `reading_network.onnx` (the seeds' probabilities averaged inside one
graph, so the sidecar needs only onnxruntime) and `reading_network.json`
(input format, window, thresholds from development people, provenance).
Input: `(n, 3, 100)` float32 from `readgaze.seqmodel.to_input` (x, y, valid
flag of a 10 s window at 10 fps). Output: `(n,)` reading probability.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from readgaze import seqmodel  # noqa: E402


class Average(torch.nn.Module):
    def __init__(self, nets: list[torch.nn.Module]) -> None:
        super().__init__()
        self.nets = torch.nn.ModuleList(nets)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.stack([torch.sigmoid(n(x)) for n in self.nets]).mean(dim=0)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("weights", nargs="+", type=Path)
    ap.add_argument("--results", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    nets = []
    for path in a.weights:
        net = seqmodel.ReadingNet()
        net.load_state_dict(torch.load(path, map_location="cpu"))
        nets.append(net.eval())
    model = Average(nets).eval()
    a.out.mkdir(parents=True, exist_ok=True)
    example = torch.from_numpy(seqmodel.to_input(np.random.default_rng(0).normal(size=(4, 100, 2)).cumsum(axis=1)))
    onnx_path = a.out / "reading_network.onnx"
    torch.onnx.export(model, (example,), str(onnx_path), input_names=["window"], output_names=["reading_probability"],
                      dynamic_axes={"window": {0: "n"}, "reading_probability": {0: "n"}}, opset_version=17, dynamo=False)

    import onnxruntime as ort

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    got = session.run(None, {"window": example.numpy()})[0]
    with torch.no_grad():
        want = model(example).numpy()
    diff = float(np.abs(got - want).max())
    assert diff < 1e-5, diff

    results = json.loads(a.results.read_text())
    cuts = results["details"]["selection"]["network"]["cuts"]
    test = results["test"]["network"]
    meta = {
        "model": "OpenMeet reading network v1: residual 1-D CNN, 3 seeds averaged",
        "input": {"name": "window", "shape": ["n", 3, 100], "fps": 10, "seconds": 10,
                  "channels": "x, y (centred on the median, divided by the window's spread, clipped to +-6), valid-frame flag",
                  "gaps": "linear interpolation, nearest value at the ends"},
        "output": "reading probability per window",
        "answer_score": "mean of the top half of the answer's window scores (10 s windows every 5 s)",
        "thresholds": {"window_balanced": cuts["window"]["balanced"], "window_low_false_positive": cuts["window"]["low_fpr"],
                       "answer_balanced": cuts["recording"]["balanced"], "answer_low_false_positive": cuts["recording"]["low_fpr"]},
        "trained_on": "GazeBase (CC BY 4.0), 258 development people, degraded to webcam quality",
        "test_people_window_balanced_accuracy": {k: v["window_balanced_accuracy"] for k, v in test.items()},
        "onnx_max_abs_diff_vs_pytorch": diff,
    }
    (a.out / "reading_network.json").write_text(json.dumps(meta, indent=1))
    print(f"wrote {onnx_path} ({onnx_path.stat().st_size / 1e6:.2f} MB), max diff {diff:.1e}")


if __name__ == "__main__":
    main()
