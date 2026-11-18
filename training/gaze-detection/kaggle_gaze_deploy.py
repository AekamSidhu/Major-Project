"""Rebuild only the shipped gaze model (no experiments) on a Kaggle GPU notebook.

    !git clone -q -b feature/gaze-detection https://github.com/AekamSidhu/Major-Project.git /kaggle/temp/Major-Project
    !cd /kaggle/working && python -u /kaggle/temp/Major-Project/training/gaze-detection/kaggle_gaze_deploy.py

Runs the cells of notebooks/gaze_transfer_learning.ipynb that the deployment
model needs, unchanged: MPIIFaceGaze download and normalisation, the training
library, the job queue, the `final_all` run (final configuration trained on
all 15 people) and the ONNX export. The experiments and LOPO evaluation are
skipped; their numbers are already in results/gaze_model_results.json.

Output in /kaggle/working/gaze_model: openmeet_gaze_resnet18.onnx,
model.pt (PyTorch state_dict) and cfg.json.
"""

import json
import os
import shutil
from pathlib import Path

NOTEBOOK = Path(__file__).parent / "notebooks" / "gaze_transfer_learning.ipynb"
CELLS = json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"]


def cell(i: int) -> None:
    src = "".join(CELLS[i]["source"])
    first, _, rest = src.partition("\n")
    if first.startswith("%%writefile "):
        Path(first.split(maxsplit=1)[1]).write_text(rest)
        print(f"[cell {i}] wrote {first.split(maxsplit=1)[1]}", flush=True)
        return
    print(f"[cell {i}] {first}", flush=True)
    exec(compile(src, f"cell{i}", "exec"), globals())


for i in (2, 3, 5, 6, 11):  # data, normalisation cache, gazelib.py, run.py, job queue
    cell(i)

# Cell 13's configuration only (its submit() calls are the development experiments).
dev_cell = "".join(CELLS[13]["source"])
config = "\n".join(line for line in dev_cell.splitlines() if not line.startswith("submit("))
exec(compile(config, "cell13-config", "exec"), globals())

# Cell 18 without the 15-fold LOPO evaluation: only the model that ships.
final_cell = "".join(CELLS[18]["source"])
exec(compile("\n".join(line for line in final_cell.splitlines() if "final_lopo" not in line), "cell18-final", "exec"), globals())
JOBS.join()  # noqa: F821 (defined by cell 11)
print(open(f"{RUNS}/final_all/out0.txt").read()[-3000:], flush=True)  # noqa: F821

import gazelib  # noqa: E402  (written by cell 5)

faces, gaze, person = gazelib.load()
cell(24)  # ONNX export, parity check and CPU timing

out = Path("/kaggle/working/gaze_model")
out.mkdir(exist_ok=True)
shutil.copy(ONNX, out / "openmeet_gaze_resnet18.onnx")  # noqa: F821 (cell 24)
shutil.copy(f"{RUNS}/final_all/model0.pt", out / "model.pt")  # noqa: F821
shutil.copy(f"{RUNS}/final_all/cfg.json", out / "cfg.json")  # noqa: F821
(out / "onnx_info.json").write_text(json.dumps(ONNX_INFO, indent=1))  # noqa: F821
print("done:", sorted(p.name for p in out.iterdir()), flush=True)
