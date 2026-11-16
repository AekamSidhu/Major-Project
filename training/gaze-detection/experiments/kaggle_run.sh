#!/usr/bin/env bash
# The whole reading-detector run, start to finish, on a Kaggle notebook
# (accelerator GPU T4, internet on). About two hours, most of it CPU work
# building the feature tables. In a notebook cell:
#
#   !git clone -q -b feature/gaze-detection https://github.com/AekamSidhu/Major-Project.git
#   !bash Major-Project/training/gaze-detection/experiments/kaggle_run.sh
#
# Data goes to /kaggle/tmp (not saved); results, logs and model weights go to
# /kaggle/working/results, which Kaggle keeps as the version's output.
set -euo pipefail
cd "$(dirname "$0")"
export OPENMEET_DATA="${OPENMEET_DATA:-/kaggle/tmp/openmeet-data}"
OUT="${OUT:-/kaggle/working/results}"
D="$OPENMEET_DATA"
WORKERS="${WORKERS:-4}"
mkdir -p "$D"/{gazebase,gazebasevr,desktopactivity,cache,tables} "$OUT"
log() { echo "=== $(date '+%H:%M:%S') $*" | tee -a "$OUT/run.log"; }

log "python packages"
pip install -q "scikit-learn>=1.5" "pandas>=2.2" "scipy>=1.13" 2>&1 | tail -1 || true
python -c "import torch, sklearn; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'sklearn', sklearn.__version__)" | tee -a "$OUT/run.log"

log "downloads"
python ../download_gazebase.py --dest "$D/gazebase" --verify | tail -2
[ -s "$D/gazebasevr/gazebasevr.zip" ] || curl -sSL -o "$D/gazebasevr/gazebasevr.zip" https://ndownloader.figshare.com/files/38844024
echo "048c04b00fd64347375cc8d37b451a22  $D/gazebasevr/gazebasevr.zip" | md5sum -c -
[ -s "$D/desktopactivity/DesktopActivity.zip" ] || curl -sSL -o "$D/desktopactivity/DesktopActivity.zip" https://raw.githubusercontent.com/EyeSyn/GazeGraph/HEAD/DesktopActivity.zip

log "30 Hz caches"
python -I -u ../build_cache.py gazebase "$D/gazebase/GazeBase_v2_0.zip" "$D/cache/gazebase.npz" --workers "$WORKERS" | tail -9
python -I -u ../build_cache.py gazebasevr "$D/gazebasevr/gazebasevr.zip" "$D/cache/gazebasevr.npz" --workers "$WORKERS" | tail -7
python -I -u ../build_cache.py desktopactivity "$D/desktopactivity/DesktopActivity.zip" "$D/cache/desktopactivity.npz" | tail -8

log "feature tables"
T() {  # one table; its summary line goes to run.log
  python -I -u make_table.py "$@" > "$OUT/table.tmp" 2>&1 || { cat "$OUT/table.tmp"; exit 1; }
  grep " windows, " "$OUT/table.tmp" | tee -a "$OUT/run.log" || true
}
T gazebase gazebase_nominal --workers "$WORKERS"
T gazebase gazebase_mixed --vary --seed 1 --workers "$WORKERS"
T gazebase gazebase_hard --noise 3.0 --drift 2.0 --workers "$WORKERS"
T gazebase gazebase_small --amplitude 0.5 --workers "$WORKERS"
T gazebase gazebase_small_hard --amplitude 0.5 --noise 3.0 --drift 2.0 --workers "$WORKERS"
T gazebase gazebase_slow --speed 0.66666667 --workers "$WORKERS"
T gazebase gazebase_clean --clean --workers "$WORKERS"
T gazebasevr gazebasevr_nominal --workers "$WORKERS"
T desktopactivity desktopactivity_raw --noise 0 --drift 0 --blinks 0 --dropout 0 --workers 1
python -I -u foldfeat.py 10 gazebase_nominal gazebase_mixed gazebase_hard gazebase_small gazebase_small_hard gazebase_slow gazebasevr_nominal desktopactivity_raw | tee -a "$OUT/run.log"

log "boosted model: person-level CV on development people"
python -I -u robust.py full "" nominal+mixed 2>&1 | grep -v Warning | tee "$OUT/boosting_cv.log"

log "network: person-level CV, then final fit (GPU)"
python -I -u cnn.py --tag cnn_gpu --seeds 0,1,2 --epochs 12 --threads 4 --final --extra gazebase_slow,gazebasevr_nominal,desktopactivity_raw 2>&1 | grep -v Warning | tee "$OUT/network_cv.log"

log "choose on development, score the locked test people"
python -I -u final.py --cnn-tag cnn_gpu --out "$OUT/reading_detector_results.json" 2>&1 | grep -v Warning | tee "$OUT/final.log"

cp "$D"/tables/*.json "$D"/tables/model_cnn_gpu_seed*.pt "$OUT"/ 2>/dev/null || true
log "done"
