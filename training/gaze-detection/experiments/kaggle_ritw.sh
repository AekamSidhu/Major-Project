#!/usr/bin/env bash
# Reading in the Wild on a Kaggle notebook (GPU T4, internet on). About 1-1.5 hours.
#
#   !git clone -q -b feature/gaze-detection https://github.com/AekamSidhu/Major-Project.git /kaggle/temp/Major-Project
#   !RITW_LINKS='<download-links .json URL from the dataset page>' bash /kaggle/temp/Major-Project/training/gaze-detection/experiments/kaggle_ritw.sh
#
# RITW_LINKS is personal and expires after about two weeks: keep it in the
# (private) notebook, never in the repository. If GAZEBASE_WEIGHTS points to
# the network weights from kaggle_run.sh, they are also tested zero-shot.
set -euo pipefail
cd "$(dirname "$0")"
: "${RITW_LINKS:?set RITW_LINKS to the download-links file (URL or path)}"
export OPENMEET_DATA="${OPENMEET_DATA:-/kaggle/temp/openmeet-data}"
OUT="${OUT:-/kaggle/working/results}"
D="$OPENMEET_DATA"
mkdir -p "$D"/{cache,tables} "$OUT"
log() { echo "=== $(date '+%H:%M:%S') $*" | tee -a "$OUT/ritw_run.log"; }

log "python packages"
pip install -q "scikit-learn>=1.5" "pandas>=2.2" "scipy>=1.13" 2>&1 | tail -1 || true

log "gaze of every labelled Seattle sequence (range reads, about 2 GB)"
python -I -u ../build_cache.py ritw "$RITW_LINKS" "$D/cache/ritw.npz" --workers 4 2>&1 | grep -v "^  [0-9]*/" | tee -a "$OUT/ritw_run.log"

T() {
  python -I -u make_table.py "$@" > "$OUT/table.tmp" 2>&1 || { cat "$OUT/table.tmp"; exit 1; }
  grep " windows, " "$OUT/table.tmp" | tee -a "$OUT/ritw_run.log" || true
}
QUIET="--noise 0 --drift 0 --blinks 0 --dropout 0"
for L in 10 5; do
  H=$(python -c "print($L / 2)")
  log "tables, $L s windows"
  T ritw "ritw_$L" $QUIET --length "$L" --hop "$H" --workers 4
  T ritw "ritw_${L}_webcam" --length "$L" --hop "$H" --workers 4
  T ritw "ritw_${L}_mixed" --vary --seed 1 --length "$L" --hop "$H" --workers 4
  T ritw "ritw_${L}_clean" --clean --length "$L" --hop "$H" --workers 4
  python -I -u foldfeat.py 10 "ritw_$L" "ritw_${L}_webcam" "ritw_${L}_mixed" | tee -a "$OUT/ritw_run.log"
done

for L in 10 5; do
  log "detectors, $L s windows"
  python -I -u ritw_run.py --length "$L" ${GAZEBASE_WEIGHTS:+--gazebase-weights "$GAZEBASE_WEIGHTS"} --out "$OUT/ritw_${L}s.json" 2>&1 | grep -v Warning | tee "$OUT/ritw_${L}s.log"
done
log "done"
