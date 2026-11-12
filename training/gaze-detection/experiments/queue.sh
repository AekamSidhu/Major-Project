#!/usr/bin/env bash
# Unattended compute queue. One heavy job at a time: the laptop has 7.7 GB of RAM
# and two jobs at once make it swap. Usage: bash queue.sh <pid of the running cnn job>
cd "$(dirname "$0")"
PY="../.venv/Scripts/python.exe"
LOGS="C:/Users/karaa/openmeet-data/logs"
step() { echo "=== $(date '+%d %H:%M:%S') $1" >> "$LOGS/queue.log"; }
step "waiting for the first network run (pid $1)"
while tasklist //FI "PID eq $1" 2>/dev/null | grep -q python; do sleep 30; done
step "boosted model with folding features, across conditions"
"$PY" -I -u robust.py full "" mixed,nominal+mixed > "$LOGS/robust_full.log" 2>&1
step "leave one behaviour out"
"$PY" -I -u loto.py gazebase_nominal,gazebase_mixed gazebase_nominal > "$LOGS/loto.log" 2>&1
step "tables: slow reader, window lengths, frame rates, second mixed draw"
T="--vary --dev-only --workers 3"
"$PY" -I -u make_table.py gazebase gazebase_slow_mixed  $T --seed 1 --speed 0.66666667   >> "$LOGS/tables2.log" 2>&1
"$PY" -I -u make_table.py gazebase gazebase_len5_mixed  $T --seed 1 --length 5 --hop 2.5  >> "$LOGS/tables2.log" 2>&1
"$PY" -I -u make_table.py gazebase gazebase_len20_mixed $T --seed 1 --length 20 --hop 10  >> "$LOGS/tables2.log" 2>&1
"$PY" -I -u make_table.py gazebase gazebase_len30_mixed $T --seed 1 --length 30 --hop 15  >> "$LOGS/tables2.log" 2>&1
"$PY" -I -u make_table.py gazebase gazebase_fps5_mixed  $T --seed 1 --fps 5               >> "$LOGS/tables2.log" 2>&1
"$PY" -I -u make_table.py gazebase gazebase_fps15_mixed $T --seed 1 --fps 15              >> "$LOGS/tables2.log" 2>&1
"$PY" -I -u make_table.py gazebase gazebase_mixed2      $T --seed 2                       >> "$LOGS/tables2.log" 2>&1
step "folding features for the new tables"
"$PY" -I -u foldfeat.py 10 gazebase_slow_mixed gazebase_len5_mixed gazebase_len20_mixed gazebase_len30_mixed gazebase_mixed2 >> "$LOGS/tables2.log" 2>&1
"$PY" -I -u foldfeat.py 5 gazebase_fps5_mixed   >> "$LOGS/tables2.log" 2>&1
"$PY" -I -u foldfeat.py 15 gazebase_fps15_mixed >> "$LOGS/tables2.log" 2>&1
step "second network run (seed 1)"
"$PY" -I -u cnn.py --tag cnn_v1_s1 --seed 1 --epochs 10 --threads 4 --final --extra gazebasevr_nominal,desktopactivity_raw > "$LOGS/cnn_v1_s1.log" 2>&1
step "queue finished"
