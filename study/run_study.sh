#!/bin/bash
# The novelty and uncertainty study of the paper, in order, on the combined ensemble cache
# (cache/astranet12, built by pipelines/build_combo_cache.py). CPU only, except the [gpu] step,
# which needs the trained checkpoints (models/tcn, models/spectranet).
# Every step writes next to its script; a step is skipped when its main output exists.
#
#   PY=python ./study/run_study.sh            # everything
#   PY=python ./study/run_study.sh --force    # recompute everything
#
# The study finds the repository as its parent directory; set ASTRANET_REPO / ASTRANET_STUDY to run
# it elsewhere. LUNA (pip install luna-ood) must be importable.
set -eu
set -o pipefail
PY=${PY:-python}; export OMP_NUM_THREADS=${OMP_NUM_THREADS:-3} OPENBLAS_NUM_THREADS=${OMP_NUM_THREADS:-3} MKL_NUM_THREADS=${OMP_NUM_THREADS:-3}
S=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd); FORCE=${1:-}; LOG=$S/run_study.log; : > "$LOG"
step() {  # step <dir> <output-that-marks-completion> <script> [args]
  local d=$1 out=$2; shift 2
  if [ -z "$FORCE" ] && [ -e "$S/$d/$out" ]; then echo "skip [$d] $1 ($out exists)" | tee -a "$LOG"; return; fi
  echo "=== [$d] $* $(date +%H:%M)" | tee -a "$LOG"
  local rc=0
  (cd "$S/$d" && "$PY" -u "$@" > "${1%.py}.log" 2>&1) || rc=$?
  echo "    rc=$rc" | tee -a "$LOG"
  if [ "$rc" -ne 0 ]; then return "$rc"; fi
}
# 1. per-spectrum LUNA scores and spectral descriptors, then one row per object
step . features.npz prepare.py
step objects object_features.npz objects.py
# 2. object-level conformal prediction sets, symmetric control, rare objects with their sets
step conformal_object objects.npz run.py
step conformal_object confidence_metrics.csv confidence_control.py
step conformal_object/symmetric summary_symmetric.csv run_symmetric.py
step conformal_object/symmetric conformal_symmetric.pdf make_figure.py conformal_symmetric.pdf
step conformal_object/examples rare_objects_conformal.csv find_examples.py
# 3. probability calibration ([gpu]: extract_masked needs the checkpoints) and abstention
step uq_conformal masked_probabilities.npz extract_masked.py        # [gpu]
step uq_conformal metrics.csv run.py
step . abstention.log abstention.py . astranet12
# 4. novelty tables and figures of the paper (novae ordinary, thresholds per redshift stratum)
PY="$PY" "$S/novelty/run_novelty.sh" 2>&1 | tee -a "$LOG"
echo "=== STUDY DONE $(date +%H:%M)" | tee -a "$LOG"
