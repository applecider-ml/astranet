#!/bin/bash
# Novelty protocol of the paper: novae ordinary, 60 rare objects in nine classes, one threshold
# per redshift stratum, alarm budgets 1, 5, 10 and 15 %. Run after study/run_study.sh (needs
# study/features.npz, study/objects/object_features.npz, cache/astranet12). CPU only.
#   PY=python ./study/novelty/run_novelty.sh
set -eu
PY=${PY:-python}; S=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd); R=$(dirname "$(dirname "$S")"); export ASTRANET_REPO=$R
mkdir -p "$S/out"
# 1. study copies with the novae moved to the ordinary set: study_nvA / study_nvB (the two nova halves swapped)
"$PY" -u "$S/build_nova_variants.py"
for v in nvA nvB; do for m in lco tier; do
  (cd "$R/study_$v/heads" && ASTRANET_STUDY="$R/study_$v" "$PY" -u experiment.py --mode $m > $m.log 2>&1)
done; done
# 2. every novelty number of Sections 4.2 to 4.4 and the per-class table
"$PY" -u "$S/paper_numbers.py"
"$PY" -u "$S/novelty_by_class.py"
"$PY" -u "$S/examples.py"
# 3. figures (need luna.style)
"$PY" -u "$S/ablation_fig.py"
"$PY" -u "$S/paper_style_figs.py"
# 4. service condition (phase withheld); needs study/service/svc_object_features.npz, written by
#    tools/build_novelty.py --dump-features study/service/svc_object_features.npz
if [ -f "$R/study/service/svc_object_features.npz" ]; then
  "$PY" -u "$S/build_svc_variants.py"
  for v in svA svB; do (cd "$R/study_$v/heads" && ASTRANET_STUDY="$R/study_$v" "$PY" -u experiment.py --mode lco > lco.log 2>&1); done
  "$PY" -u "$S/service_by_class.py"
fi
