#!/bin/bash
# Pack the trained checkpoints and the ensemble caches of the paper for Zenodo, with an MD5 manifest.
# The caches include study/features.npz and study/round2/object_features.npz (per-spectrum LUNA scores and the
# 23 spectral descriptors), so that the study runs without the raw spectra (prepare.py is then skipped).
#   ./tools/make_zenodo_bundle.sh [out_dir]      -> out_dir/astranet_v1_{checkpoints,caches}.tar.gz + MANIFEST.md5 + zenodo.json
# Unpack at the repository root: tar -xzf astranet_v1_checkpoints.tar.gz; tar -xzf astranet_v1_caches.tar.gz
set -euo pipefail
R=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd); OUT=${1:-$R/../zenodo_astranet}; mkdir -p "$OUT"; cd "$R"
CK="models/trained_v7fixmd/seed42 models/trained_v7fixmd/seed123 models/trained_v7fixmd/seed7 models/trained_v7fixsn/seed42 models/trained_v7fixsn/seed123 models/trained_v7fixsn/seed7"
for d in $CK; do for t in best swa; do [ -f "$d/checkpoint_$t.pt" ] || { echo "missing $d/checkpoint_$t.pt"; exit 1; }; done; done
CA="cache/v7fixmd6 cache/v7fixsn6 cache/v7fixcombo12 cache/challenging/v7fixcombo12.npz cache/challenging/v7fixcombo12_per_model_softmax.npz cache/challenging/v7fixmd6_recheck.npz cache/challenging/v7fixmd6_recheck_per_model_softmax.npz cache/challenging/v7fixsn6.npz cache/challenging/v7fixsn6_per_model_softmax.npz data/rare_split.json study/features.npz study/round2/object_features.npz"
for p in $CA; do [ -e "$p" ] || { echo "missing $p"; exit 1; }; done
tar -czf "$OUT/astranet_v1_checkpoints.tar.gz" $(for d in $CK; do echo $d/checkpoint_best.pt $d/checkpoint_swa.pt; done)
tar -czf "$OUT/astranet_v1_caches.tar.gz" --exclude='*.log' $CA
( cd "$OUT" && md5sum astranet_v1_checkpoints.tar.gz astranet_v1_caches.tar.gz > MANIFEST.md5 && du -sh astranet_v1_*.tar.gz )
cat > "$OUT/zenodo.json" <<EOF
{
  "title": "ASTRANet: trained checkpoints and ensemble caches (paper release)",
  "upload_type": "dataset",
  "description": "Twelve ASTRANet checkpoints (six with the temporal convolutional backbone, six with the SpectraNet backbone; three seeds, best and SWA each) and the per-split ensemble caches (embeddings, logits, per-member outputs) for the training, validation, test, rare and challenging splits, with the composite cache v7fixcombo12 used in the paper. With these files the novelty and uncertainty study of the paper (study/run_study.sh in the code repository) runs without the ZTF spectra, which are not redistributable. Code: the ASTRANet repository at the commit named in this record.",
  "creators": [{"name": "Sasli, Argyro", "affiliation": "University of Minnesota", "orcid": "0000-0001-7357-0889"}],
  "keywords": ["transients", "spectroscopy", "anomaly detection", "conformal prediction", "ZTF"],
  "license": "CC-BY-4.0",
  "related_identifiers": [{"identifier": "https://github.com/applecider-ml/astranet", "relation": "isSupplementTo", "resource_type": "software"}],
  "notes": "git commit: $(git -C "$R" rev-parse HEAD 2>/dev/null || echo unknown)"
}
EOF
echo "bundle in $OUT"
