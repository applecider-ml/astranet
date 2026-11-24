# Experimental / exploratory scripts

Alternative anomaly-detection approaches evaluated during development and **not**
part of the production pipeline (the 16-score LUNA + LightGBM Sentinel won):

- `cae_ad.py`, `dimmad_ad.py`, `distclassipy_ad.py` — alternative AD backbones
- `oe_lightgbm.py` — outlier-exposure variants at the combiner level
- `novae_exclude.py`, `recovery_strategies.py`, `robust_ad.py` — ablations on the
  detection criterion / class handling

Kept for reproducibility of the paper's ablation discussion. See `../nf_anomaly.py`
for the normalizing-flow ablation (also not adopted; results in `results/*/nf_anomaly/`).
