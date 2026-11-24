# Reproducing the paper

| item of the paper | script | output |
|---|---|---|
| checkpoints | `pipelines/train_seeds.py --config configs/{tcn,spectranet}.yaml --seed {42,123,7}` | `models/{tcn,spectranet}/seed*/checkpoint_{best,swa}.pt` |
| ensemble caches | `pipelines/build_ensemble.py`, `pipelines/extract_members.py`, `pipelines/build_combo_cache.py` | `cache/{tcn6,spectranet6,astranet12}/` |
| classification with metadata withheld (table), member agreement | `pipelines/metadata_masked_combo.py` | `results/astranet12/classification/metadata_masked.{json,md}` |
| reference without metadata dropout | the same with `configs/{tcn,spectranet}_reference.yaml` and the specs `tcn_reference6`, `spectranet_reference6` | `results/*/classification/` |
| confusion matrix (figure) | `study/novelty/paper_style_figs.py` | `study/novelty/out/confusion_known_classes_light.pdf` |
| per-spectrum LUNA scores, object rows | `study/prepare.py`, `study/objects/objects.py` | `study/features.npz`, `study/objects/object_features.npz` |
| novelty without anomaly exposure, class withheld, tier withheld, operating points at 1, 5, 10, 15 % (tables) | `study/novelty/build_nova_variants.py`, `study/heads/experiment.py --mode lco`, `--mode tier` in `study_nvA`, `study_nvB`, then `study/novelty/paper_numbers.py` | `study/novelty/out/paper_numbers.json` |
| alarm fraction per class (table) | `study/novelty/novelty_by_class.py` | `study/novelty/out/novelty_by_class.json` |
| LUNA ablation (figure) | `study/novelty/ablation_fig.py` | `study/novelty/out/ad_luna_ablation.pdf` |
| SkyPortal service condition | `pipelines/extract_service.py`, `tools/build_novelty.py --dump-features study/service/svc_object_features.npz`, `study/novelty/build_svc_variants.py`, `study/novelty/service_by_class.py` | `models/novelty.joblib`, `study/novelty/out/service_by_class.json` |
| abstention | `study/abstention.py . astranet12` | `study/abstention.log` |
| temperature scaling (table) | `study/uq_conformal/extract_masked.py` (GPU), `study/uq_conformal/run.py` | `study/uq_conformal/metrics.csv` |
| object-level conformal sets, symmetric control (table, figure) | `study/conformal_object/run.py`, `confidence_control.py`, `symmetric/run_symmetric.py`, `symmetric/make_figure.py` | `study/conformal_object/paired.json`, `symmetric/summary_symmetric.csv`, `conformal_symmetric.pdf` |
| rare objects flagged by LUNA and by confidence (table) | `study/conformal_object/examples/find_examples.py`, `study/novelty/examples.py` | `study/novelty/out/examples.{json,tex}` |
| case studies (figure) | `pipelines/case_study_figures.py --spec astranet12 --config configs/spectranet.yaml --objects ...` | `results/astranet12/figures/case_studies_light.pdf` |

`study/run_study.sh` runs the study rows in order. The JSON files in `study/novelty/out/` are the
values quoted in the paper. The spectra are described in `data/README.md`.
