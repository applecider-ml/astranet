# ASTRANet

Spectroscopic classification and object-level novelty detection for transients, the research
pipeline behind *ASTRANet: Metadata-Robust Spectroscopic Classification and Object-Level Novelty
Detection* (Sasli et al., submitted to PASP). The inference-only package that SkyPortal runs is
[astranet-public](https://github.com/applecider-ml/astranet); this repository trains, selects and
evaluates the model it ships and contains the study that produces every number of the paper.

Three modules:

1. **`astranet/`**: an observer-frame spectral classifier with optional redshift and phase (FiLM
   conditioning, metadata dropout at training), seven classes and three superclasses. Two backbones
   share the same heads: the SpectraNet backbone of Xu et al. (2026) and a dilated temporal
   convolutional backbone. The paper's model is the ensemble of both, six members each
   (`v7fixcombo12`).
2. **Novelty layer**: the 16 anomaly scores of [LUNA](https://github.com/asasli/luna) on the ensemble
   embedding and logits, calibrated per object at a nominal false-alarm rate, with and without
   exposure to rare classes, and a lifetime statistic that reports the alarm per unit time.
3. **Uncertainty layer**: object-level conformal prediction sets, stratified by anomaly score and
   classifier confidence, and temperature scaling.

## Layout

```
astranet/        classifier (model, config, dataset, losses, train_loop), predict.py, novelty.py, skyportal.py, sentinel.py, uq/
pipelines/       training and feature drivers, in run_all.sh order; experimental/ holds non-adopted ablations
configs/         v7fixmd.yaml (TCN backbone), v7fixsn.yaml (SpectraNet backbone); default.yaml = v7fixmd.yaml
study/           the novelty / uncertainty study of the paper (run_study.sh, then final/run_final.sh); see REPRODUCE.md
data/            datasets (binaries git-ignored; data/README.md has the inventory and MD5s)
models/ cache/   trained checkpoints and ensemble caches (git-ignored; Zenodo bundle, see REPRODUCE.md)
results/         regenerable outputs of pipelines/; results/model_selection/README.md is the selection record
docs/            GitHub Pages site and the history of the single-backbone analysis
```

## Quick start

```bash
pip install -r requirements.txt                       # includes luna-ood >= 0.4.1
# EITHER train from the spectra (data/README.md; the ZTF spectra are not redistributable):
CONFIG=configs/v7fixmd.yaml ./run_all.sh              # TCN members
CONFIG=configs/v7fixsn.yaml ./run_all.sh              # SpectraNet members
python pipelines/build_combo_cache.py                 # cache/v7fixcombo12
# OR unpack the Zenodo bundle (checkpoints + caches), then:
PY=python ./study/run_study.sh                        # features, heads, conformal sets, about 3 CPU-hours
PY=python ./study/final/run_final.sh                  # the novelty tables and figures of the paper
```

`REPRODUCE.md` maps each table, figure and quoted number of the paper to the script and file that
produce it.

## Headline numbers (test split of 3,394 spectra / 1,666 objects; 60 rare objects in 9 classes, novae ordinary)

| quantity | value |
|---|---|
| accuracy with redshift and phase / spectrum only | 89.0 % / 86.2 % (macro recall 83.4 / 78.6 %) |
| rare objects recovered without anomaly exposure (mean of the 16 LUNA scores), alarm budget 1 / 5 / 10 / 15 % | 51.5 / 74.1 / 82.3 / 86.9 % |
| the same for the classifier confidence at 1 % | 24.8 % |
| leave-class-out, ExtraTrees + LUNA summaries, 1 / 5 / 10 / 15 % | 50.7 / 68.4 / 82.8 / 88.2 % |
| withheld Tier 1 (35 objects), ExtraTrees + LUNA summaries, exposure to Tier 2 only | 32.0 / 57.1 / 71.4 / 82.7 % |
| object-level conformal sets, 90 % target, joint LUNA x confidence strata | 88.5 to 93.3 % coverage in every examined group, mean size 1.06 |

`study/final/` holds the scripts and the JSON records of these rows.

The protocols behind these numbers are the point of the repository: every split is grouped by
object, no object is scored by a detector or combiner that saw it, thresholds are set on
in-distribution calibration objects only, and the realised false-alarm rate is measured on
independent test objects and reported next to every recall.

## SkyPortal service

`astranet.skyportal.analyze` is what the SkyPortal analysis service runs for one source: every
spectrum of the source goes through the twelve members, the class probabilities are averaged, and
the novelty stage (`astranet/novelty.py`) scores the object.

```bash
pip install ".[novelty]"
python tools/fetch_weights.py --url <checkpoints> --novelty-url <novelty_v1.joblib>
python -m astranet.skyportal spectrum.dat --redshift 0.05
```

The novelty stage is an ExtraTrees head on the object vector (mean embedding, 23 spectral
descriptors, mean and maximum of the 16 LUNA scores). Its ordinary set is the seven classes and
novae, which therefore do not count as novel; its anomalies are the nine other rare classes. The
score is compared with ordinary calibration objects of the same redshift stratum (redshift known
or unknown), which gives a p-value and alarms at 1, 5, 10 and 15 %. The service has no
spectroscopic phase, so the bundle is fitted and calibrated with the phase withheld
(`pipelines/extract_service.py`, `tools/build_novelty.py`).

| target alarm rate | 1 % | 5 % | 10 % | 15 % |
|---|---|---|---|---|
| rare objects recovered, class withheld from the head (60 objects, 9 classes) | 52 % | 70 % | 84 % | 89 % |
| held-out novae that alarm (47 objects) | 3 % | 9 % | 15 % | 21 % |
| realised alarm rate of the shipped bundle, 1,666 test objects | 1.3 % | 4.8 % | 10.0 % | 14.6 % |
| the same, redshift known / unknown (1,602 / 64 objects) | 1.4 / 0.0 % | 4.7 / 6.3 % | 10.0 / 9.4 % | 14.5 / 17.2 % |

The first two rows come from the leave-class-out protocol of the study in the service condition (ten
splits, two swaps of the nova halves); the last two from the shipped bundle on test objects that
enter neither its fit nor its calibration. With a threshold shared by both redshift strata most
ordinary objects without a redshift alarm at the 10 % and 15 % rates, which is why the thresholds are
set per stratum.

## Earlier analysis

The single-backbone (TCN, `v7fixmd6`) analysis that preceded the paper, including the
outlier-exposure Sentinel and the model-selection record, is described in
`docs/README_single_backbone_history.md` and regenerated by `run_all.sh`; the paper's Table of
backbones compares it with the SpectraNet members and the combined ensemble.

## Citation

Sasli, A., et al. 2026, ASTRANet: Metadata-Robust Spectroscopic Classification and Object-Level
Novelty Detection, submitted to PASP. The spectroscopic backbone is SpectraNet (Xu et al. 2026,
PASP 138, 094504); the anomaly scores are LUNA (github.com/asasli/luna).

## License

MIT, see `LICENSE`.
