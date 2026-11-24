# ASTRANet

Spectroscopic classification and object-level novelty detection for transients: the code of
*ASTRANet: Metadata-Robust Spectroscopic Classification and Object-Level Novelty Detection*
(Sasli et al., submitted to PASP). The repository trains the classifier, runs the novelty and
uncertainty study of the paper, and contains the function that the SkyPortal analysis service calls.

Three parts:

1. **Classifier** (`astranet/`): an observed-frame spectral classifier with optional redshift and
   phase (FiLM conditioning, metadata dropout at training), seven classes (Ia, Ibc, SLSN, II, AGN,
   CV, TDE) and three superclasses. Two backbones share the same heads: the SpectraNet backbone of
   Xu et al. (2026) and a dilated temporal convolutional backbone. The model of the paper is the
   ensemble of both, six members each (`astranet12`).
2. **Novelty** (`astranet/novelty.py`, `study/novelty/`): the 16 anomaly scores of
   [LUNA](https://github.com/asasli/luna) on the ensemble embedding and logits, one score per
   object, compared with ordinary objects of the same redshift stratum at alarm rates of 1, 5, 10
   and 15 %.
3. **Uncertainty** (`study/conformal_object/`, `study/uq_conformal/`): object-level conformal
   prediction sets stratified by anomaly score and classifier confidence, and temperature scaling.

## Layout

```
astranet/        model, config, dataset, losses, train_loop, predict.py, novelty.py, skyportal.py, uq/
configs/         tcn.yaml, spectranet.yaml (the two backbones), *_reference.yaml (without metadata dropout)
pipelines/       training, ensemble caches, classification with metadata withheld, service features
study/           novelty and uncertainty study (run_study.sh); study/novelty/ holds the tables and figures
tools/           fetch_weights.py, build_novelty.py
data/            data/README.md describes the spectra (not part of the repository)
```

## Install

```bash
pip install ".[novelty]"          # classifier + novelty stage (scikit-learn, joblib, luna-ood)
python tools/fetch_weights.py --url <astranet_checkpoints.tar.gz> --novelty-url <novelty.joblib>
```

## Classify a spectrum

```bash
python -m astranet.skyportal spectrum.dat --redshift 0.05
```

```python
from astranet.predict import Ensemble
ens = Ensemble.from_dir("models")
out = ens.predict(wavelength, flux, redshift=0.05)      # redshift optional
```

`astranet.skyportal.analyze` is what the SkyPortal analysis service runs for one source: every
spectrum of the source goes through the twelve members, the class probabilities are averaged, and
the novelty stage scores the object. The novelty stage is an ExtraTrees head on the object vector
(mean embedding, 23 spectral descriptors, mean and maximum of the 16 LUNA scores). Its ordinary
set is the seven classes and novae; its anomalies are nine rare classes. The score is compared
with ordinary calibration objects of the same redshift stratum (redshift known or unknown), which
gives a p-value and alarms at 1, 5, 10 and 15 %. The service has no spectroscopic phase, so the
bundle is fitted and calibrated with the phase withheld.

## Results (test split of 3,394 spectra / 1,666 objects; 60 rare objects in nine classes)

| quantity | value |
|---|---|
| accuracy with redshift and phase / spectrum only | 89.0 % / 86.2 % (macro recall 83.4 / 78.6 %) |
| rare objects recovered without anomaly exposure (mean of the 16 LUNA scores), alarm rate 1 / 5 / 10 / 15 % | 51.5 / 74.1 / 82.3 / 86.9 % |
| the same for the classifier confidence at 1 % | 24.8 % |
| class withheld from the head, ExtraTrees + LUNA summaries, 1 / 5 / 10 / 15 % | 50.7 / 68.4 / 82.8 / 88.2 % |
| Tier 1 withheld (35 objects), exposure to Tier 2 only | 32.0 / 57.1 / 71.4 / 82.7 % |
| object-level conformal sets, 90 % target, joint LUNA x confidence strata | 88.5 to 93.3 % coverage in every examined group, mean size 1.06 |

Service condition (phase withheld):

| target alarm rate | 1 % | 5 % | 10 % | 15 % |
|---|---|---|---|---|
| rare objects recovered, class withheld from the head | 52 % | 70 % | 84 % | 89 % |
| novae that alarm (47 objects) | 3 % | 9 % | 15 % | 21 % |
| realised alarm rate of the shipped bundle, 1,666 test objects | 1.3 % | 4.8 % | 10.0 % | 14.6 % |

Every split is by object, thresholds are set on ordinary calibration objects only, and the realised
alarm rate is measured on test objects that enter neither the fit nor the calibration.

## Reproduce

```bash
pip install -r requirements.txt
for s in 42 123 7; do python pipelines/train_seeds.py --config configs/tcn.yaml --seed $s; python pipelines/train_seeds.py --config configs/spectranet.yaml --seed $s; done
python pipelines/build_ensemble.py --spec tcn6 --config configs/tcn.yaml
python pipelines/build_ensemble.py --spec spectranet6 --config configs/spectranet.yaml
python pipelines/extract_members.py --spec tcn6 --config configs/tcn.yaml
python pipelines/extract_members.py --spec spectranet6 --config configs/spectranet.yaml
python pipelines/build_combo_cache.py
PY=python ./study/run_study.sh
```

`REPRODUCE.md` maps each table and figure of the paper to its script and output.

## Citation

Sasli, A., et al. 2026, ASTRANet: Metadata-Robust Spectroscopic Classification and Object-Level
Novelty Detection, submitted to PASP. The SpectraNet backbone is from Xu et al. (2026, PASP 138,
094504); the anomaly scores are LUNA (github.com/asasli/luna).

## License

MIT, see `LICENSE`.
