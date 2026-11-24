# ASTRAnet

Hierarchical transient-spectra classifier (**ASTRAnet**) with a novelty layer (**ASTRANet-Sentinel**)
and a conformal uncertainty layer (**ASTRANet-CP**), packaged as a reproducible pipeline. The
deployable, inference-only package that SkyPortal runs lives in
[astranet-public](https://github.com/applecider-ml/astranet); this repository is the research
pipeline that trains, selects and evaluates the model it ships.

Three coupled modules:

1. **`astranet/`** — observer-frame spectral classifier (multi-scale TCN, attention pooling, FiLM
   conditioning on redshift/phase), 7 fine classes / 3 superclasses, deep ensemble over 3 seeds.
   Trained with **metadata dropout**, so it operates with or without redshift and phase.
2. **ASTRANet-Sentinel** (`astranet/sentinel.py`) — novelty detection on the classifier's
   embedding space, reported in three tiers that differ in *what the detector was allowed to see*
   (below). The 16 embedding-space scores come from the standalone
   [LUNA](https://github.com/asasli/luna) package.
3. **ASTRANet-CP** (`astranet/uq/`) — conformal uncertainty: APS prediction sets, Mondrian and
   AD-stratified calibration.

Every number in this README is produced by `run_all.sh` from the committed configuration; the
evaluation protocols are designed so that **no object is ever scored by a detector that saw it**,
**no model or detector choice touches the test split**, and every claim says what the method
had seen.

## Layout

```
astranet/        classifier (model, config, dataset, losses, train_loop) · sentinel.py · uq/
pipelines/       drivers, in run_all.sh order; experimental/ holds non-adopted ablations
configs/         default.yaml = v7fixmd.yaml (deployed); v7fix.yaml (no metadata dropout, reference)
data/            datasets (binaries git-ignored; data/README.md has the inventory + MD5s;
                 data/rare_split.json is the permanent object-level split of the rare set)
results/         regenerable; results/model_selection/README.md is committed (the selection record)
reports/         committed snapshots of the application-set analysis
docs/            GitHub Pages site;  notebooks/01_quickstart.ipynb — guided tour
models/ cache/   regenerable, git-ignored
```

## Quick start

```bash
pip install -r requirements.txt            # includes LUNA
# place the datasets under data/ (see data/README.md), then:
CONFIG=configs/v7fixmd.yaml ./run_all.sh   # everything below, in order (≈ 8 GPU-hours on 3 GPUs)
```

`run_all.sh` is config-driven and idempotent (each step skips if its outputs exist; `--force`
rebuilds one step). GPU steps are independent across seeds and can run on separate GPUs.

## The deployed classifier and how it was selected

Validation/test spectra all carry redshift and phase, but **56 % of the application-set spectra
have no redshift** (40 % no phase). A model trained only with full metadata collapses without it
(31 % accuracy without redshift). We therefore train with per-sample metadata dropout and select
the dropout configuration on a **pre-declared validation criterion** — the mean of validation
macro recall under three inference regimes (full metadata, no redshift, no metadata). The test
split is consulted once, for the selected model. Record: `results/model_selection/README.md`.

| ensemble | dropout z / phase | val criterion | selected |
|---|---|---|---|
| v7fix6 (no dropout, reference) | 0 / 0 | 0.540 | |
| **v7fixmd6** | 0.4 / 0.3 | **0.739** | ✓ |
| v7fixmd26 | 0.2 / 0.15 | 0.733 | |
| v7fixmd46 | 0.4 / 0.3 (robust early-stop) | 0.732 | |

Test split (3,396 spectra / 1,668 objects), the trained ensemble held fixed and metadata withheld
at inference (`pipelines/metadata_masked_eval.py`):

| inference regime | accuracy | macro recall | reference model (no dropout) |
|---|---|---|---|
| redshift + phase | 86.1 % | 81.6 % | 86.6 % / 82.8 % |
| no phase | 85.3 % | 80.5 % | 80.1 % / 65.5 % |
| no redshift | 85.7 % | 77.3 % | 31.3 % / 39.5 % |
| spectrum only | 84.3 % | 76.8 % | 53.0 % / 43.7 % |

Only SLSN depends materially on redshift (recall 0.88 → 0.56 without it). The cost of robustness is
0.5 points of accuracy with full metadata; on the hard labelled application objects (all with
redshift) it is larger (71 % → 67 % object accuracy) and is reported as such.

## ASTRANet-Sentinel: three novelty tiers

A novelty detector is only as honest as the statement of what it has seen. We evaluate every
detector at a 1 % false-alarm rate set on a held-out validation half, count detections per
**object** (a multi-epoch object is detected if any epoch is flagged), and report:

| tier | what the detector saw | protocol | pipeline |
|---|---|---|---|
| **zero-knowledge** | the training classes only | unsupervised scores, mean rank of all 16 (fixed a priori) | `discovery_eval.py` |
| **exposure-generalisation** | the training classes + *other* rare classes | leave-class-out, leave-tier-out | `discovery_eval.py`, `oe_kfold.py --split-by tier` |
| **known-rare recovery** | the training classes + the rare classes (other objects) | object-grouped 5-fold with outlier-exposure fine-tuning | `oe_kfold.py --split-by object` |

The rare set is 289 spectra / 107 objects in 11 classes, split into Tier 1 (genuinely novel
phenomena: FBOT, ILRT, luminous red nova, Ca-rich, afterglow, long GRB; 35 objects) and Tier 2
(rare members of trained families: novae ⊂ CV, QSO/blazar/BL Lac ⊂ AGN; 72 objects).

### Zero-knowledge (unsupervised)

Objects detected at 1 % FAR with no anomaly of any kind used anywhere
(`results/v7fixmd6/anomaly/discovery_unsupervised_v7fixmd6.md`):

| score (16 LUNA scores on the ensemble embedding) | objects | Tier 1 | Tier 2 |
|---|---|---|---|
| mean rank of all 16 (the deployed zero-knowledge score; chosen a priori) | 37.4 % | 54.3 % | 29.2 % |
| best single score in hindsight (typicality) | 41.1 % | 62.9 % | 30.6 % |
| density trio (typicality, kNN, class-Mahalanobis), a priori | 38.3 % | 62.9 % | 26.4 % |

**A second, classifier-free view (spectrum-level FAR only)** (`physics_view.py`): 23 spectral descriptors
(continuum colours and slope, residual line power at five velocity scales, emission/absorption
balance, pseudo-equivalent widths around strong lines) scored with the same LUNA scorers flag only
11 % of rare objects alone, but an almost disjoint set (Jaccard 0.07 with the embedding view). The
union of the two views at a combined 1 % FAR (10 validation splits) raises zero-knowledge recovery
from 37.3 ± 1.5 % to 39.6 ± 2.1 % of objects and Tier 1 from 55 % to 62 % (better in 9/10 splits,
never worse) **at a 1 % per-spectrum FAR**. Under the stricter 1 % per-object FAR of the independent
study in `~/work/Argyro/luna_ad_study/` (threshold on the maximum score over an object's epochs, FAR measured on
independent ID test objects) this union does NOT help (22–27 % vs 33 % for the LUNA mean rank alone): the gain
above is an artefact of mixing object-level detection with a spectrum-level FAR. See that study for
object-level numbers.

Three attempts to raise this ceiling without using real anomalies were run and are reported as
negative or marginal (`virtual_outliers.py`, `oe_synthetic.py`, `ssl_contrastive.py`):

| attempt | best unsupervised result | verdict |
|---|---|---|
| synthetic outliers in embedding space (class-Gaussian tails, between-class interpolation, radial shells) + any head | ≤ 21 % of objects | synthetic points fall far from the in-distribution manifold; real anomalies sit near it |
| outlier exposure of the encoder on synthetic *spectral* anomalies (featureless continua, class mix-ups, flipped / shifted / line-boosted spectra), 3 single-seed encoders | typicality 21.5 ± 4.0 % → 32.1 ± 3.9 % of objects (Tier 1 37 → 51 %); mean rank of 16 unchanged; heads trained on the synthetic anomalies ≤ 22 % | helps one density score on a single encoder, stays below the ensemble embedding (37–41 %); not adopted |
| class-conditional (per predicted class) 1 %-FAR thresholds for the unsupervised score; ensemble disagreement (mutual information) added to the combination | per-class thresholds: density trio 39.4 → 45.1 % of objects, but the realised held-out FAR rises to 2.2 %; a global threshold re-tuned to the same FAR gives 46.8 % (10 random splits). MI alone 20.6 %, and it lowers every combination it joins | no gain at matched false-alarm rate; the apparent improvement is a higher operating point |
| self-supervised contrastive encoder (NT-Xent, 60 epochs, 37,876 spectra incl. the 18,430 unlabelled ones; 0 rare-set overlap) scored with LUNA's label-free scorers | kNN 13.1 % of objects (Tier 1 17 %); with labels 13.1 % | the pretext task is too easy (loss 2.6 → 0.04) and the flags track the instrument (GMOS 27–40 % vs SEDM 2–5 % flagged); the class-discriminative embedding is the better density space |

### Exposure-generalisation (the discovery claim)

Detectors trained on validation (in-distribution) versus rare spectra of *other* classes, tested on
the class they never saw (`discovery_leave_class_out_v7fixmd6.md`). Same encoder (no outlier
exposure), same folds, same threshold rule for every detector:

| detector | objects | Tier 1 | Tier 2 |
|---|---|---|---|
| embeddings + LightGBM | **57.0 %** | **74.3 %** | 48.6 % |
| embeddings + 16 scores + LightGBM | 53.3 % | 65.7 % | 47.2 % |
| embeddings + MLP | 48.6 % | 60.0 % | 43.1 % |
| embeddings + logistic | 45.8 % | 62.9 % | 37.5 % |
| 16 scores + LightGBM (the original Sentinel combiner) | 42.1 % | 57.1 % | 34.7 % |
| 16 scores + logistic | 33.6 % | 57.1 % | 22.2 % |

Exposure to *some* anomalies helps find *unseen* ones (37 % → 57 %), and a head on the raw
embeddings generalises better than the 16-score combiner. Reducing the embedding to 16 summary
scores discards information — the control the reviewer asked for, and the reason the deployed
exposure tier is an embedding head.

**Leave-tier-out at the encoder level** (`oe_kfold.py --split-by tier`, 3 seeds, mean ± sd of
per-object detection): outlier-exposure fine-tuning on Tier 2 only, then test on Tier 1, and the
reverse. The detector never sees any class of the tested tier.

| detector | Tier 1 found, base encoder | Tier 1 found, after OE on Tier 2 | Tier 2 found, base | Tier 2 found, after OE on Tier 1 |
|---|---|---|---|---|
| embeddings + MLP | **64 ± 7 %** | 53 ± 4 % | 25 ± 2 % | 27 ± 4 % |
| embeddings + 16 scores + LightGBM | 62 ± 7 % | 52 ± 1 % | 29 ± 9 % | 28 ± 4 % |
| embeddings + LightGBM | 49 ± 6 % | 56 ± 8 % | **32 ± 5 %** | 31 ± 5 % |
| 16 scores + LightGBM | 50 ± 1 % | 53 ± 1 % | 19 ± 2 % | 27 ± 4 % |
| 16 scores + logistic | 29 ± 8 % | 56 ± 5 % | 19 ± 2 % | 29 ± 3 % |

Fine-tuning the encoder on one tier does **not** improve discovery of the other (the best
base-encoder heads lose ~10 points; the 16-score combiners gain a few). Outlier exposure
specialises the representation to what it was shown: it is a recovery mechanism, not a novelty
mechanism, which is how the next tier reports it. Tier 2 (rare members of trained families) is
hard for every detector by construction.

### Do the score signatures form regimes?

`regime_analysis.py` rank-normalises the 16 scores of every rare spectrum against the validation
split and asks whether the resulting signatures cluster. They do not, beyond "strongly vs weakly
anomalous": k-means prefers k = 2 (silhouette 0.47; k = 5 gives 0.35), the clusters carry almost
no information about class (adjusted mutual information ≤ 0.13) or tier (≤ 0.03), and the per-class
family signatures are nearly uniform across the four families (all families fire together; see
`results/v7fixmd6/anomaly/regimes/`). The five "failure modes" of the paper are therefore an
interpretive reading of the case studies, not statistically distinct modes of the detector, and are
presented as such.

### Known-rare recovery (outlier exposure)

Object-grouped 5-fold protocol (`StratifiedGroupKFold` on object id, asserted leak-free): for each
fold the encoder is OE fine-tuned on the other four folds' objects, the 16-score LightGBM
combiner is trained on half the validation pool + the exposure folds, the threshold is set at 1 %
FAR on the other validation half, and detections are counted on the held-out fold only.

| regime (3 seeds) | spectra | objects | Tier 1 | Tier 2 |
|---|---|---|---|---|
| anchor (mean score, mean threshold) | 66.4 % | **71.0 %** | 77.1 % | 68.1 % |
| per seed (mean ± sd) | 55.5 ± 3.8 % | 64.2 ± 0.9 % | | |
| seed union (~3 % FAR) | 69.6 % | 73.8 % | 77.1 % | 72.2 % |
| baseline, no OE (per seed) | | 27–40 % | | |

The earlier per-spectrum folds reported 82.4 % of spectra; the drop to 68 % is the object leakage
that per-spectrum folds allow (epochs of one object on both sides of a fold). The object number
barely moves (75.7 % → 71.0 %). An independent repetition of the per-fold fine-tuning gave 72.0 % of objects.

Every detector head under the same object-grouped folds (3-seed anchor, per object):

| detector | base encoder | after outlier exposure | Tier 1 (OE) |
|---|---|---|---|
| 16 scores + LightGBM (deployed Sentinel) | 24.3 % | **71.0 %** | 77.1 % |
| embeddings + 16 scores + LightGBM | 52.3 % | 72.0 % | 77.1 % |
| embeddings + LightGBM | 52.3 % | 69.2 % | 74.3 % |
| embeddings + logistic | 43.9 % | 67.3 % | 68.6 % |
| embeddings + MLP | 52.3 % | 62.6 % | 65.7 % |
| 16 scores + logistic | 21.5 % | 62.6 % | 68.6 % |

With outlier exposure the 16-score combiner and the embedding heads converge (67–72 %); the
combiner benefits most from OE and is what the package ships for this tier.

Deployable detectors (`oe_final.py`): OE fine-tuning on the permanent 87-object exposure part of
the rare set (`data/rare_split.json`), evaluated on the 20 held-out objects never used anywhere:
**80 % of objects** (16/20), 71 % of spectra, test-set false-alarm rate 0.8–0.9 %.

Combiner-level leave-one-class-out on the deployed features
(`experimental/oe_lightgbm.py`): 70 % of spectra overall; Ca-rich 10 %, novae 57 %, ILRT 58 %,
FBOT 71 %, blazar 82 %, QSO 93 %, afterglow 94 %, LRN 100 %.

## ASTRANet-CP

APS prediction sets on the test split: coverage 0.950 at α = 0.05 (set size 1.49 in-distribution,
2.11 on the rare set). AD-stratified Mondrian calibration gives per-quintile coverage
0.950–0.956 (standard deviation across folds 0.009–0.029, binomial s.e. 0.0085 at ≈ 665 spectra
per quintile), where marginal Mondrian drops to 0.864 in the highest-anomaly-score quintile.
Coverage is always evaluated on the fold that was not used for calibration
(`astranet/uq/_scripts/ad_stratified_mondrian.py`).

## Key design points

- **One consistent ensemble everywhere.** The deployed configuration is the within-seed 6-model
  ensemble (best + SWA × 3 seeds); every split is extracted with the same model set (asserted).
- **Say what the detector saw.** Novelty numbers are reported per tier; the discovery claim rests
  on leave-class-out / leave-tier-out, never on folds that contain the tested class.
- **Object-level everything.** Splits, folds and detection counts are per object; a rare object's
  epochs are never on both sides of any split.
- **Selection on validation only.** Dropout rate, early stopping and the unsupervised score
  combination are fixed before the test split or the application set is consulted.
- **`SN Ia` includes the peculiar-thermonuclear family** (91T, 91bg, 02cx/Iax, 03fg, CSM, pec):
  the classifier learns the intrinsic diversity of the class; the novelty layer is for genuine
  out-of-taxonomy objects.

## Application set

18,430 spectra / 9,918 objects of hard, ambiguous or unclassified survey data
(`categorize_challenging.py`, snapshots in `reports/challenging/`). With the deployed model and
the known-rare Sentinel: 82 robust anomalies among 1,441 labelled transients, 2 among 61 variable
stars, 914 among 8,400 unlabelled objects (robust = ≥ 2 epochs, ≥ 50 % flagged). The strictest
tier (every epoch flagged at 0.1 % FAR) and its catalogue cross-match are in
`reports/challenging/strict/`.

## Figures

All figures follow one standard (`luna.style` / `astranet/plotstyle.py`): CVD-validated per-class
colours, authored at final physical size, emitted as `*_light.pdf/png` (papers) and `*_dark.png`
(presentations). Restyle without re-scoring: `python pipelines/make_figures.py --spec v7fixmd6`.

## GitHub Pages

`docs/` is a self-contained static site (`build_site.py`). Publish with
**Settings → Pages → Source: `main` / `docs/`**.

## Data

See **[data/README.md](data/README.md)** for the inventory and MD5 checksums. The `.pt` binaries
are git-ignored and distributed with the paper's data release; every config resolves its paths
inside `data/`.

| dataset | role |
|---|---|
| `data/object-split-v7fix/` | train / val / test (object-level 70/15/15) |
| `data/rare_unknown.pt` + `data/rare_split.json` | 289-spectrum / 107-object rare set and its permanent object split |
| `data/challenging.pt` | 18,430-spectrum application set |

The trained-on class/subtype inventory (what counts as an anomaly by construction) is in
`results/v7fix_training_inventory.md`.
