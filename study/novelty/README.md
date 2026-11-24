# Novelty protocol

The scripts of this directory produce the novelty numbers, tables and figures of the paper for the
twelve-member model (`astranet12`). The protocol: novae belong to the ordinary set, the rare set
is 60 objects in nine classes (Tier 1: 35 objects; Tier 2, QSO, blazar and BL Lac: 25 objects), the
threshold is set per redshift stratum on ordinary calibration objects, and the alarm budgets are
1, 5, 10 and 15 %. Every number is the mean over ten calibration splits and the two swaps of the
nova halves.

| script | output in `out/` | paper item |
|---|---|---|
| `build_nova_variants.py` | `study_nvA/`, `study_nvB/` at the repository root | study copies with the novae as ordinary objects |
| `paper_numbers.py` | `paper_numbers.json` | no-exposure table, leave-class-out and whole-tier tables, operating points, paired bootstrap gains |
| `novelty_by_class.py` | `novelty_by_class.{json,md,png}` | per-class alarm fractions at each budget |
| `examples.py` | `examples.{json,tex}` | rare objects flagged by LUNA and by confidence, with their conformal sets |
| `ablation_fig.py` | `ad_luna_ablation.pdf` | LUNA ablation figure |
| `paper_style_figs.py` | `confusion_known_classes_light.pdf`, `novelty_table_light.pdf` | confusion matrix of the seven classes |
| `build_svc_variants.py`, `service_by_class.py` | `service_by_class.{json,md,png}` | SkyPortal service condition (phase withheld) |

`run_novelty.sh` runs them in order after `study/run_study.sh`. The JSON files shipped in `out/` are
the values quoted in the paper.
