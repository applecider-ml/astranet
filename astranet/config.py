# astra/config.py
# Default configuration for ASTRA.
# All hyperparameters are in one place for reproducibility.

from dataclasses import dataclass, field
from typing import List, Dict


@dataclass
class ASTRAConfig:
    # ── Paths ─────────────────────────────────────────────────
    data_dir: str = "data/object-split"   # relative or absolute
    out_dir:  str = "runs/astra"

    # ── Data loading ──────────────────────────────────────────
    batch_size:         int   = 128
    num_workers:        int   = 4
    pin_memory:         bool  = True
    augment_train:      bool  = True
    compute_derivative: bool  = True

    # ── Augmentation ──────────────────────────────────────────
    mask_prob:      float = 0.5
    mask_frac_min:  float = 0.05
    mask_frac_max:  float = 0.30

    noise_prob:     float = 0.5
    noise_snr_min:  float = 10.0
    noise_snr_max:  float = 50.0

    flux_scale_prob: float = 0.4
    flux_scale_min:  float = 0.85
    flux_scale_max:  float = 1.15

    wave_shift_prob:    float = 0.3
    wave_shift_max_pix: float = 3.0

    continuum_warp_prob:  float = 0.3
    continuum_warp_amp:   float = 0.05
    continuum_warp_knots: int   = 5

    use_balanced_sampling: bool  = True
    balance_beta:          float = 0.9999

    # ── MixUp ─────────────────────────────────────────────────
    use_mixup:   bool  = True
    mixup_alpha: float = 0.3
    mixup_prob:  float = 0.35       # reduced for more SupCon coverage (65%)
    mixup_meta:  bool  = True

    # ── Model architecture ────────────────────────────────────
    num_classes:  int   = 7        # Ia, Ibc, SLSN, II, AGN, CV, TDE
    in_channels:  int   = 2        # flux + SG derivative
    base_c:       int   = 96
    emb_dim:      int   = 192
    tcn_depth:    int   = 6
    tcn_kernel:   int   = 5
    downsample:   int   = 4
    dropout:      float = 0.2
    proj_dim:     int   = 128
    drop_path_rate:      float = 0.1
    attn_heads:          int   = 4
    w_attn_entropy:      float = 0.01
    emb_dropout:         float = 0.15
    fine_head_stop_grad: bool  = True
    w_loco:              float = 0.0      # leave-one-class-out novelty term (0 = off)
    backbone:            str   = "tcn"      # 'tcn' (ASTRANet) or 'spectranet' (Xu+2026 stages)
    z_shift:             bool  = False    # observed-frame input, moved to the rest frame inside the network when z is known

    savgol_window: int = 7
    savgol_poly:   int = 3

    meta_hidden:     int   = 128
    # Metadata dropout (training only): per-sample probability of withholding z / phase
    # (set to NaN -> validity flag 0) so the model learns to operate without metadata.
    meta_dropout_z:     float = 0.0
    meta_dropout_phase: float = 0.0
    meta_out_dim:    int   = 64

    # ── Loss weights ──────────────────────────────────────────
    w_fine:       float = 1.5
    w_coarse:     float = 0.3
    w_con_fine:   float = 0.02   # doubled for better Ibc/II separation
    w_con_coarse: float = 0.02
    temperature:  float = 0.07
    disable_contrastive_on_mixup: bool = True

    focal_gamma: float = 2.0
    focal_gamma_overrides: Dict = field(default_factory=lambda: {
        "TDE":  3.0,
        "AGN":  2.5,
        "Ibc":  2.5,    # harder focus on weak class
        "II":   2.5,    # harder focus on weak class
        "SLSN": 2.0,
    })

    focal_use_class_weights: bool = False
    balance_beta_fl: float = 0.9999
    label_smoothing: float = 0.05

    w_confusion:     float = 0.6   # stronger confusion penalty
    confusion_power: float = 2.0
    confused_pairs: List = field(default_factory=lambda: [
        ("TDE",  "Ia",  0.5),
        ("SLSN", "Ibc", 0.5),
        ("AGN",  "Ia",  0.3),
        ("Ibc",  "Ia",  0.4),   # penalize Ibc->Ia confusion
    ])

    # ── Optimizer ─────────────────────────────────────────────
    lr:           float = 3e-4
    weight_decay: float = 1e-4
    grad_clip:    float = 1.0
    grad_accum_steps: int = 1

    use_lr_scheduler:    bool  = True
    warmup_epochs:       int   = 5
    eta_min_frac:        float = 0.05
    use_cosine_restarts: bool  = True
    restart_period:      int   = 20   # earlier restart to match peak timing
    restart_mult:        int   = 2

    use_ema:   bool  = True
    ema_decay: float = 0.996

    # ── SWA (Stochastic Weight Averaging) ────────────────
    use_swa:   bool = True
    swa_start: int  = 35     # start collecting after first cosine cycle
    swa_freq:  int  = 5      # collect every N epochs

    # ── Training ──────────────────────────────────────────────
    epochs:   int = 200
    patience: int = 45
    monitor:  str = "val_macro_acc"
    seed:     int = 42
    gpu_id:   int = 0

    use_peak_assembly: bool  = True
    peak_warmup:       int   = 15
    peak_ema_alpha:    float = 0.3

    # ── Classes & hierarchy ───────────────────────────────────
    class_names: List[str] = field(default_factory=lambda: [
        "Ia", "Ibc", "SLSN", "II", "AGN", "CV", "TDE",
    ])

    hierarchy: Dict = field(default_factory=lambda: {
        "SN_Thermonuclear": ["Ia"],
        "SN_CC":            ["Ibc", "SLSN", "II"],
        "NonSN":            ["AGN", "CV", "TDE"],
    })


# Label remapping: raw survey class names → ASTRA class names
RAW_CLASS_GROUPS = {
    "Ia":   ["Ia", "Ia-norm", "Ia-99aa", "Ia-02cx", "Ia-03fg",
             "Ia-pec", "Ia-91T", "Ia-91bg", "Ia-CSM"],
    "Ibc":  ["Ib", "Ic", "Ib/c", "Ib-pec", "Ic-BL", "Ibn", "Icn", "Ien"],
    "SLSN": ["Ic-SLSN", "Ic.5-SLSN", "II-SLSN", "SLSN I"],
    "II":   ["Type II", "IIP", "IIL", "IIn", "IIb", "II-norm", "II-pec"],
    "AGN":  ["AGN", "Seyfert", "Galactic Nuclei"],
    "CV":   ["Cataclysmic", "AM CVn", "Nova-like", "U Gem"],
    "TDE":  ["Tidal Disruption Event"],
}

RAW_TO_GROUP = {
    raw: group
    for group, raws in RAW_CLASS_GROUPS.items()
    for raw in raws
}


# ── Class-set switch (Ia-pec inclusion) ───────────────────────────
# Default: the full mapping (peculiar-Ia subtypes fold into `Ia`).
# `no_ia_pec` drops the raw `Ia-pec` subtype — kept as an ablation switch.
def get_raw_class_groups(class_set: str = "with_ia_pec") -> Dict:
    """Return the raw->ASTRA label groups for the requested class set.

    with_ia_pec : the full mapping (default) — peculiar-Ia subtypes fold into Ia.
    no_ia_pec   : drop the raw 'Ia-pec' subtype (ablation switch).
    """
    import copy
    groups = copy.deepcopy(RAW_CLASS_GROUPS)
    if class_set == "no_ia_pec":
        groups["Ia"] = [r for r in groups["Ia"] if r != "Ia-pec"]
    elif class_set != "with_ia_pec":
        raise ValueError(f"unknown class_set: {class_set!r}")
    return groups


def get_raw_to_group(class_set: str = "with_ia_pec") -> Dict:
    g = get_raw_class_groups(class_set)
    return {raw: grp for grp, raws in g.items() for raw in raws}