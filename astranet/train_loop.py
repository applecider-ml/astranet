# astra/train_loop.py
# Full training loop for ASTRA.
#
# Features:
#   - MixUp augmentation with metadata mixing
#   - Contrastive loss skipped on MixUp batches
#   - Cosine LR schedule with warm restarts
#   - Gradient accumulation
#   - EMA (Exponential Moving Average) weights
#   - Attention entropy regularisation
#   - Per-class peak assembly: combines best-macro backbone with each
#     class's individually best output-layer row (fixes seesaw problem)

import os
import sys
import json
import math
import copy
import argparse
import time

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from astranet.model   import ASTRANet, build_hierarchy_mappings
from astranet.losses  import ASTRALoss
from astranet.dataset import build_loaders
from astranet.config  import ASTRAConfig


# ============================================================
# UTILITIES
# ============================================================

def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class EMA:
    """Exponential Moving Average of model weights for stable evaluation."""
    def __init__(self, model: nn.Module, decay: float = 0.996):
        self.model  = model
        self.decay  = decay
        self.shadow = {}
        self.backup = {}

    def register(self):
        for name, p in self.model.named_parameters():
            if p.requires_grad:
                self.shadow[name] = p.data.clone()

    @torch.no_grad()
    def update(self):
        for name, p in self.model.named_parameters():
            if p.requires_grad and name in self.shadow:
                self.shadow[name].mul_(self.decay).add_(p.data, alpha=1 - self.decay)

    def apply_shadow(self):
        self.backup = {}
        for name, p in self.model.named_parameters():
            if p.requires_grad and name in self.shadow:
                self.backup[name] = p.data.clone()
                p.data.copy_(self.shadow[name])

    def restore(self):
        for name, p in self.model.named_parameters():
            if p.requires_grad and name in self.backup:
                p.data.copy_(self.backup[name])
        self.backup = {}


class EarlyStopping:
    def __init__(self, patience: int = 15, mode: str = "max"):
        self.patience = patience
        self.mode     = mode
        self.best     = float("-inf") if mode == "max" else float("inf")
        self.counter  = 0

    def step(self, value: float) -> bool:
        improved = (self.mode == "max" and value > self.best) or \
                   (self.mode == "min" and value < self.best)
        if improved:
            self.best    = value
            self.counter = 0
        else:
            self.counter += 1
        return improved

    def should_stop(self) -> bool:
        return self.counter >= self.patience


def acc_from_logits(logits, labels):
    return (logits.argmax(dim=-1) == labels).float().mean().item()


def save_checkpoint(out_dir, model, optimizer, epoch, metric, tag="best"):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"checkpoint_{tag}.pt")
    torch.save({
        "epoch":     epoch,
        "model":     model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "metric":    metric,
    }, path)
    return path


# ============================================================
# PER-CLASS PEAK TRACKER
# ============================================================

class PerClassPeakTracker:
    """
    Addresses the seesaw problem in multi-class learning: improving one
    class may degrade another.  This tracker records, for each class,
    the epoch at which its smoothed recall was highest.  At the end of
    training, `assemble()` returns a state_dict that uses the overall
    best-macro-acc backbone but replaces each class's fine_head output
    row with the row from that class's individual peak epoch.
    """

    def __init__(self, num_classes: int, class_names: list,
                 warmup_epochs: int = 15, ema_alpha: float = 0.3):
        self.num_classes   = num_classes
        self.class_names   = class_names
        self.warmup_epochs = warmup_epochs
        self.ema_alpha     = ema_alpha

        self.best_recall   = [0.0] * num_classes
        self.peak_epoch    = [0]   * num_classes
        self.smooth_recall = [0.0] * num_classes

        self.peak_weight_rows = [None] * num_classes
        self.peak_bias_elems  = [None] * num_classes

        self.best_macro       = 0.0
        self.best_macro_state = None
        self.best_macro_epoch = 0

    def update(self, epoch: int, per_class_recall: list, model: nn.Module):
        for c in range(self.num_classes):
            r = per_class_recall[c]
            if self.smooth_recall[c] == 0.0:
                self.smooth_recall[c] = r
            else:
                self.smooth_recall[c] = (
                    self.ema_alpha * r + (1 - self.ema_alpha) * self.smooth_recall[c]
                )

        if epoch < self.warmup_epochs:
            return

        last_linear = None
        for module in reversed(list(model.fine_head.modules())):
            if isinstance(module, nn.Linear):
                last_linear = module
                break
        if last_linear is None:
            return

        W = last_linear.weight.data
        b = last_linear.bias.data

        for c in range(self.num_classes):
            r = self.smooth_recall[c]
            if r > self.best_recall[c]:
                self.best_recall[c]       = r
                self.peak_epoch[c]        = epoch
                self.peak_weight_rows[c]  = W[c].clone().cpu()
                self.peak_bias_elems[c]   = b[c].clone().cpu()

        macro = np.mean(per_class_recall)
        if macro > self.best_macro:
            self.best_macro       = macro
            self.best_macro_epoch = epoch
            self.best_macro_state = {k: v.clone().cpu() for k, v in model.state_dict().items()}

    def assemble(self, model: nn.Module, device) -> tuple:
        if self.best_macro_state is None:
            return None

        state = copy.deepcopy(self.best_macro_state)

        weight_key = bias_key = None
        for key in state:
            if "fine_head" in key and key.endswith(".weight"):
                weight_key = key
            if "fine_head" in key and key.endswith(".bias"):
                bias_key = key

        if weight_key is None or bias_key is None:
            print("  [PeakTracker] WARNING: could not find fine_head last linear keys")
            return None

        W, b = state[weight_key].clone(), state[bias_key].clone()
        n_replaced = 0
        for c in range(self.num_classes):
            if self.peak_weight_rows[c] is not None and self.peak_epoch[c] != self.best_macro_epoch:
                W[c] = self.peak_weight_rows[c]
                b[c] = self.peak_bias_elems[c]
                n_replaced += 1

        state[weight_key] = W
        state[bias_key]   = b
        return state, n_replaced

    def print_summary(self):
        print(f"\n  Per-class peak recall (smoothed, alpha={self.ema_alpha}):")
        print(f"  Base: epoch {self.best_macro_epoch} (macro={self.best_macro:.4f})")
        for c in range(self.num_classes):
            marker = " <- same" if self.peak_epoch[c] == self.best_macro_epoch \
                     else f" <- epoch {self.peak_epoch[c]}"
            print(f"    {self.class_names[c]:15s}: {self.best_recall[c]:.3f}{marker}")


# ============================================================
# MIXUP WITH METADATA
# ============================================================

def mixup_batch(x, y, phase, z, alpha=0.3, mix_meta=True, device="cuda"):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
        lam = max(lam, 1 - lam)
    else:
        lam = 1.0

    B = x.size(0)
    index = torch.randperm(B, device=device)
    mixed_x = lam * x + (1 - lam) * x[index]

    if mix_meta:
        ph_a, ph_b = phase, phase[index]
        z_a,  z_b  = z,     z[index]

        both_ph   = torch.isfinite(ph_a) & torch.isfinite(ph_b)
        mixed_phase = torch.where(both_ph, lam * ph_a + (1 - lam) * ph_b,
                      torch.where(torch.isfinite(ph_a), ph_a, ph_b))

        both_z   = torch.isfinite(z_a) & torch.isfinite(z_b)
        mixed_z  = torch.where(both_z, lam * z_a + (1 - lam) * z_b,
                   torch.where(torch.isfinite(z_a), z_a, z_b))
    else:
        mixed_phase, mixed_z = phase, z

    return mixed_x, y, y[index], lam, index, mixed_phase, mixed_z


# ============================================================
# EVALUATION
# ============================================================

@torch.no_grad()
def evaluate_tta(model, config, device, fine_to_super, n_tta: int = 8):
    """Run inference n_tta times with augmentation and average logits (TTA)."""
    from astranet.dataset import make_loader
    model.eval()

    test_dict = torch.load(
        os.path.join(config.data_dir, "test.pt"), map_location="cpu"
    )
    augmentor_kwargs = dict(
        mask_prob=config.mask_prob,
        mask_frac_min=config.mask_frac_min,
        mask_frac_max=config.mask_frac_max,
        noise_prob=config.noise_prob,
        noise_snr_min=config.noise_snr_min,
        noise_snr_max=config.noise_snr_max,
        flux_scale_prob=config.flux_scale_prob,
        flux_scale_min=config.flux_scale_min,
        flux_scale_max=config.flux_scale_max,
        wave_shift_prob=config.wave_shift_prob,
        wave_shift_max_pix=config.wave_shift_max_pix,
        continuum_warp_prob=config.continuum_warp_prob,
        continuum_warp_amp=config.continuum_warp_amp,
        continuum_warp_knots=config.continuum_warp_knots,
    )
    aug_loader = make_loader(
        test_dict, config.batch_size, shuffle=False,
        labeled=True, augment=True,
        num_workers=config.num_workers, pin_memory=config.pin_memory,
        compute_derivative=config.compute_derivative,
        class_names=config.class_names,
        savgol_window=config.savgol_window,
        savgol_poly=config.savgol_poly,
        augmentor_kwargs=augmentor_kwargs,
    )

    num_fine  = fine_to_super.shape[0]
    n_samples = len(aug_loader.dataset)
    all_logits = torch.zeros(n_samples, num_fine)
    all_y      = torch.zeros(n_samples, dtype=torch.long)

    for pass_idx in range(n_tta):
        idx = 0
        for batch in aug_loader:
            x  = batch["x"].to(device)
            ph = batch["phase"].to(device)
            z  = batch["z"].to(device)
            bs = x.size(0)
            out = model(x, ph, z)
            all_logits[idx:idx+bs] += out["fine_logits"].cpu()
            if pass_idx == 0:
                all_y[idx:idx+bs] = batch["y"]
            idx += bs

    all_logits /= n_tta
    preds = all_logits.argmax(dim=-1)

    class_correct = torch.zeros(num_fine)
    class_total   = torch.zeros(num_fine)
    for fi in range(num_fine):
        mask = all_y == fi
        class_total[fi]   += mask.sum().item()
        class_correct[fi] += (preds[mask] == fi).sum().item()

    per_class = (class_correct / class_total.clamp(min=1)).tolist()
    return {
        "val_loss":      0.0,
        "val_acc":       (preds == all_y).float().mean().item(),
        "val_macro_acc": float(torch.tensor(per_class).mean()),
        "coarse_acc":    0.0,
        "per_class":     per_class,
    }


@torch.no_grad()
def evaluate(model, loader, device, fine_to_super, mask_z=False, mask_phase=False):
    """Validation pass. mask_z / mask_phase withhold that metadata (NaN -> validity flag 0)
    to measure metadata-free operation with the SAME weights."""
    model.eval()
    ce = nn.CrossEntropyLoss()
    total_loss = total_fine_acc = total_coarse_acc = 0.0
    n_batches  = 0
    num_fine   = fine_to_super.shape[0]
    class_correct = torch.zeros(num_fine)
    class_total   = torch.zeros(num_fine)

    for batch in loader:
        x  = batch["x"].to(device)
        y  = batch["y"].to(device)
        ph = batch["phase"].to(device)
        z  = batch["z"].to(device)
        if mask_z:     z  = torch.full_like(z,  float("nan"))
        if mask_phase: ph = torch.full_like(ph, float("nan"))

        out = model(x, ph, z)
        coarse_labels = fine_to_super.to(device)[y]

        total_loss       += ce(out["fine_logits"], y).item()
        total_fine_acc   += acc_from_logits(out["fine_logits"],   y)
        total_coarse_acc += acc_from_logits(out["coarse_logits"], coarse_labels)
        n_batches        += 1

        preds = out["fine_logits"].argmax(dim=-1).cpu()
        y_cpu = y.cpu()
        for fi in range(num_fine):
            mask = y_cpu == fi
            class_total[fi]   += mask.sum().item()
            class_correct[fi] += (preds[mask] == fi).sum().item()

    per_class = (class_correct / class_total.clamp(min=1)).tolist()
    n = max(1, n_batches)
    return {
        "val_loss":      total_loss / n,
        "val_acc":       total_fine_acc / n,
        "val_macro_acc": float(torch.tensor(per_class).mean()),
        "coarse_acc":    total_coarse_acc / n,
        "per_class":     per_class,
    }


# ============================================================
# TRAINING EPOCH
# ============================================================

def train_epoch(model, loader, optimizer, criterion, device, config, ema):
    model.train()
    totals = dict(loss=0, fine_loss=0, coarse_loss=0,
                  con_fine=0, con_coarse=0, attn_entropy=0,
                  confusion_penalty=0, acc=0)
    n = n_mixup = 0
    accum = config.grad_accum_steps
    optimizer.zero_grad(set_to_none=True)

    for step, batch in enumerate(tqdm(loader, desc="  train", leave=False)):
        x  = batch["x"].to(device)
        y  = batch["y"].to(device)
        ph = batch["phase"].to(device)
        z  = batch["z"].to(device)

        # metadata dropout: independently withhold z / phase for a random subset of the batch
        if getattr(config, "meta_dropout_z", 0.0) > 0:
            drop = torch.rand(z.shape[0], device=device) < config.meta_dropout_z
            z = torch.where(drop, torch.full_like(z, float("nan")), z)
        if getattr(config, "meta_dropout_phase", 0.0) > 0:
            drop = torch.rand(ph.shape[0], device=device) < config.meta_dropout_phase
            ph = torch.where(drop, torch.full_like(ph, float("nan")), ph)

        apply_mix = config.use_mixup and (np.random.random() < config.mixup_prob)

        if apply_mix:
            x_in, y_a, y_b, lam, _, ph_m, z_m = mixup_batch(
                x, y, ph, z,
                alpha=config.mixup_alpha, mix_meta=config.mixup_meta, device=device,
            )
            n_mixup += 1
        else:
            x_in, y_a = x, y
            y_b, lam, ph_m, z_m = y, 1.0, ph, z

        out = model(x_in, ph_m, z_m)
        is_mixup = apply_mix and lam < 1.0

        if is_mixup:
            loss_a, ld_a = criterion(out, y_a, is_mixup=True)
            loss_b, ld_b = criterion(out, y_b, is_mixup=True)
            loss = lam * loss_a + (1 - lam) * loss_b
            ld   = {k: lam * ld_a[k] + (1 - lam) * ld_b.get(k, 0.0) for k in ld_a}
        else:
            loss, ld = criterion(out, y_a, is_mixup=False)

        (loss / accum).backward()

        if (step + 1) % accum == 0 or (step + 1) == len(loader):
            if config.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            if ema is not None:
                ema.update()

        for k in totals:
            if k == "loss":
                totals[k] += loss.item()
            elif k == "acc":
                totals[k] += acc_from_logits(out["fine_logits"], y_a)
            else:
                totals[k] += ld.get(k, 0.0)
        n += 1

    result = {k: v / max(1, n) for k, v in totals.items()}
    result["mixup_rate"] = n_mixup / max(1, n)
    return result


# ============================================================
# MAIN TRAINING FUNCTION
# ============================================================

def train(config: ASTRAConfig):
    set_seed(config.seed)
    os.makedirs(config.out_dir, exist_ok=True)

    assert config.num_classes == len(config.class_names)

    import dataclasses
    with open(os.path.join(config.out_dir, "config.json"), "w") as f:
        json.dump(dataclasses.asdict(config), f, indent=2)

    if torch.cuda.is_available() and config.gpu_id < torch.cuda.device_count():
        device = torch.device(f"cuda:{config.gpu_id}")
    else:
        device = torch.device("cpu")
    print(f"\n{'='*60}")
    print(f"ASTRA — Automated Spectral TRansient Analysis")
    print(f"Device: {device}")
    print(f"{'='*60}")

    # Data
    train_dict, train_loader, val_loader, test_loader, extras = build_loaders(config)
    print(f"\nData: {len(train_loader.dataset)} train | "
          f"{len(val_loader.dataset)} val | {len(test_loader.dataset)} test")
    for k, v in extras.items():
        print(f"       {len(v.dataset)} {k}")

    labels_np    = train_loader.dataset.y.numpy()
    class_counts = np.bincount(labels_np, minlength=config.num_classes)
    print("\nClass distribution:")
    for name, cnt in zip(config.class_names, class_counts):
        print(f"  {name:15s}: {cnt}")

    # Hierarchy
    fine_to_super, _, superclass_names = build_hierarchy_mappings(
        config.hierarchy, config.class_names
    )
    num_coarse = len(superclass_names)
    print(f"\nHierarchy ({num_coarse} superclasses): {superclass_names}")

    # Model
    model = ASTRANet(
        num_fine         = config.num_classes,
        num_coarse       = num_coarse,
        in_channels      = config.in_channels,
        base_c           = config.base_c,
        emb_dim          = config.emb_dim,
        tcn_depth        = config.tcn_depth,
        tcn_kernel       = config.tcn_kernel,
        downsample       = config.downsample,
        dropout          = config.dropout,
        drop_path_rate   = config.drop_path_rate,
        attn_heads       = config.attn_heads,
        meta_hidden      = config.meta_hidden,
        meta_out_dim     = config.meta_out_dim,
        emb_dropout      = config.emb_dropout,
        fine_head_stop_grad = config.fine_head_stop_grad,
        proj_dim         = config.proj_dim,
        backbone         = getattr(config, "backbone", "tcn"),
        z_shift          = getattr(config, "z_shift", False),
    ).to(device)

    # Optional warm-start (e.g. SSL-pretrained trunk); non-strict so heads stay random
    init_from = getattr(config, "init_from", None)
    if init_from:
        ckpt = torch.load(init_from, map_location=device, weights_only=False)
        sd = ckpt.get("state_dict", ckpt)
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(f"\nWarm-start from {init_from}: loaded {len(sd)} tensors "
              f"({len(missing)} missing, {len(unexpected)} unexpected)")

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel: {n_params:,} trainable parameters")

    # Loss
    gamma_per_class = torch.full((config.num_classes,), config.focal_gamma)
    if config.focal_gamma_overrides:
        name2idx = {n: i for i, n in enumerate(config.class_names)}
        for cls_name, g in config.focal_gamma_overrides.items():
            if cls_name in name2idx:
                gamma_per_class[name2idx[cls_name]] = g
                print(f"  Focal gamma override: {cls_name} -> {g}")

    criterion = ASTRALoss(
        fine_to_super         = fine_to_super.to(device),
        class_counts          = class_counts if config.focal_use_class_weights else None,
        num_fine              = config.num_classes,
        w_fine                = config.w_fine,
        w_coarse              = config.w_coarse,
        w_con_fine            = config.w_con_fine,
        w_con_coarse          = config.w_con_coarse,
        temperature           = config.temperature,
        w_attn_entropy        = config.w_attn_entropy,
        focal_gamma           = config.focal_gamma,
        focal_gamma_per_class = gamma_per_class,
        balance_beta          = config.balance_beta_fl,
        label_smoothing       = config.label_smoothing,
        focal_use_class_weights = config.focal_use_class_weights,
        confused_pairs        = config.confused_pairs,
        class_names           = config.class_names,
        w_confusion           = config.w_confusion,
        confusion_power       = config.confusion_power,
        w_loco                = getattr(config, "w_loco", 0.0),
    )

    print(f"\nLoss weights: fine={config.w_fine} | coarse={config.w_coarse} | "
          f"con_fine={config.w_con_fine} | con_coarse={config.w_con_coarse} "
          f"(T={config.temperature})")
    print(f"Focal: gamma={config.focal_gamma} | label_smooth={config.label_smoothing} | "
          f"class_weights={'ON' if config.focal_use_class_weights else 'OFF (balanced sampling)'}")
    print(f"Attention entropy reg: w={config.w_attn_entropy}")
    if config.w_confusion > 0:
        print(f"Confusion penalty: w={config.w_confusion} power={config.confusion_power}")
        for t, c, w in config.confused_pairs:
            print(f"  {t} -> {c}: weight={w}")
    mix_str = (f"MixUp alpha={config.mixup_alpha} prob={config.mixup_prob} "
               f"meta={'yes' if config.mixup_meta else 'no'}")
    print(f"Augmentation: {mix_str}")
    print(f"  mask={config.mask_prob} | noise={config.noise_prob} | "
          f"flux_scale={config.flux_scale_prob} | wave_shift={config.wave_shift_prob} | "
          f"warp={config.continuum_warp_prob}")
    print(f"Architecture: drop_path={config.drop_path_rate} | attn_heads={config.attn_heads} | "
          f"emb_dropout={config.emb_dropout} | stop_grad_coarse={config.fine_head_stop_grad}")
    if config.use_peak_assembly:
        print(f"Peak assembly: ON (warmup={config.peak_warmup}, ema_alpha={config.peak_ema_alpha})")

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )

    if config.use_lr_scheduler:
        if config.use_cosine_restarts:
            warmup = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=1.0 / config.warmup_epochs,
                end_factor=1.0, total_iters=config.warmup_epochs,
            )
            cosine = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, T_0=config.restart_period, T_mult=config.restart_mult,
                eta_min=config.lr * config.eta_min_frac,
            )
            scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer, schedulers=[warmup, cosine],
                milestones=[config.warmup_epochs],
            )
        else:
            def lr_lambda(ep):
                if ep < config.warmup_epochs:
                    return (ep + 1) / config.warmup_epochs
                progress = (ep - config.warmup_epochs) / max(1, config.epochs - config.warmup_epochs)
                cos = (1 + math.cos(math.pi * progress)) / 2
                return config.eta_min_frac + (1 - config.eta_min_frac) * cos
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    else:
        scheduler = None

    # EMA
    ema = EMA(model, config.ema_decay) if config.use_ema else None
    if ema is not None:
        ema.register()

    # SWA (Stochastic Weight Averaging)
    swa_model = None
    if getattr(config, 'use_swa', False):
        from torch.optim.swa_utils import AveragedModel
        swa_model = AveragedModel(model)
        swa_start = getattr(config, 'swa_start', 35)
        swa_freq  = getattr(config, 'swa_freq', 5)
        swa_n     = 0
        print(f"SWA: enabled (start={swa_start}, freq={swa_freq})")

    # Peak assembly
    peak_tracker = PerClassPeakTracker(
        num_classes=config.num_classes,
        class_names=config.class_names,
        warmup_epochs=config.peak_warmup,
        ema_alpha=config.peak_ema_alpha,
    ) if config.use_peak_assembly else None

    # Early stopping
    mode    = "min" if config.monitor == "val_loss" else "max"
    stopper = EarlyStopping(config.patience, mode=mode)
    history = []
    best_path = None

    print(f"\nTraining for up to {config.epochs} epochs "
          f"(early stop on '{config.monitor}', patience={config.patience})")
    if config.use_cosine_restarts:
        print(f"LR schedule: cosine warm restarts "
              f"T_0={config.restart_period} T_mult={config.restart_mult}")
    print()

    # ── Training loop ─────────────────────────────────────────
    for epoch in range(1, config.epochs + 1):
        t0 = time.time()
        tr = train_epoch(model, train_loader, optimizer, criterion, device, config, ema)

        if ema is not None:
            ema.apply_shadow()
        va = evaluate(model, val_loader, device, fine_to_super)
        if getattr(config, "meta_dropout_z", 0.0) > 0 or getattr(config, "meta_dropout_phase", 0.0) > 0:
            # metadata-robust training: also validate without z and without any metadata, and
            # expose a combined criterion (mean of the three macro recalls) for model selection
            va_noz  = evaluate(model, val_loader, device, fine_to_super, mask_z=True)
            va_none = evaluate(model, val_loader, device, fine_to_super, mask_z=True, mask_phase=True)
            va["val_macro_acc_noz"]  = va_noz["val_macro_acc"]
            va["val_macro_acc_none"] = va_none["val_macro_acc"]
            va["val_acc_noz"]  = va_noz["val_acc"]
            va["val_acc_none"] = va_none["val_acc"]
            va["val_macro_acc_robust"] = (va["val_macro_acc"] + va_noz["val_macro_acc"]
                                         + va_none["val_macro_acc"]) / 3.0
        if peak_tracker is not None:
            peak_tracker.update(epoch, va["per_class"], model)
        if ema is not None:
            ema.restore()

        if scheduler is not None:
            scheduler.step()

        # SWA collection
        if swa_model is not None and epoch >= swa_start and epoch % swa_freq == 0:
            swa_model.update_parameters(model)
            swa_n += 1

        lr_now = optimizer.param_groups[0]["lr"]
        record = {
            "epoch": epoch, "lr": lr_now,
            "train_loss": tr["loss"], "train_acc": tr["acc"],
            **{f"val_{k}": v for k, v in va.items() if k != "per_class"},
            "per_class": va["per_class"],
        }
        history.append(record)

        improved = stopper.step(va[config.monitor])
        star = " *" if improved else ""
        if "val_macro_acc_robust" in va:
            star = (f" | noz={va['val_macro_acc_noz']:.4f} none={va['val_macro_acc_none']:.4f} "
                    f"robust={va['val_macro_acc_robust']:.4f}") + star
        print(
            f"[{epoch:03d}/{config.epochs}] {time.time()-t0:.1f}s | lr={lr_now:.2e} | "
            f"train_loss={tr['loss']:.4f} acc={tr['acc']:.4f} | "
            f"val_loss={va['val_loss']:.4f} acc={va['val_acc']:.4f} "
            f"macro={va['val_macro_acc']:.4f} coarse={va['coarse_acc']:.4f} | "
            f"H={tr['attn_entropy']:.3f}{star}"
        )

        if epoch % 10 == 0 or improved:
            pc_str = " | ".join(f"{n[:5]}={a:.2f}" for n, a in
                                zip(config.class_names, va["per_class"]))
            print(f"         per-class: {pc_str}")

        if improved:
            if ema is not None:
                ema.apply_shadow()
            best_path = save_checkpoint(
                config.out_dir, model, optimizer, epoch, va[config.monitor], tag="best"
            )
            if ema is not None:
                ema.restore()

        if stopper.should_stop():
            print(f"\nEarly stopping. Best {config.monitor}={stopper.best:.4f}")
            break

    # ── Post-training ─────────────────────────────────────────
    with open(os.path.join(config.out_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)

    print(f"\n{'='*60}\nFinal evaluation")
    if best_path and os.path.exists(best_path):
        ckpt = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        print(f"Loaded best checkpoint (epoch {ckpt['epoch']})")

    te = evaluate(model, test_loader, device, fine_to_super)
    print(f"\n[Standard]  fine_acc={te['val_acc']:.4f} | "
          f"macro_acc={te['val_macro_acc']:.4f} | coarse_acc={te['coarse_acc']:.4f}")
    for name, acc_val in zip(config.class_names, te["per_class"]):
        print(f"  {name:15s}: {acc_val:.3f} {'█' * int(acc_val * 20)}")

    with open(os.path.join(config.out_dir, "test_results.json"), "w") as f:
        json.dump(te, f, indent=2)

    n_tta = getattr(config, "tta_n", 8)
    te_tta = evaluate_tta(model, config, device, fine_to_super, n_tta=n_tta)
    print(f"[TTA x{n_tta}]    fine_acc={te_tta['val_acc']:.4f} | "
          f"macro_acc={te_tta['val_macro_acc']:.4f}")
    with open(os.path.join(config.out_dir, "test_results_tta.json"), "w") as f:
        json.dump(te_tta, f, indent=2)

    # SWA evaluation
    if swa_model is not None and swa_n > 0:
        print(f"\n  SWA: averaged {swa_n} checkpoints (epochs {swa_start}+)")
        from torch.optim.swa_utils import update_bn
        update_bn(train_loader, swa_model, device=device)
        te_swa = evaluate(swa_model, test_loader, device, fine_to_super)
        print(f"[SWA]       fine_acc={te_swa['val_acc']:.4f} | "
              f"macro_acc={te_swa['val_macro_acc']:.4f}")
        for name, acc_val in zip(config.class_names, te_swa["per_class"]):
            print(f"  {name:15s}: {acc_val:.3f} {'█' * int(acc_val * 20)}")

        te_swa_tta = evaluate_tta(swa_model, config, device, fine_to_super, n_tta=n_tta)
        print(f"[SWA TTA]   fine_acc={te_swa_tta['val_acc']:.4f} | "
              f"macro_acc={te_swa_tta['val_macro_acc']:.4f}")

        swa_path = os.path.join(config.out_dir, "checkpoint_swa.pt")
        torch.save({
            "model": swa_model.module.state_dict(),
            "metric": te_swa["val_macro_acc"],
        }, swa_path)
        with open(os.path.join(config.out_dir, "test_results_swa.json"), "w") as f:
            json.dump(te_swa, f, indent=2)
        with open(os.path.join(config.out_dir, "test_results_swa_tta.json"), "w") as f:
            json.dump(te_swa_tta, f, indent=2)
        print(f"  SWA checkpoint saved: {swa_path}")

    # Peak assembly evaluation
    if peak_tracker is not None:
        peak_tracker.print_summary()
        result = peak_tracker.assemble(model, device)
        if result is not None:
            assembled_state, n_replaced = result
            print(f"\n  Peak assembly: {n_replaced}/{config.num_classes} classes replaced")
            model.load_state_dict(assembled_state)
            model.to(device)
            te_peak = evaluate(model, test_loader, device, fine_to_super)
            print(f"[Peak]      fine_acc={te_peak['val_acc']:.4f} | "
                  f"macro_acc={te_peak['val_macro_acc']:.4f}")

            te_peak_tta = evaluate_tta(model, config, device, fine_to_super, n_tta=n_tta)
            print(f"[Peak TTA x{n_tta}] fine_acc={te_peak_tta['val_acc']:.4f} | "
                  f"macro_acc={te_peak_tta['val_macro_acc']:.4f}")
            with open(os.path.join(config.out_dir, "test_results_peak_tta.json"), "w") as f:
                json.dump(te_peak_tta, f, indent=2)

            if te_peak["val_macro_acc"] > te["val_macro_acc"]:
                peak_path = os.path.join(config.out_dir, "checkpoint_peak_assembly.pt")
                torch.save({
                    "epoch": -1, "model": assembled_state,
                    "metric": te_peak["val_macro_acc"],
                    "peak_info": {
                        "base_epoch": peak_tracker.best_macro_epoch,
                        "per_class_peak_epochs": peak_tracker.peak_epoch,
                    },
                }, peak_path)
                print(f"  >>> Peak assembly is BETTER — saved to {peak_path}")
                best_path = peak_path

            with open(os.path.join(config.out_dir, "test_results_peak.json"), "w") as f:
                json.dump(te_peak, f, indent=2)

    print(f"\nOutputs: {config.out_dir}")
    print(f"Best checkpoint: {best_path}")
    print(f"{'='*60}")
    return model, best_path, history
