# astra/dataset.py
# Dataset and data loading for ASTRA.
#
# Each spectrum is stored as raw flux (1D). The dataset computes a
# Savitzky-Golay smoothed first derivative at load time and stacks it
# as the second input channel: x = (flux, derivative), shape (2, L).
#
# Augmentation pipeline (training only):
#   - wavelength shift (sub-pixel linear interpolation)
#   - continuum warping (low-frequency multiplicative spline)
#   - flux scaling (random multiplicative factor)
#   - Gaussian noise injection (SNR-adaptive)
#   - random masking (1-3 contiguous regions)
# All augmentations are independently toggled per sample.

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler


def _worker_init_fn(worker_id):
    seed = torch.initial_seed() % (2**32)
    np.random.seed(seed)
    import random; random.seed(seed)


# ============================================================
# SAVITZKY-GOLAY DERIVATIVE
# ============================================================

def _savgol_diff_coeffs(window: int = 7, poly: int = 3) -> np.ndarray:
    """
    Precompute Savitzky-Golay first-derivative convolution kernel.

    Smoothes the derivative to avoid amplifying high-frequency noise
    that is common in observed spectra.
    """
    assert window % 2 == 1, "window must be odd"
    assert poly < window

    half = window // 2
    x = np.arange(-half, half + 1, dtype=np.float64)
    A = np.vander(x, N=poly + 1, increasing=True)
    ATA_inv = np.linalg.inv(A.T @ A)
    coeffs = ATA_inv @ A.T
    return coeffs[1].astype(np.float32)   # row 1 = first derivative


# ============================================================
# AUGMENTATION PIPELINE
# ============================================================

class SpectraAugmentor:
    """
    Stochastic augmentation pipeline for 1D spectra.

    All augmentations operate on the raw flux *before* the derivative
    is computed, so the derivative always reflects the augmented flux.
    Each transform is applied independently with its own probability.
    """

    def __init__(
        self,
        mask_prob: float = 0.5,
        mask_frac_min: float = 0.05,
        mask_frac_max: float = 0.30,
        noise_prob: float = 0.5,
        noise_snr_min: float = 10.0,
        noise_snr_max: float = 50.0,
        flux_scale_prob: float = 0.4,
        flux_scale_min: float = 0.85,
        flux_scale_max: float = 1.15,
        wave_shift_prob: float = 0.3,
        wave_shift_max_pix: float = 3.0,
        continuum_warp_prob: float = 0.3,
        continuum_warp_amp: float = 0.05,
        continuum_warp_knots: int = 5,
    ):
        self.mask_prob = mask_prob
        self.mask_frac_min = mask_frac_min
        self.mask_frac_max = mask_frac_max
        self.noise_prob = noise_prob
        self.noise_snr_min = noise_snr_min
        self.noise_snr_max = noise_snr_max
        self.flux_scale_prob = flux_scale_prob
        self.flux_scale_min = flux_scale_min
        self.flux_scale_max = flux_scale_max
        self.wave_shift_prob = wave_shift_prob
        self.wave_shift_max_pix = wave_shift_max_pix
        self.continuum_warp_prob = continuum_warp_prob
        self.continuum_warp_amp = continuum_warp_amp
        self.continuum_warp_knots = continuum_warp_knots

    def __call__(self, flux: torch.Tensor) -> torch.Tensor:
        """Apply augmentation pipeline to a 1D flux tensor of shape (L,)."""
        if np.random.random() < self.wave_shift_prob:
            flux = self._wavelength_shift(flux)
        if np.random.random() < self.continuum_warp_prob:
            flux = self._continuum_warp(flux)
        if np.random.random() < self.flux_scale_prob:
            flux = flux * np.random.uniform(self.flux_scale_min, self.flux_scale_max)
        if np.random.random() < self.noise_prob:
            flux = self._add_noise(flux)
        if np.random.random() < self.mask_prob:
            flux = self._random_mask(flux)
        return flux

    def _random_mask(self, flux: torch.Tensor) -> torch.Tensor:
        L = flux.shape[0]
        frac = np.random.uniform(self.mask_frac_min, self.mask_frac_max)
        n_reg = np.random.randint(1, 4)
        reg_len = max(1, int(L * frac / n_reg))
        flux = flux.clone()
        for _ in range(n_reg):
            start = np.random.randint(0, max(1, L - reg_len))
            flux[start:start + reg_len] = 0.0
        return flux

    def _add_noise(self, flux: torch.Tensor) -> torch.Tensor:
        signal_std = flux.std().item()
        if signal_std < 1e-8:
            return flux
        snr = np.random.uniform(self.noise_snr_min, self.noise_snr_max)
        return flux + torch.randn_like(flux) * (signal_std / snr)

    def _wavelength_shift(self, flux: torch.Tensor) -> torch.Tensor:
        L = flux.shape[0]
        shift = np.random.uniform(-self.wave_shift_max_pix, self.wave_shift_max_pix)
        orig = torch.arange(L, dtype=torch.float32, device=flux.device)
        shifted = (orig - shift).clamp(0, L - 1)
        idx_lo = shifted.long().clamp(0, L - 2)
        frac = shifted - idx_lo.float()
        return flux[idx_lo] * (1 - frac) + flux[idx_lo + 1] * frac

    def _continuum_warp(self, flux: torch.Tensor) -> torch.Tensor:
        L = flux.shape[0]
        knot_vals = 1.0 + np.random.uniform(
            -self.continuum_warp_amp, self.continuum_warp_amp,
            size=self.continuum_warp_knots,
        )
        knot_pos = np.linspace(0, L - 1, self.continuum_warp_knots)
        warp = torch.from_numpy(
            np.interp(np.arange(L), knot_pos, knot_vals).astype(np.float32)
        ).to(flux.device)
        return flux * warp


# ============================================================
# DATASET
# ============================================================

class SpectraDataset(Dataset):
    """
    PyTorch Dataset for ASTRA.

    Inputs expected in data_dict:
      X:    FloatTensor (N, 1, L) — raw flux
      y:    list of str labels (or LongTensor of int labels)
      meta: list of dicts with keys: phase, z, instrument, class, ...

    The dataset outputs:
      x:     (2, L) — flux + SG derivative
      phase: scalar float (NaN if missing)
      z:     scalar float (NaN if missing)
      meta:  dict (pass-through)
      y:     int label (labeled=True only)
    """

    def __init__(
        self,
        data_dict: dict,
        labeled: bool = True,
        augment: bool = False,
        compute_derivative: bool = True,
        class_names: list = None,
        savgol_window: int = 7,
        savgol_poly: int = 3,
        augmentor_kwargs: dict = None,
    ):
        raw_y    = data_dict.get("y", None)
        raw_X    = data_dict["X"]
        raw_meta = data_dict["meta"]

        # Filter and encode labels
        if labeled and raw_y is not None and class_names is not None:
            label2idx = {name: i for i, name in enumerate(class_names)}
            keep, y_int = [], []
            for i, label in enumerate(raw_y):
                label_str = label if isinstance(label, str) else str(label)
                if label_str in label2idx:
                    keep.append(i)
                    y_int.append(label2idx[label_str])

            if len(keep) < len(raw_y):
                print(f"  [Dataset] Dropped {len(raw_y) - len(keep)} samples "
                      f"with labels not in class_names.")

            keep_t = torch.tensor(keep, dtype=torch.long)
            self.X    = raw_X[keep_t]
            self.meta = [raw_meta[i] for i in keep]
            self.y    = torch.tensor(y_int, dtype=torch.long)
        else:
            self.X    = raw_X
            self.meta = raw_meta
            if raw_y is not None and not isinstance(raw_y, torch.Tensor):
                self.y = raw_y if (raw_y and isinstance(raw_y[0], str)) \
                         else torch.tensor(raw_y, dtype=torch.long)
            else:
                self.y = raw_y

        self.labeled = labeled
        self.compute_derivative = compute_derivative and (self.X.shape[1] == 1)

        self._savgol_kernel = torch.from_numpy(
            _savgol_diff_coeffs(savgol_window, savgol_poly)
        )
        self._savgol_pad = savgol_window // 2

        if augment:
            self.augmentor = SpectraAugmentor(**(augmentor_kwargs or {}))
        else:
            self.augmentor = None

    def __len__(self):
        return self.X.shape[0]

    def _savgol_derivative(self, flux: torch.Tensor) -> torch.Tensor:
        kernel = self._savgol_kernel.to(flux.device)
        f = flux.unsqueeze(0).unsqueeze(0)
        k = kernel.flip(0).unsqueeze(0).unsqueeze(0)
        d = torch.nn.functional.conv1d(f, k, padding=self._savgol_pad)
        return d.squeeze(0).squeeze(0)

    def __getitem__(self, i: int) -> dict:
        x = self.X[i]
        m = self.meta[i]

        if self.compute_derivative:
            flux = x[0]
            if self.augmentor is not None:
                flux = self.augmentor(flux)
            deriv = self._savgol_derivative(flux)
            x = torch.stack([flux, deriv], dim=0)
        else:
            if self.augmentor is not None:
                for c in range(x.shape[0]):
                    x_c = x[c].clone()
                    if np.random.random() < self.augmentor.noise_prob:
                        x_c = self.augmentor._add_noise(x_c)
                    if np.random.random() < self.augmentor.mask_prob:
                        x_c = self.augmentor._random_mask(x_c)
                    x = x.clone()
                    x[c] = x_c

        phase = float(m.get("phase", float("nan")) or float("nan"))
        z     = float(m.get("z",     float("nan")) or float("nan"))
        if not np.isfinite(phase):
            phase = float("nan")
        if not np.isfinite(z):
            z = float("nan")

        out = {
            "x":     x,
            "phase": torch.tensor(phase, dtype=torch.float32),
            "z":     torch.tensor(z,     dtype=torch.float32),
            "meta":  m,
        }
        if self.labeled and self.y is not None:
            out["y"] = self.y[i]
        return out


# ============================================================
# COLLATE
# ============================================================

def collate_fn(batch: list) -> dict:
    out = {}
    for k in batch[0]:
        if k == "meta":
            out[k] = [b[k] for b in batch]
        else:
            out[k] = torch.stack([b[k] for b in batch])
    return out


# ============================================================
# LOADER FACTORIES
# ============================================================

def make_loader(
    data_dict: dict,
    batch_size: int,
    shuffle: bool,
    labeled: bool = True,
    augment: bool = False,
    num_workers: int = 0,
    pin_memory: bool = True,
    compute_derivative: bool = True,
    class_names: list = None,
    savgol_window: int = 7,
    savgol_poly: int = 3,
    augmentor_kwargs: dict = None,
) -> DataLoader:
    ds = SpectraDataset(
        data_dict, labeled=labeled, augment=augment,
        compute_derivative=compute_derivative,
        class_names=class_names,
        savgol_window=savgol_window,
        savgol_poly=savgol_poly,
        augmentor_kwargs=augmentor_kwargs,
    )
    g = torch.Generator()
    g.manual_seed(torch.initial_seed())
    return DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, pin_memory=pin_memory,
        collate_fn=collate_fn,
        generator=g, worker_init_fn=_worker_init_fn,
    )


def make_balanced_loader(
    data_dict: dict,
    batch_size: int,
    labeled: bool = True,
    augment: bool = True,
    num_workers: int = 0,
    pin_memory: bool = True,
    balance_beta: float = 0.9999,
    compute_derivative: bool = True,
    class_names: list = None,
    savgol_window: int = 7,
    savgol_poly: int = 3,
    augmentor_kwargs: dict = None,
) -> DataLoader:
    ds = SpectraDataset(
        data_dict, labeled=labeled, augment=augment,
        compute_derivative=compute_derivative,
        class_names=class_names,
        savgol_window=savgol_window,
        savgol_poly=savgol_poly,
        augmentor_kwargs=augmentor_kwargs,
    )
    labels = ds.y.numpy()
    num_classes = len(class_names) if class_names is not None else int(labels.max()) + 1
    counts = np.bincount(labels, minlength=num_classes).astype(float)
    eff    = 1.0 - np.power(balance_beta, counts)
    eff    = np.where(eff > 0, eff, 1e-8)
    w_cls  = (1.0 - balance_beta) / eff
    w_cls /= w_cls.sum()
    w_samp = torch.tensor(w_cls[labels], dtype=torch.double)
    g = torch.Generator()
    g.manual_seed(torch.initial_seed())
    sampler = WeightedRandomSampler(w_samp, num_samples=len(w_samp), replacement=True,
                                    generator=g)
    return DataLoader(
        ds, batch_size=batch_size, sampler=sampler, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
        collate_fn=collate_fn,
        generator=g, worker_init_fn=_worker_init_fn,
    )


def build_loaders(config):
    """
    Build train / val / test DataLoaders from a Config object.

    Also looks for optional unlabeled splits (rare.pt, unknown.pt, bogus.pt)
    in the same data directory and returns them in the `extras` dict.

    Returns:
        train_dict, train_loader, val_loader, test_loader, extras
    """
    def _load(name):
        return torch.load(os.path.join(config.data_dir, name), map_location="cpu")

    train_dict = _load("train.pt")
    val_dict   = _load("val.pt")
    test_dict  = _load("test.pt")

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

    common_kw = dict(
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        compute_derivative=config.compute_derivative,
        class_names=config.class_names,
        savgol_window=config.savgol_window,
        savgol_poly=config.savgol_poly,
    )

    if config.use_balanced_sampling:
        train_loader = make_balanced_loader(
            train_dict, config.batch_size, labeled=True,
            augment=config.augment_train,
            balance_beta=config.balance_beta,
            augmentor_kwargs=augmentor_kwargs,
            **common_kw,
        )
    else:
        train_loader = make_loader(
            train_dict, config.batch_size, shuffle=True,
            labeled=True, augment=config.augment_train,
            augmentor_kwargs=augmentor_kwargs,
            **common_kw,
        )

    val_loader  = make_loader(val_dict,  config.batch_size, shuffle=False,
                              labeled=True, augment=False, **common_kw)
    test_loader = make_loader(test_dict, config.batch_size, shuffle=False,
                              labeled=True, augment=False, **common_kw)

    extras = {}
    for name in ("rare", "unknown", "bogus"):
        path = os.path.join(config.data_dir, f"{name}.pt")
        if os.path.exists(path):
            extras[name] = make_loader(
                torch.load(path, map_location="cpu"),
                config.batch_size, shuffle=False,
                labeled=False, augment=False, **common_kw,
            )

    return train_dict, train_loader, val_loader, test_loader, extras
