"""Spectrum preprocessing, identical to the one that produced the training tensors.

A spectrum stays in the observed frame: it is cut to 3850-9000 A, interpolated linearly on
4096 uniform wavelength points, edge-filled, and scaled by its 5th and 95th percentiles.
The redshift enters as a model input; the network moves the spectrum to the rest frame
itself when the redshift is known (astranet.model.ZShift).
"""
from __future__ import annotations

import numpy as np

WL_MIN, WL_MAX, INTERP_LEN = 3850.0, 9000.0, 4096
MAX_UNCOVERED = 0.20


class SpectrumRejected(ValueError):
    """The spectrum cannot be put on the model grid (coverage or finite-value failure)."""


def preprocess(wavelength, flux) -> np.ndarray:
    w = np.asarray(wavelength, dtype=float); f = np.asarray(flux, dtype=float)
    keep = np.isfinite(w) & np.isfinite(f); w, f = w[keep], f[keep]
    if len(w) < 10:
        raise SpectrumRejected("fewer than 10 finite samples")
    order = np.argsort(w, kind="stable"); w, f = w[order], f[order]
    m = (w >= WL_MIN) & (w <= WL_MAX)
    if m.sum() < 10:
        raise SpectrumRejected("fewer than 10 samples inside 3850-9000 A")
    grid = np.linspace(WL_MIN, WL_MAX, INTERP_LEN)
    g = np.interp(grid, w[m], f[m], left=np.nan, right=np.nan)
    if np.isnan(g).mean() > MAX_UNCOVERED:
        raise SpectrumRejected("more than 20% of the grid is not covered")
    ok = np.flatnonzero(~np.isnan(g))
    g[: ok[0]] = g[ok[0]]; g[ok[-1] + 1:] = g[ok[-1]]
    g = g.astype(np.float32)
    p5, p95 = np.percentile(g, [5, 95])
    g = ((g - p5) / ((p95 - p5) + 1e-30)).astype(np.float32)
    if not np.isfinite(g).all():
        raise SpectrumRejected("non-finite values after scaling")
    return g


def read_dat(path: str):
    """Two-column (wavelength, flux) text file with '#' comments, as exported by SkyPortal."""
    w, f = [], []
    with open(path) as fh:
        for line in fh:
            line = line.split("#", 1)[0].split()
            if len(line) >= 2:
                try: a, b = float(line[0]), float(line[1])
                except ValueError: continue
                w.append(a); f.append(b)
    return np.array(w), np.array(f)
