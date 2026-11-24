"""Object-level novelty stage of the SkyPortal service.

One object, one or more spectra. Per spectrum the ensemble gives an embedding, logits and projections;
LUNA turns them into 16 anomaly scores and 23 spectral descriptors are measured on the flux grid. The
object vector is [mean embedding, mean descriptors, mean LUNA scores, maximum LUNA scores]. An
ExtraTrees head, fitted on ordinary objects (the seven classes and novae) against rare classes, gives
the novelty score. The score is compared with ordinary calibration objects of the same redshift
stratum (redshift known or unknown), which gives a p-value and alarms at 1, 5, 10 and 15 %.

    from astranet.novelty import Novelty
    nov = Novelty.load("models/novelty_v1.joblib")
    out = nov.score(features, grid_fluxes, redshift=0.05)   # features: list of Ensemble outputs

The bundle is built by tools/build_novelty.py.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter1d

RATES = (0.01, 0.05, 0.10, 0.15)
Z_MIN = 0.001                                   # below this the redshift counts as unknown (Galactic sources)
WL = np.linspace(3850, 9000, 4096)
WINDOWS = [(3850, 4050), (4800, 5100), (5750, 6050), (6050, 6400), (6450, 6900), (7600, 7900), (8300, 8800), (4300, 4700)]


def _norm(X):
    X = np.asarray(X, dtype=np.float32).reshape(len(X), -1)
    p5, p95 = np.percentile(X, 5, 1, keepdims=True), np.percentile(X, 95, 1, keepdims=True)
    return np.clip((X - p5) / (p95 - p5 + 1e-6), -2, 3)


def descriptors(X) -> np.ndarray:
    """23 spectral descriptors per flux grid (continuum bands and slope, texture, asymmetry, line windows)."""
    F = _norm(X); cont = gaussian_filter1d(F, 60, axis=1); res = F - cont
    bands = np.array_split(np.arange(F.shape[1]), 6); m = cont.mean(1, keepdims=True) + 1e-3
    feats = [cont[:, b].mean(1) / m[:, 0] for b in bands]
    x = (WL - WL.mean()) / WL.std(); feats.append((cont * x).mean(1) / m[:, 0])
    prev = F
    for sig in (2, 5, 12, 30, 60):
        sm = gaussian_filter1d(F, sig, axis=1); band = prev - sm; prev = sm
        feats.append(np.log(band.std(1) + 1e-4))
    for sig in (3, 10, 30):
        r = res - gaussian_filter1d(res, sig * 4, axis=1); r = gaussian_filter1d(r, sig, axis=1)
        feats.append(r.clip(0).mean(1) - (-r).clip(0).mean(1))
    for lo, hi in WINDOWS:
        w = (WL >= lo) & (WL < hi); feats.append(res[:, w].mean(1) / m[:, 0])
    return np.column_stack(feats).astype(np.float32)


def redshift_known(z) -> bool:
    try:
        return z is not None and np.isfinite(float(z)) and float(z) > Z_MIN
    except (TypeError, ValueError):
        return False


def conformal_threshold(scores, rate: float) -> float:
    """Upper conformal quantile; an alarm is score > threshold. Too few scores: the largest one."""
    s = np.sort(np.asarray(scores, dtype=float)); n = len(s)
    return float(s[min(int(np.ceil((n + 1) * (1 - rate))), n) - 1]) if n else float("inf")


class Novelty:
    def __init__(self, bundle: dict):
        self.b = bundle

    @classmethod
    def load(cls, path) -> "Novelty":
        import joblib
        return cls(joblib.load(path))

    def luna_scores(self, emb, proj_fine, fine_logits, coarse_logits) -> np.ndarray:
        s = self.b["luna"].score(dict(embeddings=np.asarray(emb, np.float32), logits=np.asarray(fine_logits, np.float32),
                                      coarse_logits=np.asarray(coarse_logits, np.float32), projections=np.asarray(proj_fine, np.float32)))
        return np.column_stack([s[m] for m in self.b["methods"]])

    def scaled_descriptors(self, grid_fluxes) -> np.ndarray:
        return np.clip((descriptors(grid_fluxes) - self.b["phys_med"]) / self.b["phys_iqr"], -20, 20).astype(np.float32)

    @staticmethod
    def object_vector(emb, phys, scores) -> np.ndarray:
        emb, phys, scores = (np.atleast_2d(np.asarray(a, dtype=np.float64)) for a in (emb, phys, scores))
        return np.hstack([emb.mean(0), phys.mean(0), scores.mean(0), scores.max(0)]).astype(np.float32)

    def score_vector(self, x, redshift=None) -> dict:
        s = float(self.b["head"].predict_proba(np.atleast_2d(x))[0, 1]); k = "redshift" if redshift_known(redshift) else "no_redshift"
        cal = self.b["calibration"][k]; p = float((1 + np.sum(cal >= s)) / (len(cal) + 1))
        alarm = {f"{int(round(r * 100))}%": bool(s > self.b["thresholds"][k][r]) for r in self.b["rates"]}
        first = next((a for a, v in alarm.items() if v), None)
        return dict(score=s, p_value=p, stratum=k, n_calibration=int(len(cal)), alarm=alarm, alarm_rate=first, version=self.b["version"])

    def score(self, features: list, grid_fluxes, redshift=None) -> dict:
        """`features`: one dict per spectrum with emb, proj_fine, fine_logits, coarse_logits (Ensemble.features)."""
        st = lambda k: np.stack([np.asarray(f[k], np.float32) for f in features])
        S = self.luna_scores(st("emb"), st("proj_fine"), st("fine_logits"), st("coarse_logits"))
        return self.score_vector(self.object_vector(st("emb"), self.scaled_descriptors(np.asarray(grid_fluxes)), S), redshift)
