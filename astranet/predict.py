"""Inference with the paper's ensemble (v7fixcombo12): six TCN members and six SpectraNet members.

    from astranet.predict import Ensemble
    ens = Ensemble.from_dir("models")              # models/trained_v7fixmd, models/trained_v7fixsn
    out = ens.predict(wavelength, flux, redshift=0.05, phase=None)

The spectrum is given in the observed frame; with a known redshift the members move it to the rest
frame internally. Member logits are averaged before the softmax, the convention of the paper's cache.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .config import ASTRAConfig
from .model import ASTRANet
from .preprocess import preprocess

PARTS = (("trained_v7fixmd", "tcn"), ("trained_v7fixsn", "spectranet"))
SEEDS = (42, 123, 7)
TAGS = ("best", "swa")


def _derivative(flux: torch.Tensor) -> torch.Tensor:
    from .dataset import _savgol_diff_coeffs
    k = torch.from_numpy(_savgol_diff_coeffs(7, 3)).to(flux.dtype)
    return torch.nn.functional.conv1d(flux.view(1, 1, -1), k.flip(0).view(1, 1, -1), padding=3).view(-1)


def _load(path: Path, backbone: str, device) -> ASTRANet:
    ac = ASTRAConfig()
    m = ASTRANet(num_fine=ac.num_classes, num_coarse=3, in_channels=ac.in_channels, base_c=ac.base_c, emb_dim=ac.emb_dim,
                 tcn_depth=ac.tcn_depth, tcn_kernel=ac.tcn_kernel, downsample=ac.downsample, dropout=ac.dropout,
                 proj_dim=ac.proj_dim, drop_path_rate=ac.drop_path_rate, attn_heads=ac.attn_heads, backbone=backbone, z_shift=True)
    m.load_state_dict(torch.load(str(path), map_location=device, weights_only=False)["model"])
    return m.to(device).eval()


class Ensemble:
    def __init__(self, members, names, device):
        self.members, self.names, self.device = members, names, device
        self.classes = list(ASTRAConfig().class_names)

    @classmethod
    def from_dir(cls, models_dir, device: str | None = None) -> "Ensemble":
        dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        members, names = [], []
        for sub, backbone in PARTS:
            for s in SEEDS:
                for tag in TAGS:
                    p = Path(models_dir) / sub / f"seed{s}" / f"checkpoint_{tag}.pt"
                    if not p.exists():
                        raise FileNotFoundError(f"missing checkpoint {p}; run tools/fetch_weights.py")
                    members.append(_load(p, backbone, dev)); names.append(f"{sub}/seed{s}/{tag}")
        return cls(members, names, dev)

    @torch.no_grad()
    def features(self, grid_flux: np.ndarray, redshift: float | None = None, phase: float | None = None) -> dict:
        """Ensemble outputs for one flux grid, in the convention of the cache: embedding and projection are the
        member means of each backbone (the projection renormalised to unit length), concatenated [TCN | SpectraNet]; logits are the mean over the members."""
        f = torch.as_tensor(np.asarray(grid_flux, dtype=np.float32))
        x = torch.stack([f, _derivative(f)]).unsqueeze(0).to(self.device)
        nan = float("nan")
        z = torch.tensor([redshift if redshift is not None and np.isfinite(redshift) else nan], dtype=torch.float32, device=self.device)
        ph = torch.tensor([phase if phase is not None and np.isfinite(phase) else nan], dtype=torch.float32, device=self.device)
        outs = [m(x, ph, z) for m in self.members]
        groups = [[o for o, n in zip(outs, self.names) if n.startswith(sub)] for sub, _ in PARTS]
        mean = lambda g, k: torch.stack([o[k][0] for o in g]).mean(0)
        return dict(member_logits=torch.stack([o["fine_logits"][0] for o in outs]).cpu().numpy(),
                    fine_logits=torch.stack([o["fine_logits"][0] for o in outs]).mean(0).cpu().numpy(),
                    coarse_logits=torch.stack([o["coarse_logits"][0] for o in outs]).mean(0).cpu().numpy(),
                    emb=torch.cat([mean(g, "embedding") for g in groups]).cpu().numpy(),
                    proj_fine=torch.cat([torch.nn.functional.normalize(mean(g, "proj_fine"), dim=-1) for g in groups]).cpu().numpy())

    def predict_grid(self, grid_flux: np.ndarray, redshift: float | None = None, phase: float | None = None,
                     return_features: bool = False) -> dict:
        """`grid_flux` is the output of astranet.preprocess.preprocess (4096 points)."""
        ft = self.features(grid_flux, redshift, phase); L = torch.as_tensor(ft["member_logits"])           # (12, K)
        p = torch.softmax(L.mean(0), -1).numpy(); votes = L.argmax(1).numpy()
        top = int(p.argmax())
        out = dict(predicted=self.classes[top], probability=float(p[top]),
                   probabilities={c: float(v) for c, v in zip(self.classes, p)},
                   member_votes={c: int((votes == i).sum()) for i, c in enumerate(self.classes)},
                   member_agreement=float((votes == top).mean()), n_members=len(self.members),
                   redshift_used=None if redshift is None or not np.isfinite(redshift) else float(redshift),
                   phase_used=None if phase is None or not np.isfinite(phase) else float(phase))
        if return_features:
            out["_features"] = ft
        return out

    def predict(self, wavelength, flux, redshift: float | None = None, phase: float | None = None,
                return_features: bool = False) -> dict:
        grid = preprocess(wavelength, flux); out = self.predict_grid(grid, redshift, phase, return_features)
        if return_features:
            out["_grid"] = grid
        return out
