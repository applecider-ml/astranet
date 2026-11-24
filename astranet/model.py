# astra/model.py
# ASTRA — Automated Spectral TRansient Analysis (simplified)
#
# Architecture: Multi-scale TCN backbone with hierarchical classification,
# multi-head attention pooling, FiLM metadata conditioning (phase + redshift).
# Instrument embedding and supervised contrastive learning removed.

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# HIERARCHY UTILITIES
# ============================================================

def build_hierarchy_mappings(hierarchy: dict, class_names: list):
    """
    Build index mappings between fine and coarse (superclass) labels.

    Args:
        hierarchy: dict mapping superclass_name -> [fine_class_name, ...]
        class_names: ordered list of fine class names

    Returns:
        fine_to_super: LongTensor of shape (num_fine,)
        super_to_fine: dict mapping superclass_idx -> [fine_idxs]
        superclass_names: list of superclass names
    """
    superclass_names = list(hierarchy.keys())
    class_to_idx = {n: i for i, n in enumerate(class_names)}
    num_fine = len(class_names)

    fine_to_super = torch.zeros(num_fine, dtype=torch.long)
    super_to_fine = {i: [] for i in range(len(superclass_names))}

    for si, (sname, fnames) in enumerate(hierarchy.items()):
        for fname in fnames:
            if fname in class_to_idx:
                fi = class_to_idx[fname]
                fine_to_super[fi] = si
                super_to_fine[si].append(fi)

    return fine_to_super, super_to_fine, superclass_names


# ============================================================
# BACKBONE COMPONENTS
# ============================================================

class ConvGNAct(nn.Module):
    """Conv1D + GroupNorm + GELU + Dropout."""
    def __init__(self, in_c, out_c, k, stride=1, dilation=1, dropout=0.0, groups=8):
        super().__init__()
        pad = (k // 2) * dilation
        self.conv = nn.Conv1d(
            in_c, out_c, kernel_size=k,
            stride=stride, padding=pad, dilation=dilation,
        )
        gn_groups = min(groups, out_c)
        if out_c % gn_groups != 0:
            gn_groups = math.gcd(out_c, gn_groups)
        self.gn = nn.GroupNorm(max(1, gn_groups), out_c)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.drop(self.act(self.gn(self.conv(x))))


class MultiScaleStem(nn.Module):
    """Parallel convolutions for narrow/medium/broad spectral features."""
    def __init__(self, in_c=2, out_c=128, kernel_sizes=(7, 31, 151), dropout=0.05):
        super().__init__()
        branch_c = out_c // len(kernel_sizes)
        self.branches = nn.ModuleList([
            ConvGNAct(in_c, branch_c, k=k, dropout=dropout) for k in kernel_sizes
        ])
        self.fuse = ConvGNAct(branch_c * len(kernel_sizes), out_c, k=1, dropout=dropout)

    def forward(self, x):
        return self.fuse(torch.cat([b(x) for b in self.branches], dim=1))


class DropPath(nn.Module):
    """Stochastic Depth: randomly drops entire residual branches during training."""
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if not self.training or self.drop_prob == 0.0:
            return x
        keep = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = torch.bernoulli(torch.full(shape, keep, device=x.device))
        return x * mask / keep


class DilatedResBlock(nn.Module):
    """Dilated residual block with optional stochastic depth."""
    def __init__(self, c, k=5, dilation=1, dropout=0.1, drop_path=0.0):
        super().__init__()
        self.conv1 = ConvGNAct(c, c, k=k, dilation=dilation, dropout=dropout)
        self.conv2 = ConvGNAct(c, c, k=k, dilation=1, dropout=dropout)
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()

    def forward(self, x):
        return x + self.drop_path(self.conv2(self.conv1(x)))


class MultiHeadAttnPool1D(nn.Module):
    """
    Multi-head learned attention pooling.

    Each head produces its own attention weights over the sequence,
    then outputs are concatenated and projected.

    Returns (pooled, attn_entropy) so the training loop can add
    an entropy bonus to prevent attention collapse.
    """
    def __init__(self, c, num_heads=4, hidden=128, dropout=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = c // num_heads
        assert c % num_heads == 0, f"emb_dim {c} must be divisible by num_heads {num_heads}"

        self.scores = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(self.head_dim, hidden // num_heads, 1),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Conv1d(hidden // num_heads, 1, 1),
            ) for _ in range(num_heads)
        ])
        self.out_proj = nn.Linear(c, c)

    def forward(self, x):
        """
        x: (B, C, L)
        Returns: pooled (B, C), attn_entropy scalar
        """
        B, C, L = x.shape
        heads = x.view(B, self.num_heads, self.head_dim, L)

        pooled_parts = []
        entropies = []

        for h in range(self.num_heads):
            h_feat = heads[:, h]                                       # (B, head_dim, L)
            attn = torch.softmax(self.scores[h](h_feat), dim=-1)       # (B, 1, L)
            pooled_parts.append((h_feat * attn).sum(dim=-1))           # (B, head_dim)

            # Entropy of attention weights (higher = more spread out)
            ent = -(attn * (attn + 1e-8).log()).sum(dim=-1).mean()     # scalar
            entropies.append(ent)

        pooled = torch.cat(pooled_parts, dim=-1)
        pooled = self.out_proj(pooled)

        avg_entropy = torch.stack(entropies).mean()
        return pooled, avg_entropy

    @torch.no_grad()
    def per_sample_entropy(self, x):
        """
        Compute per-sample attention entropy (B,) instead of the batch scalar.
        Use this for anomaly detection, not for training.

        x: (B, C, L) — the input to the attention pool (can be captured via hook)
        Returns: (B,) per-sample mean entropy across heads
        """
        B, C, L = x.shape
        heads = x.view(B, self.num_heads, self.head_dim, L)
        per_sample_ent = []
        for h in range(self.num_heads):
            h_feat = heads[:, h]                                       # (B, head_dim, L)
            attn = torch.softmax(self.scores[h](h_feat), dim=-1)       # (B, 1, L)
            ent = -(attn * (attn + 1e-8).log()).sum(dim=-1).squeeze(-1)  # (B,)
            per_sample_ent.append(ent)
        return torch.stack(per_sample_ent, dim=0).mean(dim=0)          # (B,)


# ============================================================
# METADATA ENCODER (phase + redshift only)
# ============================================================

class MetaEncoder(nn.Module):
    """
    Encodes spectral observation metadata:
      - phase (days from peak, may be NaN)
      - redshift z (may be NaN)

    NaN values are masked out via validity flags.
    Output: (B, out_dim) conditioning vector used for FiLM modulation.
    """
    def __init__(
        self,
        hidden: int = 128,
        out_dim: int = 64,
    ):
        super().__init__()
        in_dim = 12   # 6 values + 6 masks (phase + z features only)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, out_dim),
            nn.LayerNorm(out_dim),
        )
        self.res_proj = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()

    def forward(self, phase, z):
        phase_known = torch.isfinite(phase).float()
        ph = torch.nan_to_num(phase, nan=0.0) / 30.0

        z_known = torch.isfinite(z).float()
        z_norm  = torch.tanh(torch.nan_to_num(z, nan=0.0) * 5.0)

        v6 = torch.stack([
            torch.sin(ph), torch.cos(ph), torch.tanh(ph),
            phase_known, z_norm, z_known,
        ], dim=-1)

        m6 = torch.stack([
            phase_known, phase_known, phase_known,
            phase_known, z_known, z_known,
        ], dim=-1)

        x = torch.cat([v6, m6], dim=-1)
        return self.net(x) + self.res_proj(x)


# ============================================================
# FiLM CONDITIONING
# ============================================================

class FiLMLayer(nn.Module):
    """Feature-wise Linear Modulation: x -> x * (1 + gamma) + beta."""
    def __init__(self, cond_dim: int, feat_dim: int):
        super().__init__()
        self.gamma_proj = nn.Linear(cond_dim, feat_dim)
        self.beta_proj  = nn.Linear(cond_dim, feat_dim)

    def forward(self, x, cond):
        return x * (1 + self.gamma_proj(cond)) + self.beta_proj(cond)


class FiLMOnSequence(nn.Module):
    """FiLM applied to a (B, C, L) feature map."""
    def __init__(self, cond_dim: int, feat_dim: int):
        super().__init__()
        self.gamma_proj = nn.Linear(cond_dim, feat_dim)
        self.beta_proj  = nn.Linear(cond_dim, feat_dim)

    def forward(self, x, cond):
        gamma = self.gamma_proj(cond).unsqueeze(-1)
        beta  = self.beta_proj(cond).unsqueeze(-1)
        return x * (1 + gamma) + beta


# ============================================================
# PROJECTION HEAD (contrastive)
# ============================================================

class ProjectionHead(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 256, out_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return F.normalize(self.net(x), dim=-1)


# ============================================================
# MAIN MODEL: ASTRANet
# ============================================================



# ============================================================================
# SpectraNet BACKBONE (Xu et al. 2026, PASP 138:094504) — optional replacement
# for the multi-scale stem + dilated TCN. Five SpectraBlocks, each with three
# parallel 1D convolutions (small / medium / large kernels), channel concat,
# LayerNorm over channels, GELU, then a 1x1 projection and MaxPool(4) between
# stages (no pooling in the last stage). Kernel sizes and channel widths are
# the published final configuration (Tables 4-5 of the paper).
# ============================================================================
SPECTRANET_KERNELS  = ((3, 61, 1021), (3, 31, 251), (3, 15, 61), (3, 11, 31), (3, 7, 13))
SPECTRANET_CHANNELS = (64, 128, 256, 512, 1024)


class SpectraBlock(nn.Module):
    def __init__(self, in_c, out_c, kernel_sizes, do_pool: bool):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Conv1d(in_c, out_c, kernel_size=k, padding=k // 2) for k in kernel_sizes
        ])
        cat_c = out_c * len(kernel_sizes)
        self.norm = nn.LayerNorm(cat_c)
        self.do_pool = do_pool
        if do_pool:
            self.proj = nn.Conv1d(cat_c, out_c, kernel_size=1)
            self.pool = nn.MaxPool1d(4)
        self.out_channels = out_c if do_pool else cat_c

    def forward(self, x):
        x = torch.cat([conv(x) for conv in self.convs], dim=1)
        x = self.norm(x.transpose(1, 2)).transpose(1, 2)
        x = F.gelu(x)
        if self.do_pool:
            x = self.pool(self.proj(x))
        return x


class SpectraNetBackbone(nn.Module):
    """(B, in_c, L) -> (B, 3*channels[-1], L / 4^4); L=4096 -> 16 positions."""
    def __init__(self, in_c=2, kernels=SPECTRANET_KERNELS, channels=SPECTRANET_CHANNELS):
        super().__init__()
        blocks, c = [], in_c
        for i, (ks, oc) in enumerate(zip(kernels, channels)):
            b = SpectraBlock(c, oc, ks, do_pool=(i < len(channels) - 1))
            blocks.append(b); c = b.out_channels
        self.blocks = nn.Sequential(*blocks)
        self.out_channels = c

    def forward(self, x):
        return self.blocks(x)


class ZShift(nn.Module):
    """Move an observed-frame spectrum to the rest frame inside the network when the redshift is known.

    The input stays in the observed frame (3850-9000 A, uniform grid). For a sample with finite z, the flux is
    resampled at lambda_rest * (1 + z) on the same grid (linear interpolation, edge fill outside the coverage),
    rescaled by its 5th/95th percentiles and the Savitzky-Golay derivative is recomputed, which is the
    preprocessing of the rest-frame tensors. A sample with unknown or withheld z (NaN) passes unchanged, so
    metadata dropout trains the observed-frame path as well."""
    def __init__(self, length=4096, wl_min=3850.0, wl_max=9000.0, savgol_window=7, savgol_poly=3):
        super().__init__()
        from .dataset import _savgol_diff_coeffs
        self.register_buffer("grid", torch.linspace(wl_min, wl_max, length), persistent=False)
        self.register_buffer("sg", torch.from_numpy(_savgol_diff_coeffs(savgol_window, savgol_poly)).flip(0).view(1, 1, -1), persistent=False)
        self.wl_min, self.wl_max, self.step, self.pad = wl_min, wl_max, (wl_max - wl_min) / (length - 1), savgol_window // 2
        self.min_cover = 0.2

    def forward(self, x, z):
        if z is None:
            return x
        # shift only where at least 20% of the rest-frame grid falls inside the observed coverage (z <= 0.87 for 3850-9000 A);
        # beyond that the rest-frame view is almost all edge fill, and the sample takes the observed-frame path
        known = torch.isfinite(z) & (z > -1)
        known = known & ((self.wl_max / (1 + torch.where(known, z, torch.zeros_like(z))) - self.wl_min) >= self.min_cover * (self.wl_max - self.wl_min))
        if not bool(known.any()):
            return x
        zz = torch.where(known, z, torch.zeros_like(z)).to(x.dtype)
        L = x.shape[-1]
        pos = ((self.grid[None, :] * (1 + zz[:, None]) - self.wl_min) / self.step).clamp(0, L - 1)
        i0 = pos.floor().long().clamp(max=L - 2); w = pos - i0
        f = x[:, 0]
        g = f.gather(1, i0) * (1 - w) + f.gather(1, i0 + 1) * w
        q = torch.quantile(g, torch.tensor([0.05, 0.95], device=g.device, dtype=g.dtype), dim=1)
        g = (g - q[0][:, None]) / (q[1] - q[0]).clamp_min(1e-6)[:, None]
        d = F.conv1d(g[:, None], self.sg.to(g.dtype), padding=self.pad)[:, 0]
        shifted = torch.stack([g, d], 1) if x.shape[1] == 2 else g[:, None]
        return torch.where(known[:, None, None], shifted, x)


class ASTRANet(nn.Module):
    """
    ASTRA — Automated Spectral TRansient Analysis

    Hierarchical classifier for astronomical transient spectra.

    Architecture:
      1. Multi-scale TCN stem (kernels 7, 31, 151) for local/global features
      2. Temporal downsampling
      3. Dilated TCN backbone (6 residual blocks, exponential dilation)
      4. Multi-head attention pooling with entropy regularisation
      5. FiLM conditioning with observation metadata (phase, z only)
      6. Hierarchical output: coarse superclass + fine class
      7. Projection heads for supervised contrastive learning (lower weights)

    Inputs:
      x:     (B, 2, L) — flux + Savitzky-Golay derivative
      phase: (B,) float, days from peak (NaN if unknown)
      z:     (B,) float, redshift (NaN if unknown)

    Output dict:
      fine_logits:   (B, num_fine)
      coarse_logits: (B, num_coarse)
      embedding:     (B, emb_dim) — for anomaly detection / UMAP
      proj_fine:     (B, proj_dim) — L2-normalized, for SupCon
      proj_coarse:   (B, proj_dim) — L2-normalized, for SupCon
      attn_entropy:  scalar — batch-mean attention entropy (for loss)
    """

    def __init__(
        self,
        num_fine: int = 7,
        num_coarse: int = 3,
        in_channels: int = 2,
        base_c: int = 96,
        emb_dim: int = 192,
        tcn_depth: int = 6,
        tcn_kernel: int = 5,
        downsample: int = 4,
        stem_kernels: tuple = (7, 31, 151),
        dropout: float = 0.2,
        drop_path_rate: float = 0.1,
        attn_heads: int = 4,
        meta_hidden: int = 128,
        meta_out_dim: int = 64,
        proj_dim: int = 128,
        emb_dropout: float = 0.15,
        fine_head_stop_grad: bool = True,
        backbone: str = "tcn",
        z_shift: bool = False,
    ):
        super().__init__()
        self.zshift = ZShift() if z_shift else None
        self.emb_dim = emb_dim
        self.backbone = backbone
        self.fine_head_stop_grad = fine_head_stop_grad

        # Backbone
        if backbone == "spectranet":
            # SpectraNet stages replace stem + downsample + TCN; a 1x1 projection
            # brings the 3072-channel map to emb_dim so pooling / FiLM / heads are unchanged.
            self.sn = SpectraNetBackbone(in_c=in_channels)
            self.proj_in = ConvGNAct(self.sn.out_channels, emb_dim, k=1, dropout=dropout * 0.5)
        elif backbone == "tcn":
            self.stem = MultiScaleStem(
                in_c=in_channels, out_c=base_c,
                kernel_sizes=stem_kernels, dropout=dropout * 0.5,
            )
            self.proj_in = ConvGNAct(base_c, emb_dim, k=1, dropout=dropout * 0.5)
            self.downsample = downsample

            dp_rates = [drop_path_rate * i / max(1, tcn_depth - 1) for i in range(tcn_depth)]
            dilations = [2 ** (i % 6) for i in range(tcn_depth)]
            self.tcn = nn.Sequential(*[
                DilatedResBlock(emb_dim, k=tcn_kernel, dilation=d,
                                dropout=dropout, drop_path=dp)
                for d, dp in zip(dilations, dp_rates)
            ])
        else:
            raise ValueError(f"unknown backbone {backbone!r} (use 'tcn' or 'spectranet')")

        # Multi-head attention pooling
        self.attn_pool = MultiHeadAttnPool1D(
            emb_dim,
            num_heads=attn_heads,
            hidden=max(64, emb_dim // 2),
            dropout=dropout * 0.2,
        )
        self.late_norm = nn.LayerNorm(emb_dim)
        self.emb_drop  = nn.Dropout(emb_dropout)

        # Metadata conditioning
        self.meta_enc = MetaEncoder(
            hidden=meta_hidden,
            out_dim=meta_out_dim,
        )
        self.film_seq = FiLMOnSequence(cond_dim=meta_out_dim, feat_dim=emb_dim)
        self.film_emb = FiLMLayer(cond_dim=meta_out_dim, feat_dim=emb_dim)

        # Coarse head
        self.coarse_head = nn.Sequential(
            nn.Linear(emb_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.4),
            nn.Linear(256, num_coarse),
        )

        # Fine head (conditioned on coarse probs + embedding)
        self.fine_head = nn.Sequential(
            nn.Linear(emb_dim + num_coarse, 384),
            nn.LayerNorm(384),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(384, num_fine),
        )

        # Contrastive projection heads
        self.proj_fine   = ProjectionHead(emb_dim, hidden_dim=256, out_dim=proj_dim)
        self.proj_coarse = ProjectionHead(emb_dim, hidden_dim=128, out_dim=proj_dim)

    def _encode_meta(self, phase, z):
        B = phase.shape[0] if phase is not None else 1
        device = next(self.parameters()).device
        if phase is None:
            phase = torch.full((B,), float("nan"), device=device)
        if z is None:
            z = torch.full((B,), float("nan"), device=device)
        return self.meta_enc(phase, z)

    def forward(self, x, phase=None, z=None):
        meta = self._encode_meta(phase, z)
        if self.zshift is not None:
            x = self.zshift(x, z)

        if self.backbone == "spectranet":
            # SpectraNet stages, then FiLM on the final feature map (before pooling)
            feat = self.proj_in(self.sn(x))
            feat = self.film_seq(feat, meta)
        else:
            # Stem + downsample
            feat = self.stem(x)
            if self.downsample > 1:
                feat = F.avg_pool1d(feat, self.downsample, self.downsample)
            feat = self.proj_in(feat)

            # FiLM conditioning on feature map
            feat = self.film_seq(feat, meta)

            # TCN backbone
            feat = self.tcn(feat)

        # Attention pooling
        emb, attn_entropy = self.attn_pool(feat)
        emb = self.late_norm(emb)

        # FiLM conditioning on pooled embedding
        emb = self.film_emb(emb, meta)

        emb_d = self.emb_drop(emb)

        # Coarse classification
        coarse_logits = self.coarse_head(emb_d)
        coarse_probs  = F.softmax(coarse_logits, dim=-1)

        if self.fine_head_stop_grad:
            coarse_input = coarse_probs.detach()
        else:
            coarse_input = coarse_probs

        # Fine classification
        fine_logits = self.fine_head(torch.cat([emb_d, coarse_input], dim=-1))

        return {
            "fine_logits":   fine_logits,
            "coarse_logits": coarse_logits,
            "embedding":     emb,
            "proj_fine":     self.proj_fine(emb),
            "proj_coarse":   self.proj_coarse(emb),
            "attn_entropy":  attn_entropy,
        }

    def get_embedding(self, x, phase=None, z=None):
        """Convenience method: returns embedding vector only."""
        return self.forward(x, phase, z)["embedding"]
