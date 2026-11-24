# astra/losses.py
# Loss functions for ASTRA training (no instrument embedding; lower SupCon weights).
#
# Combined loss:
#   L = w_fine   * BalancedFocalLoss(fine_logits, fine_labels)
#     + w_coarse * CrossEntropy(coarse_logits, coarse_labels)
#     + w_con_f  * SupConLoss(proj_fine,   fine_labels)    [skipped on MixUp]
#     + w_con_c  * SupConLoss(proj_coarse, coarse_labels)  [skipped on MixUp]
#     + w_conf   * ConfusionPenalty(fine_logits, fine_labels)
#     - w_ent    * attn_entropy                             [entropy bonus]

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# BALANCED FOCAL LOSS
# ============================================================

class BalancedFocalLoss(nn.Module):
    """
    Focal loss with per-class gamma and optional class weighting.

    Per-class gamma allows harder focus on difficult/minority classes:
      FL_c = -alpha_c * (1 - p_t)^{gamma_c} * log(p_t)

    When use_class_weights=False, alpha_c = 1 for all classes.
    Use this when balanced sampling is already active.
    """

    def __init__(
        self,
        num_classes: int = 7,
        gamma: float = 2.0,
        gamma_per_class: torch.Tensor = None,
        beta: float = 0.9999,
        class_counts=None,
        label_smoothing: float = 0.05,
        use_class_weights: bool = True,
    ):
        super().__init__()
        self.label_smoothing = label_smoothing
        self.num_classes = num_classes

        if gamma_per_class is not None:
            self.register_buffer("gamma", gamma_per_class.float())
        else:
            self.register_buffer("gamma", torch.full((num_classes,), gamma))

        if use_class_weights and class_counts is not None:
            self.register_buffer("alpha", self._make_alpha(class_counts, beta))
        else:
            self.register_buffer("alpha", torch.ones(num_classes))

    @staticmethod
    def _make_alpha(class_counts, beta):
        counts = torch.as_tensor(class_counts, dtype=torch.float32)
        eff = 1.0 - torch.pow(beta, counts)
        w = (1.0 - beta) / (eff + 1e-8)
        return w / w.sum() * len(counts)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        C = logits.size(-1)
        y_oh = F.one_hot(targets, C).float()

        if self.label_smoothing > 0:
            y_oh = y_oh * (1 - self.label_smoothing) + self.label_smoothing / C

        log_p = F.log_softmax(logits, dim=-1)
        p     = log_p.exp()
        p_t   = (p * y_oh).sum(-1)

        gamma_t = self.gamma.to(targets.device)[targets]
        focal_w = (1 - p_t) ** gamma_t
        ce      = -(y_oh * log_p).sum(-1)
        loss    = focal_w * ce * self.alpha.to(targets.device)[targets]
        return loss.mean()


# ============================================================
# SUPERVISED CONTRASTIVE LOSS
# ============================================================

class SupConLoss(nn.Module):
    """
    Supervised Contrastive Loss (Khosla et al., 2020).
    Pulls together embeddings of the same class, pushes apart others.
    """
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        B = features.shape[0]
        device = features.device

        if B < 2:
            return torch.tensor(0.0, device=device, requires_grad=True)

        sim = features @ features.T / self.temperature
        no_diag  = ~torch.eye(B, dtype=torch.bool, device=device)
        pos_mask = (labels.unsqueeze(0) == labels.unsqueeze(1)) & no_diag

        if pos_mask.float().sum() == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)

        sim = sim - sim.max(dim=1, keepdim=True).values.detach()
        exp_sim  = torch.exp(sim) * no_diag.float()
        log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)

        n_pos = pos_mask.float().sum(dim=1).clamp(min=1)
        loss  = -(pos_mask.float() * log_prob).sum(dim=1) / n_pos

        has_pos = pos_mask.float().sum(dim=1) > 0
        if has_pos.sum() == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)
        return loss[has_pos].mean()


# ============================================================
# PAIRWISE CONFUSION PENALTY
# ============================================================

class ConfusionPenaltyLoss(nn.Module):
    """
    Extra penalty when a sample from class A is predicted as class B.

    For each pair (A, B):
      penalty += w * mean(p(B | x)^power  for all samples where y=A)

    Directly suppresses specific confusion pathways without affecting
    unrelated classes.
    """

    def __init__(self, confused_pairs: list, class_names: list, power: float = 2.0):
        """
        Args:
            confused_pairs: list of (true_class, confused_class, weight)
            class_names: ordered class names matching model output
            power: exponent on confused probability (2.0 = quadratic penalty)
        """
        super().__init__()
        self.power = power
        name2idx = {n: i for i, n in enumerate(class_names)}
        self.pairs = [
            (name2idx[t], name2idx[c], w)
            for t, c, w in confused_pairs
            if t in name2idx and c in name2idx
        ]

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if not self.pairs:
            return torch.tensor(0.0, device=logits.device, requires_grad=True)

        probs   = F.softmax(logits, dim=-1)
        penalty = torch.tensor(0.0, device=logits.device, requires_grad=True)
        n_active = 0

        for true_idx, conf_idx, w in self.pairs:
            mask = targets == true_idx
            if mask.sum() == 0:
                continue
            p_confused = probs[mask, conf_idx]
            penalty = penalty + w * (p_confused ** self.power).mean()
            n_active += 1

        return penalty / max(1, n_active)


# ============================================================
# COMBINED HIERARCHICAL LOSS
# ============================================================

def loco_novelty_loss(emb: torch.Tensor, logits: torch.Tensor, y: torch.Tensor, num_classes: int,
                      tau: float = 0.5) -> torch.Tensor:
    """Leave-one-class-out novelty loss (LOCO). One known class c, drawn at random among the classes present
    with >= 2 samples, is hidden from a LUNA-style detector built from the batch: distance to the nearest
    prototype of the OTHER classes (distance family) plus the normalised entropy of the softmax restricted to
    the other classes (uncertainty family), each standardised by its batch std. A pairwise AUROC surrogate
    asks the hidden class to score higher than the classes the detector knows. No rare object is used."""
    import math, random
    present = [k for k in range(num_classes) if int((y == k).sum()) >= 2]
    if len(present) < 3:
        return emb.sum() * 0.0
    c = random.choice(present)
    others = [k for k in present if k != c]
    mu = torch.stack([emb[y == k].mean(0) for k in others])
    d = torch.cdist(emb, mu).pow(2).min(1).values / emb.shape[1]
    keep = [k for k in range(num_classes) if k != c]
    p = F.softmax(logits[:, keep], dim=1)
    h = -(p * torch.log(p + 1e-12)).sum(1) / math.log(num_classes - 1)
    S = d / (d.detach().std() + 1e-6) + h / (h.detach().std() + 1e-6)
    i = (y == c)
    return F.softplus(-(S[i][:, None] - S[~i][None, :]) / tau).mean()


class ASTRALoss(nn.Module):
    """
    Combined training loss for ASTRA (no instrument embedding; lower SupCon weights):

    L = w_fine   * BalancedFocalLoss(fine_logits, fine_labels)
      + w_coarse * CrossEntropy(coarse_logits, coarse_labels)
      + w_con_f  * SupConLoss(proj_fine,   fine_labels)    [skipped on MixUp]
      + w_con_c  * SupConLoss(proj_coarse, coarse_labels)  [skipped on MixUp]
      + w_conf   * ConfusionPenalty(fine_logits, fine_labels)
      - w_ent    * attn_entropy

    Contrastive terms and confusion penalty are skipped on MixUp batches.
    """

    def __init__(
        self,
        fine_to_super: torch.Tensor,
        class_counts=None,
        num_fine: int = 7,
        w_fine: float = 1.5,
        w_coarse: float = 0.3,
        w_con_fine: float = 0.01,
        w_con_coarse: float = 0.02,
        w_attn_entropy: float = 0.01,
        focal_gamma: float = 2.0,
        focal_gamma_per_class: torch.Tensor = None,
        balance_beta: float = 0.9999,
        label_smoothing: float = 0.05,
        temperature: float = 0.07,
        focal_use_class_weights: bool = False,
        confused_pairs: list = None,
        class_names: list = None,
        w_confusion: float = 0.0,
        confusion_power: float = 2.0,
        w_loco: float = 0.0,
    ):
        super().__init__()
        self.register_buffer("fine_to_super", fine_to_super)
        self.w_loco = w_loco
        self.num_fine = num_fine
        self.w_fine         = w_fine
        self.w_coarse       = w_coarse
        self.w_con_fine     = w_con_fine
        self.w_con_coarse   = w_con_coarse
        self.w_attn_entropy = w_attn_entropy
        self.w_confusion    = w_confusion

        self.fine_loss = BalancedFocalLoss(
            num_classes=num_fine,
            gamma=focal_gamma,
            gamma_per_class=focal_gamma_per_class,
            beta=balance_beta,
            class_counts=class_counts,
            label_smoothing=label_smoothing,
            use_class_weights=focal_use_class_weights,
        )
        self.coarse_loss = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

        self.con_loss = SupConLoss(temperature=temperature) \
            if (w_con_fine > 0 or w_con_coarse > 0) else None

        self.confusion_penalty = ConfusionPenaltyLoss(
            confused_pairs, class_names, power=confusion_power
        ) if (w_confusion > 0 and confused_pairs and class_names) else None

    def forward(self, outputs: dict, fine_labels: torch.Tensor, is_mixup: bool = False):
        coarse_labels = self.fine_to_super[fine_labels]

        l_fine   = self.fine_loss(outputs["fine_logits"],     fine_labels)
        l_coarse = self.coarse_loss(outputs["coarse_logits"], coarse_labels)
        total    = self.w_fine * l_fine + self.w_coarse * l_coarse

        loss_dict = {
            "fine_loss":   l_fine.item(),
            "coarse_loss": l_coarse.item(),
        }

        # Contrastive — only on clean (non-MixUp) batches
        if self.con_loss is not None and not is_mixup:
            if self.w_con_fine > 0 and "proj_fine" in outputs:
                l_con_f = self.con_loss(outputs["proj_fine"], fine_labels)
                total = total + self.w_con_fine * l_con_f
                loss_dict["con_fine"] = l_con_f.item()

            if self.w_con_coarse > 0 and "proj_coarse" in outputs:
                l_con_c = self.con_loss(outputs["proj_coarse"], coarse_labels)
                total = total + self.w_con_coarse * l_con_c
                loss_dict["con_coarse"] = l_con_c.item()

        # Confusion penalty — skip on MixUp
        if self.confusion_penalty is not None and not is_mixup:
            l_conf = self.confusion_penalty(outputs["fine_logits"], fine_labels)
            total  = total + self.w_confusion * l_conf
            loss_dict["confusion_penalty"] = l_conf.item()

        # LOCO novelty term — only on clean (non-MixUp) batches, like the contrastive terms
        if self.w_loco > 0 and not is_mixup and "embedding" in outputs:
            l_loco = loco_novelty_loss(outputs["embedding"], outputs["fine_logits"], fine_labels, self.num_fine)
            total = total + self.w_loco * l_loco
            loss_dict["loco"] = l_loco.item()

        # Attention entropy bonus (encourage spread attention)
        if "attn_entropy" in outputs and self.w_attn_entropy > 0:
            attn_ent = outputs["attn_entropy"]
            total = total - self.w_attn_entropy * attn_ent
            loss_dict["attn_entropy"] = attn_ent.item()

        loss_dict["total"] = total.item()
        return total, loss_dict
