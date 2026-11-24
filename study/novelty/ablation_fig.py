"""Matched ExtraTrees ablation figure (base against base + LUNA) in the paper's magenta ramp.
Left: leave-class-out over 60 rare objects. Right: Tier 1 withheld (35 objects). Budgets 1, 5, 10, 15%. -> study/novelty/out/ad_luna_ablation.pdf"""
import json, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from pathlib import Path
R = str(Path(__import__("os").environ.get("ASTRANET_REPO", Path(__file__).resolve().parents[2])) / "study/novelty/out") + "/"; J = json.load(open(R + "paper_numbers.json")); AL = ["0.01", "0.05", "0.1", "0.15"]
plt.rcParams.update({"font.size": 8, "axes.linewidth": 0.6, "font.family": "serif"}); fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.6), sharey=True); x = np.arange(4); w = 0.36
for ax, (mode, title) in zip(axes, (("lco", "Leave-class-out (60 objects)"), ("Tier1", "Tier 1 withheld (35 objects)"))):
    for i, (h, lab, col) in enumerate((("et", "Base", "#f4a3c4"), ("et_plus_luna", "Base + LUNA", "#7d0f3b"))):
        v = [J[f"{mode}/{h}/{a}"] for a in AL]; r = [100 * q["recall"] for q in v]
        ax.bar(x + (i - .5) * w, r, w * .92, color=col, label=lab, yerr=[100 * q["sd"] for q in v], error_kw=dict(lw=0.6, capsize=1.5, ecolor="#444"))
        for k, q in enumerate(v): ax.text(x[k] + (i - .5) * w, 3, f"{100 * q['far']:.1f}", ha="center", va="bottom", fontsize=6, color="white" if i else "#333", rotation=90)
    for k, a in enumerate(AL):
        p = J[f"paired_luna/{mode}/{a}"]; ax.text(x[k], 101, f"$+{p['delta']:.1f}$\n[{p['lo']:.1f}, {p['hi']:.1f}]", ha="center", va="bottom", fontsize=6)
    ax.set_xticks(x); ax.set_xticklabels(["1%", "5%", "10%", "15%"]); ax.set_xlabel("Nominal per-object false-alarm budget"); ax.set_title(title, fontsize=8, pad=22); ax.set_ylim(0, 100)
    ax.spines[["top", "right"]].set_visible(False); ax.yaxis.grid(True, lw=0.3, color="#ddd"); ax.set_axisbelow(True)
axes[0].set_ylabel("Rare objects recovered (%)"); axes[0].legend(fontsize=7, frameon=False, loc="upper left")
fig.tight_layout(pad=0.4); fig.savefig(R + "ad_luna_ablation.pdf"); fig.savefig(R + "ad_luna_ablation.png", dpi=200); print("ABLATION FIG DONE")
