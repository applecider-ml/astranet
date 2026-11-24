"""Paper-style figures (watermelon palette, luna.viz / luna.style as in the paper):
1. plain confusion matrix of the seven known classes, z-shift combo12 on the test spectra (redshift and phase given);
2. novelty table: share of objects that alarm at 1, 5, 10, 15% target rate, per rare class (novae as a known class, one
   threshold per redshift stratum; numbers of study/novelty/out/novelty_by_class.json)."""
import json, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import confusion_matrix, f1_score
import luna.viz as lviz
from luna.style import paper_theme, theme_colors, save_fig, cmap, PAPER_COL
R = Path(__import__("os").environ.get("ASTRANET_REPO", Path(__file__).resolve().parents[2])); RES = R / "study/novelty/out"; CL = ["SN Ia", "SN Ib/c", "SLSN", "SN II", "AGN", "CV", "TDE"]
c = np.load(R / "cache/astranet12/test.npz"); y = c["labels"].astype(int); h = c["fine_logits"].argmax(1); cm = confusion_matrix(y, h, labels=range(7))
lviz.plot_confusion(cm, CL, "", RES, stem="confusion_known_classes", themes=(False,))
rec = np.diag(cm) / cm.sum(1); pre = np.diag(cm) / cm.sum(0).clip(min=1)
L = [f"Known classes, z-shift combo12, {len(y)} test spectra (redshift and phase given): accuracy {np.trace(cm) / cm.sum():.3f}, macro recall {rec.mean():.3f}, macro F1 {f1_score(y, h, average='macro'):.3f}.", "",
     "| true class (spectra) | " + " | ".join(CL) + " | recall | precision |", "|---|" + "---|" * 9]
for i, k in enumerate(CL): L.append(f"| {k} ({cm[i].sum()}) | " + " | ".join(str(v) for v in cm[i]) + f" | {rec[i]:.2f} | {pre[i]:.2f} |")
J = json.load(open(RES / "novelty_by_class.json")); AL = ["0.01", "0.05", "0.1", "0.15"]; rows = J["0.01"]["rows"]; n = J["0.01"]["n"]; rare = rows[8:]
order = sorted(rare, key=lambda k: -J["0.15"]["alarm_rate_by_true_class"][k]); lab = [f"{k} ({n[k]})" for k in order] + ["All rare (60)", "Novae (47)", "Known classes (1666)"]
M = np.array([[J[a]["alarm_rate_by_true_class"][k] for a in AL] for k in order] + [[J[a]["recall"] for a in AL], [J[a]["novae_flagged"] for a in AL], [J[a]["far_ordinary"] for a in AL]])
L += ["", "Novelty stage (novae known, one threshold per redshift stratum; leave-class-out, ET+LUNA): share of objects that alarm.", "", "| class (objects) | 1% | 5% | 10% | 15% |", "|---|---|---|---|---|"] + [f"| {k} | " + " | ".join(f"{v:.2f}" for v in r) + " |" for k, r in zip(lab, M)]
cc = theme_colors(False)
with paper_theme(False):
    fig, ax = plt.subplots(figsize=(PAPER_COL, PAPER_COL * 1.25), facecolor=cc["bg"]); ax.imshow(M, cmap=cmap(False), vmin=0, vmax=1, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(4): ax.text(j, i, f"{M[i, j] * 100:.0f}%", ha="center", va="center", color="white" if M[i, j] > 0.55 else "black", fontsize=7, fontweight="bold")
    for yy in (len(order) - 0.5, len(order) + 0.5): ax.axhline(yy, color=cc["border"], lw=1.2)
    ax.set_xticks(range(4)); ax.set_xticklabels(["1%", "5%", "10%", "15%"]); ax.set_yticks(range(len(lab))); ax.set_yticklabels(lab); ax.set_xlabel("Target false-alarm rate"); ax.xaxis.set_label_position("top"); ax.xaxis.tick_top(); ax.tick_params(length=0)
    save_fig(fig, RES / "novelty_table", False)
print("\n".join(L)); open(RES / "paper_style_figs.md", "w").write("\n".join(L) + "\n")
