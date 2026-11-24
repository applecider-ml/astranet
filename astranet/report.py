"""One-page report of an ASTRANet analysis, as a PNG for the SkyPortal analysis page.

    from astranet.report import make_report
    make_report(result, grids, "astranet_ZTF22abftjko.png", title="ZTF22abftjko")

`result` is the dictionary returned by astranet.skyportal.analyze and `grids` the accepted spectra on
the model grid (one row per accepted spectrum, in the order of result["results"]["spectra"]).
The page shows the spectra with the class of each one, the class probabilities of the object, the
votes of the twelve members, and the novelty p-value against the alarm rates."""
from __future__ import annotations

import numpy as np

ACCENT, DEEP, PINK, PALE, INK, MUTED = "#e11d6a", "#9f1239", "#f472b6", "#fce7f0", "#1f2933", "#6b7280"
MAX_SPECTRA = 6


def _label(row) -> str:
    when = str(row.get("observed_at") or "")[:10]; inst = row.get("instrument") or ""
    head = " ".join(x for x in (when, str(inst)) if x) or str(row.get("id") or "spectrum")
    return f"{head}: {row['predicted']} ({row['probability']:.2f})"


def make_report(result: dict, grids, path, title: str = "") -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    from .preprocess import WL_MIN, WL_MAX

    res = result["results"]; obj = res["object"]; nov = res.get("novelty") or {}
    rows = [r for r in res["spectra"] if r.get("status") == "ok"]; grids = np.atleast_2d(np.asarray(grids, dtype=float))
    classes = list(obj["probabilities"]); P = np.array([obj["probabilities"][c] for c in classes]); top = int(P.argmax())
    cmap = LinearSegmentedColormap.from_list("astranet", ["#ffffff", PALE, PINK, ACCENT, DEEP])
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": "#d1d5db", "axes.linewidth": 0.8, "xtick.color": MUTED, "ytick.color": MUTED,
                         "axes.labelcolor": INK, "text.color": INK, "font.family": "DejaVu Sans"})
    fig = plt.figure(figsize=(11.5, 6.6), facecolor="white")
    gs = fig.add_gridspec(3, 2, width_ratios=[1.55, 1.0], height_ratios=[1.25, 1.0, 0.62], left=0.055, right=0.975, top=0.86, bottom=0.085, hspace=0.62, wspace=0.2)

    z = obj.get("redshift_used"); head = f"ASTRANet  ·  {title}" if title else "ASTRANet"
    fig.text(0.055, 0.945, head, fontsize=15, fontweight="bold", color=DEEP, va="center")
    sub = f"{obj['predicted']}  {obj['probability']:.2f}   ·   {obj['n_spectra']} spectr{'um' if obj['n_spectra'] == 1 else 'a'}, {obj['spectra_agreeing']} agreeing"
    sub += f"   ·   redshift {z:.4g}" if z is not None else "   ·   no redshift"
    if obj.get("n_rejected"): sub += f"   ·   {obj['n_rejected']} rejected"
    fig.text(0.055, 0.905, sub, fontsize=10.5, color=INK, va="center")

    # spectra
    ax = fig.add_subplot(gs[:, 0]); wl = np.linspace(WL_MIN, WL_MAX, grids.shape[1]); show = list(range(len(rows)))[:MAX_SPECTRA]
    for k, i in enumerate(show):
        g = grids[i]; lo, hi = np.nanpercentile(g, [1, 99]); g = (g - lo) / (hi - lo + 1e-12); off = (len(show) - 1 - k) * 1.15
        same = rows[i]["predicted"] == obj["predicted"]
        ax.plot(wl, g + off, lw=0.8, color=ACCENT if same else MUTED)
        ax.text(wl[-1] - 40, off + 1.02, _label(rows[i]), ha="right", va="bottom", fontsize=8, color=DEEP if same else MUTED)
    ax.set_xlim(WL_MIN, WL_MAX); ax.set_ylim(-0.1, 1.15 * len(show) + 0.2); ax.set_yticks([]); ax.set_xlabel("observed wavelength (Å)")
    more = f" (first {MAX_SPECTRA} of {len(rows)})" if len(rows) > MAX_SPECTRA else ""
    ax.set_title("Spectra on the model grid, scaled and offset" + more, loc="left", fontsize=9.5, color=INK)
    for s in ("top", "right", "left"): ax.spines[s].set_visible(False)

    # object probabilities
    ax = fig.add_subplot(gs[0, 1]); y = np.arange(len(classes))[::-1]
    ax.barh(y, P, color=[ACCENT if i == top else PINK for i in range(len(classes))], height=0.68)
    for yi, p in zip(y, P): ax.text(min(p + 0.015, 0.93), yi, f"{p:.2f}", va="center", fontsize=8, color=INK)
    ax.set_yticks(y); ax.set_yticklabels(classes); ax.set_xlim(0, 1); ax.set_title("Class probability of the object", loc="left", fontsize=9.5, color=INK)
    ax.tick_params(length=0)
    for s in ("top", "right"): ax.spines[s].set_visible(False)

    # members voting for each class, per spectrum
    ax = fig.add_subplot(gs[1, 1]); nm = rows[0].get("n_members") or 1
    V = np.array([[r.get("member_votes", {}).get(c, 0) for c in classes] for r in rows[:MAX_SPECTRA]], dtype=float)
    ax.imshow(V, cmap=cmap, vmin=0, vmax=nm, aspect="auto")
    for i in range(V.shape[0]):
        for j in range(V.shape[1]):
            if V[i, j] > 0: ax.text(j, i, f"{int(V[i, j])}", ha="center", va="center", fontsize=8, color="white" if V[i, j] > 0.55 * nm else INK)
    ax.set_xticks(range(len(classes))); ax.set_xticklabels(classes); ax.set_yticks(range(V.shape[0])); ax.set_yticklabels([str(r.get("observed_at") or r.get("id") or k + 1)[:10] for k, r in enumerate(rows[:MAX_SPECTRA])], fontsize=7.5)
    ax.set_title(f"Members voting for each class (of {nm})", loc="left", fontsize=9.5, color=INK); ax.tick_params(length=0)

    # novelty
    ax = fig.add_subplot(gs[2, 1]); ax.set_title("Novelty against ordinary objects", loc="left", fontsize=9.5, color=INK)
    if nov.get("status") == "ok":
        p = max(float(nov["p_value"]), 1e-3); rates = [0.15, 0.10, 0.05, 0.01]; edges = [1.0] + rates + [1e-3]; shades = ["#f9fafb", PALE, "#fbcfe1", PINK, ACCENT]
        for a, b, c in zip(edges[:-1], edges[1:], shades): ax.axvspan(b, a, color=c, lw=0)
        for r in rates: ax.text(r, 1.04, f"{int(round(r * 100))}%", ha="center", va="bottom", fontsize=7.5, color=MUTED, transform=ax.get_xaxis_transform())
        ax.plot([p], [0.5], marker="v", ms=11, color=DEEP, mec="white", mew=1.2, clip_on=False)
        ax.set_xscale("log"); ax.set_xlim(1.0, 1e-3); ax.set_ylim(0, 1); ax.set_yticks([]); ax.set_xlabel("p-value (alarm rates marked above)")
        word = f"alarm at {nov['alarm_rate']}" if nov.get("alarm_rate") else "no alarm"
        strat = "redshift known" if nov.get("stratum") == "redshift" else "redshift unknown"
        fig.text(0.975, 0.905, f"novelty p = {nov['p_value']:.3f}   ·   {word}", fontsize=10.5, color=DEEP if nov.get("alarm_rate") else INK, ha="right", va="center", fontweight="bold" if nov.get("alarm_rate") else "normal")
        ax.text(0.995, 0.08, f"{strat}, {nov.get('n_calibration')} calibration objects", transform=ax.transAxes, ha="right", va="bottom", fontsize=7.5, color=INK)
    else:
        ax.axis("off"); ax.text(0.0, 0.5, f"novelty stage {nov.get('status', 'off')}" + (f": {nov['reason']}" if nov.get("reason") else ""), va="center", fontsize=9, color=MUTED)
    fig.savefig(str(path), dpi=130); plt.close(fig)
    return str(path)
