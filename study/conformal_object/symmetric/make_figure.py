"""Coverage by subgroup for the four stratifications (alpha=0.1), from summary_symmetric.csv."""
import csv, sys
import numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
out = sys.argv[1] if len(sys.argv) > 1 else 'conformal_symmetric.pdf'
S = [r for r in csv.DictReader(open('summary_symmetric.csv')) if float(r['alpha']) == 0.1]
# colours aligned with the paper's confusion-matrix magenta ramp (#fdf2f8 ... #e2216e ... #b51549), plus a neutral baseline
methods = [('global', 'Global', '#c9c4c7'), ('luna', 'LUNA strata', '#f4a3c4'),
           ('confidence', 'Confidence strata', '#e2216e'), ('joint', 'Joint (4 cells)', '#7d0f3b')]
subs = [('all', 'All'), ('luna_high', 'High\nLUNA'), ('confidence_high', 'Low\nconf.'),
        ('luna_high_confidence_low', 'High\nLUNA\nonly'), ('confidence_high_luna_low', 'Low\nconf.\nonly'),
        ('misclassified', 'Mis-\nclassified')]
get = lambda m, s, k: float(next(r for r in S if r['method'] == m and r['subset'] == s)[k])
plt.rcParams.update({'font.size': 7, 'axes.linewidth': 0.6, 'font.family': 'serif'})
fig, ax = plt.subplots(figsize=(3.45, 2.35))
x = np.arange(len(subs)) + np.array([0, 0, 0, 0, 0, 0.35]); w = 0.19
for i, (m, lab, col) in enumerate(methods):
    c = [100*get(m, s, 'coverage_mean') for s, _ in subs]; e = [100*get(m, s, 'coverage_sd') for s, _ in subs]
    ax.bar(x + (i-1.5)*w, c, w*0.9, color=col, label=lab, yerr=e, error_kw=dict(lw=0.6, capsize=1.2, ecolor='#444'))
ax.text(x[4]-0.5*w, 2, '0%\n(empty\nsets)', ha='center', va='bottom', fontsize=5, color='#444')
ax.axhline(90, ls='--', lw=0.8, color='#222'); ax.text(x[-1]+0.5, 90.5, 'target', ha='right', va='bottom', fontsize=6)
ax.axvline((x[4]+x[5])/2, lw=0.5, color='#bbb')
ax.set_xticks(x); ax.set_xticklabels([l for _, l in subs], fontsize=6)
ax.set_ylabel('True-class coverage (%)'); ax.set_ylim(0, 105)
ax.spines[['top', 'right']].set_visible(False); ax.yaxis.grid(True, lw=0.3, color='#ddd'); ax.set_axisbelow(True)
ax.legend(ncol=2, fontsize=6, frameon=False, loc='lower left', bbox_to_anchor=(0, 1.0), handlelength=1.2, columnspacing=1.0)
fig.tight_layout(pad=0.3); fig.savefig(out); fig.savefig(out.replace('.pdf', '.png'), dpi=200)
