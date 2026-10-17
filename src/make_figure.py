"""Build the README figure from the committed report JSON. No hand data needed."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
INK, ACC, MUT = '#1b2430', '#c2410c', '#94a3b8'
cen = json.load(open(ROOT / 'reports/risk_blend_review.json'))['v2_reference']['score_dependent_censor_sensitivity']
ks = [0, 50, 100, 200, 400]
ys = [cen[f'censor_{k}'] for k in ks]

fig, ax = plt.subplots(figsize=(6.6, 4.1))
ax.plot(ks, ys, '-o', color=ACC, lw=2, ms=6, zorder=3)
for k, y in zip(ks, ys):
    ax.annotate(f'{y:.2f}', (k, y), textcoords='offset points', xytext=(0, 9), ha='center', fontsize=8.5, color=INK)
ax.annotate('every unlabelled pair\ncounted as innocent', xy=(0, ys[0]), xytext=(55, .745), fontsize=8.5,
            color=INK, arrowprops=dict(arrowstyle='->', color=INK, alpha=.5, lw=.9))
ax.set_xlabel('most suspicious unlabelled pairs excluded from scoring')
ax.set_ylabel('pair average precision')
ax.set_title('One model, five scores', loc='left', fontsize=11, color=INK)
ax.set_ylim(.68, 1.0)
ax.grid(axis='y', alpha=.18, lw=.7)
for s in ('top', 'right'):
    ax.spines[s].set_visible(False)
for s in ('left', 'bottom'):
    ax.spines[s].set_color(MUT)
ax.tick_params(colors=INK, labelsize=8.5)
fig.tight_layout()
fig.savefig(ROOT / 'assets/results.png', dpi=200, facecolor='white')
print('wrote assets/results.png')
