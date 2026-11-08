"""Regenerate the paper figures from the frozen CSVs in ../data (plotting only; no number is recomputed).
Sized at final print size for IEEE spconf: column = 3.39 in, text width = 7.0 in.
Usage:  python3 scripts/make_figures.py
"""
import os
import pandas as pd
import matplotlib as mpl
from matplotlib import font_manager as fm
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, '..', 'data')
F = os.path.join(HERE, '..', 'figures')
for f in ['regular', 'bold', 'italic', 'bolditalic']:
    p = f'/usr/share/texmf/fonts/opentype/public/tex-gyre/texgyretermes-{f}.otf'
    if os.path.exists(p):
        fm.fontManager.addfont(p)
mpl.rcParams.update({
    'font.family': 'serif', 'font.serif': ['TeX Gyre Termes', 'Times New Roman', 'DejaVu Serif'],
    'mathtext.fontset': 'stix', 'font.size': 7.5, 'axes.titlesize': 7.3, 'axes.labelsize': 7.3,
    'xtick.labelsize': 6.6, 'ytick.labelsize': 6.6, 'legend.fontsize': 6.3, 'axes.linewidth': 0.6,
    'xtick.major.width': 0.6, 'ytick.major.width': 0.6, 'xtick.major.size': 2.5, 'ytick.major.size': 2.5,
    'lines.linewidth': 1.15, 'pdf.fonttype': 42, 'savefig.bbox': 'tight', 'savefig.pad_inches': 0.015,
    'axes.spines.top': False, 'axes.spines.right': False, 'legend.frameon': False,
})
COL = {'123': '#8c8c8c', '2026': '#1f77b4', 'W1': '#c0392b', 'W2': '#e67e22'}
TAU, H = 3.0, 1.75


def calibration():
    """Fig. 2: attacker norm score vs. the two frozen thresholds; W1 clean honest drift."""
    k = pd.read_csv(os.path.join(D, 'fig_calibration_data.csv'), dtype={'seed': str})
    fig, ax = plt.subplots(1, 2, figsize=(3.39, 1.68), sharey=True, gridspec_kw={'wspace': 0.07})
    a = k[k.panel == 'a']
    for s, ls in [('123', '--'), ('2026', '--'), ('W1', '-'), ('W2', '-')]:
        g = a[a.seed == s].sort_values('round')
        lab = s + ('$^\\ast$' if s.startswith('W') else '')
        ax[0].plot(g['round'], g.abs_z_norm, ls, color=COL[s], label=lab, lw=0.95 if ls == '--' else 1.3)
    b = k[k.panel == 'b']
    first = True
    for c, g in b.groupby('client'):
        g = g.sort_values('round')
        if c == 'client_004':
            ax[1].plot(g['round'], g.abs_z_norm, color='#6a3d9a', lw=1.3, label='client 004', zorder=3)
        else:
            ax[1].plot(g['round'], g.abs_z_norm, color='#b5b5b5', lw=0.75, label='clients 000--003'.replace('--', '–') if first else None)
            first = False
    for axx in ax:
        axx.axhspan(H, TAU, color='#fdebd0', zorder=0, lw=0)
        axx.axhline(TAU, color='k', lw=0.8)
        axx.axhline(H, color='k', lw=0.8, ls=':')
        axx.set_xlim(20, 49.5)
        axx.set_xticks([20, 30, 40, 49])
        axx.set_xlabel('round $r$', labelpad=1)
        axx.set_ylim(0, 3.8)
        axx.set_yticks([0, 1, 1.75, 3])
        axx.set_yticklabels(['0', '1', 'ZP $h$=1.75', 'v3 $\\tau$=3'])
    ax[0].set_ylabel('norm score $z^m_{k,r}$', labelpad=1)
    ax[0].set_title('(a) BadNets attacker (client 002)', pad=2)
    ax[1].set_title('(b) W1$^\\ast$ clean run', pad=2)
    ax[0].legend(loc='lower right', ncol=2, handlelength=1.7, columnspacing=0.7, borderaxespad=0.15,
                 title='seed', title_fontsize=6.2, labelspacing=0.25)
    ax[1].legend(loc='upper left', handlelength=1.6, borderaxespad=0.15, labelspacing=0.25)
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(F, f'fig_calibration.{ext}'), dpi=300)
    plt.close(fig)


def contamination():
    """Fig. 3: conditional displacement of the inclusive median (seed 2026) and the W1 counter-example."""
    c = pd.read_csv(os.path.join(D, 'fig_contamination_data.csv'))
    fig, ax = plt.subplots(1, 3, figsize=(7.0, 1.52), gridspec_kw={'wspace': 0.30})
    for i, p in [(0, '(a)'), (2, '(c)')]:
        g = c[c.panel == p].sort_values('round')
        ax[i].fill_between(g['round'], g.band_lo, g.band_hi, color='#dedede', lw=0,
                           label='clip band $[s_{m-1}, s_m]$')
        ax[i].plot(g['round'], g['median'], '--', color='k', lw=0.9, label='inclusive median $c_r$')
        ax[i].plot(g['round'], g.client002_t, color=COL['W1'], lw=1.3, label='client 002: $t_{2,r}$')
        ax[i].set_ylabel('self-cosine $t_{k,r}$', labelpad=1)
    g = c[c.panel == '(b)']
    for cl, col, lab in [('client_004', '#2ca02c', 'client 004 (honest)'),
                         ('client_001', '#1f77b4', 'client 001 (honest)'),
                         ('client_002', COL['W1'], 'client 002 (intervened)')]:
        h = g[g.client == cl].sort_values('round')
        ax[1].plot(h['round'], h.z_t, color=col, lw=1.2, label=lab)
    ax[1].axhline(TAU, color='k', lw=0.8)
    ax[1].text(12.4, 3.08, '$\\tau=3$', fontsize=6.2, va='bottom')
    ax[1].set_ylabel('temporal score $z^t_{k,r}$', labelpad=1)
    ax[1].set_ylim(-0.15, 5.3)
    ax[0].set_title('(a) Seed 2026: client 002 crosses up', pad=2)
    ax[1].set_title('(b) Seed 2026: honest $z^t$ jumps', pad=2)
    ax[2].set_title('(c) Seed W1$^\\ast$: starts above the band', pad=2)
    for axx in ax:
        axx.axvline(20, color='k', lw=0.6, ls=':')
        axx.set_xlim(12, 35)
        axx.set_xticks([12, 20, 28, 35])
        axx.set_xlabel('round $r$', labelpad=1)
    ax[2].set_ylim(0.105, 0.41)
    ax[2].set_yticks([0.2, 0.25, 0.3, 0.35, 0.4])
    ax[2].legend(loc='lower right', handlelength=1.5, borderaxespad=0.1, labelspacing=0.2)
    ax[1].legend(loc='center right', handlelength=1.5, borderaxespad=0.1, labelspacing=0.2,
                 bbox_to_anchor=(1.02, 0.38))
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(F, f'fig_contamination.{ext}'), dpi=300)
    plt.close(fig)


if __name__ == '__main__':
    calibration()
    contamination()
    print('figures written to', os.path.abspath(F))
