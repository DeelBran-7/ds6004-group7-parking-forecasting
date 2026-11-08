"""Compact report figure from the two published buffered-gate summary tables.

This renderer does not fit models, choose parameters, or alter experiment data.
"""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'examples/rq2_buffered_gate'
OUTPUT = ROOT / 'reports/rq2_gate/figures/buffered_gate_summary.png'
STRATEGIES = ('C0', 'CURRENT_BUFFERED_GATE', 'FORECAST_BUFFERED_GATE')
LABELS = ('C0 current score', 'Current + buffer', 'Forecast + buffer')
COLORS = ('#96A8B5', '#087D84', '#D57A29')


def build():
    metrics = pd.read_csv(DATA / 'recommendation_metrics.csv')
    pairs = pd.read_csv(DATA / 'paired_policy_comparisons.csv')
    metrics = metrics.loc[metrics.cohort.eq('open_at_departure') & metrics.strategy.isin(STRATEGIES)]
    pairs = pairs.loc[pairs.cohort.eq('open_at_departure')]
    if len(metrics) != 6 or metrics.duplicated(['horizon_min', 'strategy']).any():
        raise ValueError('Expected the three strategies at both horizons')
    for _, rows in metrics.groupby('horizon_min'):
        if rows.matched_cases.nunique() != 1:
            raise ValueError('Strategy comparisons require identical denominators')
    rule = pairs.loc[pairs.comparison.isin(('CURRENT_BUFFERED_GATE-C0', 'FORECAST_BUFFERED_GATE-C0'))
                     & pairs.metric.eq('mean_distance_m')]
    prediction = pairs.loc[pairs.comparison.eq('FORECAST_BUFFERED_GATE-CURRENT_BUFFERED_GATE')
                           & pairs.metric.eq('mean_distance_m')]
    if len(rule) != 4 or len(prediction) != 2:
        raise ValueError('Missing paired distance comparisons')
    savings, increments = -rule.estimate.to_numpy(), -prediction.estimate.to_numpy()

    # 7.1 x 3.75 inches gives a 270-point high figure suitable for A4 text width.
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8})
    fig, axes = plt.subplots(2, 3, figsize=(7.1, 3.75))
    fig.subplots_adjust(left=.205, right=.97, top=.76, bottom=.235, wspace=.48, hspace=.50)
    fig.text(.035, .955, 'RQ2: distance savings come mainly from the ranking rule',
             fontsize=12.1, fontweight='bold', color='#183042')
    fig.text(.035, .897, 'Empirical buffers | same departures and candidates | open at departure',
             fontsize=8.3, color='#526B80')
    titles = ('Mean distance (m)', 'Full: 0 spaces', 'Near-full: <10% free')
    fields = ('mean_distance_m', 'top1_full_count', 'top1_near_full_count')
    for row, horizon in enumerate((15, 30)):
        rows = metrics.loc[metrics.horizon_min.eq(horizon)].set_index('strategy').loc[list(STRATEGIES)]
        for col, field in enumerate(fields):
            ax = axes[row, col]
            values = rows[field].to_numpy()
            ax.barh(np.arange(3), values, height=.58, color=COLORS)
            for y, value in enumerate(values):
                text = f'{value:.1f}' if col == 0 else f'{int(value)}'
                ax.annotate(text, (value, y), xytext=(4, 0), textcoords='offset points',
                            va='center', fontsize=8.2, color='#183042')
            ax.set_yticks(np.arange(3), LABELS if col == 0 else [])
            ax.set_ylim(2.55, -.55)
            ax.set_xlim(0, max(1, metrics[field].max()) * (1.22 if col == 0 else 1.35))
            ax.xaxis.set_major_locator(MaxNLocator(3, integer=col != 0))
            ax.tick_params(axis='both', length=0, labelsize=7.5)
            ax.grid(axis='x', color='#E9EDF0', lw=.7); ax.set_axisbelow(True)
            for spine in ax.spines.values(): spine.set_visible(False)
            if row == 0:
                ax.set_title(titles[col], fontsize=8.4, fontweight='bold', loc='left', pad=12)
        count = int(rows.matched_cases.iloc[0])
        axes[row, 0].text(-.87, 1.05, f'{horizon} min | n = {count:,}', transform=axes[row, 0].transAxes,
                          fontsize=8.1, fontweight='bold', color='#183042', ha='left', va='bottom')
    fig.text(.035, .13,
             f'Buffers vs C0: {savings.min():.1f}–{savings.max():.1f} m shorter. '
             f'Forecast vs current buffer: only {increments.min():.2f}–{increments.max():.2f} m shorter.',
             fontsize=8.2, fontweight='bold', color='#183042')
    fig.text(.035, .075, 'Near-full is a separate warning state, not parking failure; its count includes full arrivals.',
             fontsize=7.8, color='#8A4529')
    fig.text(.035, .028, 'Counts are point estimates. Rare or equal full-event counts do not establish safety or equivalence.',
             fontsize=7.6, color='#526B80')
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=300, facecolor='white')
    plt.close(fig)
    return OUTPUT


if __name__ == '__main__':
    print(build())
