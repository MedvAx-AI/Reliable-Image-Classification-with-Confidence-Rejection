"""Final figures computed from shared metrics, never used to select policies."""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.evaluate import condition_key


def plot_final_results(result: dict, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    families = [('clean', 'none'), ('gaussian_blur', 'severe'),
                ('gaussian_noise', 'severe'), ('jpeg', 'severe')]
    titles = ['Clean', 'Severe blur', 'Severe noise', 'Severe JPEG']
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.5), layout='constrained')
    for ax, (family, severity), title in zip(axes, families, titles):
        subset = [r for r in result['metrics'] if (r['corruption'], r['severity']) == (family, severity)]
        for condition, color in [('A', '#087D8B'), ('C', '#D07824')]:
            bins = next(r for r in subset if r['condition'] == condition)['reliability_bins']
            populated = [b for b in bins if b['count']]
            ax.plot([b['mean_confidence'] for b in populated], [b['accuracy'] for b in populated],
                    marker='o', markersize=3, label='raw' if condition == 'A' else 'calibrated', color=color)
        ax.plot([0, 1], [0, 1], ':', color='gray')
        ax.set(title=title + ', N=2000', xlabel='Mean confidence', xlim=(0, 1), ylim=(0, 1))
        ax.grid(alpha=.2)
    axes[0].set_ylabel('Bin accuracy (all images)')
    axes[-1].legend(fontsize=8)
    fig.savefig(out / 'reliability.png', dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 4, figsize=(13, 3.5), layout='constrained')
    for ax, (family, severity), title in zip(axes, families, titles):
        analysis = result['sensitivity'][f'{family}_{severity}']
        for name, color in [('raw', '#087D8B'), ('calibrated', '#D07824')]:
            rows = [r for r in analysis[name]['threshold_sweep'] if r['selective_risk'] is not None]
            ax.plot([r['coverage'] for r in rows], [r['selective_risk'] for r in rows], label=name, color=color)
        ax.set(title=title + ', N=2000', xlabel='Coverage', xlim=(0, 1), ylim=(0, 1))
        ax.grid(alpha=.2)
    axes[0].set_ylabel('Selective risk (errors / accepted)')
    axes[-1].legend(fontsize=8)
    fig.suptitle('Fixed threshold sensitivity - descriptive only; policies remain frozen', fontsize=11)
    fig.savefig(out / 'risk_coverage.png', dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), layout='constrained')
    keys = list(result['sensitivity'])
    labels = ['clean', 'blur .5', 'blur 1', 'blur 1.5', 'noise .03', 'noise .06', 'noise .12', 'JPEG 70', 'JPEG 40', 'JPEG 15']
    for condition, color in [('A', '#087D8B'), ('B', '#405C98'), ('C', '#D07824'), ('D', '#96434C')]:
        rows = [r for r in result['metrics'] if r['condition'] == condition]
        axes[0].plot(labels, [r['accepted_accuracy'] for r in rows], 'o-', label=condition, color=color)
        axes[1].plot(labels, [r['coverage'] for r in rows], 'o-', label=condition, color=color)
    for ax, title in zip(axes, ['Accepted accuracy', 'Coverage']):
        ax.set(title=title, ylim=(0, 1))
        ax.tick_params(axis='x', rotation=45, labelsize=8)
        ax.grid(alpha=.2)
    axes[1].legend(ncol=4)
    fig.savefig(out / 'accuracy_coverage.png', dpi=180)
    plt.close(fig)
