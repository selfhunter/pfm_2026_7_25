#!/usr/bin/env python3
"""
PFM Comparison: Baseline (fixed-interval global reinit) vs Adaptive
(deformation-triggered local reinit + vorticity-guided density).

Generates publication-quality comparison charts from saved metrics.
Run this AFTER running both baseline and adaptive simulations:
    python run_2D_baseline.py --steps 200
    python run_2D_adaptive.py --steps 200
    python plot_comparison.py
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import os
import sys
import glob

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'legend.fontsize': 10,
    'figure.dpi': 150,
    'savefig.dpi': 150,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
})

# Color scheme
C_BASELINE = '#3B82F6'    # blue
C_ADAPTIVE = '#EF4444'    # red
C_DIFF = '#10B981'        # green
C_THRESHOLD = '#F59E0B'   # amber

OUTPUT_DIR = 'comparison_plots'


def find_metrics():
    """Find baseline and adaptive metrics files."""
    baseline_dir = 'logs/2D_baseline_1024x256_reinit-20-8/metrics/metrics.npz'
    adaptive_dir = 'logs/2D_adaptive_1024x256/metrics/metrics.npz'

    baseline_path = baseline_dir if os.path.exists(baseline_dir) else None
    adaptive_path = adaptive_dir if os.path.exists(adaptive_dir) else None

    # Try to find by glob
    if baseline_path is None:
        candidates = glob.glob('logs/*baseline*/metrics/metrics.npz')
        if candidates:
            baseline_path = candidates[0]
    if adaptive_path is None:
        candidates = glob.glob('logs/*adaptive*/metrics/metrics.npz')
        if candidates:
            adaptive_path = candidates[0]

    return baseline_path, adaptive_path


def load_metrics(path, label):
    """Load metrics from .npz file."""
    if path is None or not os.path.exists(path):
        print(f"[WARNING] Metrics file not found: {path}")
        return None
    data = np.load(path, allow_pickle=True)
    metrics = {k: data[k] for k in data.files}
    print(f"[LOAD] {label}: {len(metrics.get('frame_times', []))} steps")
    return metrics


def fig_save(fig, name):
    """Save figure to output directory."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, name)
    fig.savefig(path)
    print(f"[SAVE] {path}")
    plt.close(fig)


# ===================================================================
# Chart 1: Frame Time Comparison
# ===================================================================
def plot_frame_time(baseline, adaptive):
    """Per-step wall-clock time comparison."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 1a: Time series
    ax = axes[0]
    steps_b = np.arange(len(baseline['frame_times']))
    steps_a = np.arange(len(adaptive['frame_times']))

    ax.plot(steps_b, baseline['frame_times'], color=C_BASELINE, alpha=0.7,
            linewidth=0.8, label='Baseline (global reinit)')
    ax.plot(steps_a, adaptive['frame_times'], color=C_ADAPTIVE, alpha=0.7,
            linewidth=0.8, label='Adaptive (local reinit)')

    # Moving average for clarity
    window = max(5, len(steps_a) // 20)
    if len(steps_b) >= window:
        ma_b = np.convolve(baseline['frame_times'], np.ones(window)/window, mode='valid')
        ma_a = np.convolve(adaptive['frame_times'], np.ones(window)/window, mode='valid')
        ax.plot(steps_b[window-1:], ma_b, color=C_BASELINE, linewidth=2.0,
                label=f'Baseline (MA-{window})')
        ax.plot(steps_a[window-1:], ma_a, color=C_ADAPTIVE, linewidth=2.0,
                label=f'Adaptive (MA-{window})')

    ax.set_xlabel('Step')
    ax.set_ylabel('Frame Time (s)')
    ax.set_title('Per-Step Execution Time')
    ax.legend(loc='upper right', framealpha=0.8)
    ax.grid(True, alpha=0.3)

    # 1b: Distribution comparison
    ax = axes[1]
    times_b = baseline['frame_times']
    times_a = adaptive['frame_times']

    bins = np.linspace(min(times_b.min(), times_a.min()),
                        max(times_b.max(), times_a.max()), 30)
    ax.hist(times_b, bins=bins, alpha=0.5, color=C_BASELINE, label='Baseline', density=True)
    ax.hist(times_a, bins=bins, alpha=0.5, color=C_ADAPTIVE, label='Adaptive', density=True)

    # Add mean lines
    ax.axvline(times_b.mean(), color=C_BASELINE, linestyle='--', linewidth=2,
               label=f'Baseline mean: {times_b.mean():.3f}s')
    ax.axvline(times_a.mean(), color=C_ADAPTIVE, linestyle='--', linewidth=2,
               label=f'Adaptive mean: {times_a.mean():.3f}s')

    speedup = (times_b.mean() - times_a.mean()) / times_b.mean() * 100
    ax.set_xlabel('Frame Time (s)')
    ax.set_ylabel('Density')
    ax.set_title(f'Time Distribution (Speedup: {speedup:.1f}%)')
    ax.legend(loc='upper right', framealpha=0.8)
    ax.grid(True, alpha=0.3)

    fig.suptitle('Computational Performance: Baseline vs Adaptive PFM', fontsize=14, y=1.01)
    fig_save(fig, '01_frame_time_comparison.png')

    return speedup


# ===================================================================
# Chart 2: Kinetic Energy Preservation
# ===================================================================
def plot_kinetic_energy(baseline, adaptive):
    """Kinetic energy decay over time — KEY accuracy metric."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 2a: Absolute energy
    ax = axes[0]
    ke_b = baseline['kinetic_energy']
    ke_a = adaptive['kinetic_energy']
    steps_b = np.arange(len(ke_b))
    steps_a = np.arange(len(ke_a))

    ax.plot(steps_b, ke_b, color=C_BASELINE, linewidth=1.5, alpha=0.8, label='Baseline')
    ax.plot(steps_a, ke_a, color=C_ADAPTIVE, linewidth=1.5, alpha=0.8, label='Adaptive')
    ax.set_xlabel('Step')
    ax.set_ylabel('Kinetic Energy')
    ax.set_title('Kinetic Energy vs Time')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    # 2b: Energy relative to initial (preservation ratio)
    ax = axes[1]
    ke_b_norm = ke_b / ke_b[0] if ke_b[0] > 0 else ke_b
    ke_a_norm = ke_a / ke_a[0] if ke_a[0] > 0 else ke_a

    ax.plot(steps_b, ke_b_norm * 100, color=C_BASELINE, linewidth=1.5, alpha=0.8,
            label=f'Baseline (final: {ke_b_norm[-1]*100:.1f}%)')
    ax.plot(steps_a, ke_a_norm * 100, color=C_ADAPTIVE, linewidth=1.5, alpha=0.8,
            label=f'Adaptive (final: {ke_a_norm[-1]*100:.1f}%)')

    # Highlight the improvement
    improvement = (ke_a_norm[-1] - ke_b_norm[-1]) * 100  # percentage points
    ax.fill_between(steps_a, ke_b_norm[:len(ke_a_norm)] * 100, ke_a_norm * 100,
                     alpha=0.15, color=C_DIFF, label=f'Improvement: {improvement:.1f}pp')

    ax.set_xlabel('Step')
    ax.set_ylabel('Energy Retention (%)')
    ax.set_title(f'Kinetic Energy Preservation (Δ = {improvement:.1f} pp)')
    ax.legend(loc='lower left')
    ax.grid(True, alpha=0.3)
    ax.axhline(100, color='gray', linestyle=':', alpha=0.5)

    fig.suptitle('Energy Conservation: Less Numerical Dissipation with Adaptive Reinit', fontsize=14, y=1.01)
    fig_save(fig, '02_kinetic_energy_comparison.png')

    return improvement


# ===================================================================
# Chart 3: Enstrophy Preservation
# ===================================================================
def plot_enstrophy(baseline, adaptive):
    """Enstrophy (vorticity squared) over time — KEY vortex preservation metric."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ens_b = baseline['enstrophy']
    ens_a = adaptive['enstrophy']
    steps_b = np.arange(len(ens_b))
    steps_a = np.arange(len(ens_a))

    # 3a: Absolute enstrophy
    ax = axes[0]
    ax.plot(steps_b, ens_b, color=C_BASELINE, linewidth=1.5, alpha=0.8, label='Baseline')
    ax.plot(steps_a, ens_a, color=C_ADAPTIVE, linewidth=1.5, alpha=0.8, label='Adaptive')
    ax.set_xlabel('Step')
    ax.set_ylabel('Enstrophy (∫ω²/2 dA)')
    ax.set_title('Enstrophy vs Time')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    # 3b: Normalized enstrophy
    ax = axes[1]
    ens_b_norm = ens_b / ens_b[0] if ens_b[0] > 0 else ens_b
    ens_a_norm = ens_a / ens_a[0] if ens_a[0] > 0 else ens_a

    ax.plot(steps_b, ens_b_norm * 100, color=C_BASELINE, linewidth=1.5, alpha=0.8,
            label=f'Baseline (final: {ens_b_norm[-1]*100:.1f}%)')
    ax.plot(steps_a, ens_a_norm * 100, color=C_ADAPTIVE, linewidth=1.5, alpha=0.8,
            label=f'Adaptive (final: {ens_a_norm[-1]*100:.1f}%)')

    improvement = (ens_a_norm[-1] - ens_b_norm[-1]) * 100
    ax.fill_between(steps_a, ens_b_norm[:len(ens_a_norm)] * 100, ens_a_norm * 100,
                     alpha=0.15, color=C_DIFF, label=f'Improvement: {improvement:.1f}pp')

    ax.set_xlabel('Step')
    ax.set_ylabel('Enstrophy Retention (%)')
    ax.set_title(f'Vorticity Preservation (Δ = {improvement:.1f} pp)')
    ax.legend(loc='lower left')
    ax.grid(True, alpha=0.3)
    ax.axhline(100, color='gray', linestyle=':', alpha=0.5)

    fig.suptitle('Vortex Structure Preservation: Higher Enstrophy = Better Eddy Capture', fontsize=14, y=1.01)
    fig_save(fig, '03_enstrophy_comparison.png')

    return improvement


# ===================================================================
# Chart 4: Adaptive-only metrics
# ===================================================================
def plot_adaptive_metrics(adaptive):
    """Adaptive PFM specific metrics: active particles, marked cells, deformation."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    n_steps = len(adaptive['frame_times'])
    steps = np.arange(n_steps)

    # 4a: Active particle ratio
    ax = axes[0, 0]
    total = adaptive.get('total_particles', adaptive['active_particles'].max())
    active_ratio = adaptive['active_particles'] / total * 100
    ax.fill_between(steps, 0, active_ratio, color=C_ADAPTIVE, alpha=0.3)
    ax.plot(steps, active_ratio, color=C_ADAPTIVE, linewidth=1.5)
    ax.axhline(100, color='gray', linestyle=':', alpha=0.5, label='Baseline (100%)')
    ax.set_xlabel('Step')
    ax.set_ylabel('Active Particles (%)')
    ax.set_title(f'Adaptive Particle Density\n(Avg: {active_ratio.mean():.1f}%)')
    ax.legend(loc='lower left')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 105)

    # 4b: Marked cells for reinit
    ax = axes[0, 1]
    total_cells = adaptive.get('total_cells', res_x * res_y if 'res_x' in dir() else 1024*256)
    marked_ratio = adaptive['marked_cells'] / total_cells * 100
    ax.fill_between(steps, 0, marked_ratio, color=C_THRESHOLD, alpha=0.3)
    ax.plot(steps, marked_ratio, color=C_THRESHOLD, linewidth=1.5)
    ax.axhline(100, color='gray', linestyle=':', alpha=0.5, label='Global reinit (100%)')
    ax.set_xlabel('Step')
    ax.set_ylabel('Cells Marked (%)')
    ax.set_title(f'Local Reinit Coverage\n(Avg: {marked_ratio.mean():.1f}%)')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    # 4c: Average deformation factor
    ax = axes[1, 0]
    ax.plot(steps, adaptive['avg_deformation'], color=C_ADAPTIVE, linewidth=1.5)
    threshold = adaptive.get('deformation_threshold', 0.3)
    ax.axhline(threshold, color=C_THRESHOLD, linestyle='--', linewidth=2,
               label=f'Reinit threshold: {threshold}')
    ax.set_xlabel('Step')
    ax.set_ylabel('Avg ||T-I||_F')
    ax.set_title('Average Particle Deformation')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    # 4d: High-density cell count
    ax = axes[1, 1]
    high_ratio = adaptive.get('cell_density_high', np.zeros(n_steps)) / total_cells * 100
    ax.fill_between(steps, 0, high_ratio, color=C_DIFF, alpha=0.3)
    ax.plot(steps, high_ratio, color=C_DIFF, linewidth=1.5)
    ax.set_xlabel('Step')
    ax.set_ylabel('High-Density Cells (%)')
    ax.set_title(f'Vortex-Region (16/cell) Coverage\n(Avg: {high_ratio.mean():.1f}%)')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, max(high_ratio.max() * 1.2, 10))

    fig.suptitle('Adaptive PFM: Internal Metrics', fontsize=14, y=1.01)
    fig_save(fig, '04_adaptive_internal_metrics.png')


# ===================================================================
# Chart 5: Cumulative Performance Summary
# ===================================================================
def plot_summary(baseline, adaptive):
    """Summary bar chart comparing key metrics."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 5a: Total compute time
    ax = axes[0]
    total_b = baseline['frame_times'].sum()
    total_a = adaptive['frame_times'].sum()
    mean_b = baseline['frame_times'].mean()
    mean_a = adaptive['frame_times'].mean()

    bars = ax.bar(['Baseline', 'Adaptive'], [total_b, total_a],
                  color=[C_BASELINE, C_ADAPTIVE], alpha=0.7, edgecolor='black', linewidth=0.5)
    # Add value labels
    for bar, val in zip(bars, [total_b, total_a]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + total_b*0.01,
                f'{val:.1f}s\n({val/total_b*100:.1f}%)', ha='center', fontsize=10)
    ax.set_ylabel('Total Time (s)')
    ax.set_title(f'Total Compute Time\nSpeedup: {(1-total_a/total_b)*100:.1f}%')
    ax.grid(True, alpha=0.3, axis='y')

    # 5b: Energy preservation (final / initial)
    ax = axes[1]
    ke_b_ratio = baseline['kinetic_energy'][-1] / baseline['kinetic_energy'][0] * 100
    ke_a_ratio = adaptive['kinetic_energy'][-1] / adaptive['kinetic_energy'][0] * 100

    bars = ax.bar(['Baseline', 'Adaptive'], [ke_b_ratio, ke_a_ratio],
                  color=[C_BASELINE, C_ADAPTIVE], alpha=0.7, edgecolor='black', linewidth=0.5)
    for bar, val in zip(bars, [ke_b_ratio, ke_a_ratio]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{val:.1f}%', ha='center', fontsize=11)
    ax.set_ylabel('Energy Retention (%)')
    ax.set_title(f'Kinetic Energy Preservation\nΔ = {ke_a_ratio - ke_b_ratio:.1f} pp')
    ax.grid(True, alpha=0.3, axis='y')

    # 5c: Enstrophy preservation
    ax = axes[2]
    ens_b_ratio = baseline['enstrophy'][-1] / baseline['enstrophy'][0] * 100
    ens_a_ratio = adaptive['enstrophy'][-1] / adaptive['enstrophy'][0] * 100

    bars = ax.bar(['Baseline', 'Adaptive'], [ens_b_ratio, ens_a_ratio],
                  color=[C_BASELINE, C_ADAPTIVE], alpha=0.7, edgecolor='black', linewidth=0.5)
    for bar, val in zip(bars, [ens_b_ratio, ens_a_ratio]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{val:.1f}%', ha='center', fontsize=11)
    ax.set_ylabel('Enstrophy Retention (%)')
    ax.set_title(f'Vorticity Preservation\nΔ = {ens_a_ratio - ens_b_ratio:.1f} pp')
    ax.grid(True, alpha=0.3, axis='y')

    fig.suptitle('PFM Optimization: Performance & Accuracy Summary', fontsize=14, y=1.02)
    fig_save(fig, '05_summary_comparison.png')


# ===================================================================
# Chart 6: Step-by-step Speedup
# ===================================================================
def plot_speedup_timeline(baseline, adaptive):
    """Per-step speedup of adaptive vs baseline."""
    fig, ax = plt.subplots(figsize=(14, 5))

    min_len = min(len(baseline['frame_times']), len(adaptive['frame_times']))
    steps = np.arange(min_len)
    times_b = baseline['frame_times'][:min_len]
    times_a = adaptive['frame_times'][:min_len]

    speedup = (times_b - times_a) / times_b * 100  # positive = faster

    # Color-code: green when faster, red when slower
    colors = [C_DIFF if s > 0 else C_ADAPTIVE for s in speedup]
    ax.bar(steps, speedup, color=colors, alpha=0.7, width=1.0)
    ax.axhline(0, color='black', linewidth=0.5)
    ax.axhline(np.mean(speedup), color=C_THRESHOLD, linestyle='--', linewidth=2,
               label=f'Mean speedup: {np.mean(speedup):.1f}%')

    ax.set_xlabel('Step')
    ax.set_ylabel('Speedup (%)')
    ax.set_title(f'Per-Step Speedup (Adaptive vs Baseline)')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    fig_save(fig, '06_speedup_timeline.png')


# ===================================================================
# Chart 7: Reinit Pattern Comparison
# ===================================================================
def plot_reinit_pattern(baseline, adaptive):
    """Visualize the difference in reinit patterns."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    min_len = min(len(baseline['frame_times']), len(adaptive['frame_times']))
    steps = np.arange(min_len)

    # 7a: Baseline — fixed interval global reinit
    ax = axes[0]
    reinit_every = baseline.get('reinit_every', 20)
    reinit_steps_b = np.arange(0, min_len, reinit_every)
    y_vals = np.ones(len(reinit_steps_b)) * 100
    ax.vlines(reinit_steps_b, 0, 100, colors=C_BASELINE, linewidth=2, alpha=0.7, label='Global reinit')
    ax.scatter(reinit_steps_b, y_vals, color=C_BASELINE, s=40, zorder=5)
    ax.set_xlabel('Step')
    ax.set_ylabel('Cells Reinitialized (%)')
    ax.set_title(f'Baseline: Fixed-Interval Global Reinit\n(every {reinit_every} steps → 100% cells)')
    ax.set_ylim(-5, 120)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    # 7b: Adaptive — variable, local reinit
    ax = axes[1]
    total_cells = adaptive.get('total_cells', 1024*256)
    marked_ratio = adaptive['marked_cells'][:min_len] / total_cells * 100

    # Mark where reinit actually happened
    reinit_type = adaptive.get('reinit_type', np.zeros(min_len))
    local_reinit_mask = reinit_type[:min_len] == 2

    ax.plot(steps, marked_ratio, color=C_ADAPTIVE, linewidth=1, alpha=0.5, label='Marked cells')
    ax.scatter(steps[local_reinit_mask], marked_ratio[local_reinit_mask],
               color=C_ADAPTIVE, s=30, zorder=5, label='Local reinit triggered')

    ax.set_xlabel('Step')
    ax.set_ylabel('Cells Reinitialized (%)')
    avg_marked = marked_ratio[local_reinit_mask].mean() if local_reinit_mask.any() else 0
    ax.set_title(f'Adaptive: Deformation-Triggered Local Reinit\n(Avg {avg_marked:.1f}% cells per reinit event)')
    ax.set_ylim(-5, 120)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    fig.suptitle('Reinitialization Pattern: Global vs Adaptive Local', fontsize=14, y=1.01)
    fig_save(fig, '07_reinit_pattern_comparison.png')


# ===================================================================
# Chart 8: Efficiency-Accuracy Pareto
# ===================================================================
def plot_pareto(baseline, adaptive):
    """Pareto-like visualization: accuracy vs cost."""
    fig, ax = plt.subplots(figsize=(8, 7))

    # Normalize: baseline at (1, 1)
    baseline_time = baseline['frame_times'].mean()
    adaptive_time = adaptive['frame_times'].mean()
    baseline_ke = baseline['kinetic_energy'][-1] / baseline['kinetic_energy'][0]
    adaptive_ke = adaptive['kinetic_energy'][-1] / adaptive['kinetic_energy'][0]
    baseline_ens = baseline['enstrophy'][-1] / baseline['enstrophy'][0]
    adaptive_ens = adaptive['enstrophy'][-1] / adaptive['enstrophy'][0]

    # Accuracy = average of KE and enstrophy preservation
    baseline_acc = (baseline_ke + baseline_ens) / 2
    adaptive_acc = (adaptive_ke + adaptive_ens) / 2

    # Normalized: baseline = (1, 1)
    norm_time_b = 1.0
    norm_time_a = adaptive_time / baseline_time
    norm_acc_b = 1.0
    norm_acc_a = adaptive_acc / baseline_acc

    # Draw
    ax.scatter([norm_time_b], [norm_acc_b], color=C_BASELINE, s=200, zorder=5,
               label='Baseline (reference)', edgecolors='black', linewidth=1)
    ax.scatter([norm_time_a], [norm_acc_a], color=C_ADAPTIVE, s=200, zorder=5,
               label='Adaptive', edgecolors='black', linewidth=1)

    # Arrow showing improvement direction
    ax.annotate('', xy=(norm_time_a, norm_acc_a), xytext=(norm_time_b, norm_acc_b),
                arrowprops=dict(arrowstyle='->', color=C_DIFF, lw=2.5,
                               connectionstyle='arc3,rad=-0.2'))

    # Pareto frontier hint
    ax.axhline(1.0, color='gray', linestyle=':', alpha=0.5)
    ax.axvline(1.0, color='gray', linestyle=':', alpha=0.5)

    # Quadrant labels
    ax.text(0.5, 1.02, 'Better Accuracy\nSlower', ha='center', fontsize=9, alpha=0.5, transform=ax.transAxes)
    ax.text(1.02, 1.02, 'Better Accuracy\nFaster (IDEAL)', ha='left', fontsize=9, alpha=0.7,
            color=C_DIFF, fontweight='bold', transform=ax.transAxes)
    ax.text(0.5, -0.02, 'Worse Accuracy\nSlower (WORST)', ha='center', fontsize=9, alpha=0.5, transform=ax.transAxes)
    ax.text(1.02, -0.02, 'Worse Accuracy\nFaster', ha='left', fontsize=9, alpha=0.5, transform=ax.transAxes)

    ax.set_xlabel('Normalized Time (↓ better)')
    ax.set_ylabel('Normalized Accuracy (↑ better)')
    ax.set_title('Efficiency-Accuracy Trade-off')
    ax.legend(loc='lower left')
    ax.grid(True, alpha=0.3)

    # Set limits with some padding
    ax.set_xlim(0.3, 1.2)
    ax.set_ylim(0.95, 1.15)

    fig_save(fig, '08_pareto_efficiency.png')


# ===================================================================
# Generate synthetic data for demonstration (when no real data exists)
# ===================================================================
def generate_demo_data():
    """Generate plausible demo metrics for when real simulation data
    isn't available yet. This allows the plotting code to be tested."""
    np.random.seed(42)
    n_steps = 200
    reinit_every = 20

    # Baseline: regular pattern
    baseline = {
        'frame_times': np.ones(n_steps) * 1.5 + np.random.randn(n_steps) * 0.1,
        'kinetic_energy': 0.05 * np.exp(-np.arange(n_steps) * 0.002) + np.random.randn(n_steps) * 0.0005,
        'enstrophy': 0.1 * np.exp(-np.arange(n_steps) * 0.003) + np.random.randn(n_steps) * 0.001,
        'reinit_every': reinit_every,
        'reinit_every_grad_m': 8,
        'total_cells': 1024 * 256,
        'total_particles': 1024 * 256 * 16,
        'exp_name': 'baseline_demo',
        'mode': 'baseline',
    }
    # Add reinit spikes to baseline frame times
    for step in range(0, n_steps, reinit_every):
        baseline['frame_times'][step] *= 1.5
    baseline['kinetic_energy'][0] = baseline['kinetic_energy'][0] * 1.5
    baseline['enstrophy'][0] = baseline['enstrophy'][0] * 1.5
    baseline['kinetic_energy'] = np.abs(baseline['kinetic_energy'])
    baseline['enstrophy'] = np.abs(baseline['enstrophy'])

    # Adaptive: faster steps, better energy preservation
    adaptive = {
        'frame_times': np.ones(n_steps) * 1.0 + np.random.randn(n_steps) * 0.08,
        'kinetic_energy': baseline['kinetic_energy'] * 1.08 - np.arange(n_steps) * 0.00002,
        'enstrophy': baseline['enstrophy'] * 1.12 - np.arange(n_steps) * 0.00003,
        'active_particles': (1024 * 256 * 16) * (0.55 + 0.15 * np.sin(np.arange(n_steps) * 0.05)),
        'marked_cells': (1024 * 256 * 0.15 * np.ones(n_steps)).astype(int),
        'avg_deformation': 0.25 + 0.1 * np.sin(np.arange(n_steps) * 0.08),
        'max_deformation': 0.5 + 0.2 * np.sin(np.arange(n_steps) * 0.08),
        'reinit_type': np.array([2 if i % 5 == 0 else 0 for i in range(n_steps)]),
        'cell_density_high': (1024 * 256 * 0.2 * np.ones(n_steps)).astype(int),
        'total_cells': 1024 * 256,
        'total_particles': 1024 * 256 * 16,
        'deformation_threshold': 0.3,
        'exp_name': 'adaptive_demo',
        'mode': 'adaptive',
    }
    adaptive['kinetic_energy'] = np.abs(adaptive['kinetic_energy'])
    adaptive['enstrophy'] = np.abs(adaptive['enstrophy'])

    return baseline, adaptive


# ===================================================================
# Main
# ===================================================================
def main():
    print("=" * 60)
    print("PFM Optimization Comparison Plotter")
    print("=" * 60)

    baseline_path, adaptive_path = find_metrics()

    use_demo = False
    if baseline_path is None or adaptive_path is None:
        print("\n[MISSING] Real metrics not found. Using DEMO data for visualization.")
        print("To generate real data, run:")
        print("  python run_2D_baseline.py --steps 200")
        print("  python run_2D_adaptive.py --steps 200")
        print()
        baseline, adaptive = generate_demo_data()
        use_demo = True
    else:
        baseline = load_metrics(baseline_path, 'Baseline')
        adaptive = load_metrics(adaptive_path, 'Adaptive')

        if baseline is None or adaptive is None:
            print("[ERROR] Failed to load metrics.")
            sys.exit(1)

    print(f"\n[INFO] Baseline: {baseline.get('exp_name', 'unknown')}, "
          f"{len(baseline['frame_times'])} steps")
    print(f"[INFO] Adaptive: {adaptive.get('exp_name', 'unknown')}, "
          f"{len(adaptive['frame_times'])} steps")
    if use_demo:
        print("[INFO] *** USING DEMO DATA ***")

    # Generate all comparison charts
    print("\n[PLOT] Generating comparison charts...")

    speedup = plot_frame_time(baseline, adaptive)
    ke_improvement = plot_kinetic_energy(baseline, adaptive)
    ens_improvement = plot_enstrophy(baseline, adaptive)
    plot_adaptive_metrics(adaptive)
    plot_summary(baseline, adaptive)
    plot_speedup_timeline(baseline, adaptive)
    plot_reinit_pattern(baseline, adaptive)
    plot_pareto(baseline, adaptive)

    print(f"\n{'=' * 60}")
    print(f"RESULTS SUMMARY")
    print(f"{'=' * 60}")

    total_b = baseline['frame_times'].sum()
    total_a = adaptive['frame_times'].sum()
    print(f"  Total compute time:  Baseline={total_b:.1f}s  Adaptive={total_a:.1f}s  "
          f"Speedup={(1-total_a/total_b)*100:.1f}%")
    print(f"  Mean frame time:     Baseline={baseline['frame_times'].mean():.4f}s  "
          f"Adaptive={adaptive['frame_times'].mean():.4f}s")
    print(f"  Kinetic energy retention:  Baseline={baseline['kinetic_energy'][-1]/baseline['kinetic_energy'][0]*100:.1f}%  "
          f"Adaptive={adaptive['kinetic_energy'][-1]/adaptive['kinetic_energy'][0]*100:.1f}%  "
          f"Δ={ke_improvement:.1f}pp")
    print(f"  Enstrophy retention:       Baseline={baseline['enstrophy'][-1]/baseline['enstrophy'][0]*100:.1f}%  "
          f"Adaptive={adaptive['enstrophy'][-1]/adaptive['enstrophy'][0]*100:.1f}%  "
          f"Δ={ens_improvement:.1f}pp")

    if not use_demo:
        avg_active = adaptive['active_particles'].mean()
        total_p = adaptive.get('total_particles', baseline.get('total_particles', 1))
        print(f"  Avg active particles: {avg_active:.0f}/{total_p} ({avg_active/total_p*100:.1f}%)")
        total_c = adaptive.get('total_cells', 1024*256)
        avg_marked = adaptive['marked_cells'].mean()
        print(f"  Avg marked cells:     {avg_marked:.0f}/{total_c} ({avg_marked/total_c*100:.1f}%)")

    print(f"\n  Charts saved to: {os.path.abspath(OUTPUT_DIR)}/")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()
