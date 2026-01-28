"""
Plot comparison: Baseline vs Gradient Clipping experiment
"""
import json
import os
import numpy as np
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Read original (baseline) results
original_dir = os.path.join(SCRIPT_DIR, 'original_experiment', 'extracted_results', 'gradient_asymmetry_results')
original = {}
for d in os.listdir(original_dir):
    if d.startswith('actor_'):
        result_file = os.path.join(original_dir, d, 'results.json')
        if os.path.exists(result_file):
            with open(result_file) as f:
                data = json.load(f)
            key = (data['actor_lr'], data['critic_lr'])
            original[key] = data['stopping_episode']

# Read gradient clipping results
clip_dir = os.path.join(SCRIPT_DIR, 'gradient_clipping_experiment', 'extracted_results', 'gradient_clipping_results')
clipped = {}
for d in os.listdir(clip_dir):
    if d.startswith('gradclip_'):
        result_file = os.path.join(clip_dir, d, 'results.json')
        if os.path.exists(result_file):
            with open(result_file) as f:
                data = json.load(f)
            key = (data['actor_lr'], data['critic_lr'])
            clipped[key] = data['stopping_episode']

# Common keys sorted high to low
common_keys = sorted(set(original.keys()) & set(clipped.keys()), key=lambda x: (-x[0], -x[1]))

labels = [f"({k[0]}, {k[1]})" for k in common_keys]
orig_stops = [original[k] for k in common_keys]
clip_stops = [clipped[k] for k in common_keys]

# Create comparison bar chart
fig, ax = plt.subplots(figsize=(14, 6))

x = np.arange(len(labels))
width = 0.35

bars1 = ax.bar(x - width/2, orig_stops, width, label='Baseline (No Clipping)', color='#e74c3c', alpha=0.8)
bars2 = ax.bar(x + width/2, clip_stops, width, label='Gradient Clipping (norm=1.0)', color='#2ecc71', alpha=0.8)

ax.axhline(y=2000, color='blue', linestyle='--', linewidth=2, label='Max Episodes (2000)')

ax.set_ylabel('Stopping Episode', fontsize=12)
ax.set_xlabel('(Actor LR, Critic LR)', fontsize=12)
ax.set_title('Gradient Clipping Does NOT Prevent Actor Saturation\nStopping Episodes: Baseline vs Gradient Clipping (max_grad_norm=1.0)', fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
ax.legend(loc='upper right')
ax.set_ylim(0, 2300)

def add_labels(bars):
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{int(height)}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)

add_labels(bars1)
add_labels(bars2)

plt.tight_layout()

# Save to ColabExperiments/figures
figures_dir = os.path.join(SCRIPT_DIR, 'figures')
os.makedirs(figures_dir, exist_ok=True)
plt.savefig(os.path.join(figures_dir, 'gradient_clipping_comparison.png'), dpi=150, bbox_inches='tight')

# Save to paper/figures
paper_fig_dir = os.path.join(SCRIPT_DIR, '..', 'paper', 'figures')
if os.path.exists(paper_fig_dir):
    plt.savefig(os.path.join(paper_fig_dir, 'gradient_clipping_comparison.pdf'), bbox_inches='tight')
    plt.savefig(os.path.join(paper_fig_dir, 'gradient_clipping_comparison.png'), dpi=150, bbox_inches='tight')

print("Saved: figures/gradient_clipping_comparison.png")
print("Saved: paper/figures/gradient_clipping_comparison.pdf")

# Print summary
print("\n" + "="*60)
print("SUMMARY")
print("="*60)
for k in common_keys:
    diff = clipped[k] - original[k]
    print(f"  ({k[0]}, {k[1]}): baseline={original[k]}, clipped={clipped[k]}, diff={diff:+d}")
print(f"\nBaseline mean stopping: {np.mean(orig_stops):.1f}")
print(f"Clipped mean stopping: {np.mean(clip_stops):.1f}")
print(f"Mean difference: {np.mean([clipped[k] - original[k] for k in common_keys]):+.1f}")
print("="*60)

# Count how many reached 2000 in each
orig_converged = sum(1 for s in orig_stops if s >= 2000)
clip_converged = sum(1 for s in clip_stops if s >= 2000)
print(f"\nConfigurations reaching 2000 episodes:")
print(f"  Baseline: {orig_converged}/16")
print(f"  Gradient Clipping: {clip_converged}/16")
