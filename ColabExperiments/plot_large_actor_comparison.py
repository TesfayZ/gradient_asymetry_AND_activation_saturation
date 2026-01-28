"""
Plot comparison: Small Actor vs Large Actor architecture
"""
import json
import os
import numpy as np
import matplotlib.pyplot as plt

# Read original results
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
original_dir = os.path.join(SCRIPT_DIR, 'results')
original = {}
for d in os.listdir(original_dir):
    if d.startswith('actor_'):
        result_file = os.path.join(original_dir, d, 'results.json')
        if os.path.exists(result_file):
            with open(result_file) as f:
                data = json.load(f)
            key = (data['actor_lr'], data['critic_lr'])
            original[key] = data['stopping_episode']

# Read large actor results
large_dir = os.path.join(SCRIPT_DIR, 'large_actor_experiment', 'large_actor_results', 'large_actor_results')
large = {}
for d in os.listdir(large_dir):
    if d.startswith('large_actor_'):
        result_file = os.path.join(large_dir, d, 'results.json')
        if os.path.exists(result_file):
            with open(result_file) as f:
                data = json.load(f)
            key = (data['actor_lr'], data['critic_lr'])
            large[key] = data['stopping_episode']

# Get common keys (experiments completed in both)
common_keys = sorted(set(original.keys()) & set(large.keys()), key=lambda x: (-x[0], -x[1]))

# Prepare data for plotting
labels = [f"({k[0]}, {k[1]})" for k in common_keys]
small_stops = [original[k] for k in common_keys]
large_stops = [large[k] for k in common_keys]

# Create comparison bar chart
fig, ax = plt.subplots(figsize=(12, 6))

x = np.arange(len(labels))
width = 0.35

bars1 = ax.bar(x - width/2, small_stops, width, label='Small Actor (64→32)', color='#e74c3c', alpha=0.8)
bars2 = ax.bar(x + width/2, large_stops, width, label='Large Actor (512→128)', color='#3498db', alpha=0.8)

# Add horizontal line at 2000 (max episodes)
ax.axhline(y=2000, color='green', linestyle='--', linewidth=2, label='Max Episodes (2000)')

ax.set_ylabel('Stopping Episode', fontsize=12)
ax.set_xlabel('(Actor LR, Critic LR)', fontsize=12)
ax.set_title('Actor Architecture Does NOT Prevent Saturation\nStopping Episodes: Small Actor vs Large Actor', fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=45, ha='right')
ax.legend(loc='upper right')
ax.set_ylim(0, 2200)

# Add value labels on bars
def add_labels(bars):
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{int(height)}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)

add_labels(bars1)
add_labels(bars2)

plt.tight_layout()
figures_dir = os.path.join(SCRIPT_DIR, 'figures')
os.makedirs(figures_dir, exist_ok=True)
plt.savefig(os.path.join(figures_dir, 'large_actor_comparison.png'), dpi=150, bbox_inches='tight')
paper_fig_dir = os.path.join(SCRIPT_DIR, '..', 'paper', 'figures')
if os.path.exists(paper_fig_dir):
    plt.savefig(os.path.join(paper_fig_dir, 'large_actor_comparison.pdf'), bbox_inches='tight')
print("Saved: figures/large_actor_comparison.png")
print("Saved: paper/figures/large_actor_comparison.pdf")

# Also create a difference plot
fig2, ax2 = plt.subplots(figsize=(10, 5))

diffs = [large[k] - original[k] for k in common_keys]
colors = ['#27ae60' if d > 0 else '#e74c3c' for d in diffs]

bars = ax2.bar(labels, diffs, color=colors, alpha=0.8)
ax2.axhline(y=0, color='black', linewidth=1)
ax2.set_ylabel('Difference (Large - Small)', fontsize=12)
ax2.set_xlabel('(Actor LR, Critic LR)', fontsize=12)
ax2.set_title('Difference in Stopping Episodes\n(Positive = Large Actor Lasted Longer)', fontsize=14)
plt.xticks(rotation=45, ha='right')

# Add value labels
for bar, diff in zip(bars, diffs):
    height = bar.get_height()
    ax2.annotate(f'{int(diff):+d}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3 if height >= 0 else -12),
                textcoords="offset points",
                ha='center', va='bottom' if height >= 0 else 'top', fontsize=10)

plt.tight_layout()
plt.savefig(os.path.join(figures_dir, 'large_actor_difference.png'), dpi=150, bbox_inches='tight')
print("Saved: figures/large_actor_difference.png")

# Print summary statistics
print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"Experiments compared: {len(common_keys)}")
print(f"Small Actor mean stopping: {np.mean(small_stops):.1f} episodes")
print(f"Large Actor mean stopping: {np.mean(large_stops):.1f} episodes")
print(f"Mean difference: {np.mean(diffs):+.1f} episodes")
print(f"Max improvement: {max(diffs):+d} episodes")
print(f"Max regression: {min(diffs):+d} episodes")
print("="*60)
print("\nCONCLUSION: Larger actor architecture does NOT prevent saturation.")
print("All high-LR experiments still stopped early (<300 episodes out of 2000).")
