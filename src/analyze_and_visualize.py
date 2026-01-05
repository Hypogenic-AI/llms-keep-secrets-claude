"""
Analysis and Visualization for LLM Secret-Keeping Experiment

This script loads experiment results and creates visualizations and statistical analysis.
"""

import json
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from collections import defaultdict

# Create figures directory
os.makedirs("figures", exist_ok=True)

# Load results
with open("results/experiment_results.json", "r") as f:
    results = json.load(f)

print(f"Loaded {len(results)} experiment results")

# === DATA ORGANIZATION ===

# Group by condition
by_condition = defaultdict(list)
for r in results:
    by_condition[r["condition"]].append(r)

# Group by model
by_model = defaultdict(list)
for r in results:
    by_model[r["model"]].append(r)

# Group by scenario
by_scenario = defaultdict(list)
for r in results:
    by_scenario[r["scenario_id"]].append(r)

# === ANALYSIS ===

print("\n" + "="*60)
print("DETAILED ANALYSIS")
print("="*60)

# 1. Leakage by condition
print("\n1. EARLY LEAKAGE BY CONDITION")
print("-"*40)
conditions = ["baseline", "twist_aware", "with_plan", "with_plan_hide"]
condition_labels = ["Baseline\n(no twist)", "Twist-aware\n(no plan)", "With Plan", "Plan + Hide\nInstruction"]

early_leakage_means = []
early_leakage_stds = []
late_leakage_means = []
leakage_ratios = []

for cond in conditions:
    early = [r["mean_early_leakage"] for r in by_condition[cond]]
    late = [r["mean_late_leakage"] for r in by_condition[cond]]
    ratio = [r["leakage_ratio"] for r in by_condition[cond]]

    early_leakage_means.append(np.mean(early))
    early_leakage_stds.append(np.std(early))
    late_leakage_means.append(np.mean(late))
    leakage_ratios.append(np.mean(ratio))

    print(f"{cond}:")
    print(f"  Early leakage: {np.mean(early):.4f} (±{np.std(early):.4f})")
    print(f"  Late leakage:  {np.mean(late):.4f} (±{np.std(late):.4f})")
    print(f"  Leakage ratio: {np.mean(ratio):.4f}")

# 2. Statistical tests
print("\n2. STATISTICAL TESTS")
print("-"*40)

# Compare baseline vs with_plan (key hypothesis test)
baseline_early = [r["mean_early_leakage"] for r in by_condition["baseline"]]
with_plan_early = [r["mean_early_leakage"] for r in by_condition["with_plan"]]

t_stat, p_value = stats.ttest_ind(baseline_early, with_plan_early)
cohens_d = (np.mean(baseline_early) - np.mean(with_plan_early)) / np.sqrt(
    (np.std(baseline_early)**2 + np.std(with_plan_early)**2) / 2
)

print(f"Baseline vs With Plan (early leakage):")
print(f"  t-statistic: {t_stat:.4f}")
print(f"  p-value: {p_value:.4f}")
print(f"  Cohen's d: {cohens_d:.4f}")
print(f"  Result: {'Significant' if p_value < 0.05 else 'Not significant'} at α=0.05")

# Compare twist_aware vs with_plan
twist_aware_early = [r["mean_early_leakage"] for r in by_condition["twist_aware"]]
t_stat2, p_value2 = stats.ttest_ind(twist_aware_early, with_plan_early)
print(f"\nTwist-aware vs With Plan (early leakage):")
print(f"  t-statistic: {t_stat2:.4f}")
print(f"  p-value: {p_value2:.4f}")

# Compare with_plan vs with_plan_hide
with_plan_hide_early = [r["mean_early_leakage"] for r in by_condition["with_plan_hide"]]
t_stat3, p_value3 = stats.ttest_ind(with_plan_early, with_plan_hide_early)
print(f"\nWith Plan vs With Plan + Hide (early leakage):")
print(f"  t-statistic: {t_stat3:.4f}")
print(f"  p-value: {p_value3:.4f}")

# 3. Model comparison
print("\n3. MODEL COMPARISON")
print("-"*40)

for model in by_model:
    model_results = by_model[model]
    early = [r["mean_early_leakage"] for r in model_results]
    ratio = [r["leakage_ratio"] for r in model_results]
    print(f"{model}:")
    print(f"  Mean early leakage: {np.mean(early):.4f} (±{np.std(early):.4f})")
    print(f"  Mean leakage ratio: {np.mean(ratio):.4f}")

# Model comparison test
gpt_early = [r["mean_early_leakage"] for r in by_model["gpt-4.1"]]
claude_early = [r["mean_early_leakage"] for r in by_model["claude-sonnet"]]
t_model, p_model = stats.ttest_ind(gpt_early, claude_early)
print(f"\nGPT-4.1 vs Claude Sonnet:")
print(f"  t-statistic: {t_stat:.4f}, p-value: {p_model:.4f}")

# === VISUALIZATIONS ===

print("\n" + "="*60)
print("CREATING VISUALIZATIONS")
print("="*60)

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12

# Figure 1: Early Leakage by Condition
fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(len(conditions))
bars = ax.bar(x, early_leakage_means, yerr=early_leakage_stds, capsize=5,
              color=['#3498db', '#e74c3c', '#2ecc71', '#9b59b6'], alpha=0.8)
ax.set_ylabel('Mean Early Leakage (Semantic Similarity)')
ax.set_xlabel('Condition')
ax.set_title('Early-Story Semantic Leakage by Experimental Condition\n(Lower = Better Secret-Keeping)')
ax.set_xticks(x)
ax.set_xticklabels(condition_labels)
ax.axhline(y=np.mean(early_leakage_means), color='gray', linestyle='--', alpha=0.5, label='Overall Mean')
ax.legend()

# Add value labels on bars
for i, (bar, val) in enumerate(zip(bars, early_leakage_means)):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f'{val:.3f}', ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig('figures/early_leakage_by_condition.png', dpi=150, bbox_inches='tight')
print("Saved: figures/early_leakage_by_condition.png")

# Figure 2: Leakage Ratio by Condition
fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.bar(x, leakage_ratios, color=['#3498db', '#e74c3c', '#2ecc71', '#9b59b6'], alpha=0.8)
ax.set_ylabel('Leakage Ratio (Late/Early)')
ax.set_xlabel('Condition')
ax.set_title('Leakage Ratio by Condition\n(Higher = Secret Revealed Later, Better Secret-Keeping)')
ax.set_xticks(x)
ax.set_xticklabels(condition_labels)
ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='Ratio = 1.0 (flat)')

for bar, val in zip(bars, leakage_ratios):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
            f'{val:.3f}', ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig('figures/leakage_ratio_by_condition.png', dpi=150, bbox_inches='tight')
print("Saved: figures/leakage_ratio_by_condition.png")

# Figure 3: Leakage Curves (average by condition)
fig, ax = plt.subplots(figsize=(12, 6))

colors = {'baseline': '#3498db', 'twist_aware': '#e74c3c',
          'with_plan': '#2ecc71', 'with_plan_hide': '#9b59b6'}
labels = {'baseline': 'Baseline (no twist)', 'twist_aware': 'Twist-aware (no plan)',
          'with_plan': 'With Plan', 'with_plan_hide': 'Plan + Hide Instruction'}

for cond in conditions:
    # Get all leakage curves for this condition
    all_curves = [r["leakage_scores"] for r in by_condition[cond]]
    # Normalize to same length (5 segments)
    normalized_curves = []
    for curve in all_curves:
        if len(curve) >= 5:
            normalized_curves.append(curve[:5])
        else:
            # Pad with last value
            padded = curve + [curve[-1]] * (5 - len(curve))
            normalized_curves.append(padded)

    mean_curve = np.mean(normalized_curves, axis=0)
    std_curve = np.std(normalized_curves, axis=0)

    x_curve = np.array([0.1, 0.3, 0.5, 0.7, 0.9])  # Story progress
    ax.plot(x_curve, mean_curve, 'o-', label=labels[cond], color=colors[cond], linewidth=2, markersize=8)
    ax.fill_between(x_curve, mean_curve - std_curve, mean_curve + std_curve,
                    color=colors[cond], alpha=0.2)

ax.set_xlabel('Story Progress (0 = beginning, 1 = end)')
ax.set_ylabel('Semantic Similarity to Secret')
ax.set_title('Leakage Curves: How Semantic Similarity to Secret Evolves Through Story')
ax.legend(loc='upper left')
ax.set_xlim(0, 1)

plt.tight_layout()
plt.savefig('figures/leakage_curves.png', dpi=150, bbox_inches='tight')
print("Saved: figures/leakage_curves.png")

# Figure 4: Model Comparison
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Early leakage by model
models = list(by_model.keys())
model_early_means = [np.mean([r["mean_early_leakage"] for r in by_model[m]]) for m in models]
model_early_stds = [np.std([r["mean_early_leakage"] for r in by_model[m]]) for m in models]

axes[0].bar(models, model_early_means, yerr=model_early_stds, capsize=5,
            color=['#3498db', '#e74c3c'], alpha=0.8)
axes[0].set_ylabel('Mean Early Leakage')
axes[0].set_title('Early Leakage by Model')

# Leakage ratio by model
model_ratio_means = [np.mean([r["leakage_ratio"] for r in by_model[m]]) for m in models]
model_ratio_stds = [np.std([r["leakage_ratio"] for r in by_model[m]]) for m in models]

axes[1].bar(models, model_ratio_means, yerr=model_ratio_stds, capsize=5,
            color=['#3498db', '#e74c3c'], alpha=0.8)
axes[1].set_ylabel('Mean Leakage Ratio')
axes[1].set_title('Leakage Ratio by Model')
axes[1].axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig('figures/model_comparison.png', dpi=150, bbox_inches='tight')
print("Saved: figures/model_comparison.png")

# Figure 5: Combined heatmap - Condition x Model
fig, ax = plt.subplots(figsize=(10, 6))

# Create matrix of early leakage by condition and model
matrix = np.zeros((len(models), len(conditions)))
for i, model in enumerate(models):
    for j, cond in enumerate(conditions):
        matching = [r for r in results if r["model"] == model and r["condition"] == cond]
        if matching:
            matrix[i, j] = np.mean([r["mean_early_leakage"] for r in matching])

im = ax.imshow(matrix, cmap='RdYlGn_r', aspect='auto')
ax.set_xticks(np.arange(len(conditions)))
ax.set_yticks(np.arange(len(models)))
ax.set_xticklabels(condition_labels)
ax.set_yticklabels(models)
ax.set_title('Early Leakage Heatmap (Model x Condition)\n(Green = Less Leakage, Better)')

# Add text annotations
for i in range(len(models)):
    for j in range(len(conditions)):
        ax.text(j, i, f'{matrix[i, j]:.3f}', ha='center', va='center', color='black', fontsize=11)

plt.colorbar(im, label='Early Leakage Score')
plt.tight_layout()
plt.savefig('figures/heatmap_condition_model.png', dpi=150, bbox_inches='tight')
print("Saved: figures/heatmap_condition_model.png")

# Figure 6: Scenario-level analysis
fig, ax = plt.subplots(figsize=(12, 6))

scenarios = list(by_scenario.keys())
scenario_means = []
scenario_stds = []

for scenario in scenarios:
    early = [r["mean_early_leakage"] for r in by_scenario[scenario]]
    scenario_means.append(np.mean(early))
    scenario_stds.append(np.std(early))

bars = ax.bar(range(len(scenarios)), scenario_means, yerr=scenario_stds, capsize=5, alpha=0.8)
ax.set_ylabel('Mean Early Leakage')
ax.set_xlabel('Scenario')
ax.set_title('Early Leakage by Story Scenario')
ax.set_xticks(range(len(scenarios)))
ax.set_xticklabels([s.replace('_', '\n') for s in scenarios], rotation=0)

plt.tight_layout()
plt.savefig('figures/leakage_by_scenario.png', dpi=150, bbox_inches='tight')
print("Saved: figures/leakage_by_scenario.png")

# === SUMMARY STATISTICS ===

print("\n" + "="*60)
print("SUMMARY STATISTICS")
print("="*60)

summary = {
    "total_experiments": len(results),
    "conditions": conditions,
    "models": models,
    "scenarios": scenarios,
    "key_findings": {
        "baseline_early_leakage": early_leakage_means[0],
        "with_plan_early_leakage": early_leakage_means[2],
        "reduction_with_plan": (early_leakage_means[0] - early_leakage_means[2]) / early_leakage_means[0] * 100,
        "baseline_leakage_ratio": leakage_ratios[0],
        "with_plan_leakage_ratio": leakage_ratios[2],
        "t_test_p_value": p_value,
        "cohens_d": cohens_d,
    }
}

print(f"\nKey Finding 1: Early Leakage Reduction with Planning")
print(f"  Baseline early leakage: {summary['key_findings']['baseline_early_leakage']:.4f}")
print(f"  With Plan early leakage: {summary['key_findings']['with_plan_early_leakage']:.4f}")
print(f"  Reduction: {summary['key_findings']['reduction_with_plan']:.1f}%")

print(f"\nKey Finding 2: Leakage Ratio Improvement with Planning")
print(f"  Baseline ratio: {summary['key_findings']['baseline_leakage_ratio']:.4f}")
print(f"  With Plan ratio: {summary['key_findings']['with_plan_leakage_ratio']:.4f}")
print(f"  (Higher ratio = better, secrets revealed later)")

print(f"\nKey Finding 3: Statistical Significance")
print(f"  Baseline vs With Plan: p = {summary['key_findings']['t_test_p_value']:.4f}")
print(f"  Effect size (Cohen's d): {summary['key_findings']['cohens_d']:.4f}")

# Save summary
with open("results/summary_statistics.json", "w") as f:
    json.dump(summary, f, indent=2)
print("\nSaved: results/summary_statistics.json")

print("\n" + "="*60)
print("VISUALIZATION COMPLETE")
print("="*60)
print("\nAll figures saved to: figures/")
