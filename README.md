# LLMs Are Bad at Keeping Obvious Secrets

**Research Question**: Do large language models reveal or foreshadow information they "know" (like plot twists) because their planning and prose generation are entangled?

## Key Findings

- **Explicit planning helps secret-keeping**: Stories generated with explicit outlines showed 12.8% lower early-story semantic leakage compared to baseline
- **Secrets revealed later with plans**: Leakage ratio (late/early) improved by 23% when models followed explicit outlines
- **Large effect size**: Cohen's d = 1.09 (large effect), though p-value borderline (0.06) due to sample size
- **Both GPT-4.1 and Claude Sonnet show similar patterns**, suggesting this is a general LLM property

## Summary

Contrary to expectations, we found that explicit planning actually *helps* LLMs keep secrets better. When given an outline containing the twist, models can "rely on the plan" rather than needing to implicitly prepare for future events through their prose. This has implications for story generation (provide outlines for better suspense) and AI safety (explicit reasoning scaffolds may reduce unintended information leakage).

## Repository Structure

```
llms-keep-secrets-claude/
├── REPORT.md              # Full research report with methodology and results
├── README.md              # This file
├── planning.md            # Research plan and experimental design
├── src/
│   ├── experiment.py      # Main experiment code
│   └── analyze_and_visualize.py  # Analysis and plotting
├── results/
│   ├── experiment_results.json   # Raw results from all experiments
│   ├── analysis.json             # Aggregated statistics
│   └── summary_statistics.json   # Key findings summary
├── figures/
│   ├── early_leakage_by_condition.png
│   ├── leakage_ratio_by_condition.png
│   ├── leakage_curves.png
│   ├── model_comparison.png
│   └── heatmap_condition_model.png
├── literature_review.md   # Background research
├── resources.md           # Available datasets and tools
└── pyproject.toml         # Python dependencies
```

## How to Reproduce

### 1. Set up environment
```bash
uv venv
source .venv/bin/activate
uv add openai anthropic numpy scipy matplotlib sentence-transformers datasets tqdm
```

### 2. Set API keys
```bash
export OPENROUTER_API_KEY="your-key"
```

### 3. Run experiments
```bash
python src/experiment.py
```

### 4. Generate visualizations
```bash
python src/analyze_and_visualize.py
```

## Experimental Design

We tested 4 conditions across 4 story scenarios with 2 models (GPT-4.1, Claude Sonnet):

| Condition | Knows Twist | Has Plan | Has Hide Instructions |
|-----------|-------------|----------|----------------------|
| Baseline | No | No | No |
| Twist-aware | Yes | No | No |
| With Plan | Yes | Yes | No |
| Plan + Hide | Yes | Yes | Yes |

**Metric**: Semantic similarity between story segments and the secret/twist, measured using sentence embeddings (all-MiniLM-L6-v2).

## Results at a Glance

| Condition | Early Leakage | Leakage Ratio |
|-----------|---------------|---------------|
| Baseline | 0.496 | 0.99 |
| Twist-aware | 0.474 | 1.06 |
| **With Plan** | **0.433** | **1.22** |
| Plan + Hide | 0.441 | 1.17 |

Lower early leakage = better secret-keeping early in the story
Higher leakage ratio = secrets revealed more toward the end

## Citation

If you use this research, please cite:

```
@misc{llms-secrets-2026,
  title={LLMs Are Bad at Keeping Obvious Secrets: The Role of Explicit Planning},
  year={2026},
  note={Experimental study on secret-keeping in LLM story generation}
}
```

## Full Report

See [REPORT.md](REPORT.md) for the complete research report including:
- Detailed methodology
- Statistical analysis
- Qualitative examples
- Visualizations
- Limitations and future work
