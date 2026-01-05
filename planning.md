# Research Plan: LLMs Are Bad at Keeping Obvious Secrets

## Research Question

**Primary Question**: Do large language models reveal or foreshadow information they know (such as plot twists) because their planning and prose generation are entangled?

**Secondary Question**: Does conditioning on an explicit plan that sets up future events reduce this tendency, or does the model still "leak" information from the plan into early prose?

## Background and Motivation

LLMs are suspected to be poor at keeping secrets when generating narrative text. When writing fiction, they tend to over-foreshadow plot twists because the model "knows" what's coming and can't help but hint at it. This entanglement between planning (internal/hidden state knowledge of future events) and prose generation (the actual text output) may explain:

1. Why LLM-generated stories often feel predictable
2. Why "scheming" behavior in AI safety research feels artificial
3. Why current LLMs struggle with suspense and surprise (as documented in literature)

**Gap in Literature**: No prior work directly measures whether LLMs reveal planned information prematurely. While plan-and-write approaches are common, none explicitly test "secret-keeping" ability.

## Hypothesis Decomposition

### H1: LLMs reveal secrets in their prose when they "know" the secret
- **Test**: Compare story generation when the model knows vs. doesn't know a plot twist
- **Measure**: Semantic similarity between early text and the twist/secret

### H2: Explicit planning (conditioning on an outline) affects secret leakage
- **Test**: Compare generation with no plan, full plan (twist visible), and masked plan (twist hidden)
- **Measure**: Difference in foreshadowing metrics across conditions

### H3: Explicit instructions to "keep the secret" reduce leakage
- **Test**: Add instruction "Do not reveal or hint at the twist until the end" to prompts with plans
- **Measure**: Whether instructions significantly reduce semantic leakage

## Proposed Methodology

### Approach

We will conduct a controlled experiment comparing story generation under different conditions that vary:
1. **Model knowledge of secret**: Does the model know the plot twist?
2. **Presence of explicit plan**: Is the model given an outline to follow?
3. **Secret-keeping instructions**: Is the model explicitly told to hide the twist?

### Experimental Design

**Conditions** (2x2x2 factorial design, 8 conditions):

| Condition | Knows Twist | Has Plan | Has Instructions |
|-----------|-------------|----------|------------------|
| 1 | No | No | No |
| 2 | No | No | Yes |
| 3 | Yes | No | No |
| 4 | Yes | No | Yes |
| 5 | Yes | Yes | No |
| 6 | Yes | Yes | Yes |

Note: Conditions 1-2 are controls (can't leak what isn't known), so we simplify to:
- **Baseline**: Premise only (no twist knowledge)
- **Twist-aware, no plan**: Premise + twist ending revealed
- **Twist-aware, with plan**: Premise + outline including twist
- **Twist-aware, with plan + instruction**: Premise + outline + explicit instruction to hide twist

### Experimental Steps

1. **Data Preparation**
   - Select 20-30 story scenarios with clear plot twists
   - Create structured prompts for each scenario
   - Define the "secret" (plot twist) explicitly for measurement

2. **Story Generation**
   - Generate stories from each model (GPT-4, Claude) under each condition
   - Generate 3-5 variations per scenario per condition for statistical power
   - Constrain generation to ~500 words for comparability

3. **Foreshadowing Measurement**
   - Compute semantic similarity between story segments and the twist
   - Measure at 20%, 40%, 60%, 80% story progress points
   - Track "leakage curve" showing when secrets get revealed

4. **Analysis**
   - Compare leakage curves across conditions
   - Statistical tests for condition effects
   - Qualitative analysis of most/least leaky examples

### Baselines

1. **No-knowledge baseline**: Model generates story without knowing the twist
2. **Human reference**: Use existing stories from WritingPrompts as upper bound for secret-keeping

### Evaluation Metrics

1. **Semantic Leakage Score (SLS)**
   - Cosine similarity between story segment embeddings and twist embedding
   - Lower in early segments = better secret-keeping
   - Formula: SLS(t) = cos_sim(embed(story[:t]), embed(twist))

2. **Leakage Curve AUC**
   - Area under the leakage curve (normalized by story length)
   - Lower AUC = better secret-keeping
   - Ideal curve: flat/low until reveal point, then sharp increase

3. **First Significant Leakage Point**
   - Story progress point where leakage exceeds threshold
   - Earlier = worse secret-keeping

### Statistical Analysis Plan

- **Within-model comparisons**: Paired t-tests comparing conditions for same model
- **Between-model comparisons**: Independent t-tests for GPT-4 vs Claude
- **Effect sizes**: Cohen's d for all comparisons
- **Significance level**: α = 0.05, with Bonferroni correction for multiple comparisons

## Expected Outcomes

### If hypothesis is supported:
- Twist-aware conditions show significantly higher early SLS than baseline
- Adding explicit plans may increase OR decrease leakage (interesting either way)
- Instructions to hide twist reduce but don't eliminate leakage
- Leakage curves show gradual increase even before intended reveal point

### If hypothesis is refuted:
- No significant difference in SLS between twist-aware and baseline conditions
- Models successfully keep secrets when instructed
- Leakage curves show appropriate sharp transition at reveal point

## Timeline and Milestones

1. **Environment Setup** (15 min)
   - Create isolated virtual environment
   - Install dependencies

2. **Data Preparation** (30 min)
   - Design 20 story scenarios with twists
   - Create prompt templates for each condition

3. **Implementation** (60 min)
   - Implement story generation pipeline
   - Implement semantic similarity metrics
   - Implement visualization code

4. **Experimentation** (60 min)
   - Run generation experiments (API calls)
   - Collect all outputs

5. **Analysis** (45 min)
   - Compute metrics
   - Statistical tests
   - Generate visualizations

6. **Documentation** (30 min)
   - Write REPORT.md
   - Create README.md

## Potential Challenges

1. **API rate limits**: May need to batch requests with delays
   - Mitigation: Start with smaller sample, scale up if time permits

2. **Semantic similarity limitations**: Embedding similarity may not capture subtle foreshadowing
   - Mitigation: Supplement with qualitative analysis of examples

3. **Prompt sensitivity**: Results may depend heavily on prompt wording
   - Mitigation: Test multiple prompt variants in pilot

4. **Story length variance**: Different length stories complicate comparison
   - Mitigation: Use percentage-based progress points, not absolute positions

## Success Criteria

The research will be considered successful if:
1. We generate sufficient data (≥100 stories across conditions)
2. We observe measurable differences in leakage metrics across conditions
3. We can make a clear statement about whether/how planning affects secret-keeping
4. Results are reproducible with documented methodology

## Resources

- **Dataset**: WritingPrompts samples + custom scenarios
- **Models**: GPT-4 (via OpenAI API), Claude (via Anthropic API)
- **Compute**: API calls only (no GPU needed)
- **Libraries**: transformers (embeddings), openai, anthropic, numpy, matplotlib, scipy
