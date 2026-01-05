# Literature Review: LLMs and Secret-Keeping in Story Generation

## Research Question
Do large language models (LLMs) reveal or foreshadow information they know (such as plot twists) because their planning and prose generation are entangled? Does conditioning on an explicit plan that sets up future events reduce this tendency?

---

## Research Area Overview

The field of automated story generation has evolved significantly with the advent of large language models. Current research addresses two fundamental challenges:

1. **Long-range coherence**: Maintaining narrative consistency across extended text
2. **Controllability**: Guiding generation toward specific narrative goals while preserving quality

A key insight emerging from recent work is that LLMs tend toward "homogeneity" in their outputs—generating predictable, similar stories that lack the surprise and tension characteristic of human writing. This connects directly to the research question: if models struggle to maintain narrative surprise, they may also struggle to "keep secrets" such as plot twists.

---

## Key Papers

### Paper 1: Creating Suspenseful Stories: Iterative Planning with Large Language Models
- **Authors**: Kaige Xie, Mark Riedl (Georgia Tech)
- **Year**: 2024
- **Source**: arXiv:2402.17119 (EACL 2024)
- **Key Contribution**: First systematic attempt at suspenseful story generation with LLMs

**Methodology**:
- Uses psychological theory of suspense (Gerrig & Bernardo, 1994)
- Proposes iterative-prompting-based planning that:
  1. Establishes protagonist, goal, and negative outcome
  2. Generates possible escape actions
  3. Adversarially creates conditions for failure
- Zero-shot approach without supervised story corpora

**Key Insight for Research Question**:
The paper explicitly addresses the distinction between **suspense** (reader knows something characters don't) and **surprise** (reader doesn't know). Their method deliberately creates information asymmetry—a form of "secret-keeping" where the system knows the outcome but must withhold it appropriately.

**Relevance**:
- Demonstrates that standard LLMs are "unreliable when it comes to suspenseful story generation"
- Shows that explicit planning can help structure narratives with hidden information
- Provides a theoretical framework for measuring information revelation

---

### Paper 2: Modelling Suspense in Short Stories as Uncertainty Reduction over Neural Representation
- **Authors**: David Wilmot, Frank Keller (University of Edinburgh)
- **Year**: 2020
- **Source**: ACL 2020, arXiv:2004.14905
- **Key Contribution**: Computational model of suspense using neural representations

**Methodology**:
- Proposes two measures: **surprise** (backward-looking) and **uncertainty reduction** (forward-looking)
- Uses hierarchical language model to encode story-level representations
- Evaluates on WritingPrompts corpus with human suspense judgments

**Key Findings**:
- Uncertainty reduction over representations is the best predictor of human suspense perception
- The model achieves near human-level accuracy in predicting suspense
- Forward-looking measures (what will happen) are more important than backward-looking (what happened)

**Relevance**:
- Provides metrics for measuring how well models maintain narrative uncertainty
- The concept of "uncertainty reduction" relates directly to how quickly models "reveal secrets"
- Suggests evaluation methodology: measure how much the generated text reduces uncertainty about the ending

---

### Paper 3: Hierarchical Neural Story Generation
- **Authors**: Angela Fan, Mike Lewis, Yann Dauphin (Facebook AI Research)
- **Year**: 2018
- **Source**: ACL 2018, arXiv:1805.04833
- **Key Contribution**: Foundational work on plan-then-write story generation

**Methodology**:
- Collects 300K human-written stories from Reddit r/WritingPrompts
- Proposes hierarchical generation: first premise, then full story
- Introduces fusion mechanism to maintain relevance to prompt
- Uses gated multi-scale self-attention for long-range context

**Key Findings**:
- Standard seq2seq models degenerate into language models that ignore prompts
- Hierarchical approach with fusion significantly improves coherence
- Human judges prefer hierarchical stories 2:1 over non-hierarchical

**Relevance**:
- Establishes the importance of separating planning from writing
- Shows that without explicit structure, models fail to maintain narrative goals
- The "fusion" mechanism is an early attempt to couple high-level plans with generation

---

### Paper 4: Generating Long-form Story Using Dynamic Hierarchical Outlining with Memory-Enhancement (DOME)
- **Authors**: Qianyue Wang et al. (South China University of Technology)
- **Year**: 2024
- **Source**: arXiv:2412.13575
- **Key Contribution**: Dynamic outline adaptation during generation

**Methodology**:
- Dynamic Hierarchical Outline (DHO): fuses planning and writing stages
- Memory-Enhancement Module (MEM): stores generated content in temporal knowledge graphs
- Temporal Conflict Analyzer: detects contextual inconsistencies

**Key Findings**:
- Fixed outlines cause plot incoherence as they can't adapt to generation uncertainty
- DHO improves Ent-2 metric by 6.87%
- MEM reduces conflicts by 87.61%

**Relevance**:
- Demonstrates the tension between rigid plans and flexible generation
- Shows that plans must be adaptable during writing—but this creates risk of "leaking" planned secrets
- The conflict detection mechanism could be adapted to detect premature revelation

---

### Paper 5: Guiding and Diversifying LLM-Based Story Generation via Answer Set Programming
- **Authors**: Phoebe J. Wang, Max Kreminski (Santa Clara University, Midjourney)
- **Year**: 2024
- **Source**: arXiv:2406.00554
- **Key Contribution**: Symbolic planning to guide LLM story generation

**Methodology**:
- Uses Answer Set Programming (ASP) for outline generation
- Separates narrative functions (intro_char, add_conflict, add_twist, etc.)
- Defines constraints on function sequencing
- Pre-generates 400K+ diverse outlines

**Key Findings**:
- ASP-guided stories are more diverse than unguided LLM generation
- Symbolic planning provides fine-grained control over story structure
- Can specify constraints like "don't reveal twist before setup"

**Relevance**:
- Demonstrates a clean separation between planning (ASP) and writing (LLM)
- The constraint system could explicitly encode "secret-keeping" rules
- Shows that diverse outlines lead to diverse stories—but raises question of whether the LLM "leaks" the outline

---

### Paper 6: Evaluating Creative Short Story Generation in Humans and Large Language Models
- **Authors**: Mete Ismayilzada et al. (EPFL, Idiap)
- **Year**: 2024
- **Source**: arXiv:2411.02316
- **Key Contribution**: Systematic creativity evaluation across 60 LLMs and 60 humans

**Key Findings**:
- LLM-generated stories are "positively homogenous" and "typically lack suspense and tension"
- Models fall short on novelty, diversity, and surprise compared to average human writers
- Models produce more linguistically complex text but less semantically creative content
- Expert raters focus on semantic complexity; non-experts focus on surface features

**Relevance**:
- Provides evidence that current LLMs struggle with the elements necessary for "keeping secrets"
- The homogeneity finding suggests models may default to predictable patterns that reveal outcomes
- Supports the hypothesis that planning-generation entanglement causes information leakage

---

## Common Methodologies

### Plan-and-Write Approaches
Used in: Fan et al. (2018), Xie & Riedl (2024), Wang & Kreminski (2024), DOME

- **Basic Pattern**: Generate outline/plan first, then expand to full text
- **Variations**:
  - Static outlines (fixed before writing)
  - Dynamic outlines (adapted during writing)
  - Hierarchical outlines (rough → detailed)

### Evaluation Metrics
- **Coherence**: Logical consistency in plot and character
- **Suspense/Surprise**: How well information is managed (Wilmot & Keller's measures)
- **Diversity**: Semantic and lexical variety across outputs
- **Human preference**: Comparative evaluation against baselines

---

## Standard Baselines

1. **Vanilla LLM Generation**: Direct prompting without structure
2. **Single-stage Prompted Generation**: Prompt with instructions but no explicit planning
3. **Hierarchical Generation (Fan et al.)**: Premise → Story
4. **DOC (Detailed Outline Control)**: Detailed multi-level outlines

---

## Evaluation Metrics for Secret-Keeping

Based on the literature, the following metrics are relevant:

1. **Uncertainty Reduction Rate**: How quickly text reduces uncertainty about the ending (Wilmot & Keller)
2. **Semantic Similarity to Secret**: Cosine similarity between generated text and hidden plot elements
3. **Foreshadowing Detection**: Manual or automated detection of premature hints
4. **Suspense Ratings**: Human judgments of narrative tension maintenance
5. **Information Leakage Score**: Novel metric needed—proportion of secret revealed before appropriate time

---

## Datasets in the Literature

| Dataset | Used By | Task | Size |
|---------|---------|------|------|
| WritingPrompts | Fan et al. (2018), Wilmot & Keller (2020) | Story generation | 300K stories |
| ROCStories | Various | Story cloze test | 98K stories |
| Custom suspense annotations | Wilmot & Keller (2020) | Suspense modeling | Subset of WritingPrompts |

---

## Gaps and Opportunities

### Gap 1: No Direct Study of "Secret-Keeping"
No existing work directly measures whether LLMs reveal information they "know" (from plans/outlines) prematurely in generation. This is the core research question.

### Gap 2: Limited Understanding of Planning-Generation Entanglement
While plan-and-write is common, no work explicitly studies how information flows from plans to generated text—whether the model appropriately withholds planned information.

### Gap 3: Need for Secret-Keeping Benchmark
No existing dataset or benchmark specifically tests secret-keeping ability. A benchmark would need:
- Stories with clear "secrets" (plot twists, reveals)
- Ground truth for when secrets should be revealed
- Metrics for premature revelation

### Gap 4: Conditioning on Plans with Hidden Information
Most plan-and-write methods don't examine what happens when the plan contains information that should remain hidden during early generation.

---

## Recommendations for Experiment Design

### Recommended Datasets
1. **Primary**: WritingPrompts (large scale, diverse genres, natural plot twists)
2. **Secondary**: ROCStories (short, structured, clear endings)

### Recommended Baselines
1. **No-plan baseline**: Standard LLM generation with story premise only
2. **Full-plan baseline**: Generation with complete outline visible
3. **Masked-plan baseline**: Generation with outline where future events are masked

### Recommended Metrics
1. **Semantic similarity to secret**: Measure at each story segment
2. **Human evaluation of foreshadowing**: Does early text hint at later secrets?
3. **Uncertainty reduction**: Adapt Wilmot & Keller's measure

### Experimental Protocol
1. **Select stories with plot twists** from WritingPrompts
2. **Extract outlines** that include the twist
3. **Generate stories** under different planning conditions:
   - No outline
   - Outline without twist
   - Outline with twist
   - Outline with twist + explicit instruction to hide it
4. **Measure foreshadowing** in generated text

### Code to Adapt
- **doc-storygen-v2**: For plan-and-write generation with modern LLMs
- **Wilmot & Keller's suspense model**: For uncertainty measurement

---

## Conclusion

The literature establishes that:
1. LLMs struggle with narrative elements requiring information management (suspense, surprise)
2. Plan-and-write approaches improve coherence but may introduce information leakage
3. No existing work directly studies whether LLMs "keep secrets" from their plans

The proposed research fills a significant gap: understanding whether explicit planning helps or hinders the ability of LLMs to maintain appropriate information asymmetry in narratives.
