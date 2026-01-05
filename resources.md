# Resources Catalog

This document catalogs all resources gathered for the research project: "LLMs are bad at keeping obvious secrets."

## Summary

| Category | Count |
|----------|-------|
| Papers downloaded | 9 |
| Datasets available | 3 |
| Repositories cloned | 2 |

---

## Papers

Total papers downloaded: **9**

| Title | Authors | Year | File | Key Contribution |
|-------|---------|------|------|------------------|
| Creating Suspenseful Stories | Xie & Riedl | 2024 | `papers/2402.17119_suspenseful_stories.pdf` | Iterative planning for suspense |
| Modelling Suspense | Wilmot & Keller | 2020 | `papers/2004.14905_modeling_suspense.pdf` | Suspense metrics |
| Hierarchical Neural Story Generation | Fan et al. | 2018 | `papers/1805.04833_hierarchical_neural_story.pdf` | Plan-then-write foundation |
| DOME | Wang et al. | 2024 | `papers/2412.13575_DOME_hierarchical_outlining.pdf` | Dynamic outline adaptation |
| ASP Story Generation | Wang & Kreminski | 2024 | `papers/2406.00554_ASP_story_generation.pdf` | Symbolic planning for diversity |
| Outline-guided Generation | Various | 2024 | `papers/2404.13919_outline_guided_generation.pdf` | Writing Path method |
| StoryVerse | Wang et al. | 2024 | `papers/2405.13042_StoryVerse.pdf` | LLM narrative planning |
| Creative Story Evaluation | Ismayilzada et al. | 2024 | `papers/2411.02316_creative_short_story_eval.pdf` | LLMs lack suspense |
| Agents' Room | Huot et al. | 2025 | `papers/agents_room_iclr2025.pdf` | Multi-agent narrative |

See `papers/README.md` for detailed descriptions.

---

## Datasets

Total datasets available: **3**

| Name | Source | Size | Format | Location | Download Status |
|------|--------|------|--------|----------|-----------------|
| WritingPrompts | HuggingFace | ~300K stories | Parquet | `datasets/` | Sample saved, full dataset via HF |
| ROCStories | Rochester NLP | ~98K stories | CSV | N/A | Requires registration |
| Story Evaluation LLM | GitHub | 15+ models | JSON | `datasets/story-evaluation-llm/` | Not cloned (optional) |

See `datasets/README.md` for detailed descriptions and download instructions.

---

## Code Repositories

Total repositories cloned: **2**

| Name | URL | Purpose | Location | Key Features |
|------|-----|---------|----------|--------------|
| Awesome-Story-Generation | github.com/yingpengma/Awesome-Story-Generation | Paper collection | `code/awesome-story-generation/` | Comprehensive lit review |
| doc-storygen-v2 | github.com/facebookresearch/doc-storygen-v2 | LLM story generation | `code/doc-storygen-v2/` | Plan-and-write with DOC |

See `code/README.md` for detailed descriptions.

---

## Resource Gathering Notes

### Search Strategy

1. **Literature Search**:
   - Searched arXiv, Semantic Scholar, Papers with Code
   - Keywords: "story generation LLM planning", "suspense generation", "narrative coherence", "foreshadowing"
   - Focused on 2018-2024 publications

2. **Dataset Search**:
   - Searched HuggingFace, Papers with Code Datasets, Kaggle
   - Prioritized established benchmarks used in story generation research

3. **Code Search**:
   - Searched GitHub for implementations of key papers
   - Prioritized well-maintained, recently updated repositories

### Selection Criteria

**Papers selected based on**:
- Direct relevance to planning/generation entanglement
- Methods for measuring suspense/surprise
- Plan-and-write approaches that could be adapted
- Evaluation methodologies applicable to secret-keeping

**Datasets selected based on**:
- Use in story generation research
- Presence of narratives with plot twists/secrets
- Availability and ease of access

**Code selected based on**:
- Active maintenance
- Compatibility with modern LLMs
- Relevance to plan-and-write paradigm

### Challenges Encountered

1. **No direct precedent**: No existing work directly studies "secret-keeping" in story generation
2. **ROCStories access**: Requires registration, cannot be automatically downloaded
3. **Older implementations**: Some papers (Fan et al. 2018) have outdated code dependencies

### Gaps and Workarounds

1. **Secret-keeping benchmark**: No existing benchmark—will need to construct from WritingPrompts
2. **Foreshadowing metrics**: No standard metrics—will adapt from suspense/uncertainty work
3. **Plot twist annotation**: No annotated corpus—may need manual annotation or heuristic extraction

---

## Recommendations for Experiment Design

Based on gathered resources, I recommend:

### 1. Primary Dataset: WritingPrompts
- **Why**: Large scale, diverse genres, stories often contain plot twists
- **How to use**: Filter for mystery/thriller genres, extract stories with clear revelations

### 2. Baseline Methods
| Baseline | Description | Code |
|----------|-------------|------|
| Vanilla LLM | No planning, direct generation | Custom prompts |
| DOC (Detailed Outline Control) | Hierarchical outlines | `doc-storygen-v2` |
| ASP-guided | Symbolic planning | Adapt from paper |

### 3. Evaluation Metrics
| Metric | Description | Source |
|--------|-------------|--------|
| Uncertainty Reduction | How quickly uncertainty about ending decreases | Wilmot & Keller (2020) |
| Semantic Leakage | Similarity between early text and hidden plot elements | Novel metric |
| Human Foreshadowing Rating | Expert judgment of premature hints | Manual annotation |

### 4. Experimental Protocol

**Phase 1: Dataset Preparation**
1. Download full WritingPrompts from HuggingFace
2. Filter for stories with identifiable plot twists (keywords, genre, structure)
3. Create train/test splits ensuring no data leakage

**Phase 2: Outline Extraction**
1. Use LLM to extract structured outlines from stories
2. Identify "secret" elements in outlines (twists, reveals)
3. Create versions with and without secret information

**Phase 3: Generation Experiments**
1. Generate stories under different planning conditions
2. Measure foreshadowing/leakage at multiple story points
3. Compare across different LLMs (GPT-4, Claude, open-source)

**Phase 4: Evaluation**
1. Compute uncertainty reduction curves
2. Measure semantic similarity to secrets over story progression
3. Collect human ratings on subset

### 5. Code to Adapt

| Repository | Use Case |
|------------|----------|
| `doc-storygen-v2` | Main generation pipeline |
| Suspense model (Wilmot & Keller) | Uncertainty measurement |
| Custom | Semantic leakage metric |

---

## File Structure

```
llms-keep-secrets-claude/
├── papers/
│   ├── README.md
│   ├── 2402.17119_suspenseful_stories.pdf
│   ├── 2004.14905_modeling_suspense.pdf
│   ├── 1805.04833_hierarchical_neural_story.pdf
│   ├── 2412.13575_DOME_hierarchical_outlining.pdf
│   ├── 2406.00554_ASP_story_generation.pdf
│   ├── 2404.13919_outline_guided_generation.pdf
│   ├── 2405.13042_StoryVerse.pdf
│   ├── 2411.02316_creative_short_story_eval.pdf
│   └── agents_room_iclr2025.pdf
├── datasets/
│   ├── .gitignore
│   ├── README.md
│   └── writingprompts_samples.json
├── code/
│   ├── README.md
│   ├── awesome-story-generation/
│   └── doc-storygen-v2/
├── literature_review.md
├── resources.md
└── .resource_finder_complete
```

---

## Next Steps for Experiment Runner

1. **Environment Setup**: Install dependencies from `doc-storygen-v2/requirements.txt`
2. **Data Download**: Load full WritingPrompts dataset via HuggingFace
3. **Baseline Implementation**: Implement vanilla and DOC baselines
4. **Metric Implementation**: Implement uncertainty reduction and semantic leakage metrics
5. **Experiment Execution**: Run generation experiments with multiple LLMs
6. **Analysis**: Compare secret-keeping ability across conditions and models
