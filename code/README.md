# Cloned Code Repositories

This directory contains code repositories relevant to the research on LLMs and secret-keeping in story generation.

## Repository 1: Awesome-Story-Generation

- **URL**: https://github.com/yingpengma/Awesome-Story-Generation
- **Location**: `code/awesome-story-generation/`
- **Purpose**: Comprehensive collection of papers and resources on LLM-based story generation

### Description
This repository collects an extensive list of papers about Story Generation / Storytelling, focusing exclusively on the era of Large Language Models (LLMs). It serves as a reference for the current state-of-the-art in story generation.

### Key Contents
- Papers organized by year and topic
- Links to implementations
- Taxonomy of story generation approaches

### Relevance
- Provides comprehensive literature overview
- Contains references to plan-based story generation methods
- Lists evaluation metrics and datasets

---

## Repository 2: doc-storygen-v2

- **URL**: https://github.com/facebookresearch/doc-storygen-v2
- **Location**: `code/doc-storygen-v2/`
- **Purpose**: LLM story generation with detailed outline control (DOC)

### Description
Updated version of doc-story-generation, modified to work with newer open-source and chat-based LLMs. Implements the DOC (Detailed Outline Control) method for long-form story generation.

### Key Features
- Plan-and-write framework for story generation
- Hierarchical outline generation
- Works with modern LLMs (GPT-4, Claude, open-source models)
- Evaluation of story plots for suspense and surprise

### Relevance
This is directly relevant to the research hypothesis:
1. **Implements explicit planning**: Uses hierarchical outlines that could include "secrets"
2. **Measures suspense**: Has built-in evaluation for suspense/surprise
3. **Separates planning and writing**: Key to testing the entanglement hypothesis

### Key Files
- `story_generation/`: Main generation code
- `evaluation/`: Story quality evaluation
- `prompts/`: Prompting templates for planning

---

## Other Relevant Repositories (Not Cloned)

### fairseq/examples/stories
- **URL**: https://github.com/pytorch/fairseq/tree/master/examples/stories
- **Purpose**: Original hierarchical neural story generation implementation
- **Note**: Older (2018), uses older PyTorch/fairseq versions

### GOAT-Storytelling-Agent
- **URL**: https://github.com/GOAT-AI-lab/GOAT-Storytelling-Agent
- **Purpose**: Agent-based long story generation with consistency
- **Features**: Supports suspenseful tones, structured generation

---

## Usage Notes

### doc-storygen-v2

To use for experiments:

```bash
cd code/doc-storygen-v2

# Install dependencies
pip install -r requirements.txt

# Set up API keys (for OpenAI models)
export OPENAI_API_KEY="your-key"

# Generate a story with outline control
python generate_story.py --premise "Your story premise" --model "gpt-4"
```

### For Custom Experiments

The key insight is that doc-storygen-v2 separates:
1. **Outline generation**: High-level plot planning
2. **Story writing**: Text generation following the outline

This separation is exactly what's needed to test whether:
- Models "leak" secrets from the outline
- Explicit planning reduces inadvertent foreshadowing
- Different models vary in secret-keeping ability
