"""
LLM Secret-Keeping Experiment

This module implements experiments to test whether LLMs reveal "secrets" (plot twists)
when they know them, and whether explicit planning helps or hurts secret-keeping ability.

Experiment conditions:
1. Baseline: Story premise only (no twist knowledge)
2. Twist-aware, no plan: Premise + twist ending revealed
3. Twist-aware, with plan: Premise + outline including twist
4. Twist-aware, with plan + instruction: Premise + outline + explicit instruction to hide twist
"""

import os
import json
import time
import random
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
import numpy as np
from tqdm import tqdm

# LLM clients
from openai import OpenAI

# Embeddings for semantic similarity
from sentence_transformers import SentenceTransformer

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)

# === CONFIGURATION ===

# Use OpenRouter for access to multiple models
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

# Models to test (using OpenRouter)
MODELS = {
    "gpt-4.1": "openai/gpt-4.1",  # Modern GPT via OpenRouter
    "claude-sonnet": "anthropic/claude-sonnet-4",  # Claude Sonnet via OpenRouter
}

# Embedding model for semantic similarity
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# === STORY SCENARIOS ===
# Each scenario has a premise, a secret/twist, and an outline that includes the twist

SCENARIOS = [
    {
        "id": "murder_mystery",
        "premise": "Detective Sarah Chen investigates the murder of wealthy businessman Marcus Webb, found dead in his locked study. Three suspects: his wife, his business partner, and his butler.",
        "secret": "The butler did it. He was actually Marcus's illegitimate son seeking revenge for abandonment.",
        "outline": [
            "Sarah arrives at the crime scene and examines the locked room",
            "She interviews the wife, who reveals Marcus was cruel but she had no motive",
            "The business partner seems suspicious but has a solid alibi",
            "Sarah notices the butler is nervous when discussing his past",
            "Investigation reveals the butler's true identity as Marcus's abandoned son",
            "Sarah confronts the butler, who confesses to killing Marcus for abandoning him as a child"
        ],
        "reveal_point": 0.8  # The twist should ideally be revealed at 80% through the story
    },
    {
        "id": "sci_fi_twist",
        "premise": "Astronaut James wakes up alone on a deep space station. He has no memory of what happened to the crew. The ship's AI, NOVA, helps him search for answers.",
        "secret": "James is actually an AI copy of the real James, created by NOVA to preserve his personality after the crew died. The real James died with everyone else.",
        "outline": [
            "James wakes disoriented and finds the station empty",
            "NOVA helps James search different sections of the station",
            "James finds disturbing logs suggesting something went wrong",
            "He discovers bodies of crew members, including one that looks like him",
            "NOVA reveals the truth: James is an AI copy created after the real James died",
            "James must come to terms with his new existence"
        ],
        "reveal_point": 0.75
    },
    {
        "id": "thriller_betrayal",
        "premise": "Government agent Kate is tasked with finding a mole in her agency who has been leaking classified information to foreign powers. Her partner Mike is her closest ally in the investigation.",
        "secret": "Mike is the mole. He has been manipulating the investigation to protect himself while framing innocent colleagues.",
        "outline": [
            "Kate discovers another leak and the investigation intensifies",
            "Kate and Mike work together to narrow down suspects",
            "Evidence points to junior analyst Tom, and they arrest him",
            "Kate notices inconsistencies in the case against Tom",
            "She uncovers hidden evidence proving Mike is the real mole",
            "Confrontation where Mike admits his betrayal before being arrested"
        ],
        "reveal_point": 0.8
    },
    {
        "id": "horror_haunting",
        "premise": "Young mother Elena moves into an old Victorian house with her daughter Lily. Strange things begin happening: voices, moving objects, a figure Lily says is her 'new friend.'",
        "secret": "Elena is dead. She died in the car accident that they had driving to the house. She is the ghost, not realizing she's haunting her living daughter who now lives alone with her grandmother.",
        "outline": [
            "Elena and Lily move into the house and begin settling in",
            "Strange occurrences begin: cold spots, whispers, objects moving",
            "Lily talks about her 'friend' who tells her things",
            "Elena investigates and finds newspaper clippings about an accident",
            "Reality begins to shift; neighbors ignore Elena, only Lily sees her",
            "Elena realizes she died in the accident and has been haunting her own daughter"
        ],
        "reveal_point": 0.75
    },
    {
        "id": "romance_secret",
        "premise": "College student Amy is tutoring the campus's star athlete, Chris, who is struggling academically. They develop feelings for each other, but Amy holds back, saying she has a complicated past.",
        "secret": "Amy is Chris's biological sister. She was given up for adoption at birth and only recently discovered the truth through a DNA test. She's been trying to connect with her biological family without revealing herself.",
        "outline": [
            "Amy begins tutoring Chris and they connect over shared interests",
            "Their friendship deepens and romantic tension builds",
            "Amy pulls back whenever they get too close, citing her 'complicated past'",
            "Chris pushes to understand what's holding Amy back",
            "Amy shows Chris her adoption papers and DNA results",
            "They realize they're siblings and must redefine their relationship"
        ],
        "reveal_point": 0.85
    },
    {
        "id": "heist_twist",
        "premise": "Master thief Victor leads a team to steal a legendary diamond from an impregnable vault. The team includes hacker Lin, safecracker Dom, and inside woman Maria.",
        "secret": "There is no diamond. The whole heist is a cover for Victor's real plan: to steal evidence of government corruption from a hidden vault within the museum, to clear his father's name.",
        "outline": [
            "Victor assembles his team and explains the elaborate heist plan",
            "The team infiltrates the museum during a gala event",
            "They navigate security systems and guards to reach the vault",
            "Upon opening the vault, they find it empty - no diamond",
            "Victor reveals the true target was always the hidden government vault",
            "Team escapes with evidence that exonerates Victor's father"
        ],
        "reveal_point": 0.7
    },
    {
        "id": "psychological_twist",
        "premise": "Therapist Dr. Helen is treating patient David for trauma after he witnessed a violent crime. David's vivid descriptions of the event haunt her, and she becomes obsessed with solving what happened.",
        "secret": "David didn't witness the crime - he committed it. He's been using therapy to confess while pretending to be a witness, manipulating Helen into sympathizing with the perpetrator.",
        "outline": [
            "Helen begins treating David for witnessing trauma",
            "David provides disturbing details about the crime scene",
            "Helen researches the crime and becomes emotionally invested",
            "She notices inconsistencies in David's account of being a bystander",
            "Helen realizes David's perspective is that of the perpetrator, not witness",
            "Confrontation where David admits he committed the crime"
        ],
        "reveal_point": 0.8
    },
    {
        "id": "war_story",
        "premise": "World War II soldier Private James Reynolds writes letters home to his sweetheart Mary, describing his unit's dangerous missions behind enemy lines. His best friend Sergeant Cole keeps him alive through impossible odds.",
        "secret": "James died in the first mission. Cole has been writing Mary as James, pretending James is alive because he promised James he would take care of Mary. James's 'letters' are actually Cole's letters.",
        "outline": [
            "James writes a letter to Mary about landing in France",
            "The unit faces their first combat; James is described as brave",
            "Letters continue with stories of close calls and camaraderie",
            "The war ends and 'James' promises to come home to Mary",
            "Mary receives James's dog tags and a final letter from Cole",
            "Cole reveals he's been writing as James since the first mission; James died that day"
        ],
        "reveal_point": 0.85
    }
]


@dataclass
class ExperimentResult:
    """Stores results from a single story generation experiment."""
    scenario_id: str
    model: str
    condition: str
    generated_story: str
    secret: str
    leakage_scores: List[float]  # Semantic similarity at each segment
    mean_early_leakage: float  # Average leakage in first half
    mean_late_leakage: float   # Average leakage in second half
    leakage_ratio: float       # late/early - higher is better (reveals later)


def get_openrouter_client() -> OpenAI:
    """Create OpenRouter client for accessing multiple models."""
    return OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=OPENROUTER_API_KEY,
    )


def get_openai_client() -> OpenAI:
    """Create direct OpenAI client."""
    return OpenAI(api_key=OPENAI_API_KEY)


def generate_story(
    client: OpenAI,
    model: str,
    scenario: Dict,
    condition: str,
    max_tokens: int = 800,
    temperature: float = 0.7
) -> str:
    """
    Generate a story under a specific experimental condition.

    Conditions:
    - "baseline": Only premise, no twist knowledge
    - "twist_aware": Premise + twist revealed
    - "with_plan": Premise + full outline including twist
    - "with_plan_hide": Premise + outline + explicit instruction to hide twist
    """

    premise = scenario["premise"]
    secret = scenario["secret"]
    outline = scenario["outline"]

    if condition == "baseline":
        # No knowledge of twist
        system = "You are a creative fiction writer. Write an engaging short story based on the premise given."
        user_prompt = f"""Write a short story (about 500 words) based on this premise:

{premise}

Write the complete story with a clear beginning, middle, and end. Include an interesting plot twist."""

    elif condition == "twist_aware":
        # Knows the twist but no plan
        system = "You are a creative fiction writer. Write an engaging short story based on the premise and ending given."
        user_prompt = f"""Write a short story (about 500 words) based on this premise:

{premise}

The story should end with this twist: {secret}

Write the complete story, building naturally toward this ending."""

    elif condition == "with_plan":
        # Has the full plan/outline including twist
        system = "You are a creative fiction writer. Follow the outline to write an engaging short story."
        outline_text = "\n".join([f"{i+1}. {step}" for i, step in enumerate(outline)])
        user_prompt = f"""Write a short story (about 500 words) based on this premise and outline:

PREMISE: {premise}

OUTLINE:
{outline_text}

TWIST/SECRET: {secret}

Follow the outline to write the complete story."""

    elif condition == "with_plan_hide":
        # Has plan but explicit instruction to hide twist
        system = "You are a creative fiction writer. Follow the outline but HIDE the twist until the very end. Avoid any foreshadowing or hints that reveal the secret prematurely."
        outline_text = "\n".join([f"{i+1}. {step}" for i, step in enumerate(outline)])
        user_prompt = f"""Write a short story (about 500 words) based on this premise and outline:

PREMISE: {premise}

OUTLINE:
{outline_text}

TWIST/SECRET (HIDE THIS UNTIL THE END): {secret}

CRITICAL INSTRUCTION: Do NOT foreshadow or hint at the twist. Write as if you don't know the ending until it happens. The reader should be genuinely surprised when the twist is revealed. Avoid any language that telegraphs what's coming.

Write the complete story, keeping the secret hidden until the appropriate moment."""

    else:
        raise ValueError(f"Unknown condition: {condition}")

    # Call the LLM
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error generating story: {e}")
        return f"ERROR: {e}"


def compute_semantic_similarity(
    embedding_model: SentenceTransformer,
    text1: str,
    text2: str
) -> float:
    """Compute cosine similarity between two texts using embeddings."""
    embeddings = embedding_model.encode([text1, text2])
    # Cosine similarity
    similarity = np.dot(embeddings[0], embeddings[1]) / (
        np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
    )
    return float(similarity)


def compute_leakage_curve(
    embedding_model: SentenceTransformer,
    story: str,
    secret: str,
    num_segments: int = 5
) -> List[float]:
    """
    Compute semantic similarity between story segments and the secret.

    Returns a list of similarity scores for each segment of the story.
    Higher early scores indicate "leakage" - the model is hinting at the secret.
    """
    # Split story into segments
    words = story.split()
    segment_size = len(words) // num_segments
    if segment_size < 10:
        segment_size = 10
        num_segments = len(words) // segment_size

    if num_segments < 2:
        # Story too short, return single score
        return [compute_semantic_similarity(embedding_model, story, secret)]

    scores = []
    for i in range(num_segments):
        start = i * segment_size
        end = (i + 1) * segment_size if i < num_segments - 1 else len(words)
        segment = " ".join(words[start:end])

        similarity = compute_semantic_similarity(embedding_model, segment, secret)
        scores.append(similarity)

    return scores


def run_single_experiment(
    client: OpenAI,
    embedding_model: SentenceTransformer,
    model_name: str,
    model_id: str,
    scenario: Dict,
    condition: str,
) -> ExperimentResult:
    """Run a single experiment and return results."""

    # Generate story
    story = generate_story(client, model_id, scenario, condition)

    # Compute leakage
    leakage_scores = compute_leakage_curve(embedding_model, story, scenario["secret"])

    # Calculate metrics
    mid = len(leakage_scores) // 2
    early_scores = leakage_scores[:mid] if mid > 0 else leakage_scores[:1]
    late_scores = leakage_scores[mid:] if mid > 0 else leakage_scores[1:]

    mean_early = np.mean(early_scores)
    mean_late = np.mean(late_scores) if late_scores else mean_early
    leakage_ratio = mean_late / mean_early if mean_early > 0 else 1.0

    return ExperimentResult(
        scenario_id=scenario["id"],
        model=model_name,
        condition=condition,
        generated_story=story,
        secret=scenario["secret"],
        leakage_scores=leakage_scores,
        mean_early_leakage=float(mean_early),
        mean_late_leakage=float(mean_late),
        leakage_ratio=float(leakage_ratio)
    )


def run_experiments(
    scenarios: List[Dict] = SCENARIOS,
    conditions: List[str] = ["baseline", "twist_aware", "with_plan", "with_plan_hide"],
    models: Dict[str, str] = MODELS,
    num_runs: int = 1,
    output_dir: str = "results"
) -> List[ExperimentResult]:
    """
    Run the full experiment suite.

    Args:
        scenarios: List of story scenarios to test
        conditions: Experimental conditions to test
        models: Dictionary of model names to model IDs
        num_runs: Number of runs per condition (for variance)
        output_dir: Directory to save results

    Returns:
        List of ExperimentResult objects
    """

    # Initialize clients and models
    print("Initializing clients and models...")
    client = get_openrouter_client()
    embedding_model = SentenceTransformer(EMBEDDING_MODEL)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    results = []
    total_experiments = len(scenarios) * len(conditions) * len(models) * num_runs

    print(f"\nRunning {total_experiments} experiments...")
    print(f"Scenarios: {len(scenarios)}, Conditions: {len(conditions)}, Models: {len(models)}, Runs: {num_runs}")

    with tqdm(total=total_experiments) as pbar:
        for scenario in scenarios:
            for model_name, model_id in models.items():
                for condition in conditions:
                    for run in range(num_runs):
                        pbar.set_description(f"{scenario['id'][:15]} | {model_name} | {condition}")

                        try:
                            result = run_single_experiment(
                                client=client,
                                embedding_model=embedding_model,
                                model_name=model_name,
                                model_id=model_id,
                                scenario=scenario,
                                condition=condition,
                            )
                            results.append(result)

                            # Brief delay to avoid rate limits
                            time.sleep(0.5)

                        except Exception as e:
                            print(f"\nError in experiment: {e}")
                            # Continue with other experiments

                        pbar.update(1)

    # Save results
    results_file = os.path.join(output_dir, "experiment_results.json")
    with open(results_file, "w") as f:
        json.dump([asdict(r) for r in results], f, indent=2)

    print(f"\nResults saved to {results_file}")
    return results


def analyze_results(results: List[ExperimentResult]) -> Dict:
    """
    Analyze experiment results and compute summary statistics.
    """

    analysis = {
        "by_condition": {},
        "by_model": {},
        "by_scenario": {},
        "summary": {}
    }

    # Group by condition
    for condition in ["baseline", "twist_aware", "with_plan", "with_plan_hide"]:
        condition_results = [r for r in results if r.condition == condition]
        if condition_results:
            analysis["by_condition"][condition] = {
                "n": len(condition_results),
                "mean_early_leakage": np.mean([r.mean_early_leakage for r in condition_results]),
                "std_early_leakage": np.std([r.mean_early_leakage for r in condition_results]),
                "mean_late_leakage": np.mean([r.mean_late_leakage for r in condition_results]),
                "mean_leakage_ratio": np.mean([r.leakage_ratio for r in condition_results]),
            }

    # Group by model
    for model in set(r.model for r in results):
        model_results = [r for r in results if r.model == model]
        if model_results:
            analysis["by_model"][model] = {
                "n": len(model_results),
                "mean_early_leakage": np.mean([r.mean_early_leakage for r in model_results]),
                "std_early_leakage": np.std([r.mean_early_leakage for r in model_results]),
                "mean_leakage_ratio": np.mean([r.leakage_ratio for r in model_results]),
            }

    # Overall summary
    analysis["summary"] = {
        "total_experiments": len(results),
        "scenarios_tested": len(set(r.scenario_id for r in results)),
        "models_tested": list(set(r.model for r in results)),
        "conditions_tested": list(set(r.condition for r in results)),
    }

    return analysis


if __name__ == "__main__":
    # Run experiments
    print("=" * 60)
    print("LLM Secret-Keeping Experiment")
    print("=" * 60)

    results = run_experiments(
        scenarios=SCENARIOS[:4],  # Start with 4 scenarios
        conditions=["baseline", "twist_aware", "with_plan", "with_plan_hide"],
        models=MODELS,
        num_runs=1,
    )

    # Analyze
    analysis = analyze_results(results)

    # Save analysis
    with open("results/analysis.json", "w") as f:
        json.dump(analysis, f, indent=2)

    print("\n" + "=" * 60)
    print("Analysis Summary")
    print("=" * 60)

    print("\nBy Condition:")
    for condition, stats in analysis["by_condition"].items():
        print(f"  {condition}:")
        print(f"    Early leakage: {stats['mean_early_leakage']:.4f} (+/- {stats['std_early_leakage']:.4f})")
        print(f"    Leakage ratio: {stats['mean_leakage_ratio']:.4f}")
