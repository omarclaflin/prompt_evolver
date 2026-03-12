"""Scenario sampling, evaluation dispatch, and result aggregation."""

import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Callable, Union, Optional
from statistics import mean


@dataclass
class ScenarioSplit:
    """Split scenarios into evaluation, validation, and unused sets."""
    evaluation_scenarios: List[Dict]
    validation_scenarios: List[Dict]
    unused_scenarios: List[Dict]


@dataclass
class EvalResult:
    """Result from a single eval on a single scenario."""
    scenario_id: str
    category: str
    eval_name: str
    score: float
    reason: str
    passed: bool


@dataclass
class PromptScore:
    """Aggregated evaluation results for a prompt."""
    prompt_id: str
    mean_score: float
    scores_by_category: Dict[str, float]
    scores_by_eval: Dict[str, float]
    eval_results: List[EvalResult]
    delta_from_baseline: float


def stratified_split(
    scenarios: List[Dict],
    eval_split: Union[float, int],
    validation_split: Union[float, int],
    random_seed: int = 42
) -> ScenarioSplit:
    """
    Split scenarios into evaluation and validation sets with stratified sampling per category.

    Args:
        scenarios: List of scenario dicts with 'category' field
        eval_split: Float (percentage) or int (count per category) for evaluation set
        validation_split: Float (percentage) or int (count per category) for validation set
        random_seed: Random seed for reproducibility

    Returns:
        ScenarioSplit with evaluation, validation, and unused scenarios
    """
    random.seed(random_seed)

    # Group scenarios by category
    by_category = defaultdict(list)
    for scenario in scenarios:
        category = scenario.get('category', 'unknown')
        by_category[category].append(scenario)

    evaluation_scenarios = []
    validation_scenarios = []
    unused_scenarios = []

    for category, cat_scenarios in by_category.items():
        # Shuffle for randomness
        shuffled = cat_scenarios.copy()
        random.shuffle(shuffled)

        # Determine counts
        if isinstance(eval_split, float):
            eval_count = int(len(cat_scenarios) * eval_split)
        else:
            eval_count = min(eval_split, len(cat_scenarios))

        if isinstance(validation_split, float):
            val_count = int(len(cat_scenarios) * validation_split)
        else:
            val_count = min(validation_split, len(cat_scenarios))

        # Sample evaluation set
        eval_set = shuffled[:eval_count]
        evaluation_scenarios.extend(eval_set)

        # Sample validation set from remaining scenarios (held-out)
        remaining = shuffled[eval_count:]
        val_set = remaining[:val_count]
        validation_scenarios.extend(val_set)

        # Unused: scenarios not in either set
        eval_ids = {s['id'] for s in eval_set}
        val_ids = {s['id'] for s in val_set}
        used_ids = eval_ids | val_ids
        unused = [s for s in cat_scenarios if s['id'] not in used_ids]
        unused_scenarios.extend(unused)

    return ScenarioSplit(
        evaluation_scenarios=evaluation_scenarios,
        validation_scenarios=validation_scenarios,
        unused_scenarios=unused_scenarios
    )


def subsample_scenarios(
    scenarios: List[Dict],
    subsample_fraction: float = 0.5,
    per_iteration: bool = True,
    random_seed: Optional[int] = None
) -> List[Dict]:
    """
    Subsample scenarios with fresh categorical sampling per call.

    Args:
        scenarios: Source scenario list
        subsample_fraction: Fraction of scenarios to sample per category (default: 0.5)
        per_iteration: If True, returns a fresh subsample on each call
        random_seed: Optional seed for reproducibility

    Returns:
        Subsampled scenarios (stratified by category)
    """
    if not per_iteration:
        return scenarios

    if random_seed is not None:
        random.seed(random_seed)

    # Group by category
    by_category = defaultdict(list)
    for scenario in scenarios:
        category = scenario.get('category', 'unknown')
        by_category[category].append(scenario)

    # Subsample each category
    subsampled = []
    for category, cat_scenarios in by_category.items():
        shuffled = cat_scenarios.copy()
        random.shuffle(shuffled)

        # Take fraction of scenarios from this category
        count = max(1, int(len(cat_scenarios) * subsample_fraction))
        subsampled.extend(shuffled[:count])

    return subsampled


def evaluate_prompt(
    prompt: str,
    scenarios: List[Dict],
    eval_runner: Callable,
    prompt_id: str = "prompt",
    baseline_score: float = 0.0
) -> PromptScore:
    """
    Evaluate a prompt on a set of scenarios.

    Args:
        prompt: The prompt string to evaluate
        scenarios: List of scenarios to evaluate on
        eval_runner: Callable(prompt, scenario) -> List[Dict] with eval results
        prompt_id: Identifier for this prompt
        baseline_score: Baseline score to compute delta

    Returns:
        PromptScore with aggregated results
    """
    all_eval_results = []
    scores = []

    for scenario in scenarios:
        scenario_id = scenario.get('id', 'unknown')
        category = scenario.get('category', 'unknown')

        try:
            # Call eval_runner
            results = eval_runner(prompt, scenario)

            for result in results:
                eval_result = EvalResult(
                    scenario_id=scenario_id,
                    category=category,
                    eval_name=result['eval_name'],
                    score=result['score'],
                    reason=result['reason'],
                    passed=result['passed']
                )
                all_eval_results.append(eval_result)
                scores.append(result['score'])

        except Exception as e:
            # Gracefully handle failures
            print(f"Warning: Evaluation failed for scenario {scenario_id}: {e}")
            continue

    # Aggregate scores
    mean_score_val = mean(scores) if scores else 0.0

    # Scores by category
    scores_by_cat = defaultdict(list)
    for er in all_eval_results:
        scores_by_cat[er.category].append(er.score)
    scores_by_category = {cat: mean(s) for cat, s in scores_by_cat.items()}

    # Scores by eval
    scores_by_ev = defaultdict(list)
    for er in all_eval_results:
        scores_by_ev[er.eval_name].append(er.score)
    scores_by_eval = {ev: mean(s) for ev, s in scores_by_ev.items()}

    return PromptScore(
        prompt_id=prompt_id,
        mean_score=mean_score_val,
        scores_by_category=scores_by_category,
        scores_by_eval=scores_by_eval,
        eval_results=all_eval_results,
        delta_from_baseline=mean_score_val - baseline_score
    )
