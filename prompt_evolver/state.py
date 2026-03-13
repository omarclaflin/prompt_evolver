"""State management for cross-iteration optimization tracking."""

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional
import math


@dataclass
class ComponentVersionRecord:
    """Record of a component version."""
    component_name: str
    version_id: str
    text: str
    meta_prompt_used: Optional[str]
    iteration_created: int
    coefficient: Optional[float] = None


@dataclass
class IterationState:
    """State captured at end of each iteration."""
    iteration: int
    baseline_score: float
    best_prompt_score: float
    best_prompt: str
    component_delta_gains: Dict[str, float]
    meta_prompt_efficacies: Dict[str, float]
    version_allocations: Dict[str, int]


@dataclass
class OptimizerState:
    """Complete optimizer state across iterations."""
    current_iteration: int
    version_pool: List[ComponentVersionRecord]
    iteration_history: List[IterationState]
    meta_prompt_weights: Dict[str, float]
    lambda_decay: float = 0.5
    global_best_prompt: Optional[str] = None
    global_best_score: float = -1.0


def load_state(path: Path) -> Optional[OptimizerState]:
    """Load optimizer state from JSON file."""
    if not path.exists():
        return None

    with open(path, 'r') as f:
        data = json.load(f)

    # Reconstruct dataclasses
    version_pool = [ComponentVersionRecord(**v) for v in data['version_pool']]
    iteration_history = [IterationState(**h) for h in data['iteration_history']]

    return OptimizerState(
        current_iteration=data['current_iteration'],
        version_pool=version_pool,
        iteration_history=iteration_history,
        meta_prompt_weights=data['meta_prompt_weights'],
        lambda_decay=data.get('lambda_decay', 0.5),
        global_best_prompt=data.get('global_best_prompt'),
        global_best_score=data.get('global_best_score', -1.0)
    )


def save_state(state: OptimizerState, path: Path) -> None:
    """Save optimizer state to JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        'current_iteration': state.current_iteration,
        'version_pool': [asdict(v) for v in state.version_pool],
        'iteration_history': [asdict(h) for h in state.iteration_history],
        'meta_prompt_weights': state.meta_prompt_weights,
        'lambda_decay': state.lambda_decay,
        'global_best_prompt': state.global_best_prompt,
        'global_best_score': state.global_best_score,
    }

    with open(path, 'w') as f:
        json.dump(data, f, indent=2)


def apply_recency_weights(history: List[IterationState], lambda_decay: float) -> List[float]:
    """
    Apply exponential-Gaussian recency weighting to iteration history.

    w(t) = exp(-λ * (T - t)) where T = current iteration, t = past iteration

    Args:
        history: List of iteration states
        lambda_decay: Decay rate (higher = more recency bias)

    Returns:
        List of weights (one per iteration)
    """
    if not history:
        return []

    T = len(history) - 1  # Current iteration index (0-based)
    weights = []

    for t in range(len(history)):
        w = math.exp(-lambda_decay * (T - t))
        weights.append(w)

    # Normalize to sum to 1
    total = sum(weights)
    if total > 0:
        weights = [w / total for w in weights]

    return weights
