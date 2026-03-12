"""Tests for state management."""

import pytest
import tempfile
import json
from pathlib import Path

from prompt_evolver.state import (
    ComponentVersionRecord,
    IterationState,
    OptimizerState,
    load_state,
    save_state,
    apply_recency_weights
)


def test_load_state_nonexistent():
    """Test loading state from nonexistent file."""
    result = load_state(Path("/nonexistent/path.json"))
    assert result is None


def test_save_and_load_state():
    """Test saving and loading state."""
    version_pool = [
        ComponentVersionRecord("comp1", "v1", "text1", None, 0, 0.5),
        ComponentVersionRecord("comp2", "v2", "text2", "meta1", 1, 0.3),
    ]

    iteration_history = [
        IterationState(
            iteration=1,
            baseline_score=0.7,
            best_prompt_score=0.75,
            best_prompt="best prompt text",
            component_delta_gains={"comp1": 0.1, "comp2": 0.2},
            meta_prompt_efficacies={"meta1": 0.3},
            version_allocations={"comp1": 2, "comp2": 2}
        )
    ]

    state = OptimizerState(
        current_iteration=1,
        version_pool=version_pool,
        iteration_history=iteration_history,
        meta_prompt_weights={"meta1": 1.2},
        lambda_decay=0.5
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        state_path = Path(tmpdir) / "test_state.json"

        # Save
        save_state(state, state_path)
        assert state_path.exists()

        # Load
        loaded = load_state(state_path)

        assert loaded is not None
        assert loaded.current_iteration == 1
        assert len(loaded.version_pool) == 2
        assert len(loaded.iteration_history) == 1
        assert loaded.lambda_decay == 0.5

        # Check version pool
        assert loaded.version_pool[0].component_name == "comp1"
        assert loaded.version_pool[0].coefficient == 0.5
        assert loaded.version_pool[1].meta_prompt_used == "meta1"

        # Check iteration history
        assert loaded.iteration_history[0].iteration == 1
        assert loaded.iteration_history[0].baseline_score == 0.7


def test_state_json_format():
    """Test that state is saved in valid JSON format."""
    version_pool = [
        ComponentVersionRecord("comp1", "v1", "text1", None, 0, 0.5),
    ]

    state = OptimizerState(
        current_iteration=0,
        version_pool=version_pool,
        iteration_history=[],
        meta_prompt_weights={},
        lambda_decay=0.5
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        state_path = Path(tmpdir) / "test_state.json"

        save_state(state, state_path)

        # Verify it's valid JSON
        with open(state_path, 'r') as f:
            data = json.load(f)

        assert "current_iteration" in data
        assert "version_pool" in data
        assert "iteration_history" in data
        assert "meta_prompt_weights" in data


def test_apply_recency_weights():
    """Test recency weight calculation."""
    history = [
        IterationState(0, 0.7, 0.7, "p1", {}, {}, {}),
        IterationState(1, 0.7, 0.75, "p2", {}, {}, {}),
        IterationState(2, 0.7, 0.8, "p3", {}, {}, {}),
    ]

    weights = apply_recency_weights(history, lambda_decay=0.5)

    assert len(weights) == 3

    # Weights should sum to 1
    assert pytest.approx(sum(weights), abs=1e-6) == 1.0

    # More recent iterations should have higher weight
    assert weights[2] > weights[1]
    assert weights[1] > weights[0]


def test_apply_recency_weights_empty():
    """Test recency weights with empty history."""
    weights = apply_recency_weights([], lambda_decay=0.5)
    assert weights == []


def test_apply_recency_weights_single():
    """Test recency weights with single iteration."""
    history = [
        IterationState(0, 0.7, 0.7, "p1", {}, {}, {})
    ]

    weights = apply_recency_weights(history, lambda_decay=0.5)

    assert len(weights) == 1
    assert weights[0] == 1.0  # Single weight should be 1


def test_apply_recency_weights_decay_parameter():
    """Test that lambda_decay affects weight distribution."""
    history = [
        IterationState(i, 0.7, 0.7 + i * 0.05, f"p{i}", {}, {}, {})
        for i in range(5)
    ]

    weights_low_decay = apply_recency_weights(history, lambda_decay=0.1)
    weights_high_decay = apply_recency_weights(history, lambda_decay=2.0)

    # With high decay, most recent should be much more dominant
    assert weights_high_decay[-1] > weights_low_decay[-1]

    # With low decay, distribution should be more even
    assert max(weights_low_decay) - min(weights_low_decay) < max(weights_high_decay) - min(weights_high_decay)
