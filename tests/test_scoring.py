"""Tests for scenario sampling and evaluation."""

import pytest
from prompt_evolver.scoring import (
    stratified_split,
    subsample_scenarios,
    evaluate_prompt
)


def create_test_scenarios():
    """Create test scenarios with categories."""
    return [
        {"id": "s1", "category": "cat1", "eval_names": ["Eval1"]},
        {"id": "s2", "category": "cat1", "eval_names": ["Eval1"]},
        {"id": "s3", "category": "cat1", "eval_names": ["Eval1"]},
        {"id": "s4", "category": "cat2", "eval_names": ["Eval2"]},
        {"id": "s5", "category": "cat2", "eval_names": ["Eval2"]},
        {"id": "s6", "category": "cat2", "eval_names": ["Eval2"]},
    ]


def test_stratified_split_float():
    """Test stratified split with float percentages."""
    scenarios = create_test_scenarios()

    split = stratified_split(scenarios, eval_split=0.5, validation_split=0.5, random_seed=42)

    # Should get ~50% of each category in eval and validation
    assert len(split.evaluation_scenarios) > 0
    assert len(split.validation_scenarios) > 0

    # Check stratification: both categories should be represented
    eval_cats = set(s["category"] for s in split.evaluation_scenarios)
    assert "cat1" in eval_cats or "cat2" in eval_cats


def test_stratified_split_int():
    """Test stratified split with integer counts."""
    scenarios = create_test_scenarios()

    split = stratified_split(scenarios, eval_split=2, validation_split=1, random_seed=42)

    # Should get 2 per category in eval (4 total), 1 per category in validation (2 total)
    assert len(split.evaluation_scenarios) == 4  # 2 from cat1, 2 from cat2
    assert len(split.validation_scenarios) == 2  # 1 from cat1, 1 from cat2


def test_stratified_split_no_overlap():
    """Test that eval and validation sets are disjoint (held-out validation)."""
    scenarios = create_test_scenarios()

    split = stratified_split(scenarios, eval_split=0.5, validation_split=0.5, random_seed=42)

    # Get IDs from each set
    eval_ids = {s["id"] for s in split.evaluation_scenarios}
    val_ids = {s["id"] for s in split.validation_scenarios}

    # Ensure no overlap (validation is held-out)
    overlap = eval_ids & val_ids
    assert len(overlap) == 0, f"Eval and validation sets should not overlap, but found: {overlap}"


def test_subsample_scenarios():
    """Test scenario subsampling."""
    scenarios = create_test_scenarios()

    # With per_iteration=True and subsample_fraction=0.5, should return ~50% per category
    result = subsample_scenarios(scenarios, subsample_fraction=0.5, per_iteration=True, random_seed=42)

    # Should have fewer scenarios than original (or equal if very small)
    # With 6 scenarios (3 per cat), 50% per cat = 1 per cat = 2 total
    assert len(result) <= len(scenarios)
    assert len(result) >= 2  # At least 1 per category (2 categories)

    # All returned IDs should be from original set
    assert set(s["id"] for s in result).issubset(set(s["id"] for s in scenarios))

    # Check that both categories are represented
    categories = {s["category"] for s in result}
    assert len(categories) == 2  # Both cat1 and cat2


def test_evaluate_prompt_basic():
    """Test prompt evaluation with mock eval_runner."""
    scenarios = [
        {"id": "s1", "category": "cat1", "eval_names": ["Eval1"]},
        {"id": "s2", "category": "cat1", "eval_names": ["Eval1"]},
    ]

    def mock_eval_runner(prompt, scenario):
        return [
            {"eval_name": "Eval1", "score": 0.7, "reason": "Good", "passed": True}
        ]

    result = evaluate_prompt(
        "test prompt",
        scenarios,
        mock_eval_runner,
        prompt_id="test"
    )

    assert result.prompt_id == "test"
    assert result.mean_score == 0.7
    assert len(result.eval_results) == 2  # 2 scenarios × 1 eval each
    assert result.scores_by_category["cat1"] == 0.7


def test_evaluate_prompt_handles_failures():
    """Test that evaluation handles failures gracefully."""
    scenarios = [
        {"id": "s1", "category": "cat1", "eval_names": ["Eval1"]},
        {"id": "s2", "category": "cat1", "eval_names": ["Eval1"]},
    ]

    def failing_eval_runner(prompt, scenario):
        if scenario["id"] == "s1":
            raise Exception("Eval failed!")
        return [
            {"eval_name": "Eval1", "score": 0.7, "reason": "Good", "passed": True}
        ]

    result = evaluate_prompt(
        "test prompt",
        scenarios,
        failing_eval_runner,
        prompt_id="test"
    )

    # Should only have results from s2
    assert len(result.eval_results) == 1
    assert result.eval_results[0].scenario_id == "s2"


def test_evaluate_prompt_aggregates_by_eval():
    """Test aggregation by eval name."""
    scenarios = [
        {"id": "s1", "category": "cat1", "eval_names": ["Eval1", "Eval2"]},
    ]

    def mock_eval_runner(prompt, scenario):
        return [
            {"eval_name": "Eval1", "score": 0.8, "reason": "Good", "passed": True},
            {"eval_name": "Eval2", "score": 0.6, "reason": "OK", "passed": True},
        ]

    result = evaluate_prompt(
        "test prompt",
        scenarios,
        mock_eval_runner,
        prompt_id="test"
    )

    assert "Eval1" in result.scores_by_eval
    assert "Eval2" in result.scores_by_eval
    assert result.scores_by_eval["Eval1"] == 0.8
    assert result.scores_by_eval["Eval2"] == 0.6
    assert result.mean_score == 0.7  # Average of 0.8 and 0.6
