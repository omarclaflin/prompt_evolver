"""Tests for regression models."""

import pytest
import numpy as np
from collections import OrderedDict

from prompt_evolver.regression import (
    drop_reference_categories,
    fit_component_version_regression,
    compute_delta_gains
)
from prompt_evolver.state import ComponentVersionRecord
from prompt_evolver.population import Population, PromptCandidate


def create_test_population():
    """Create a minimal test population."""
    # 3 versions: 2 for comp_a, 1 for comp_b
    version_pool = [
        ComponentVersionRecord("comp_a", "comp_a_v0", "text0", None, 0, None),
        ComponentVersionRecord("comp_a", "comp_a_v1", "text1", None, 1, None),
        ComponentVersionRecord("comp_b", "comp_b_v0", "text2", None, 0, None),
    ]

    # 2 prompts
    prompts = [
        PromptCandidate(
            "p1",
            {"comp_a": "comp_a_v0", "comp_b": "comp_b_v0"},
            "full_text_1",
            {"comp_a_v0": 1, "comp_a_v1": 0, "comp_b_v0": 1}
        ),
        PromptCandidate(
            "p2",
            {"comp_a": "comp_a_v1", "comp_b": "comp_b_v0"},
            "full_text_2",
            {"comp_a_v0": 0, "comp_a_v1": 1, "comp_b_v0": 1}
        ),
    ]

    return Population(prompts, version_pool, "baseline", "preamble")


def test_drop_reference_categories():
    """Test that reference categories are correctly dropped."""
    population = create_test_population()

    # Full indicator matrix (2 prompts × 3 versions)
    X = np.array([
        [1, 0, 1],  # p1: comp_a_v0, comp_b_v0
        [0, 1, 1],  # p2: comp_a_v1, comp_b_v0
    ])

    X_encoded, kept_version_ids = drop_reference_categories(X, population)

    # Should drop comp_a_v0 (first for comp_a) and comp_b_v0 (first for comp_b)
    # Only comp_a_v1 should remain
    assert X_encoded.shape == (2, 1)
    assert len(kept_version_ids) == 1
    assert "comp_a_v1" in kept_version_ids


def test_fit_component_version_regression():
    """Test component-version regression fitting."""
    population = create_test_population()

    # Scores: p2 (with comp_a_v1) scores higher than p1
    scores = [0.6, 0.8]
    baseline_score = 0.5

    result = fit_component_version_regression(population, scores, baseline_score)

    # Check structure
    assert result is not None
    assert "comp_a_v0" in result.coefficients
    assert "comp_a_v1" in result.coefficients
    assert "comp_b_v0" in result.coefficients

    # Reference categories should have 0 coefficient
    assert result.coefficients["comp_a_v0"] == 0.0
    assert result.coefficients["comp_b_v0"] == 0.0

    # comp_a_v1 should have positive coefficient (delta = 0.8 - baseline vs 0.6 - baseline)
    assert result.coefficients["comp_a_v1"] > 0


def test_compute_delta_gains():
    """Test delta gain computation."""
    coefficients = {
        "comp_a_v0": 0.0,
        "comp_a_v1": 0.3,
        "comp_b_v0": 0.0,
        "comp_b_v1": -0.1,
    }

    version_pool = [
        ComponentVersionRecord("comp_a", "comp_a_v0", "text0", None, 0, 0.0),
        ComponentVersionRecord("comp_a", "comp_a_v1", "text1", None, 1, 0.3),
        ComponentVersionRecord("comp_b", "comp_b_v0", "text2", None, 0, 0.0),
        ComponentVersionRecord("comp_b", "comp_b_v1", "text3", None, 1, -0.1),
    ]

    delta_gains = compute_delta_gains(coefficients, version_pool)

    assert "comp_a" in delta_gains
    assert "comp_b" in delta_gains

    # Delta gain = max - min
    assert delta_gains["comp_a"] == pytest.approx(0.3 - 0.0)
    assert delta_gains["comp_b"] == pytest.approx(0.0 - (-0.1))


def test_regression_with_empty_population():
    """Test regression handles empty population gracefully."""
    version_pool = []
    prompts = []
    population = Population(prompts, version_pool, "baseline", "preamble")

    scores = []
    baseline_score = 0.5

    result = fit_component_version_regression(population, scores, baseline_score)

    # Should return empty coefficients
    assert result.coefficients == {}
    assert result.r_squared == 0.0
