"""Tests for population building."""

import pytest
from collections import OrderedDict
from prompt_evolver.population import (
    build_population,
    build_indicator_matrix,
    get_versions_by_component
)
from prompt_evolver.state import ComponentVersionRecord


def create_test_version_pool():
    """Create test version pool."""
    return [
        ComponentVersionRecord("greeting", "greeting_v0", "Hello!", None, 0, None),
        ComponentVersionRecord("greeting", "greeting_v1", "Hi there!", None, 1, None),
        ComponentVersionRecord("instruction", "instruction_v0", "Learn.", None, 0, None),
    ]


def test_get_versions_by_component():
    """Test filtering versions by component."""
    version_pool = create_test_version_pool()

    greeting_versions = get_versions_by_component(version_pool, "greeting")

    assert len(greeting_versions) == 2
    assert all(v.component_name == "greeting" for v in greeting_versions)


def test_build_population_basic():
    """Test basic population building."""
    version_pool = create_test_version_pool()
    component_names = ["greeting", "instruction"]
    baseline_components = OrderedDict([
        ("greeting", "<!-- @component: greeting -->\nHello!"),
        ("instruction", "<!-- @component: instruction -->\nLearn.")
    ])

    population = build_population(
        preamble="Preamble\n",
        component_names=component_names,
        version_pool=version_pool,
        population_size=4,
        baseline_components=baseline_components,
        random_seed=42
    )

    assert len(population.prompts) == 4
    assert len(population.version_pool) == 3

    # Each prompt should have indicators for all versions
    for prompt in population.prompts:
        assert len(prompt.version_indicators) == 3
        # Each prompt uses exactly one version per component
        greeting_count = sum(
            1 for vid in ["greeting_v0", "greeting_v1"]
            if prompt.version_indicators.get(vid, 0) == 1
        )
        assert greeting_count == 1


def test_build_indicator_matrix():
    """Test indicator matrix construction."""
    version_pool = create_test_version_pool()
    component_names = ["greeting", "instruction"]
    baseline_components = OrderedDict([
        ("greeting", "<!-- @component: greeting -->\nHello!"),
        ("instruction", "<!-- @component: instruction -->\nLearn.")
    ])

    population = build_population(
        preamble="Preamble\n",
        component_names=component_names,
        version_pool=version_pool,
        population_size=3,
        baseline_components=baseline_components,
        random_seed=42
    )

    matrix = build_indicator_matrix(population)

    # Shape: n_prompts × n_versions
    assert matrix.shape == (3, 3)

    # Each row (prompt) should have exactly 2 ones (one per component)
    for row in matrix:
        assert row.sum() == 2

    # All entries should be 0 or 1
    assert set(matrix.flatten()) <= {0, 1}


def test_population_reproducibility():
    """Test that same seed produces same population."""
    version_pool = create_test_version_pool()
    component_names = ["greeting", "instruction"]
    baseline_components = OrderedDict([
        ("greeting", "<!-- @component: greeting -->\nHello!"),
        ("instruction", "<!-- @component: instruction -->\nLearn.")
    ])

    pop1 = build_population(
        preamble="Preamble\n",
        component_names=component_names,
        version_pool=version_pool,
        population_size=3,
        baseline_components=baseline_components,
        random_seed=42
    )

    pop2 = build_population(
        preamble="Preamble\n",
        component_names=component_names,
        version_pool=version_pool,
        population_size=3,
        baseline_components=baseline_components,
        random_seed=42
    )

    # Should produce identical prompts
    for p1, p2 in zip(pop1.prompts, pop2.prompts):
        assert p1.component_versions == p2.component_versions
