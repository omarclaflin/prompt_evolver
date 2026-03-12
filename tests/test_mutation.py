"""Tests for mutation and feedback filtering."""

import pytest
from prompt_evolver.mutation import (
    normalize_name,
    find_component_eval_mappings,
    filter_feedback_for_component,
    FeedbackItem
)


def test_normalize_name():
    """Test name normalization."""
    assert normalize_name("SocraticMethod") == "socraticmethod"
    assert normalize_name("socratic_method") == "socraticmethod"
    assert normalize_name("socratic-method") == "socraticmethod"
    assert normalize_name("SOCRATIC_METHOD") == "socraticmethod"


def test_find_component_eval_mappings():
    """Test auto-detection of component-eval mappings with exact match."""
    components = ["socratic_method", "relevance", "greeting"]
    evals = ["SocraticMethod", "Relevance", "Correctness"]

    mappings = find_component_eval_mappings(components, evals)

    # Exact match after normalization: socratic_method -> socraticmethod == SocraticMethod -> socraticmethod
    assert "socratic_method" in mappings
    assert "SocraticMethod" in mappings["socratic_method"]
    assert "relevance" in mappings
    assert "Relevance" in mappings["relevance"]
    assert "greeting" not in mappings  # No match


def test_filter_feedback_no_mapping():
    """Test feedback filtering without mapping (all feedback returned)."""
    feedback = [
        FeedbackItem("Eval1", 0.7, "Good", True, "s1", "cat1"),
        FeedbackItem("Eval2", 0.5, "Poor", False, "s2", "cat1"),
        FeedbackItem("Eval3", 0.8, "Great", True, "s3", "cat2"),
    ]

    # No mapping -> all feedback
    result = filter_feedback_for_component("greeting", feedback, None, False)
    assert len(result) == 3


def test_filter_feedback_with_mapping_matched():
    """Test feedback filtering with mapping for matched component."""
    feedback = [
        FeedbackItem("SocraticMethod", 0.7, "Good", True, "s1", "cat1"),
        FeedbackItem("Relevance", 0.5, "Poor", False, "s2", "cat1"),
        FeedbackItem("Correctness", 0.8, "Great", True, "s3", "cat2"),
    ]

    mapping = {"socratic_method": ["SocraticMethod"]}

    result = filter_feedback_for_component("socratic_method", feedback, mapping, False)

    assert len(result) == 1
    assert result[0].eval_name == "SocraticMethod"


def test_filter_feedback_with_mapping_unmapped():
    """Test feedback filtering with mapping for unmapped component (all feedback returned)."""
    feedback = [
        FeedbackItem("SocraticMethod", 0.7, "Good", True, "s1", "cat1"),
        FeedbackItem("Relevance", 0.5, "Poor", False, "s2", "cat1"),
    ]

    mapping = {"socratic_method": ["SocraticMethod"]}

    # "greeting" not in mapping -> all feedback
    result = filter_feedback_for_component("greeting", feedback, mapping, False)

    assert len(result) == 2


def test_filter_feedback_failed_only():
    """Test failed_only filter."""
    feedback = [
        FeedbackItem("Eval1", 0.7, "Good", True, "s1", "cat1"),
        FeedbackItem("Eval2", 0.5, "Poor", False, "s2", "cat1"),
        FeedbackItem("Eval3", 0.8, "Great", True, "s3", "cat2"),
    ]

    result = filter_feedback_for_component("greeting", feedback, None, failed_only=True)

    assert len(result) == 1
    assert result[0].eval_name == "Eval2"
    assert not result[0].passed


def test_filter_feedback_mapping_and_failed_only():
    """Test combined mapping and failed_only filters."""
    feedback = [
        FeedbackItem("SocraticMethod", 0.7, "Good", True, "s1", "cat1"),
        FeedbackItem("SocraticMethod", 0.5, "Poor", False, "s2", "cat1"),
        FeedbackItem("Relevance", 0.8, "Great", True, "s3", "cat2"),
    ]

    mapping = {"socratic_method": ["SocraticMethod"]}

    result = filter_feedback_for_component("socratic_method", feedback, mapping, failed_only=True)

    assert len(result) == 1
    assert result[0].eval_name == "SocraticMethod"
    assert not result[0].passed
    assert result[0].score == 0.5
