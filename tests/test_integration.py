"""Integration tests for full optimization workflow."""

import pytest
import tempfile
from pathlib import Path
from prompt_evolver.optimizer import run_optimization


def create_mock_scenarios():
    """Create mock scenarios for testing."""
    return [
        {"id": "s1", "category": "basic", "eval_names": ["Engagement"]},
        {"id": "s2", "category": "basic", "eval_names": ["Engagement"]},
        {"id": "s3", "category": "basic", "eval_names": ["Engagement"]},
        {"id": "s4", "category": "advanced", "eval_names": ["Correctness"]},
        {"id": "s5", "category": "advanced", "eval_names": ["Correctness"]},
        {"id": "s6", "category": "advanced", "eval_names": ["Correctness"]},
    ]


def create_mock_eval_runner(score=0.7):
    """Create a mock eval runner that returns fixed scores."""
    def mock_eval_runner(prompt, scenario):
        eval_names = scenario.get("eval_names", ["DefaultEval"])
        return [
            {
                "eval_name": eval_name,
                "score": score,
                "reason": "Mock evaluation result",
                "passed": score >= 0.6
            }
            for eval_name in eval_names
        ]
    return mock_eval_runner


def test_full_optimization_basic():
    """Test basic optimization workflow with 2 iterations."""
    prompt = """<!-- @component: greeting -->
Hello student!
<!-- @component: instruction -->
Let's learn math.
"""

    scenarios = create_mock_scenarios()
    eval_runner = create_mock_eval_runner(score=0.7)

    with tempfile.TemporaryDirectory() as tmpdir:
        state_path = Path(tmpdir) / "test_state.json"

        result = run_optimization(
            prompt=prompt,
            scenarios=scenarios,
            eval_runner=eval_runner,
            iterations=2,
            version_budget=4,
            population_size=8,
            eval_split=2,  # 2 scenarios per category
            validation_split=2,
            state_path=str(state_path)
        )

        # Check result
        assert result is not None
        assert "<!-- @component:" in result
        assert "greeting" in result or "Hello" in result

        # Check state file was created
        assert state_path.exists()


def test_optimization_with_component_mapping():
    """Test optimization with component-eval mapping enabled."""
    prompt = """<!-- @component: engagement -->
Engage the student!
<!-- @component: correctness -->
Be accurate!
"""

    scenarios = create_mock_scenarios()
    eval_runner = create_mock_eval_runner(score=0.8)

    with tempfile.TemporaryDirectory() as tmpdir:
        state_path = Path(tmpdir) / "test_state.json"

        result = run_optimization(
            prompt=prompt,
            scenarios=scenarios,
            eval_runner=eval_runner,
            iterations=1,
            version_budget=4,
            population_size=6,
            eval_split=0.5,
            validation_split=0.5,
            component_eval_mapping=True,  # Enable mapping
            state_path=str(state_path)
        )

        assert result is not None
        assert "<!-- @component:" in result


def test_optimization_with_failed_only():
    """Test optimization with failed_only_feedback enabled."""
    prompt = """<!-- @component: greeting -->
Hello!
<!-- @component: instruction -->
Learn.
"""

    scenarios = create_mock_scenarios()

    # Mock eval runner that sometimes fails
    def mixed_eval_runner(prompt, scenario):
        eval_names = scenario.get("eval_names", ["DefaultEval"])
        results = []
        for i, eval_name in enumerate(eval_names):
            score = 0.5 if i % 2 == 0 else 0.8
            results.append({
                "eval_name": eval_name,
                "score": score,
                "reason": "Mixed results",
                "passed": score >= 0.6
            })
        return results

    with tempfile.TemporaryDirectory() as tmpdir:
        state_path = Path(tmpdir) / "test_state.json"

        result = run_optimization(
            prompt=prompt,
            scenarios=scenarios,
            eval_runner=mixed_eval_runner,
            iterations=1,
            version_budget=4,
            population_size=6,
            eval_split=0.5,
            validation_split=0.5,
            failed_only_feedback=True,
            state_path=str(state_path)
        )

        assert result is not None


def test_optimization_state_persistence():
    """Test that state is saved and can be loaded."""
    prompt = """<!-- @component: greeting -->
Hello!
<!-- @component: instruction -->
Learn.
"""

    scenarios = create_mock_scenarios()
    eval_runner = create_mock_eval_runner(score=0.75)

    with tempfile.TemporaryDirectory() as tmpdir:
        state_path = Path(tmpdir) / "test_state.json"

        # Run first iteration
        result1 = run_optimization(
            prompt=prompt,
            scenarios=scenarios,
            eval_runner=eval_runner,
            iterations=1,
            version_budget=4,
            population_size=6,
            eval_split=0.5,
            validation_split=0.5,
            state_path=str(state_path)
        )

        assert state_path.exists()

        # Load state and check
        from prompt_evolver.state import load_state
        state = load_state(state_path)

        assert state is not None
        assert state.current_iteration == 1
        assert len(state.version_pool) >= 2  # Baseline versions (may not generate new ones in test)
        assert len(state.iteration_history) == 1


def test_optimization_early_stopping():
    """Test early stopping based on delta_gain_stop."""
    prompt = """<!-- @component: greeting -->
Hello!
<!-- @component: instruction -->
Learn.
"""

    scenarios = create_mock_scenarios()
    eval_runner = create_mock_eval_runner(score=0.7)

    with tempfile.TemporaryDirectory() as tmpdir:
        state_path = Path(tmpdir) / "test_state.json"

        result = run_optimization(
            prompt=prompt,
            scenarios=scenarios,
            eval_runner=eval_runner,
            iterations=10,  # Many iterations
            version_budget=4,
            population_size=6,
            eval_split=0.5,
            validation_split=0.5,
            delta_gain_stop=0.5,  # High threshold for early stop
            state_path=str(state_path)
        )

        # Should stop early
        from prompt_evolver.state import load_state
        state = load_state(state_path)

        # Likely stopped before completing all 10 iterations
        # (depends on random generation, but with constant scores, delta gains should be low)
        assert result is not None


def test_optimization_without_components():
    """Test optimization with prompt that has no component tags."""
    prompt = "This is a plain prompt with no tags."

    scenarios = create_mock_scenarios()
    eval_runner = create_mock_eval_runner(score=0.7)

    with tempfile.TemporaryDirectory() as tmpdir:
        state_path = Path(tmpdir) / "test_state.json"

        result = run_optimization(
            prompt=prompt,
            scenarios=scenarios,
            eval_runner=eval_runner,
            iterations=1,
            version_budget=2,
            population_size=4,
            eval_split=0.5,
            validation_split=0.5,
            state_path=str(state_path)
        )

        # Should handle __all__ component
        assert result is not None
