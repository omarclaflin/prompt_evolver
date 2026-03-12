"""Example usage of the prompt_evolver library.

This script demonstrates how to use the prompt optimizer with mock data.
For real usage, replace the mock eval_runner with your actual evaluation logic.
"""

from prompt_evolver import run_optimization


def create_example_prompt():
    """Create an example prompt with component tags."""
    return """<!-- @component: greeting -->
Hello! I'm your AI tutor, and I'm here to help you learn.

<!-- @component: instruction -->
Let's work through this problem step by step. I'll guide you with questions.

<!-- @component: encouragement -->
You're doing great! Keep up the good work.
"""


def create_example_scenarios():
    """Create example scenarios for testing."""
    scenarios = []

    # Algebra scenarios
    for i in range(10):
        scenarios.append({
            "id": f"algebra_{i}",
            "category": "algebra",
            "eval_names": ["Clarity", "Engagement", "Correctness"],
            "input": f"Solve for x: {i+2}x + 3 = {(i+2)*5 + 3}"
        })

    # Geometry scenarios
    for i in range(10):
        scenarios.append({
            "id": f"geometry_{i}",
            "category": "geometry",
            "eval_names": ["Clarity", "Engagement", "Correctness"],
            "input": f"Find the area of a circle with radius {i+1}"
        })

    return scenarios


def create_mock_eval_runner():
    """
    Create a mock eval runner for demonstration.

    In real usage, this would:
    1. Use the prompt to configure your agent/model
    2. Send the scenario input to the agent
    3. Collect the agent's response
    4. Run your eval metrics on the response
    5. Return eval results
    """
    def eval_runner(prompt, scenario):
        # Mock: Assign random-ish scores based on prompt length
        # (In reality, you'd run actual evals)
        base_score = 0.6
        prompt_bonus = (len(prompt) % 100) / 500  # 0 to 0.2

        results = []
        for eval_name in scenario["eval_names"]:
            # Vary scores slightly by eval type
            score_variation = hash(eval_name) % 30 / 100  # -0.15 to 0.15
            score = max(0.0, min(1.0, base_score + prompt_bonus + score_variation))

            results.append({
                "eval_name": eval_name,
                "score": score,
                "reason": f"Mock evaluation result for {eval_name}",
                "passed": score >= 0.7
            })

        return results

    return eval_runner


def main():
    """Run example optimization."""
    print("Prompt Evolver - Example Usage")
    print("=" * 70)

    # Prepare inputs
    prompt = create_example_prompt()
    scenarios = create_example_scenarios()
    eval_runner = create_mock_eval_runner()

    print(f"\nStarting with {len(scenarios)} scenarios")
    print(f"Prompt has {len(prompt)} characters")
    print("\nInitial prompt:")
    print(prompt)
    print("\n" + "=" * 70)

    # Run optimization
    try:
        optimized_prompt = run_optimization(
            prompt=prompt,
            scenarios=scenarios,
            eval_runner=eval_runner,
            model="gpt-4o-mini",
            iterations=3,
            eval_split=5,  # 5 scenarios per category for evaluation
            validation_split=3,  # 3 scenarios per category for validation
            version_budget=9,  # 3 versions per component
            population_size=18,  # 2x version budget
            component_eval_mapping=False,
            failed_only_feedback=False,
            condense_feedback=False,
            use_meta_prompts=False,
            delta_gain_stop=None,
            state_path="example_evolver_state.json"
        )

        print("\n" + "=" * 70)
        print("OPTIMIZATION COMPLETE")
        print("=" * 70)
        print("\nOptimized prompt:")
        print(optimized_prompt)

        print("\n" + "=" * 70)
        print("State saved to: example_evolver_state.json")
        print("You can inspect the state file to see:")
        print("  - All component versions generated")
        print("  - Regression coefficients per version")
        print("  - Iteration history with scores and delta gains")

    except Exception as e:
        print(f"\n✗ Optimization failed: {e}")
        print("\nNote: This example requires OpenAI API key to be set.")
        print("Set OPENAI_API_KEY environment variable or add to .env file.")


if __name__ == "__main__":
    main()
