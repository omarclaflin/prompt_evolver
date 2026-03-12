"""Main optimization loop and entry point."""

from pathlib import Path
from typing import Callable, List, Dict, Optional, Union
from collections import OrderedDict

from .components import parse_components, reassemble, list_component_names
from .state import (
    ComponentVersionRecord,
    IterationState,
    OptimizerState,
    load_state,
    save_state
)
from .scoring import (
    stratified_split,
    subsample_scenarios,
    evaluate_prompt,
    EvalResult
)
from .mutation import (
    find_component_eval_mappings,
    filter_feedback_for_component,
    condense_feedback as condense_feedback_fn,
    generate_component_version,
    load_meta_prompts,
    sample_meta_prompt,
    MutationRequest,
    FeedbackItem
)
from .population import build_population, get_versions_by_component
from .regression import (
    fit_component_version_regression,
    compute_delta_gains,
    fit_meta_prompt_regression
)


def run_optimization(
    prompt: str,
    scenarios: List[Dict],
    eval_runner: Callable,
    model: str = "gpt-4o-mini",
    iterations: int = 5,
    eval_split: Union[float, int] = 0.5,
    validation_split: Union[float, int] = 0.5,
    version_budget: Optional[int] = None,
    population_size: Optional[int] = None,
    component_eval_mapping: Optional[bool] = None,
    failed_only_feedback: bool = False,
    condense_feedback: bool = False,
    use_meta_prompts: bool = False,
    delta_gain_stop: Optional[float] = None,
    state_path: str = "evolver_state.json",
) -> str:
    """
    Run component-aware prompt optimization.

    Args:
        prompt: Initial prompt with <!-- @component: name --> tags
        scenarios: List of scenario dicts (id, category, eval_names, ...)
        eval_runner: Callable(prompt, scenario) -> List[Dict] with eval results
        model: LLM model for generating mutations
        iterations: Number of optimization iterations
        eval_split: Float (%) or int (count) for evaluation scenarios per category
        validation_split: Float (%) or int (count) for validation scenarios per category
        version_budget: Total component versions to generate (default: components × 3)
        population_size: Number of prompts in population (default: version_budget × 2)
        component_eval_mapping: Enable auto name-matching between components and evals
        failed_only_feedback: Only use feedback where passed=False
        condense_feedback: Use LLM to condense feedback before mutation
        use_meta_prompts: Sample meta-prompts from metaprompt_instructions.txt
        delta_gain_stop: Early stopping threshold for max delta_gain
        state_path: Path to save/load state file

    Returns:
        Optimized prompt string
    """
    print("=" * 70)
    print("PROMPT EVOLVER - Component-Aware Optimization")
    print("=" * 70)

    # Parse components
    preamble, baseline_components = parse_components(prompt)
    component_names = list(baseline_components.keys())
    n_components = len(component_names)

    print(f"\nParsed {n_components} components: {component_names}")
    print(f"Preamble length: {len(preamble)} chars")

    # Set defaults
    if version_budget is None:
        version_budget = n_components * 3
    if population_size is None:
        population_size = version_budget * 2

    print(f"\nConfiguration:")
    print(f"  Iterations: {iterations}")
    print(f"  Version budget: {version_budget}")
    print(f"  Population size: {population_size}")
    print(f"  Eval split: {eval_split}")
    print(f"  Validation split: {validation_split}")
    print(f"  Component-eval mapping: {component_eval_mapping}")
    print(f"  Failed-only feedback: {failed_only_feedback}")
    print(f"  Condense feedback: {condense_feedback}")
    print(f"  Use meta-prompts: {use_meta_prompts}")
    print(f"  Delta gain stop: {delta_gain_stop}")

    # Load or initialize state
    state_file = Path(state_path)
    state = load_state(state_file)

    if state is None:
        print(f"\nInitializing new state...")
        state = OptimizerState(
            current_iteration=0,
            version_pool=[],
            iteration_history=[],
            meta_prompt_weights={},
            lambda_decay=0.5
        )
    else:
        print(f"\nLoaded existing state from {state_path}")
        print(f"  Current iteration: {state.current_iteration}")
        print(f"  Version pool size: {len(state.version_pool)}")

    # Split scenarios
    print(f"\nSplitting {len(scenarios)} scenarios...")
    scenario_split = stratified_split(scenarios, eval_split, validation_split)
    print(f"  Evaluation: {len(scenario_split.evaluation_scenarios)}")
    print(f"  Validation: {len(scenario_split.validation_scenarios)}")
    print(f"  Unused: {len(scenario_split.unused_scenarios)}")

    # Step 1: Baseline Evaluation
    print("\n" + "=" * 70)
    print("STEP 1: Baseline Evaluation")
    print("=" * 70)

    baseline_prompt = reassemble(preamble, baseline_components)
    baseline_result = evaluate_prompt(
        baseline_prompt,
        scenario_split.evaluation_scenarios,
        eval_runner,
        prompt_id="baseline"
    )
    baseline_score = baseline_result.mean_score

    print(f"\nBaseline score: {baseline_score:.4f}")
    print(f"  By category: {baseline_result.scores_by_category}")
    print(f"  By eval: {baseline_result.scores_by_eval}")

    # Convert eval results to feedback items
    baseline_feedback = [
        FeedbackItem(
            eval_name=er.eval_name,
            score=er.score,
            reason=er.reason,
            passed=er.passed,
            scenario_id=er.scenario_id,
            category=er.category
        )
        for er in baseline_result.eval_results
    ]

    # Find component-eval mappings
    component_eval_map = None
    if component_eval_mapping:
        all_eval_names = list(set(er.eval_name for er in baseline_result.eval_results))
        component_eval_map = find_component_eval_mappings(component_names, all_eval_names)
        print(f"\nComponent-Eval Mappings:")
        for comp, evals in component_eval_map.items():
            print(f"  {comp} → {evals}")
        unmapped = [c for c in component_names if c not in component_eval_map]
        if unmapped:
            print(f"  Unmapped components: {unmapped}")

    # Add baseline versions to pool if empty
    if not state.version_pool:
        for comp_name in component_names:
            version = ComponentVersionRecord(
                component_name=comp_name,
                version_id=f"{comp_name}_v0",
                text=baseline_components[comp_name],
                meta_prompt_used=None,
                iteration_created=0,
                coefficient=0.0
            )
            state.version_pool.append(version)
        print(f"\nAdded {len(component_names)} baseline versions to pool")

    # Load meta-prompts if needed
    meta_prompts = []
    used_combinations = set()
    if use_meta_prompts:
        meta_prompts_file = Path(__file__).parent / "metaprompt_instructions.txt"
        meta_prompts = load_meta_prompts(meta_prompts_file)
        print(f"\nLoaded {len(meta_prompts)} meta-prompts")

        # Initialize weights if empty
        if not state.meta_prompt_weights:
            state.meta_prompt_weights = {mp: 1.0 for mp in meta_prompts}

        # Track already-used combinations
        for version in state.version_pool:
            if version.meta_prompt_used:
                used_combinations.add((version.meta_prompt_used, version.component_name))

    # Main optimization loop
    for iter_num in range(state.current_iteration, iterations):
        print("\n" + "=" * 70)
        print(f"ITERATION {iter_num + 1}/{iterations}")
        print("=" * 70)

        # Step 2: Generate Component Versions
        print("\nSTEP 2: Generate Component Versions")

        # Compute version allocations
        if iter_num == 0 or not state.iteration_history:
            # First iteration: equal allocation
            versions_per_component = version_budget // n_components
            version_allocations = {comp: versions_per_component for comp in component_names}
            # Distribute remainder
            remainder = version_budget % n_components
            for i, comp in enumerate(component_names[:remainder]):
                version_allocations[comp] += 1
        else:
            # Proportional to delta_gain
            last_iter = state.iteration_history[-1]
            delta_gains = last_iter.component_delta_gains
            total_gain = sum(delta_gains.values())

            if total_gain > 0:
                # Compute proportional allocation
                version_allocations = {}
                for comp in component_names:
                    gain = delta_gains.get(comp, 0.0)
                    alloc = int((gain / total_gain) * version_budget)
                    version_allocations[comp] = max(0, alloc)

                # Redistribute remainder to components with highest delta_gain
                allocated_total = sum(version_allocations.values())
                remainder = version_budget - allocated_total

                if remainder > 0:
                    # Sort components by delta_gain (descending)
                    sorted_comps = sorted(component_names, key=lambda c: delta_gains.get(c, 0.0), reverse=True)
                    for i in range(remainder):
                        comp = sorted_comps[i % len(sorted_comps)]
                        version_allocations[comp] += 1
            else:
                # Fallback to equal
                versions_per_component = version_budget // n_components
                version_allocations = {comp: versions_per_component for comp in component_names}
                # Distribute remainder
                remainder = version_budget % n_components
                for i, comp in enumerate(component_names[:remainder]):
                    version_allocations[comp] += 1

        print(f"\nVersion allocations: {version_allocations}")

        # Generate versions
        new_versions = []
        for comp_name in component_names:
            n_versions = version_allocations.get(comp_name, 0)
            if n_versions == 0:
                continue

            # Get current text (latest version or baseline)
            existing_versions = get_versions_by_component(state.version_pool, comp_name)
            if existing_versions:
                current_text = existing_versions[-1].text
            else:
                current_text = baseline_components[comp_name]

            # Filter feedback
            filtered_feedback = filter_feedback_for_component(
                comp_name,
                baseline_feedback,
                component_eval_map,
                failed_only_feedback
            )

            print(f"\n  Generating {n_versions} versions for component '{comp_name}'")
            print(f"    Feedback items: {len(filtered_feedback)}")

            for v_idx in range(n_versions):
                # Sample meta-prompt or use default
                if use_meta_prompts and meta_prompts:
                    instruction = sample_meta_prompt(
                        meta_prompts,
                        state.meta_prompt_weights,
                        used_combinations,
                        comp_name
                    )
                    if instruction is None:
                        instruction = "Rewrite this section. Keep it concise and actionable. Do not include extraneous text."
                        meta_prompt_id = None
                    else:
                        meta_prompt_id = instruction
                        used_combinations.add((instruction, comp_name))
                else:
                    instruction = "Rewrite this section. Keep it concise and actionable. Do not include extraneous text."
                    meta_prompt_id = None

                # Condense feedback if requested
                feedback_to_use = filtered_feedback
                if condense_feedback and filtered_feedback:
                    condensed_text = condense_feedback_fn(filtered_feedback, model)
                    # Create single feedback item with condensed text
                    feedback_to_use = [
                        FeedbackItem(
                            eval_name="Condensed",
                            score=0.5,
                            reason=condensed_text,
                            passed=False,
                            scenario_id="condensed",
                            category="all"
                        )
                    ]

                # Generate version
                request = MutationRequest(
                    component_name=comp_name,
                    current_text=current_text,
                    feedback=feedback_to_use,
                    instruction=instruction,
                    meta_prompt_id=meta_prompt_id
                )

                try:
                    new_text = generate_component_version(request, model)
                    version_id = f"{comp_name}_v{len(existing_versions) + v_idx + 1}"

                    version = ComponentVersionRecord(
                        component_name=comp_name,
                        version_id=version_id,
                        text=new_text,
                        meta_prompt_used=meta_prompt_id,
                        iteration_created=iter_num + 1,
                        coefficient=None
                    )
                    new_versions.append(version)
                    state.version_pool.append(version)

                    print(f"    ✓ Generated {version_id}")

                except Exception as e:
                    print(f"    ✗ Failed to generate version: {e}")
                    continue

        print(f"\nGenerated {len(new_versions)} new versions")
        print(f"Total version pool size: {len(state.version_pool)}")

        # Step 3: Build Population & Evaluate
        print("\nSTEP 3: Build Population & Evaluate")

        population = build_population(
            preamble,
            component_names,
            state.version_pool,
            population_size,
            baseline_components,
            random_seed=42 + iter_num
        )

        print(f"\nBuilt population of {len(population.prompts)} prompts")
        print("Evaluating population...")

        # Subsample scenarios for this evaluation (50% per category)
        eval_scenarios = subsample_scenarios(
            scenario_split.evaluation_scenarios,
            subsample_fraction=0.5,
            per_iteration=True,
            random_seed=42 + iter_num
        )

        population_scores = []
        for prompt_candidate in population.prompts:
            result = evaluate_prompt(
                prompt_candidate.full_text,
                eval_scenarios,
                eval_runner,
                prompt_id=prompt_candidate.prompt_id,
                baseline_score=baseline_score
            )
            population_scores.append(result.mean_score)

        print(f"Mean population score: {sum(population_scores) / len(population_scores):.4f}")
        print(f"Best population score: {max(population_scores):.4f}")
        print(f"Worst population score: {min(population_scores):.4f}")

        # Step 4: Statistical Scoring
        print("\nSTEP 4: Statistical Scoring")

        # Component-version regression
        regression_result = fit_component_version_regression(
            population,
            population_scores,
            baseline_score
        )

        print(f"\nComponent-version regression R²: {regression_result.r_squared:.4f}")

        # Update coefficients in version pool
        for version in state.version_pool:
            version.coefficient = regression_result.coefficients.get(version.version_id, 0.0)

        # Compute delta gains
        delta_gains = compute_delta_gains(regression_result.coefficients, state.version_pool)
        print(f"\nDelta gains by component:")
        for comp, gain in sorted(delta_gains.items(), key=lambda x: -x[1]):
            print(f"  {comp}: {gain:.4f}")

        # Meta-prompt regression
        meta_prompt_efficacies = {}
        if use_meta_prompts and meta_prompts:
            meta_result = fit_meta_prompt_regression(
                state.version_pool,
                regression_result.coefficients,
                use_interactions=False
            )

            if meta_result:
                print(f"\nMeta-prompt regression R²: {meta_result.r_squared:.4f}")
                meta_prompt_efficacies = meta_result.efficacies

                # Update weights with recency bias
                # Recent efficacies are weighted more heavily than historical ones
                if len(state.iteration_history) > 0:
                    from .state import apply_recency_weights
                    # Apply exponential decay to historical data
                    recency_weights = apply_recency_weights(state.iteration_history, state.lambda_decay)
                    current_weight = recency_weights[-1] if recency_weights else 1.0

                    # Update meta-prompt weights with recency-weighted efficacies
                    for mp, efficacy in meta_prompt_efficacies.items():
                        # Blend current efficacy with historical weight
                        old_weight = state.meta_prompt_weights.get(mp, 1.0)
                        new_weight = max(0.1, 1.0 + efficacy)
                        # Weight recent observations more heavily
                        state.meta_prompt_weights[mp] = old_weight * (1 - current_weight) + new_weight * current_weight
                else:
                    # First iteration: just use efficacies directly
                    for mp, efficacy in meta_prompt_efficacies.items():
                        state.meta_prompt_weights[mp] = max(0.1, 1.0 + efficacy)

                print(f"\nMeta-prompt efficacies:")
                for mp, eff in sorted(meta_prompt_efficacies.items(), key=lambda x: -x[1])[:5]:
                    print(f"  {mp[:50]}...: {eff:.4f}")

        # Step 5: Select & Save
        print("\nSTEP 5: Select & Save")

        # Best from population
        best_idx = population_scores.index(max(population_scores))
        best_population_prompt = population.prompts[best_idx]
        best_population_score = population_scores[best_idx]

        print(f"\nBest population prompt: {best_population_prompt.prompt_id}")
        print(f"  Score: {best_population_score:.4f}")

        # Best from regression (assemble from best versions)
        best_components = OrderedDict()
        for comp_name in component_names:
            versions = get_versions_by_component(state.version_pool, comp_name)
            best_version = max(versions, key=lambda v: v.coefficient or 0.0)
            best_components[comp_name] = best_version.text

        best_regression_prompt = reassemble(preamble, best_components)

        # Validate both
        print("\nValidating candidates on validation set...")

        validation_result_pop = evaluate_prompt(
            best_population_prompt.full_text,
            scenario_split.validation_scenarios,
            eval_runner,
            prompt_id="best_population"
        )

        validation_result_reg = evaluate_prompt(
            best_regression_prompt,
            scenario_split.validation_scenarios,
            eval_runner,
            prompt_id="best_regression"
        )

        print(f"\nValidation scores:")
        print(f"  Best population: {validation_result_pop.mean_score:.4f}")
        print(f"  Best regression: {validation_result_reg.mean_score:.4f}")

        # Keep winner
        if validation_result_pop.mean_score >= validation_result_reg.mean_score:
            best_prompt = best_population_prompt.full_text
            best_score = validation_result_pop.mean_score
            print(f"\n→ Selected: Best population prompt")
        else:
            best_prompt = best_regression_prompt
            best_score = validation_result_reg.mean_score
            print(f"\n→ Selected: Best regression prompt")

        # Save iteration state
        iter_state = IterationState(
            iteration=iter_num + 1,
            baseline_score=baseline_score,
            best_prompt_score=best_score,
            best_prompt=best_prompt,
            component_delta_gains=delta_gains,
            meta_prompt_efficacies=meta_prompt_efficacies,
            version_allocations=version_allocations
        )
        state.iteration_history.append(iter_state)
        state.current_iteration = iter_num + 1

        # Save state
        save_state(state, state_file)
        print(f"\n✓ State saved to {state_path}")

        # Step 6: Check early stopping
        max_delta_gain = max(delta_gains.values()) if delta_gains else 0.0
        print(f"\nMax delta gain: {max_delta_gain:.4f}")

        if delta_gain_stop is not None and max_delta_gain < delta_gain_stop:
            print(f"\n→ Early stopping: max delta gain {max_delta_gain:.4f} < threshold {delta_gain_stop:.4f}")
            break

    # Final summary
    print("\n" + "=" * 70)
    print("OPTIMIZATION COMPLETE")
    print("=" * 70)

    if state.iteration_history:
        final_iter = state.iteration_history[-1]
        print(f"\nFinal iteration: {final_iter.iteration}")
        print(f"Baseline score: {final_iter.baseline_score:.4f}")
        print(f"Final score: {final_iter.best_prompt_score:.4f}")
        print(f"Improvement: {final_iter.best_prompt_score - final_iter.baseline_score:.4f}")

        print(f"\nProgress over iterations:")
        for i, hist in enumerate(state.iteration_history):
            print(f"  Iteration {hist.iteration}: {hist.best_prompt_score:.4f}")

        return final_iter.best_prompt
    else:
        print("\nNo iterations completed, returning baseline prompt")
        return baseline_prompt
