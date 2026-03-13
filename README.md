# Prompt Evolver

Component-aware prompt optimization library using statistical regression and evolutionary search.

## Overview

Prompt Evolver systematically optimizes LLM prompts by:
1. Parsing prompts into tagged components
2. Generating variations using LLM-based mutations
3. Evaluating combinations on real scenarios
4. Using OLS regression to identify high-impact components
5. Adapting search strategy based on statistical evidence

## Installation

```bash
source .venv/bin/activate
pip install scikit-learn numpy
```

## Quick Start

```python
from prompt_evolver import run_optimization

# 1. Tag your prompt with components
prompt = """
<!-- @component: greeting -->
Hello! I'm your tutor.

<!-- @component: instruction -->
Let's work together to solve this problem step by step.
"""

# 2. Prepare scenarios
scenarios = [
    {
        "id": "scenario_1",
        "category": "algebra",
        "eval_names": ["Clarity", "Engagement"],
        "input": "Solve 2x + 3 = 7"
    },
    # ... more scenarios
]

# 3. Define eval runner
def eval_runner(prompt, scenario):
    # Run your agent with the prompt
    # Evaluate response against scenario's evals
    return [
        {
            "eval_name": "Clarity",
            "score": 0.85,
            "reason": "Explanation was clear",
            "passed": True
        },
        # ... more eval results
    ]

# 4. Run optimization
optimized_prompt = run_optimization(
    prompt=prompt,
    scenarios=scenarios,
    eval_runner=eval_runner,
    iterations=5,
    version_budget=12,
    population_size=24
)

print("Optimized prompt:")
print(optimized_prompt)
```

## API Reference

### Main Entry Point

```python
run_optimization(
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
    condense_feedback_flag: bool = False,
    use_meta_prompts: bool = False,
    delta_gain_stop: Optional[float] = None,
    state_path: str = "evolver_state.json",
) -> str
```

#### Parameters

- **prompt**: Initial prompt with `<!-- @component: name -->` tags
- **scenarios**: List of scenario dicts with `id`, `category`, `eval_names` fields
- **eval_runner**: Callable(prompt, scenario) → List[Dict] with eval results
- **model**: LLM model for generating mutations (default: "gpt-4o-mini")
- **iterations**: Number of optimization iterations (default: 5)
- **eval_split**: Float (%) or int (count) for evaluation scenarios per category
- **validation_split**: Float (%) or int (count) for validation scenarios per category
- **version_budget**: Total component versions to generate (default: components × 3)
- **population_size**: Number of prompts in population (default: version_budget × 2)
- **component_eval_mapping**: Auto-match components to evals by name (default: None)
- **failed_only_feedback**: Only use feedback where passed=False (default: False)
- **condense_feedback_flag**: Use LLM to condense feedback (default: False)
- **use_meta_prompts**: Sample meta-prompts from file (default: False)
- **delta_gain_stop**: Early stopping threshold for max delta_gain (default: None)
- **state_path**: Path to save/load state file (default: "evolver_state.json")

#### Returns

Optimized prompt string with component tags preserved.

### Eval Runner Contract

Your `eval_runner` function must accept:
- `prompt` (str): The prompt to evaluate
- `scenario` (dict): Scenario object with fields you defined

And return a list of dicts, each containing:
```python
{
    "eval_name": str,    # Name of the eval
    "score": float,      # Numeric score (0.0 to 1.0)
    "reason": str,       # Explanation / feedback text
    "passed": bool       # Pass/fail based on threshold
}
```

### Scenario Format

Each scenario dict must have:
```python
{
    "id": str,           # Unique identifier
    "category": str,     # Category for stratified sampling
    "eval_names": list   # List of eval names to run
}
```

Additional fields (input, expected_output, etc.) are passed through to your `eval_runner`.

## Features

### Component Tagging

Mark sections of your prompt with HTML-style comments:

```
<!-- @component: greeting -->
Your greeting text here

<!-- @component: instruction -->
Your instruction text here
```

The optimizer will:
- Parse components automatically
- Generate variations for each component
- Test combinations to find best overall prompt

### Stratified Sampling

Scenarios are split by category to ensure balanced evaluation:
- `eval_split`: Scenarios used for training (default: 50%)
- `validation_split`: Scenarios used for validation (default: 50%)
- Can overlap if splits sum > 100%
- Sampling is independent per split

### Statistical Scoring

Uses OLS regression to:
1. Predict score improvement from component versions
2. Identify which components have highest impact (delta_gain)
3. Allocate more mutations to high-impact components
4. (Optional) Track meta-prompt efficacy

### Adaptive Allocation

First iteration: Equal mutations across all components
Later iterations: Proportional to delta_gain from previous iteration

Components with low impact get fewer mutations; high-impact components get more.

### Meta-Prompt Evolution

When `use_meta_prompts=True`:
- Samples mutation strategies from `metaprompt_instructions.txt`
- Tracks which strategies produce better versions
- Adapts sampling weights based on efficacy

### State Persistence

Optimization state is saved after each iteration:
- Resume interrupted optimizations
- Analyze version pool and coefficients
- Track progress across iterations

State file contains:
- All component versions generated
- Regression coefficients per version
- Iteration history with scores and delta gains
- Meta-prompt weights (if enabled)

## Advanced Usage

### Component-Eval Mapping

Auto-match components to specific evals by name:

```python
optimized = run_optimization(
    prompt=prompt,
    scenarios=scenarios,
    eval_runner=eval_runner,
    component_eval_mapping=True  # Enable auto-matching
)
```

Example: Component named `socratic_method` matches eval named `SocraticMethod`
- Name matching is case-insensitive, ignores underscores/hyphens
- Mapped components only see feedback from their evals
- Unmapped components see all feedback

### Failed-Only Feedback

Only use feedback from failed evaluations:

```python
optimized = run_optimization(
    prompt=prompt,
    scenarios=scenarios,
    eval_runner=eval_runner,
    failed_only_feedback=True
)
```

Useful when you want mutations to focus on fixing failures rather than maintaining strengths.

### Feedback Condensation

Use LLM to condense feedback before mutation:

```python
optimized = run_optimization(
    prompt=prompt,
    scenarios=scenarios,
    eval_runner=eval_runner,
    condense_feedback_flag=True
)
```

Reduces token usage and focuses mutation on key issues.

### Early Stopping

Stop when component improvements plateau:

```python
optimized = run_optimization(
    prompt=prompt,
    scenarios=scenarios,
    eval_runner=eval_runner,
    delta_gain_stop=0.02  # Stop if max delta_gain < 0.02
)
```

## File Structure

```
prompt_evolver/
├── __init__.py              # Public API
├── optimizer.py             # Main orchestration loop
├── components.py            # Component parsing/reassembly
├── state.py                 # State persistence
├── scoring.py               # Scenario sampling, evaluation
├── mutation.py              # Version generation with LLM
├── population.py            # Random combinations
├── regression.py            # OLS models
├── metaprompt_instructions.txt  # Meta-prompt library
├── EVO_OPTIMIZER_SPEC.md    # Full specification
├── README.md                # This file
└── tests/                   # Test suite
    ├── test_components.py
    ├── test_mutation.py
    ├── test_scoring.py
    ├── test_population.py
    ├── test_regression.py
    ├── test_state.py
    └── test_integration.py
```

## Testing

Run the test suite:

```bash
cd /AITutor/deep-eval
source evals/.venv/bin/activate
PYTHONPATH=$PWD:$PYTHONPATH python -m pytest prompt_evolver/tests/ -v
```

All 41 tests should pass.

## Examples

See `EVO_OPTIMIZER_SPEC.md` for the complete specification and detailed examples.

## License

Internal use only.
