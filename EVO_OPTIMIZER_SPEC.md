# Prompt Evolver — Specification

## Overview

A component-aware prompt optimizer that uses statistical regression to evaluate component-version contributions, meta-prompt efficacy tracking, and adaptive allocation across iterations. Uses a fractional-factorial evolutionary approach.

---

## Workflow

### Step 1: Baseline Evaluation

Split scenarios per category into evaluation and validation sets using `eval_split` and `validation_split`. Each accepts a float (percentage, e.g. 0.5) or an int (exact count per category, e.g. 10). These are independent — they can sum to less than 100% (e.g., 10%/10% leaves 80% unused).

Run categorically subsampled scenarios (from evaluation set) × each scenario's assigned evals against the current prompt. Collect per scenario-eval pair:
- `score` (float)
- `reason` (string — eval explanation)
- `passed` (bool — p/f based on eval threshold)

This is the baseline.

### Step 2: Generate Component Versions

Establish total component-version pool (`version_budget`, default: components × 3). Allocate versions across components — first iteration: equal across all components; later iterations: proportional to delta_gain from the previous iteration (components with near-zero delta get fewer or zero new versions).

For each version to generate:

1. If `use_meta_prompts=True`: pick a meta-prompt instruction (initially even weights) — skip any meta-prompt × component pair already in the current version pool. If `use_meta_prompts=False`: use the default mutation prompt (see Default Mutation Prompt section).
2. Collect feedback/scores from Step 1 for this component. Feedback filtering:
   - If `component_eval_mapping=True`: normalized exact match between component names and eval names (lowercase, strip underscores/hyphens — e.g., `socratic_method` ↔ `SocraticMethod` both normalize to `socraticmethod`). Mapped components only see feedback from their matched evals. Unmapped components see all feedback the prompt received. On first run, print found mappings and list unmapped components/evals.
   - If `failed_only_feedback=True`: only include feedback (`reason`, `score`) from evals where `passed` = false.
   - If `condense_feedback=True`: run collected feedback through an LLM to produce a condensed summary before sending to the mutation prompt.
3. Send to LLM using the mutation prompt: system message + `[Current Section]` (component text) + `[Feedback]` (filtered feedback) + `[Instruction]` (default rewrite instruction, or sampled meta-prompt instruction) → new component version.
4. Record which mutation instruction produced this version for which component (relevant when `use_meta_prompts=True`).

### Step 3: Build Population & Evaluate

Combine component-versions randomly to fill population pool (`population_size`, default: `version_budget × 2`). Each prompt is a random combination of one version per component. Evaluate each prompt against a fresh categorically subsampled scenario set × each scenario's assigned evals. Collect `score`, `reason`, `passed` for each.

### Step 4: Statistical Scoring

1. **Component-version regression:** Predict Δfitness (score change vs baseline) from binary 0/1 indicators for which version of each component a prompt uses. One coefficient per version — tells you how much that version helped or hurt. Delta_gain per component = spread between best and worst version coefficients.
2. **Meta-prompt regression:** (Only when `use_meta_prompts=True`.) Predict version quality (its coefficient from #1) from binary 0/1 indicators for which meta-prompt produced it. One coefficient per meta-prompt — tells you which mutation strategies produce better versions.
3. **Meta-prompt × component interaction regression** (separate, when enough data, only when `use_meta_prompts=True`): Same as #2 but adds interaction terms (meta-prompt × component) to capture which mutation strategies work best for which components.

These update:
- Proportional allocation of versions to components (more to high delta_gain).
- Sampling weights for meta-prompts (more to high efficacy, de-emphasized as delta contributions shrink).

### Step 5: Select & Save

1. Pick best-scoring prompt from the population.
2. Assemble a prompt from best version per component (from regression).
3. Evaluate both on the validation scenario set. Keep the winner.
4. Save cross-iteration scores to state file, weighted by exponential-Gaussian recency.

### Step 6: Repeat

Go to Step 2 with updated weights and allocations. Repeat for `iterations` (default 5), or stop early if `delta_gain_stop` threshold is set and met.

---

## Caller Contract

The library does not read workbooks, run agents, or execute evals directly. The caller provides:

### 1. Scenarios

A list of scenario objects. Each scenario has:

```python
{
    "id": "core_tutoring_042",
    "category": "core_tutoring",       # used for stratified sampling
    "eval_names": ["SocraticMethod", "Relevance", "Correctness"]  # which evals apply
}
```

### 2. Eval Runner

A callable with the signature:

```python
def eval_runner(prompt: str, scenario: dict) -> list[dict]:
    """
    Run the agent with the given prompt, evaluate the response against
    this scenario's evals, return results.
    """
    # ... caller's code: run agent, collect response, run evals ...
    return [
        {"eval_name": "SocraticMethod", "score": 0.8, "reason": "Used open-ended questions...", "passed": True},
        {"eval_name": "Relevance",      "score": 0.6, "reason": "Drifted off-topic...",        "passed": False},
    ]
```

The library calls this function. It does not know or care what happens inside.

### 3. Prompt String

A string containing `<!-- @component: xxx -->` tags. The library parses components from it at runtime — component count is discovered, not configured. Any text before the first tag is treated as a frozen preamble (never mutated). The library mutates tagged components and returns an optimized prompt string.

---

## Default Mutation Prompt

Used for all mutations when `use_meta_prompts=False` (default). Adapted from DeepEval GEPA's rewriter:

**System:**
```
You are refining a section of a prompt used in an LLM pipeline.
Given the current section and concise feedback, produce a revised version
that addresses the issues while preserving intent and style.
Return only the new section text, no explanations.
```

**User:**
```
[Current Section]
{component_text}

[Feedback]
{feedback_text}

[Instruction]
Rewrite this section. Keep it concise and actionable. Do not include extraneous text.
```

When `use_meta_prompts=True`, the `[Instruction]` block is replaced with the sampled meta-prompt instruction from `metaprompt_instructions.txt`.

---

## API

### Entry Point

```python
from prompt_evolver.optimizer import run_optimization

optimized_prompt = run_optimization(
    prompt=current_prompt,               # string with <!-- @component --> tags
    scenarios=scenarios,                 # list of scenario dicts (id, category, eval_names)
    eval_runner=eval_runner,             # callable(prompt, scenario) -> list of result dicts
    model="gpt-4o-mini",                # LLM for generating component mutations
    iterations=5,
    eval_split=0.5,                      # float (%) or int (count) of scenarios per category for evaluation
    validation_split=0.5,                # float (%) or int (count) of scenarios per category for validation
    version_budget="components * 3",     # total component-versions to generate (computed from prompt if not set)
    population_size="version_budget * 2",# number of prompts in population (computed from version_budget if not set)
    component_eval_mapping=None,         # None=off, True=auto name-match
    failed_only_feedback=False,
    condense_feedback=False,
    use_meta_prompts=False,              # True = sample from metaprompt_instructions.txt; False = single rewrite prompt
    delta_gain_stop=None,                # float threshold or None
    state_path="evolver_state.json",     # where to save/load cross-iteration state
)
# Returns: optimized prompt string
```

### Eval Result Format

Each item returned by `eval_runner` must have:

```python
{
    "eval_name": str,    # name of the eval
    "score": float,      # numeric score
    "reason": str,       # eval explanation / feedback text
    "passed": bool       # pass/fail based on threshold
}
```

### Scenario Format

```python
{
    "id": str,           # unique scenario identifier
    "category": str,     # category label for stratified sampling
    "eval_names": list   # list of eval name strings assigned to this scenario
}
```

The scenario dict may contain additional keys (e.g., input text, expected output) — the library passes the full dict through to `eval_runner` untouched.

---

## File Structure

```
prompt-evolver/
├── optimizer.py              # run_optimization() entry point, orchestrates the workflow
├── population.py             # Population assembly, random combination of component-versions
├── mutation.py               # Meta-prompt templates, version generation, feedback filtering
├── regression.py             # Component-version regression, meta-prompt regression, interaction model
├── scoring.py                # Scenario sampling, eval_runner dispatch, result collection
├── state.py                  # Cross-iteration state, recency weighting, file I/O
├── components.py             # Parse/reassemble <!-- @component --> tags
└── metaprompt_instructions.txt  # One mutation strategy per line (used when use_meta_prompts=True)
```

---

## Statistical Details

### Component-Version Regression

The dependent variable is the **difference score** (population prompt score minus baseline score), not the raw score. This isolates version effects from the baseline prompt's contribution.

```
Δfitness_i = β_0 + Σ_c Σ_v β_cv * I(prompt_i uses version v of component c) + ε_i

where Δfitness_i = score(prompt_i) - score(baseline_prompt)
```

- `i` indexes prompts in the population
- Reference version (v1/original) absorbed into intercept — so β_cv estimates the marginal lift of version v over the original for component c
- Σ_c (versions_c - 1) dummy variables total, where versions_c varies per component due to non-equal allocation
- Fit via OLS. Coefficients are estimated marginal effects on score change from baseline.
- `delta_gain_c = max(β_cv) - min(β_cv)` for component c — spread of version impact within that component

### Meta-Prompt Regression

```
quality_j = γ_0 + Σ_m γ_m * I(version_j produced by meta-prompt m) + ε_j
```

- `j` indexes all component-versions produced this iteration
- `quality_j` = that version's β coefficient from the component-version regression
- Fit via OLS. Coefficients estimate each meta-prompt's contribution to version quality.

### Meta-Prompt × Component Interactions (when data permits)

```
quality_j = γ_0 + Σ_m γ_m * I(mp_m) + Σ_m Σ_c γ_mc * I(mp_m) * I(comp_c) + ε_j
```

Requires enough observations per meta-prompt × component cell. Minimum ~3 observations per cell.

### Recency Weighting

Cross-iteration scores weighted by exponential-Gaussian:

```
w(t) = exp(-λ * (T - t)) where T = current iteration, t = past iteration
```

λ controls decay rate. Higher λ = more recency bias.
