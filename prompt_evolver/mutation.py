"""Component version generation using LLM with feedback filtering."""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set
from collections import defaultdict


@dataclass
class FeedbackItem:
    """Single piece of feedback from an eval."""
    eval_name: str
    score: float
    reason: str
    passed: bool
    scenario_id: str
    category: str


@dataclass
class MutationRequest:
    """Request to generate a new component version."""
    component_name: str
    current_text: str
    feedback: List[FeedbackItem]
    instruction: str
    meta_prompt_id: Optional[str]


def normalize_name(name: str) -> str:
    """Normalize component/eval name for matching.

    Strips brackets and their contents (e.g. [GEval]), then removes
    underscores, hyphens, and spaces, and lowercases.
    """
    name = re.sub(r'\[.*?\]', '', name)
    return re.sub(r'[_\-\s]', '', name.strip().lower())


def find_component_eval_mappings(
    component_names: List[str],
    eval_names: List[str]
) -> Dict[str, List[str]]:
    """
    Auto-detect component -> eval mappings based on name similarity.

    Args:
        component_names: List of component names
        eval_names: List of eval names

    Returns:
        Dict mapping component_name -> list of matched eval_names
    """
    mappings = {}

    for comp_name in component_names:
        norm_comp = normalize_name(comp_name)
        matched_evals = []

        for eval_name in eval_names:
            norm_eval = normalize_name(eval_name)
            # Check for exact match after normalization
            if norm_comp == norm_eval:
                matched_evals.append(eval_name)

        if matched_evals:
            mappings[comp_name] = matched_evals

    return mappings


def filter_feedback_for_component(
    component_name: str,
    feedback: List[FeedbackItem],
    component_eval_mapping: Optional[Dict[str, List[str]]],
    failed_only: bool
) -> List[FeedbackItem]:
    """
    Filter feedback based on component-eval mappings and pass/fail status.

    Args:
        component_name: Name of component being mutated
        feedback: All available feedback
        component_eval_mapping: Dict of component -> evals, or None to disable
        failed_only: If True, only include feedback where passed=False

    Returns:
        Filtered feedback list
    """
    filtered = feedback

    # Apply component-eval mapping filter
    if component_eval_mapping is not None:
        if component_name in component_eval_mapping:
            # Component is mapped: only use its mapped evals
            mapped_evals = set(component_eval_mapping[component_name])
            filtered = [f for f in filtered if f.eval_name in mapped_evals]
        # else: component unmapped, use all feedback (no filter)

    # Apply failed_only filter
    if failed_only:
        filtered = [f for f in filtered if not f.passed]

    return filtered


def condense_feedback(feedback: List[FeedbackItem], model: str = "gpt-4o-mini") -> str:
    """
    Condense feedback using LLM summarization.

    Args:
        feedback: List of feedback items
        model: Model to use for condensation

    Returns:
        Condensed feedback text
    """
    from openai import OpenAI

    if not feedback:
        return "No feedback available."

    # Format feedback
    feedback_text = "\n\n".join([
        f"[{f.eval_name}] Score: {f.score}, Passed: {f.passed}\n{f.reason}"
        for f in feedback
    ])

    # Condense using LLM
    client = OpenAI()
    prompt = f"""Condense the following evaluation feedback into 2-3 concise bullet points highlighting the key issues:

{feedback_text}

Condensed feedback:"""

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=300
    )

    return response.choices[0].message.content.strip()


def generate_component_version(
    request: MutationRequest,
    model: str = "gpt-4o-mini"
) -> str:
    """
    Generate a new component version using LLM.

    Args:
        request: Mutation request with component text and feedback
        model: Model to use for generation

    Returns:
        New component version text
    """
    from openai import OpenAI

    # Format feedback
    if request.feedback:
        feedback_text = "\n\n".join([
            f"[{f.eval_name}] Score: {f.score:.2f}, Passed: {f.passed}\n{f.reason}"
            for f in request.feedback
        ])
    else:
        feedback_text = "No specific feedback available."

    # Build prompt
    system_message = """You are refining a section of a prompt used in an LLM pipeline.
Given the current section and concise feedback, produce a revised version
that addresses the issues while preserving intent and style.
Return only the new section text, no explanations."""

    user_message = f"""[Current Section]
{request.current_text}

[Feedback]
{feedback_text}

[Instruction]
{request.instruction}"""

    # Call LLM
    client = OpenAI()
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ],
        temperature=0.7,
        max_tokens=2000
    )

    return response.choices[0].message.content.strip()


def load_meta_prompts(path: Path) -> List[str]:
    """
    Load meta-prompt instructions from file.

    Args:
        path: Path to metaprompt_instructions.txt

    Returns:
        List of instruction strings
    """
    with open(path, 'r') as f:
        lines = f.readlines()

    # Parse numbered instructions
    instructions = []
    for line in lines:
        line = line.strip()
        if line and line[0].isdigit():
            # Remove number prefix (e.g., "1. " -> "")
            instruction = re.sub(r'^\d+\.\s*', '', line)
            instructions.append(instruction)

    return instructions


def sample_meta_prompt(
    meta_prompts: List[str],
    weights: Dict[str, float],
    used_combinations: Set[tuple[str, str]],
    component_name: str
) -> Optional[str]:
    """
    Sample a meta-prompt instruction with weighted sampling, avoiding duplicates.

    Args:
        meta_prompts: List of meta-prompt instructions
        weights: Dict mapping instruction -> weight
        used_combinations: Set of (instruction, component_name) tuples already used
        component_name: Current component being mutated

    Returns:
        Sampled instruction string, or None if all combinations exhausted
    """
    import random

    # Find available instructions
    available = []
    available_weights = []

    for instruction in meta_prompts:
        if (instruction, component_name) not in used_combinations:
            available.append(instruction)
            weight = weights.get(instruction, 1.0)
            available_weights.append(weight)

    if not available:
        return None

    # Normalize weights
    total = sum(available_weights)
    if total > 0:
        probabilities = [w / total for w in available_weights]
    else:
        probabilities = [1.0 / len(available)] * len(available)

    # Sample
    sampled = random.choices(available, weights=probabilities, k=1)[0]
    return sampled
