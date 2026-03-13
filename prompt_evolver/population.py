"""Population assembly and indicator matrix construction."""

import random
from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, List
import numpy as np

from .state import ComponentVersionRecord
from .components import reassemble


@dataclass
class PromptCandidate:
    """A single prompt candidate in the population."""
    prompt_id: str
    component_versions: Dict[str, str]  # component_name -> version_id
    full_text: str
    version_indicators: Dict[str, int]  # version_id -> 1/0


@dataclass
class Population:
    """Population of prompt candidates."""
    prompts: List[PromptCandidate]
    version_pool: List[ComponentVersionRecord]
    baseline_prompt: str
    preamble: str


def get_versions_by_component(
    version_pool: List[ComponentVersionRecord],
    component_name: str
) -> List[ComponentVersionRecord]:
    """Get all versions for a specific component."""
    return [v for v in version_pool if v.component_name == component_name]


def build_population(
    preamble: str,
    component_names: List[str],
    version_pool: List[ComponentVersionRecord],
    population_size: int,
    baseline_components: OrderedDict,
    random_seed: int = 42
) -> Population:
    """
    Build a population by randomly combining component versions.

    Args:
        preamble: Frozen text before first component tag
        component_names: List of component names
        version_pool: All available component versions
        population_size: Number of prompts to generate
        baseline_components: Original components (for reassembly)
        random_seed: Random seed for reproducibility

    Returns:
        Population with prompt candidates
    """
    random.seed(random_seed)

    # Group versions by component
    versions_by_component = {}
    for comp_name in component_names:
        versions_by_component[comp_name] = get_versions_by_component(version_pool, comp_name)

    prompts = []
    for i in range(population_size):
        # Randomly select one version per component
        component_versions = {}
        version_indicators = {v.version_id: 0 for v in version_pool}

        for comp_name in component_names:
            versions = versions_by_component[comp_name]
            if versions:
                selected = random.choice(versions)
                component_versions[comp_name] = selected.version_id
                version_indicators[selected.version_id] = 1

        # Reassemble full prompt
        # Build components dict with selected versions
        components_for_reassembly = OrderedDict()
        for comp_name in component_names:
            version_id = component_versions.get(comp_name)
            if version_id:
                # Find the version text
                version_record = next((v for v in version_pool if v.version_id == version_id), None)
                if version_record:
                    components_for_reassembly[comp_name] = version_record.text
                else:
                    # Fallback to baseline
                    components_for_reassembly[comp_name] = baseline_components[comp_name]
            else:
                components_for_reassembly[comp_name] = baseline_components[comp_name]

        full_text = reassemble(preamble, components_for_reassembly)

        prompt = PromptCandidate(
            prompt_id=f"prompt_{i}",
            component_versions=component_versions,
            full_text=full_text,
            version_indicators=version_indicators
        )
        prompts.append(prompt)

    # Baseline prompt
    baseline_prompt = reassemble(preamble, baseline_components)

    return Population(
        prompts=prompts,
        version_pool=version_pool,
        baseline_prompt=baseline_prompt,
        preamble=preamble
    )


def build_indicator_matrix(population: Population) -> np.ndarray:
    """
    Build binary indicator matrix for regression.

    Rows = prompts, Columns = versions
    Entry [i,j] = 1 if prompt i uses version j, else 0

    Args:
        population: Population with prompts and version pool

    Returns:
        Binary indicator matrix (n_prompts × n_versions)
    """
    n_prompts = len(population.prompts)
    version_ids = [v.version_id for v in population.version_pool]
    n_versions = len(version_ids)

    matrix = np.zeros((n_prompts, n_versions), dtype=int)

    for i, prompt in enumerate(population.prompts):
        for j, version_id in enumerate(version_ids):
            if prompt.version_indicators.get(version_id, 0) == 1:
                matrix[i, j] = 1

    return matrix
