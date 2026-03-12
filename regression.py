"""Statistical regression models for component and meta-prompt analysis."""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional
from collections import defaultdict

try:
    from sklearn.linear_model import LinearRegression
except ImportError:
    LinearRegression = None

from .state import ComponentVersionRecord
from .population import Population


@dataclass
class ComponentVersionRegressionResult:
    """Results from component-version regression."""
    coefficients: Dict[str, float]  # version_id -> coefficient
    intercept: float
    r_squared: float


@dataclass
class MetaPromptRegressionResult:
    """Results from meta-prompt regression."""
    efficacies: Dict[str, float]  # meta_prompt_id -> coefficient
    intercept: float
    r_squared: float
    used_interactions: bool


def drop_reference_categories(
    X: np.ndarray,
    population: Population
) -> tuple[np.ndarray, List[str]]:
    """
    Drop reference categories (first version per component) to avoid collinearity.

    Args:
        X: Full indicator matrix (n_prompts × n_versions)
        population: Population with version pool

    Returns:
        (X_encoded, kept_version_ids) where X_encoded has reference categories removed
    """
    # Group versions by component
    versions_by_component = defaultdict(list)
    for idx, version in enumerate(population.version_pool):
        versions_by_component[version.component_name].append((idx, version.version_id))

    # Determine which columns to keep (drop first version per component)
    keep_indices = []
    kept_version_ids = []

    for comp_name, versions in versions_by_component.items():
        # Sort by version_id for consistency
        versions.sort(key=lambda x: x[1])
        # Keep all except first
        for idx, version_id in versions[1:]:
            keep_indices.append(idx)
            kept_version_ids.append(version_id)

    if not keep_indices:
        # No versions to keep (all are reference categories)
        return np.empty((X.shape[0], 0)), []

    X_encoded = X[:, keep_indices]
    return X_encoded, kept_version_ids


def fit_component_version_regression(
    population: Population,
    scores: List[float],
    baseline_score: float
) -> ComponentVersionRegressionResult:
    """
    Fit OLS regression to predict delta fitness from component version indicators.

    Args:
        population: Population with prompts and version pool
        scores: Mean scores for each prompt in population
        baseline_score: Baseline prompt score

    Returns:
        ComponentVersionRegressionResult with coefficients and R²
    """
    if LinearRegression is None:
        raise ImportError("scikit-learn is required for regression. Install with: pip install scikit-learn")

    # Build indicator matrix
    from .population import build_indicator_matrix
    X_full = build_indicator_matrix(population)

    # Compute delta fitness
    y = np.array([score - baseline_score for score in scores])

    # Drop reference categories
    X, kept_version_ids = drop_reference_categories(X_full, population)

    if X.shape[1] == 0:
        # No versions to fit (all are reference categories)
        coefficients = {v.version_id: 0.0 for v in population.version_pool}
        return ComponentVersionRegressionResult(
            coefficients=coefficients,
            intercept=0.0,
            r_squared=0.0
        )

    # Fit OLS
    try:
        model = LinearRegression()
        model.fit(X, y)

        # Extract coefficients
        coefficients = {}
        for version_id, coef in zip(kept_version_ids, model.coef_):
            coefficients[version_id] = float(coef)

        # Reference categories get 0.0
        for version in population.version_pool:
            if version.version_id not in coefficients:
                coefficients[version.version_id] = 0.0

        r_squared = model.score(X, y) if X.shape[0] > X.shape[1] else 0.0

        return ComponentVersionRegressionResult(
            coefficients=coefficients,
            intercept=float(model.intercept_),
            r_squared=r_squared
        )

    except Exception as e:
        print(f"Warning: Regression failed: {e}. Using uniform coefficients.")
        # Fallback to uniform coefficients
        coefficients = {v.version_id: 0.0 for v in population.version_pool}
        return ComponentVersionRegressionResult(
            coefficients=coefficients,
            intercept=0.0,
            r_squared=0.0
        )


def compute_delta_gains(
    coefficients: Dict[str, float],
    version_pool: List[ComponentVersionRecord]
) -> Dict[str, float]:
    """
    Compute delta_gain per component = max(coef) - min(coef).

    Args:
        coefficients: version_id -> coefficient
        version_pool: All component versions

    Returns:
        Dict mapping component_name -> delta_gain
    """
    # Group coefficients by component
    coefs_by_component = defaultdict(list)
    for version in version_pool:
        coef = coefficients.get(version.version_id, 0.0)
        coefs_by_component[version.component_name].append(coef)

    # Compute delta_gain per component
    delta_gains = {}
    for comp_name, coefs in coefs_by_component.items():
        if coefs:
            delta_gains[comp_name] = max(coefs) - min(coefs)
        else:
            delta_gains[comp_name] = 0.0

    return delta_gains


def fit_meta_prompt_regression(
    version_pool: List[ComponentVersionRecord],
    coefficients: Dict[str, float],
    use_interactions: bool = False
) -> Optional[MetaPromptRegressionResult]:
    """
    Fit regression to predict version quality from meta-prompt indicators.

    Args:
        version_pool: All component versions
        coefficients: version_id -> coefficient (quality measure)
        use_interactions: If True, include meta-prompt × component interactions

    Returns:
        MetaPromptRegressionResult or None if not enough data
    """
    if LinearRegression is None:
        return None

    # Filter versions that have meta_prompt_used
    versions_with_meta = [v for v in version_pool if v.meta_prompt_used is not None]
    if len(versions_with_meta) < 3:
        return None  # Not enough data

    # Build feature matrix
    # Collect unique meta-prompts and components
    meta_prompts = sorted(set(v.meta_prompt_used for v in versions_with_meta))
    component_names = sorted(set(v.component_name for v in versions_with_meta))

    # Build X and y
    y = np.array([coefficients.get(v.version_id, 0.0) for v in versions_with_meta])

    if use_interactions:
        # Include interactions
        feature_names = []
        X_list = []

        for v in versions_with_meta:
            row = []
            # Meta-prompt indicators
            for mp in meta_prompts:
                row.append(1 if v.meta_prompt_used == mp else 0)
            # Interaction terms: meta-prompt × component
            for mp in meta_prompts:
                for comp in component_names:
                    row.append(1 if v.meta_prompt_used == mp and v.component_name == comp else 0)
            X_list.append(row)

        X = np.array(X_list)

        # Check if enough data
        if X.shape[0] < X.shape[1] * 3:
            # Not enough observations per feature
            use_interactions = False
    else:
        # Meta-prompt indicators only
        X = np.array([
            [1 if v.meta_prompt_used == mp else 0 for mp in meta_prompts]
            for v in versions_with_meta
        ])

    # Fit
    try:
        model = LinearRegression()
        model.fit(X, y)

        efficacies = {}
        for mp, coef in zip(meta_prompts, model.coef_[:len(meta_prompts)]):
            efficacies[mp] = float(coef)

        r_squared = model.score(X, y) if X.shape[0] > X.shape[1] else 0.0

        return MetaPromptRegressionResult(
            efficacies=efficacies,
            intercept=float(model.intercept_),
            r_squared=r_squared,
            used_interactions=use_interactions
        )

    except Exception as e:
        print(f"Warning: Meta-prompt regression failed: {e}")
        return None
