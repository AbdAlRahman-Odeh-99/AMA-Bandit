# -*- coding: utf-8 -*-
"""
core/two_stage_utils.py

Standalone home for the handful of two_stage_asymmetric.py functions that
two_stage_multiclass.py (and two_stage_multiclass_runner.py) import and
reuse UNCHANGED -- because they were already class-count-agnostic, nothing
about them needed to change for the multiclass generalization (see
two_stage_multiclass.py's module docstring, which explains exactly this).

Moved here (out of two_stage_asymmetric.py) since that file has been
deleted from this repo; two_stage_multiclass.py now imports these from
core.two_stage_utils instead of two_stage.two_stage_asymmetric.

Each function below is a byte-for-byte-equivalent port of the original
two_stage_asymmetric.py definition (only docstrings/formatting added):

    generate_view_combinations           (originally line 59)
    generate_combination_costs_heterogeneous  (originally line 55)
    generate_per_view_costs              (originally line 48)
    match_cluster_labels                 (originally line 16)
    calculate_two_stage_error            (originally line 458, as calculate_two_phase_error)

No other two_stage_asymmetric.py functions are needed by the multiclass
path -- everything else there (predict_combo / predict_single_combination,
initialize_centers, combo_closed_form_reward, run_alg,
run_inference_lp_colgen, solve_LP / run_inference_lp) was either K=2-
specific and re-implemented in two_stage_multiclass.py, or unused by the
multiclass runner (the legacy exhaustive-enumeration LP path).
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
from scipy.optimize import linear_sum_assignment


def match_cluster_labels(learned_centers, true_centers):
    """Hungarian-match rows of `learned_centers` (K, M) to the known
    generative `true_centers` (dict: modality index -> (K, 1) array of
    that modality's K true means), by squared Euclidean distance summed
    over all M modalities.

    Already K-agnostic (builds a general K x K cost matrix) -- used
    UNCHANGED by two_stage_multiclass.py's SYNTHETIC-data inference path
    (run_inference_lp_colgen_multiclass), where learned centers need to be
    matched against known true means. NOT needed for real-data callers,
    since initialize_centers_multiclass is supervised (learned center row
    k already tracks true class k by construction) -- see
    two_stage_multiclass_runner.py's module docstring.

    Returns
    -------
    dict: learned-center row index -> true-class index.
    """
    # Convert dict of arrays (modality -> (K,1)) to (K,M)
    true_mean_array = np.hstack([true_centers[m] for m in sorted(true_centers.keys())])
    K = learned_centers.shape[0]

    # Cost matrix: squared Euclidean distances
    cost_matrix = np.zeros((K, K))
    for i in range(K):
        for j in range(K):
            cost_matrix[i, j] = np.sum((learned_centers[i] - true_mean_array[j]) ** 2)

    # Hungarian assignment
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    mapping = {row: col for row, col in zip(row_ind, col_ind)}
    return mapping


def generate_per_view_costs(m_modalities, rng):
    """Draw random per-view acquisition costs ~ Uniform(0,1), force view 0
    (the first/free view) to cost 0, and normalize the rest to sum to 1.

    Note: most callers in this codebase (two_stage_multiclass_runner.py
    included) use core.datasets.generate_modality_costs_heterogeneous
    instead, which draws costs from a configurable lognormal/uniform
    distribution keyed by dataset name for reproducibility across runs.
    This simpler version is kept only because the original
    two_stage_asymmetric.py exposed it and some caller may still depend on
    generating costs directly from an rng without a dataset name.
    """
    costs = rng.random(size=(m_modalities,))
    costs[0] = 0
    costs = costs / np.sum(costs)
    return costs


def generate_combination_costs_heterogeneous(view_combinations, per_view_costs):
    """Build a {combo: total_cost} dict for every combo in
    view_combinations, where a combo's cost is the sum of its 1-indexed
    views' entries in per_view_costs (view v's cost is
    per_view_costs[v - 1])."""
    return {combo: sum(per_view_costs[v - 1] for v in combo) for combo in view_combinations}


def generate_view_combinations(m_modalities):
    """Enumerate every subset of the PAID views {2, ..., m_modalities},
    each returned with the FREE view 1 forced in (so every combo has at
    least view 1). This is the full action set used by the enumerated
    acquisition policies --
    2^(m_modalities - 1) combos total, which is why nviews must stay small
    (see MAX_RECOMMENDED_MODALITIES in core/datasets.py).
    """
    modalities = list(range(2, m_modalities + 1))
    all_views = []
    for r in range(len(modalities) + 1):
        for combo in combinations(modalities, r):
            all_views.append((1,) + combo)
    return all_views


def calculate_two_stage_error(T, T1, stage2_error, init_err):
    """Blend the Stage-1 (init_err, over T1 rounds) and Stage-2
    (stage2_error, over T - T1 rounds) error rates into a single sample-
    weighted error over all T training rounds. Does NOT include any
    inference-phase error -- see two_stage_multiclass_runner.py's
    run_experiment docstring for how the separate Total Reward figure
    folds inference back in."""
    stage1_scale = T1 / T
    stage2_scale = (T - T1) / T
    return (stage1_scale * init_err) + (stage2_scale * stage2_error)
