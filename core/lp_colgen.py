# -*- coding: utf-8 -*-
"""
core/lp_colgen.py

Column-generation solver for the stochastic inference LP. It solves the same
linear program that
`solve_lp_policy` solves (maximize expected reward over a distribution
on view subsets, subject to a per-round budget constraint and the
probability simplex), but WITHOUT ever materializing the full
`2**nviews - 1` power set of subsets.

Instead, it solves a small "restricted master problem" LP over a growing
set of discovered columns (subsets), and at each iteration uses the
master problem's dual variables to run a pruned branch-and-bound search
(the "pricing subproblem") for a single new subset with negative reduced
cost -- i.e. one that would improve the LP objective if added. This
relies on the reward function (`lp_reward`) being monotone in the
included views, so the "assume every remaining feature gets added" bound
is a valid relaxation for pruning.

`lp_reward` uses the monotone gain formula required by the pricing bound.

This makes inference tractable at ANY nviews, including physionet (41) and
miniboone (50), where `enumerate_subsets(nviews)` cannot even be built in
memory, let alone optimized over.

The pricing implementation is shared by both proposed method families.

One deliberate adaptation relative to the notebook's version: the free
view (index 0) is seeded as the initial active column instead of starting
column generation from the empty set, matching this codebase's "free view
always observed" convention used by the acquisition policies. The
branch-and-bound pricing
subproblem then only ever branches over PAID views (indices 1..nviews-1);
the free view is fixed True in every candidate subset it proposes.

=== Instrumentation note ===
Both solvers split their loop into the two halves that actually cost
something, and time them separately: t_master_lp for the restricted-master
linprog call and t_pricing for the branch-and-bound subproblem, with
n_colgen_iters counting the rounds. This is the decomposition the previous
single inference_time_sec could not give: at 40+ views the pricing search
dominates and the master is free, while at low view counts with many
discovered columns the reverse can hold, and the two point at completely
different fixes.

COLGEN_ITER_WARN below is a WARNING THRESHOLD, not a cap -- the loop's
termination conditions are unchanged. It exists because an inference solve
that quietly takes thousands of iterations is a real failure mode with no
visible symptom other than a slow job.
"""

import math

import numba
import numpy as np
import scipy.optimize as opt
from sklearn.metrics import f1_score, roc_auc_score
from core.logging_utils import bump, get_logger, tick
from core.acquisition_policies import multiclass_reward, pairwise_diff_sq_from_means

#: Iterations after which column generation is reported as suspicious.
#: Purely diagnostic -- nothing is truncated.
COLGEN_ITER_WARN = 500

_log = get_logger("afa.colgen")


@numba.njit
def gain_func(snrs):
    snr_sum = np.sum(snrs)
    arg = (0.5 * np.sqrt(snr_sum)) / np.sqrt(2.0)
    error_rate = 0.5 * (1.0 - math.erf(arg))
    return 1.0 - error_rate

def solve_lp_policy_colgen(est_means, costs, inference_budget, n_inference, noise_var=1.0, thres=1e-6):
    nviews = len(costs)
    mean_diff_sq = np.square(0.5 * (est_means[0, :] - est_means[1, :]) / np.sqrt(noise_var))
    budget_ratio = inference_budget / n_inference

    free_mask = np.zeros(nviews, dtype=bool)
    free_mask[0] = True

    active_subsets = [free_mask]
    active_c = [gain_func(mean_diff_sq[free_mask])]
    active_g = [float(np.sum(costs[free_mask]))]

    def solve_subproblem_bb(y_ub, y_eq):
        best_subset = None
        best_reduced_cost = 0.0

        def branch(feature_idx, current_selection):
            nonlocal best_subset, best_reduced_cost

            optimistic_selection = np.copy(current_selection)
            optimistic_selection[feature_idx:] = True
            max_potential_reward = gain_func(mean_diff_sq[optimistic_selection])

            # ORGINAL
            #guaranteed_penalty = y_ub * np.sum(costs[current_selection]) + y_eq
            #optimistic_bound = -max_potential_reward + guaranteed_penalty
            # FIXED
            guaranteed_penalty = -(y_ub * np.sum(costs[current_selection]) + y_eq)
            optimistic_bound = -max_potential_reward + guaranteed_penalty

            if optimistic_bound > best_reduced_cost:
                return  # prune -- this branch cannot beat what we have

            if feature_idx == nviews:
                reward = gain_func(mean_diff_sq[current_selection])
                cost = np.sum(costs[current_selection])
                reduced_cost = -reward - y_ub * cost - y_eq   # FIX cost
                #reduced_cost = -reward + y_ub * cost + y_eq  # Orignial cost
                if reduced_cost < best_reduced_cost:
                    best_reduced_cost = reduced_cost
                    best_subset = np.copy(current_selection)
                return

            # --- Branching ---
            # Branch 1: Exclude the current feature
            branch(feature_idx + 1, current_selection)
            # Branch 2: Include the current feature
            current_selection[feature_idx] = True
            branch(feature_idx + 1, current_selection)
            current_selection[feature_idx] = False  # backtrack

        # Start the search from the first paid feature (index 1), with the free view always included
        branch(1, np.copy(free_mask))

        # Do not return the empty set, as it's already in the master problem
        if best_subset is not None and not np.any(best_subset[1:]):
            return None, 0.0
        return best_subset, best_reduced_cost

    opt_dist = np.array([1.0])
    n_iter = 0
    while True:
        n_iter += 1
        bump("n_colgen_iters")
        if n_iter == COLGEN_ITER_WARN:
            _log.warning("binary column generation still running after %d iterations "
                         "(%d active columns, nviews=%d, budget_ratio=%.6g) -- "
                         "not truncating, but this solve is pathological",
                         n_iter, len(active_c), nviews, budget_ratio)
        # --- Step A: Master Problem ---
        with tick("t_master_lp"):
            obj_map = -np.array(active_c)
            A_ub = np.array([active_g])
            b_ub = np.array([budget_ratio])
            A_eq = np.ones((1, len(active_c)))
            b_eq = np.array([1.0])
            res_rmp = opt.linprog(obj_map, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=(0, None), method="highs")
            y_ub = res_rmp.ineqlin.marginals[0] if res_rmp.ineqlin is not None else 0.0
            y_eq = res_rmp.eqlin.marginals[0] if res_rmp.eqlin.marginals is not None else 0.0
        # --- Step B: Run Branch and Bound ---
        # This replaces the giant matrix search
        with tick("t_pricing"):
            new_subset, min_reduced_cost = solve_subproblem_bb(y_ub, y_eq)
        opt_dist = res_rmp.x
        # --- Step C: Check and Update -
        if new_subset is None or min_reduced_cost >= -thres:
            break  # no improving column left -- optimal

        # Check if the new subset is already in the active set to prevent infinite loops
        is_duplicate = any(np.array_equal(new_subset, s) for s in active_subsets)
        if is_duplicate:
            # Pricing re-proposed a column it had already generated: the
            # duplicate guard below is what stops an infinite loop, so
            # reaching it means the solve terminated on a degeneracy rather
            # than on optimality. Previously indistinguishable from a clean
            # exit; now it says so.
            _log.warning("binary column generation stopped on a DUPLICATE column at "
                         "iteration %d (%d active columns) -- the returned policy is "
                         "the last feasible master solution, not a certified optimum",
                         n_iter, len(active_c))
            break

        active_subsets.append(new_subset)
        active_c.append(gain_func(mean_diff_sq[new_subset]))
        active_g.append(float(np.sum(costs[new_subset])))

    _log.debug("binary colgen: %d iterations, %d columns, nviews=%d",
               n_iter, len(active_subsets), nviews)

    probs = np.clip(opt_dist, 0, None)
    probs = probs / probs.sum() if probs.sum() > 0 else np.ones(len(active_subsets)) / len(active_subsets)

    return active_subsets, probs

# --------------------------------------------------------------------------
# Multiclass variant (used by adaptive/adaptive.py)
# --------------------------------------------------------------------------
# The multiclass reward functions live here so both algorithm families can
# import them without a circular dependency.

def solve_lp_policy_colgen_multiclass(est_means, costs, inference_budget,
                                       n_inference, thres=1e-6):
    """Multiclass counterpart of solve_lp_policy_colgen: identical
    restricted-master / branch-and-bound-pricing structure and identical
    sign conventions, with the per-subset reward swapped from the binary
    Q-function (gain_func over pooled SNR) to the multiclass average
    pairwise Bhattacharyya accuracy proxy (multiclass_reward over the
    pairwise squared-mean-difference tensor). The pruning bound
    ("assume every remaining paid view gets added") remains valid because
    multiclass_reward is monotone in the included views -- see its
    docstring.

    est_means: (nclasses, nviews). With nclasses == 2 this solves the same
    KIND of LP as the binary solver but under a DIFFERENT reward formula
    (average pairwise Bhattacharyya proxy vs. pooled-SNR Q-function), so
    binary results are NOT expected to match solve_lp_policy_colgen
    numerically -- use the right solver for the right method.

    Returns (active_subsets, probs) exactly like the binary version.
    """
    est_means = np.asarray(est_means, dtype=np.float64)
    costs = np.asarray(costs, dtype=np.float64)
    nviews = len(costs)
    diff_sq = pairwise_diff_sq_from_means(est_means)  # (nviews, nc, nc)
    budget_ratio = inference_budget / n_inference

    def reward_of(mask):
        return float(multiclass_reward(diff_sq[mask]))

    free_mask = np.zeros(nviews, dtype=bool)
    free_mask[0] = True

    active_subsets = [free_mask]
    active_c = [reward_of(free_mask)]
    active_g = [float(np.sum(costs[free_mask]))]

    def solve_subproblem_bb(y_ub, y_eq):
        best_subset = None
        best_reduced_cost = 0.0

        def branch(feature_idx, current_selection):
            nonlocal best_subset, best_reduced_cost

            optimistic_selection = np.copy(current_selection)
            optimistic_selection[feature_idx:] = True
            max_potential_reward = reward_of(optimistic_selection)

            guaranteed_penalty = y_ub * np.sum(costs[current_selection]) + y_eq
            optimistic_bound = -max_potential_reward + guaranteed_penalty
            if optimistic_bound > best_reduced_cost:
                return  # prune -- this branch cannot beat what we have

            if feature_idx == nviews:
                reward = reward_of(current_selection)
                cost = np.sum(costs[current_selection])
                reduced_cost = -reward - y_ub * cost - y_eq
                if reduced_cost < best_reduced_cost:
                    best_reduced_cost = reduced_cost
                    best_subset = np.copy(current_selection)
                return

            branch(feature_idx + 1, current_selection)
            current_selection[feature_idx] = True
            branch(feature_idx + 1, current_selection)
            current_selection[feature_idx] = False  # backtrack

        branch(1, np.copy(free_mask))

        if best_subset is not None and not np.any(best_subset[1:]):
            return None, 0.0
        return best_subset, best_reduced_cost

    opt_dist = np.array([1.0])
    n_iter = 0
    while True:
        n_iter += 1
        bump("n_colgen_iters")
        if n_iter == COLGEN_ITER_WARN:
            _log.warning("multiclass column generation still running after %d "
                         "iterations (%d active columns, nviews=%d, "
                         "budget_ratio=%.6g) -- not truncating, but this solve is "
                         "pathological", n_iter, len(active_c), nviews, budget_ratio)
        with tick("t_master_lp"):
            obj_map = -np.array(active_c)
            A_ub = np.array([active_g])
            b_ub = np.array([budget_ratio])
            A_eq = np.ones((1, len(active_c)))
            b_eq = np.array([1.0])
            res_rmp = opt.linprog(obj_map, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq,
                                   b_eq=b_eq, bounds=(0, None), method="highs")
            y_ub = res_rmp.ineqlin.marginals[0] if res_rmp.ineqlin is not None else 0.0
            y_eq = res_rmp.eqlin.marginals[0] if res_rmp.eqlin.marginals is not None else 0.0

        with tick("t_pricing"):
            new_subset, min_reduced_cost = solve_subproblem_bb(y_ub, y_eq)
        opt_dist = res_rmp.x
        if new_subset is None or min_reduced_cost >= -thres:
            break  # no improving column left -- optimal

        is_duplicate = any(np.array_equal(new_subset, s) for s in active_subsets)
        if is_duplicate:
            _log.warning("multiclass column generation stopped on a DUPLICATE column "
                         "at iteration %d (%d active columns) -- the returned policy "
                         "is the last feasible master solution, not a certified "
                         "optimum", n_iter, len(active_c))
            break

        active_subsets.append(new_subset)
        active_c.append(reward_of(new_subset))
        active_g.append(float(np.sum(costs[new_subset])))

    _log.debug("multiclass colgen: %d iterations, %d columns, nviews=%d",
               n_iter, len(active_subsets), nviews)

    probs = np.clip(opt_dist, 0, None)
    probs = probs / probs.sum() if probs.sum() > 0 else np.ones(len(active_subsets)) / len(active_subsets)

    return active_subsets, probs


def sample_lp_colgen_policy(X_inference, Y_inference, learned_centers, costs,
                             inference_budget, rng, label_map, predict_fn, noise_var=1.0):
    """Shared "solve + physically sample" routine used by
    two_stage_asymmetric.run_inference_lp_colgen (synthetic, label_map from
    match_cluster_labels) and two_stage_runner.run_inference_lp_dataset_colgen
    (real data, identity label_map) -- previously duplicated near-verbatim
    in both files; the only real difference between the two callers is how
    `label_map` gets built, so that's now the caller's job, not this
    function's.

    `predict_fn` is passed in (rather than imported here) to avoid a
    circular import: two_stage_asymmetric.py imports from core/, so core/
    cannot import back from two_stage_asymmetric.py. Pass
    `predict_single_combination` from two_stage_asymmetric.py.

    Solves the LP via solve_lp_policy_colgen (0.5-factor `lp_reward`
    convention), converts the discovered 0-indexed boolean masks to this
    codebase's 1-indexed combo-tuple convention (index 0 == view 1, the
    free view), then physically samples a combo per inference round subject
    to a running budget, falling back to the free view alone if the budget
    would go negative -- identical policy-execution logic in both former
    duplicates.

    Returns
    -------
    dict with keys: combos_list, probabilities, inference_accuracy,
    inference_error, inference_f1, inference_auroc, actual_cost.
    """
    n = len(X_inference)

    with tick("t_inference_solve"):
        masks, probs = solve_lp_policy_colgen(
            est_means=learned_centers, costs=costs,
            inference_budget=inference_budget, n_inference=n, noise_var=noise_var,
        )
    combos_list = [tuple(i + 1 for i, flag in enumerate(m) if flag) for m in masks]
    combo_costs_discovered = {c: float(np.sum(costs[np.array(c) - 1])) for c in combos_list}
    probabilities = np.clip(probs, 0, 1)
    probabilities /= np.sum(probabilities)

    actual_inference_rewards = []
    actual_inference_cost = 0.0
    remaining_inference_budget = inference_budget

    sign_factor = 1.0 if label_map.get(0, 0) == 0 else -1.0
    y_pred_mapped = np.zeros(n, dtype=int)
    evidence_for_class1 = np.zeros(n)
    n_fallbacks = 0

    with tick("t_inference_sample"):
        for i in range(n):
            sampled_idx = rng.choice(len(combos_list), p=probabilities)
            sampled_combo = combos_list[sampled_idx]

            cost_of_combo = combo_costs_discovered[sampled_combo]
            if remaining_inference_budget - cost_of_combo < 0:
                sampled_combo = (1,)  # fallback: the free view alone
                cost_of_combo = costs[0]
                n_fallbacks += 1
            else:
                remaining_inference_budget -= cost_of_combo
                actual_inference_cost += cost_of_combo

            raw_pred, score = predict_fn(X_inference[i], learned_centers, sampled_combo, return_score=True)
            matched_pred = label_map.get(raw_pred, raw_pred)
            actual_inference_rewards.append(int(matched_pred == Y_inference[i]))
            y_pred_mapped[i] = matched_pred
            evidence_for_class1[i] = sign_factor * score

    if n_fallbacks:
        # The budget ran out partway through inference, so the tail of the
        # test set was classified on the free view alone. That is a
        # legitimate policy outcome, but it depresses inference_accuracy for
        # a reason nothing in the workbook otherwise records.
        _log.info("inference budget exhausted on %d/%d rounds (%.1f%%) -- those "
                  "rounds fell back to the free view", n_fallbacks, n,
                  100.0 * n_fallbacks / max(1, n))

    inference_accuracy = float(np.mean(actual_inference_rewards))

    inference_f1 = f1_score(Y_inference, y_pred_mapped, zero_division=0)
    try:
        inference_auroc = roc_auc_score(Y_inference, evidence_for_class1)
    except ValueError:
        inference_auroc = float("nan")

    return {
        "combos_list": combos_list,
        "probabilities": probabilities,
        "inference_accuracy": inference_accuracy,
        "inference_error": float(1 - inference_accuracy),
        "inference_f1": inference_f1,
        "inference_auroc": inference_auroc,
        "actual_cost": actual_inference_cost,
    }
