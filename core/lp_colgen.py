# -*- coding: utf-8 -*-
"""Column-generation solver for adaptive multiclass inference.

The solver finds a distribution over modality subsets under the inference
budget using a restricted master LP and branch-and-bound pricing.
"""

import numpy as np
import scipy.optimize as opt
from core.logging_utils import bump, get_logger, tick
from core.acquisition_policies import multiclass_reward, pairwise_diff_sq_from_means

#: Iterations after which column generation is reported as suspicious.
#: Purely diagnostic -- nothing is truncated.
COLGEN_ITER_WARN = 500

_log = get_logger("afa.colgen")




def solve_lp_policy_colgen_multiclass(est_means, costs, inference_budget,
                                       n_inference, thres=1e-6):
    """Solve the budgeted inference LP using multiclass reward estimates.

    Returns the discovered subset masks and their probabilities.
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
