# -*- coding: utf-8 -*-
"""
core/acquisition_policies.py

Shared acquisition policies and enumerated-arm bookkeeping for Adaptive and
Two-stage. The standalone greedy acquisition has been removed; the nested
chain builder remains because it defines the action space used by
``lp_chain``.

=== Instrumentation note ===
The acquisition entry points below carry a
@timed("t_acquisition") decorator from core.logging_utils. That single
placement is why the new t_acquisition column is populated for BOTH method
families and for EVERY acquisition mode without either method module being
edited: both of them reach their per-round subset choice through exactly
these functions. Decorators, not `with` blocks, so no numerical body was
re-indented. The decorator is a no-op when timing is disabled and when no
runner is collecting, so importing this module standalone costs nothing.
Do NOT add a t_acquisition tick at any CALLER of these -- ticks accumulate
per name, and a caller-side tick around a call to a decorated function
would count the same span twice.
"""

from __future__ import annotations

import numba
import numpy as np
import scipy.optimize as opt

from core.logging_utils import bump, timed

ACQUISITION_MODES = ("lp_chain", "lp_full_opt", "ucb_argmax", "hedge")
LP_ACQUISITION_MODES = ("lp_chain", "lp_full_opt")
ORACLE_ACQUISITION_MODES = ("lp_full_opt",)
ARGMAX_ACQUISITION_MODES = ("ucb_argmax",)
HEDGE_ACQUISITION_MODES = ("hedge",)
FULL_ENUMERATION_MODES = ACQUISITION_MODES
REWARD_UPDATE_SCOPES = ("subsets", "selected")
MAX_REWARD_ESTIMATE_VIEWS = 20
UCB_BOUNDS = ("vc", "vector")


def validate_ucb_bound(ucb_bound):
    """Validate and return the empirical-arm confidence-bound option."""
    if ucb_bound not in UCB_BOUNDS:
        raise ValueError(
            f"ucb_bound must be one of {UCB_BOUNDS}; got {ucb_bound!r}.")
    return ucb_bound


def ucb_confidence_bonus(combo_counts, alpha_ucb, round_idx,
                         ucb_bound="vc", vc_dimension=None):
    """VC- or vector-complexity confidence radius for empirical rewards.

    ``vc`` uses the model-complexity radius
    ``alpha_ucb * sqrt((vc_dimension + log(round_idx + 2)) / n)``.  Callers
    supply ``vc_dimension = K * d_arm**2`` for each arm.

    ``vector`` uses the same radius with ``vc_dimension = d_arm``.
    """
    validate_ucb_bound(ucb_bound)
    counts = np.asarray(combo_counts, dtype=np.float64)
    if np.any(counts <= 0):
        raise ValueError("All arm counts must be positive.")
    if float(alpha_ucb) < 0:
        raise ValueError("alpha_ucb must be non-negative.")
    if vc_dimension is None:
        raise ValueError(f"vc_dimension is required for ucb_bound={ucb_bound!r}.")
    complexity = np.asarray(vc_dimension, dtype=np.float64)
    if complexity.shape != counts.shape:
        raise ValueError("vc_dimension and combo_counts must have the same shape.")
    if np.any(complexity < 0):
        raise ValueError("vc_dimension must be non-negative.")
    return float(alpha_ucb) * np.sqrt(
        (complexity + np.log(int(round_idx) + 2)) / counts)


def resolve_step_size(step_size, time_horizon):
    """Validate the OMD step-size option.

    ``None`` denotes the automatic per-round schedule ``1 / sqrt(t + 1)``;
    an explicit number denotes a fixed override.
    """
    time_horizon = int(time_horizon)
    if time_horizon <= 0:
        raise ValueError(
            f"time_horizon must be positive, got {time_horizon}")
    if step_size is None:
        return None
    step_size = float(step_size)
    if not np.isfinite(step_size) or step_size < 0:
        raise ValueError(
            f"step_size must be a finite non-negative number or None, got "
            f"{step_size!r}")
    return step_size


def step_size_at_round(step_size, round_idx):
    """Return the OMD step size at zero-based global round ``round_idx``."""
    round_idx = int(round_idx)
    if round_idx < 0:
        raise ValueError(f"round_idx must be non-negative, got {round_idx}")
    if step_size is None:
        return float(1.0 / np.sqrt(round_idx + 1))
    step_size = float(step_size)
    if not np.isfinite(step_size) or step_size < 0:
        raise ValueError(
            f"step_size must be a finite non-negative number or None, got "
            f"{step_size!r}")
    return step_size


def step_size_tag(step_size):
    """Stable filename representation of explicit and automatic step sizes."""
    return "invSqrtRound" if step_size is None else f"{float(step_size):g}"


def uses_empirical_arm_rewards(acquisition):
    return acquisition in ("lp_chain", "ucb_argmax", "hedge")

# Reward
@numba.njit
def bhattacharyya_accuracy_proxy(diff_mean_sq_mat):
    d_norm = np.sum(diff_mean_sq_mat, axis=0)  # (nc, nc)
    # Bhattacharyya overlap-based proxy for classification error
    error = np.exp(-0.125 * d_norm)  # 1/8 * |\delta\mu|^2
    return np.maximum(1.0 - error, 0.0)
@numba.njit
def multiclass_reward(diff_mean_sq_mat):
    nc = diff_mean_sq_mat.shape[1]
    pairwise_acc = bhattacharyya_accuracy_proxy(diff_mean_sq_mat)  # (nc, nc)
    denom = 1.0 / nc / (nc - 1)
    return 0.5 * denom * np.sum(pairwise_acc)
def pairwise_diff_sq_from_means(est_means):
    mean_tr = np.asarray(est_means, dtype=np.float64).T  # (nviews, nc)
    w_diff = mean_tr[:, :, None] - mean_tr[:, None, :]   # (nviews, nc, nc)
    return np.square(w_diff)


# Enumerated action spaces and policies.
@timed("t_acquisition")
def greedy_chain(centers, costs, free_indices, gain_func, force_free=True,
                 empty_value=0.0):
    """The GREEDY CHAIN: the nested action space that replaces the
    full 2^(nviews-1) enumeration, cutting the arm count to nviews+1.

    Free views are added first, then paid views are appended by the best
    marginal-gain-per-unit-cost ratio:
        S_0 = {free views} subset S_1 subset ... subset S_p = everything
    Returns
    -------
    list of 1-INDEXED view tuples ordered by increasing size, so callers'
    bit_index / combo_masks / combo_cost construction (and
    generate_view_combinations' output format) consume it with no changes.
    """
    nviews = centers.shape[1]
    sel, objective = [], empty_value
    for i in list(free_indices):
        gain = gain_func(sel + [i]) - objective
        if force_free or gain > 0:
            sel.append(i)
            objective += gain
    if not sel:            # no free view in this cost model
        sel = [int(np.argmin(costs))]
        objective = gain_func(sel)
    chain = [sorted(sel)]                                        # S_0
    remaining = [i for i in range(nviews) if i not in sel]
    while remaining:
        best_ratio, best_add = -np.inf, None
        for i in remaining:
            gain = gain_func(sel + [i]) - objective
            ratio = gain / (costs[i] + 1e-9)
            if ratio > best_ratio:
                best_ratio, best_add = ratio, i
        sel.append(best_add)
        objective = gain_func(sel)
        remaining.remove(best_add)
        chain.append(sorted(sel))                                # S_1 .. S_p
    chain.sort(key=len)
    return [tuple(v + 1 for v in s) for s in chain]              # 1-indexed

@timed("t_acquisition")
def linprog_policy_over_estimates(ucb, combo_cost, budget_per_round):
    """Exact optimum of the per-round budgeted LP over the ENUMERATED
    combinations, solved with a general LP solver:

        maximise    sum_c p_c * ucb_c
        subject to  sum_c p_c * cost_c <= budget_per_round
                    sum_c p_c = 1,  p >= 0

    Returns (p, lam):
        p    (n_combos,) probability vector -- SAMPLE from it. The previous
             implementation returned a two-point mixture (idx_lo, idx_hi,
             p_hi) because one inequality plus the simplex means SOME
             optimal vertex has at most two nonzeros; the solver may
             instead land on an interior optimal face when arms tie, so the
             full vector is returned and callers draw from it.
        lam  shadow price on the budget constraint, >= 0, taken from the
             LP dual rather than a hull slope. Same meaning as before --
             what the OMD dual was approximating -- so Lagrangian traces
             stay comparable across acquisition modes.
    """
    bump("n_lp_solves")
    ucb = np.asarray(ucb, dtype=np.float64).ravel()
    combo_cost = np.asarray(combo_cost, dtype=np.float64).ravel()
    n = ucb.shape[0]
    b = max(0.0, float(budget_per_round))

    res = opt.linprog(
        -ucb,
        A_ub=combo_cost.reshape(1, -1),
        b_ub=np.array([b]),
        A_eq=np.ones((1, n)),
        b_eq=np.array([1.0]),
        bounds=(0.0, 1.0),
        method="highs",
    )

    if not res.success:
        from core.logging_utils import get_logger
        get_logger("afa.acquisition").warning(
            "linprog FAILED (status=%s, %s) at budget_per_round=%.6g over %d arms; "
            "falling back to the cheapest arm for this round",
            getattr(res, "status", "?"), getattr(res, "message", ""), b, n)
        p = np.zeros(n)
        p[int(np.argmin(combo_cost))] = 1.0
        return p, 0.0

    p = np.clip(res.x, 0.0, None)
    total = p.sum()
    if total <= 0:
        p = np.zeros(n)
        p[int(np.argmin(combo_cost))] = 1.0
    else:
        p = p / total          # exact simplex sum, for rng.choice

    # HiGHS reports d(objective)/d(b_ub) for the MINIMISED objective -ucb@p,
    # so the marginal is <= 0 and its negation is the gain in expected ucb
    # per unit of budget -- matching the old hull slope's sign convention.
    lam = 0.0
    marg = getattr(res, "ineqlin", None)
    if marg is not None and len(marg.marginals):
        lam = -float(marg.marginals[0])

    return p, max(lam, 0.0)

@timed("t_acquisition")
def argmax_policy_over_estimates(ucb, combo_cost, omd_lambda,
                                 remain_budget=None):
    """The NOTEBOOK's acquisition rule (acquisition="ucb_argmax"): pick the
    single arm maximising the Lagrangian

        score_c = ucb_c - omd_lambda * cost_c

    over the ENUMERATED arms, with no randomisation and no LP.

    Unlike ``linprog_policy_over_estimates``:

      * DETERMINISTIC. The LP returns a distribution and the caller draws
        from it, so the LP can hit a per-round budget exactly by mixing two
        arms. This returns one index; the budget is respected only in
        expectation through ``omd_lambda``. The caller must therefore keep
        running its OMD dual update --
        with omd_lambda pinned at 0 this degenerates to "always buy the
        highest-ucb arm", which is usually the full view set.
      * COST ENTERS LINEARLY, not as a constraint. An arm whose cost exceeds
        what is left is still scored; `remain_budget` (optional) filters it
        out afterwards.

    Parameters
    ----------
    ucb : (n_arms,) array
        Arm values, already including any exploration bonus. NOT capped at
        1.0 here, deliberately -- see the CAPPING note in the two callers'
        module docstrings: capping an argmax collapses saturated arms into a
        tie that np.argmax always breaks toward index 0.
    combo_cost : (n_arms,) array, total cost of each arm.
    omd_lambda : float, current dual variable / shadow price.
    remain_budget : float or None
        If given, arms costing more than this are excluded. When nothing is
        affordable the cheapest arm is returned rather than raising -- with
        this codebase's cost convention (arm 0 is the free view alone, cost
        0) that is the free-only fallback every other mode also uses.

    Returns
    -------
    int : index into `ucb` / `combo_cost`.
    """
    ucb = np.asarray(ucb, dtype=np.float64).ravel()
    combo_cost = np.asarray(combo_cost, dtype=np.float64).ravel()
    score = ucb - float(omd_lambda) * combo_cost
    if remain_budget is not None:
        affordable = combo_cost <= float(remain_budget) + 1e-12
        if not affordable.any():
            return int(np.argmin(combo_cost))
        score = np.where(affordable, score, -np.inf)
    return int(np.argmax(score))


def build_arm_tables(combos, costs, nviews):
    """Bookkeeping shared by every enumerated-action-space caller.

    Parameters
    ----------
    combos : list of 1-INDEXED view tuples
        Either greedy_chain(...) or
        core.two_stage_utils.generate_view_combinations(nviews).
    costs : (nviews,) array, 0-indexed per-view costs.

    Returns
    -------
    dict with
        combo_masks : (n_arms, nviews) bool
        combo_cost  : (n_arms,) float   -- total cost of each arm
        arm_bits    : (n_arms,) int64   -- bitmask of each arm, arm j's bit i
                                           set iff view i is in arm j
        bit_index   : dict bitmask -> arm index

    Deliberately does NOT build the sub_arms containment list: it is
    O(n_arms^2) memory, which is fine for the nviews+1 chain and fatal for
    the 2^(nviews-1) enumeration. Callers get arm_bits and test containment
    with the vectorised `(arm_bits & played_bits) == arm_bits`.
    """
    n_arms = len(combos)
    combo_masks = np.zeros((n_arms, nviews), dtype=bool)
    combo_cost = np.zeros(n_arms, dtype=np.float64)
    arm_bits = np.zeros(n_arms, dtype=np.int64)
    bit_index = {}
    for j, combo in enumerate(combos):
        idx = np.asarray(combo, dtype=int) - 1
        combo_masks[j, idx] = True
        combo_cost[j] = float(costs[combo_masks[j]].sum())
        bits = int(sum(1 << int(i) for i in idx))
        arm_bits[j] = bits
        bit_index[bits] = j
    return {
        "combo_masks": combo_masks,
        "combo_cost": combo_cost,
        "arm_bits": arm_bits,
        "bit_index": bit_index,
    }


def mask_to_bits(mask):
    """0-indexed boolean view mask -> integer bitmask (bit i = view i)."""
    return int(sum(1 << i for i in range(len(mask)) if mask[i]))


def arm_accuracies_from_means(X_rows, Y_rows, means, combo_masks):
    """Per-arm EMPIRICAL nearest-centroid accuracy of `means`, restricted to
    each arm's view set, measured on (X_rows, Y_rows).

    This is the arm-value function for the ORACLE acquisition modes: pass
    the TRUE generative means and the rows the policy is about to run over,
    and the result is each subset's exact achievable accuracy -- the same
    quantity r_hat is a running estimate of under the learned policies, on the
    same 0/1 scale, so an oracle run and a learning run are directly
    comparable arm for arm.

    Returns (n_arms,) float. An empty row set yields a flat chance-level
    1/nclasses, which keeps the LP well posed (it then just buys the
    cheapest arm) instead of raising inside the solver.
    """
    combo_masks = np.asarray(combo_masks, dtype=bool)
    means = np.asarray(means, dtype=np.float64)
    n_arms = combo_masks.shape[0]
    nclasses = means.shape[0]

    xs = np.asarray(X_rows, dtype=np.float64)
    ys = np.asarray(Y_rows, dtype=int)
    if len(xs) == 0:
        return np.full(n_arms, 1.0 / nclasses, dtype=np.float64)
    if means.shape[1] != combo_masks.shape[1]:
        raise ValueError(
            f"means has {means.shape[1]} views but combo_masks has "
            f"{combo_masks.shape[1]}. Pass true means at the POST-truncation "
            f"width -- see core.optimal_static.synthetic_true_means' "
            f"n_views_used parameter, which exists for exactly this mismatch.")

    acc = np.empty(n_arms, dtype=np.float64)
    for j in range(n_arms):
        m = combo_masks[j]
        d = ((xs[:, None, m] - means[None, :, m]) ** 2).sum(axis=2)  # (n, nc)
        acc[j] = float(np.mean(d.argmin(axis=1) == ys))
    return acc
