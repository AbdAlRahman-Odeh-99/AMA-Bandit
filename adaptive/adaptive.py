# -*- coding: utf-8 -*-
"""
adaptive.py
"""

import numba
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score
from core.two_stage_utils import generate_view_combinations
from core.acquisition_policies import (
    ACQUISITION_MODES,
    ARGMAX_ACQUISITION_MODES,
    HEDGE_ACQUISITION_MODES,
    FULL_ENUMERATION_MODES,
    MAX_REWARD_ESTIMATE_VIEWS,
    ORACLE_ACQUISITION_MODES,
    REWARD_UPDATE_SCOPES,
    arm_accuracies_from_means,
    argmax_policy_over_estimates,
    build_arm_tables,
    greedy_chain,
    linprog_policy_over_estimates,
    mask_to_bits,
    resolve_step_size,
    step_size_at_round,
    ucb_confidence_bonus,
    uses_empirical_arm_rewards as _uses_empirical_arm_rewards,
    validate_ucb_bound,
)

from core.logging_utils import get_logger

_log = get_logger("afa.adaptive")

# ─────────────────────────────────────────────────────────────────────────
# Prediction
# ─────────────────────────────────────────────────────────────────────────
@numba.njit
def pred_linear_cla(x_observe, class_means):
    mean_tr = class_means.T  # (v, nc)
    diff_mean_sq = mean_tr[:, :, None] - mean_tr[:, None, :]  # (sel, nc, nc)
    pairwise_mean_avg = 0.5 * (mean_tr[:, :, None] + mean_tr[:, None, :])
    inner_prod = np.sum(
        (x_observe[:, None, None] - pairwise_mean_avg) * diff_mean_sq,
        axis=0) > 0  # bool (nc, nc)
    np.fill_diagonal(inner_prod, False)
    y_pred = np.argmax(np.sum(inner_prod, axis=1))
    return y_pred

def class_posterior_scores(x_observe, class_means_sub):
    """softmax(-0.5 * ||x_obs - mu_k[obs]||^2) over classes k -- the exact
    class posterior under equal priors and unit shared variance, restricted
    to the observed views. Used ONLY for AUROC (a continuous per-class
    score); the hard prediction stays pred_linear_cla's for verbatim parity
    with the notebook. Returns shape (nclasses,), rows sum to 1."""
    d2 = np.sum(np.square(x_observe[None, :] - class_means_sub), axis=1)
    logits = -0.5 * d2
    logits -= logits.max()  # numerical stability
    p = np.exp(logits)
    return p / p.sum()


def _macro_f1(y_true, y_pred):
    return f1_score(y_true, y_pred, average="macro", zero_division=0)


def _macro_ovr_auroc(y_true, score_mat, nclasses):
    """Macro one-vs-rest AUROC from an (n, nclasses) posterior-score
    matrix; falls back to the standard binary AUROC (positive-class score
    column) when nclasses == 2, and to NaN when a class is missing from
    y_true (roc_auc_score raises ValueError in that case)."""
    try:
        if nclasses == 2:
            return roc_auc_score(y_true, score_mat[:, 1])
        return roc_auc_score(y_true, score_mat, multi_class="ovr",
                             average="macro", labels=np.arange(nclasses))
    except ValueError:
        return float("nan")


# ─────────────────────────────────────────────────────────────────────────
# Training phase: adaptive online training under a training budget
# ─────────────────────────────────────────────────────────────────────────
def run_training_phase(nviews, nclasses, costs, n_train, training_budget,
                        X_train, Y_train, est_means_init, feedback="full",
                        alpha_ucb=1.0, lr=1e-2, step_size=None,
                        lambda_max=10.0, rng=None,
                        acquisition="lp_chain", ucb_bound="vc",
                        reward_update="subsets",
                        force_free=True, true_means=None):
    """
    Adaptive multiclass training phase, wrapping the notebook's `simulation`
    (feedback="bandit") / `simulation_full_feedback` (feedback="full")
    update rules in the repo's phase structure. Returns the same dict
    shape as the binary version (est_means, est_counts, train_reward,
    cum_train_reward, train_f1, train_auroc, label_mapping, spent), with
    label_mapping always the identity (see module docstring, item 3), plus
    a few acquisition-mode diagnostics that callers may ignore.

    feedback : {"full", "bandit"}
        est_means update rule. Unchanged.
    acquisition : {"lp_chain", "lp_full_opt", "ucb_argmax", "hedge"}
        Per-round subset selection. See the module docstring's ACQUISITION
        MODES section. Learned policies use empirical arm rewards;
        ``lp_full_opt`` uses exact rewards from the supplied true means.
    true_means : (nclasses, nviews) array, REQUIRED for "lp_full_opt"
        The generative class means. Ignored by every other mode. Supply
        core.optimal_static.synthetic_true_means(...) at the POST-truncation
        view width (pass X_train.shape[1] as n_views_used).
    reward_update : {"subsets", "selected"}, default "subsets"
        Controls empirical arm-reward updates for learned policies and is
        ignored by ``lp_full_opt``.
        "subsets" replays every arm contained in the played subset;
        "selected" scores only the played arm.
    force_free : bool, default True
        Passed to greedy_chain -- keeps the free view(s) in every acquired
        subset, matching the LP's invariant.

    rng: warmup rounds' uniform-random predictions (notebook behavior) and,
        under the LP modes, the per-round mixture draw; a fresh
        default_rng(0) is created if None.

    """
    if feedback not in ("full", "bandit"):
        raise ValueError(f"feedback must be 'full' or 'bandit', got {feedback!r}")
    if acquisition not in ACQUISITION_MODES:
        raise ValueError(f"acquisition must be one of {ACQUISITION_MODES}, "
                         f"got {acquisition!r}")
    validate_ucb_bound(ucb_bound)
    if reward_update not in REWARD_UPDATE_SCOPES:
        raise ValueError(f"reward_update must be one of {REWARD_UPDATE_SCOPES}, "
                         f"got {reward_update!r}")
    uses_empirical_arm_rewards = _uses_empirical_arm_rewards(acquisition)
    is_argmax = acquisition in ARGMAX_ACQUISITION_MODES
    is_hedge = acquisition in HEDGE_ACQUISITION_MODES

    if (feedback == "bandit" and reward_update == "subsets" and uses_empirical_arm_rewards):
        raise ValueError(
            "feedback='bandit' with reward_update='subsets' is incoherent: the "
            "counterfactual replay reads y_true, which bandit feedback does not "
            "reveal. Use reward_update='selected' with feedback='bandit', or "
            "feedback='full' with reward_update='subsets'.")
    if acquisition in FULL_ENUMERATION_MODES and nviews > MAX_REWARD_ESTIMATE_VIEWS:
        raise ValueError(
            f"acquisition={acquisition!r} enumerates 2^(nviews-1) = 2^{nviews - 1} "
            f"subsets eagerly; nviews={nviews} exceeds "
            f"MAX_REWARD_ESTIMATE_VIEWS={MAX_REWARD_ESTIMATE_VIEWS}. Trim "
            f"the input with max_modalities.")

    is_oracle = acquisition in ORACLE_ACQUISITION_MODES
    if is_oracle:
        if true_means is None:
            raise ValueError(
                f"acquisition={acquisition!r} needs the TRUE generative means, "
                f"which exist only for the synthetic datasets. Pass "
                f"true_means=core.optimal_static.synthetic_true_means(...).")
        true_means = np.asarray(true_means, dtype=np.float64)
        if true_means.shape != (nclasses, nviews):
            raise ValueError(
                f"true_means must have shape ({nclasses}, {nviews}), got "
                f"{true_means.shape}. synthetic_true_means must be called with "
                f"n_views_used=X_train.shape[1] -- the generator's width and "
                f"the post-max_modalities width are not the same thing.")
    if rng is None:
        rng = np.random.default_rng(0)
    step_size = resolve_step_size(step_size, n_train)
    budget_window = int(n_train)

    est_means = np.asarray(est_means_init, dtype=np.float64).copy()
    if est_means.shape != (nclasses, nviews):
        raise ValueError(
            f"est_means_init must have shape ({nclasses}, {nviews}), got {est_means.shape}"
        )

    est_counts = np.ones((nclasses, nviews))
    one_vec = np.ones(nclasses)

    free_indices = [idx for idx in range(nviews) if costs[idx] == 0]
    free_only_subset = np.zeros(nviews, dtype=bool)
    free_only_subset[free_indices] = True

    remaining_budget = training_budget
    spending_ratio = training_budget / budget_window
    omd_lambda = 0.0

    record_pred = np.zeros(n_train, dtype=int)
    record_scores = np.zeros((n_train, nclasses))
    total_spent = 0.0

    # Learned policies maintain one permanent empirical table over the full
    # arm universe. lp_chain selects its per-round nested action space from
    # this table; ucb_argmax and hedge score the full table directly.
    combo_masks = combo_cost = arm_bits = None
    bit_index = r_hat = combo_counts = None
    if uses_empirical_arm_rewards:
        if nviews > MAX_REWARD_ESTIMATE_VIEWS:
            raise ValueError(
                f"this empirical-arm acquisition enumerates 2^(nviews-1) = "
                f"2^{nviews - 1} arms; nviews={nviews} exceeds "
                f"MAX_REWARD_ESTIMATE_VIEWS={MAX_REWARD_ESTIMATE_VIEWS}.")
        tables = build_arm_tables(generate_view_combinations(nviews), costs, nviews)
        combo_masks = tables["combo_masks"]
        combo_cost = tables["combo_cost"]
        arm_bits = tables["arm_bits"]
        bit_index = tables["bit_index"]
        r_hat = np.full(len(arm_bits), 1.0 / nclasses, dtype=np.float64)
        combo_counts = np.ones(len(arm_bits), dtype=np.float64)

    def confidence_bonus(indices, round_idx):
        counts = combo_counts[indices]
        arm_dimension = np.count_nonzero(combo_masks[indices], axis=-1)
        vc_dimension = (nclasses * np.square(arm_dimension)
                        if ucb_bound == "vc" else arm_dimension)
        return ucb_confidence_bonus(
            counts, alpha_ucb, round_idx,
            ucb_bound=ucb_bound, vc_dimension=vc_dimension)

    b_allowance = spending_ratio
    p_oracle = None
    views_trace = np.zeros(n_train)
    cost_trace = np.zeros(n_train, dtype=np.float64)
    budget_before_trace = np.zeros(n_train, dtype=np.float64)
    budget_remaining_trace = np.zeros(n_train, dtype=np.float64)
    seen_masks = set()
    selected_subsets = [None] * n_train



    # ORACLE SETUP (acquisition="lp_full_opt" only)
    if is_oracle:
        tables = build_arm_tables(generate_view_combinations(nviews), costs, nviews)
        combo_masks = tables["combo_masks"]
        combo_cost = tables["combo_cost"]
        arm_bits = tables["arm_bits"]
        bit_index = tables["bit_index"]
        r_hat = arm_accuracies_from_means(X_train, Y_train, true_means, combo_masks)
        combo_counts = np.full(len(arm_bits), float(n_train))
        p_oracle, omd_lambda = linprog_policy_over_estimates(r_hat, combo_cost, spending_ratio)

    # ONE-TIME HEDGE SETUP (acquisition="hedge" only)
    if is_hedge:
        hedge_v = np.ones(2, dtype=np.float64)
        hedge_epsilon = np.sqrt(np.log(2.0) / n_train)

    for t in range(n_train):
        budget_before_trace[t] = remaining_budget
        if t < nclasses:
            subset = np.ones(nviews, dtype=bool)
            is_init = True
        elif is_argmax:
            if remaining_budget <= 0:
                subset = free_only_subset.copy()
            else:
                round_idx = max(t - nclasses, 0)
                raw_ucb = r_hat + confidence_bonus(slice(None), round_idx)
                affordable_idx = np.flatnonzero(combo_cost <= remaining_budget + 1e-12)
                candidate_cost = combo_cost[affordable_idx]
                candidate_ucb = raw_ucb[affordable_idx]
                # j_local indexes candidate_ucb/candidate_cost.
                j_local = argmax_policy_over_estimates(candidate_ucb, candidate_cost, omd_lambda, remaining_budget,)
                # Convert the local result back to the permanent arm table.
                j_global = int(affordable_idx[j_local])
                subset = combo_masks[j_global].copy()
            is_init = False
        elif is_hedge:
            if remaining_budget <= 0:
                subset = free_only_subset.copy()
            else:
                round_idx = max(t - nclasses, 0)
                candidate_idx = np.flatnonzero(
                    combo_cost <= remaining_budget + 1e-12)
                candidate_cost = combo_cost[candidate_idx]
                candidate_ucb = (r_hat[candidate_idx]
                                 + confidence_bonus(candidate_idx, round_idx))
                hedge_y = hedge_v / hedge_v.sum()
                candidate_effective_cost = (hedge_y[0] + hedge_y[1] * candidate_cost)
                candidate_score = candidate_ucb / candidate_effective_cost
                # j_local indexes candidate_score.
                j_local = int(np.argmax(candidate_score))
                # Convert to the global arm index.
                j_global = int(candidate_idx[j_local])
                # Adaptive result.
                subset = combo_masks[j_global].copy()
            is_init = False
        elif is_oracle:
            if remaining_budget <= 0:
                subset = free_only_subset.copy()
            else:
                subset = combo_masks[int(rng.choice(len(p_oracle), p=p_oracle))].copy()
            is_init = False
        else:
            # lp_chain: construct a nested chain from the current empirical
            # UCB table, then solve the budgeted LP over that chain.
            rounds_left_in_window = budget_window - (t % budget_window)
            b_allowance = max(0.0, remaining_budget) / max(1, rounds_left_in_window)
            if remaining_budget <= 0:
                subset = free_only_subset.copy()
            else:
                round_idx = max(t - nclasses, 0)
                def gain_func(sel):
                    bits = sum(1 << int(i) for i in sel)
                    j_global = bit_index[bits]
                    return float(r_hat[j_global]
                                 + confidence_bonus(j_global, round_idx))
                combos = greedy_chain(
                    est_means, costs, free_indices, gain_func,
                    force_free=force_free, empty_value=1.0 / nclasses)
                candidate_idx = np.array([
                    bit_index[sum(1 << (v - 1) for v in combo)]
                    for combo in combos
                ], dtype=int)
                candidate_ucb = (r_hat[candidate_idx]
                                 + confidence_bonus(candidate_idx, round_idx))
                candidate_cost = combo_cost[candidate_idx]
                p_lp, omd_lambda = linprog_policy_over_estimates(
                    candidate_ucb, candidate_cost, b_allowance)
                j_local = int(rng.choice(len(p_lp), p=p_lp))
                j_global = int(candidate_idx[j_local])
                subset = combo_masks[j_global].copy()
            is_init = False

        # budget check
        inst_cost = np.sum(costs[subset])
        if remaining_budget >= inst_cost:
            remaining_budget -= inst_cost
        else:
            subset = free_only_subset.copy()
            inst_cost = np.sum(costs[subset])  # == 0
            remaining_budget -= inst_cost
        if is_hedge:
            resource_z = np.array([1.0, inst_cost], dtype=np.float64,)
            hedge_v *= ((1.0 + hedge_epsilon)** resource_z)
        total_spent += inst_cost
        cost_trace[t] = inst_cost
        budget_remaining_trace[t] = remaining_budget
        views_trace[t] = int(subset.sum())
        seen_masks.add(subset.tobytes())
        selected_subsets[t] = tuple((np.flatnonzero(subset) + 1).tolist())
        # observe acquired views only (features are semi-bandit in BOTH modes)
        x_obs = X_train[t, subset]
        means_sub = est_means[:, subset]
        if not is_init:
            y_pred = int(pred_linear_cla(x_obs, means_sub))
        else:
            y_pred = int(rng.integers(nclasses))
        record_pred[t] = y_pred
        record_scores[t] = class_posterior_scores(x_obs, means_sub)

        y_true = int(Y_train[t])
        reward = y_pred == y_true

        # Per-arm empirical reward update. During the forced full-modality
        # initialization, full-feedback/subset replay can evaluate every arm
        # contained in the acquired full set, so retain those observations.
        # Keep selected-only empirical initialization excluded: its recorded
        # initialization prediction is uniform-random rather than the arm
        # classifier's prediction and would therefore be a spurious reward.
        skip_empirical_init_update = (
            acquisition == "lp_chain" and is_init
            and reward_update == "selected"
        )
        if (combo_masks is not None and not is_oracle
                and not skip_empirical_init_update):
            played_bits = mask_to_bits(subset)
            if reward_update == "selected":
                # BANDIT scope
                j0 = bit_index.get(played_bits)
                targets = [] if j0 is None else [(j0, float(reward))]
            else:
                # COUNTERFACTUAL REPLAY
                targets = []
                contained_idx = np.flatnonzero(
                    (arm_bits & played_bits) == arm_bits)
                for k in contained_idx:
                    m_k = combo_masks[k]
                    y_sub = int(pred_linear_cla(X_train[t, m_k], est_means[:, m_k]))
                    targets.append((int(k), float(y_sub == y_true)))

            for j0, r_obs in targets:
                combo_counts[j0] += 1.0
                r_hat[j0] += (r_obs - r_hat[j0]) / combo_counts[j0]

        # Update means
        if feedback == "full":
            # y_true revealed every round: running mean of the TRUE class
            est_counts[y_true, subset] += 1
            est_means[y_true, subset] += ((1.0 / est_counts[y_true, subset]) * (x_obs - est_means[y_true, subset]))
        else:  # bandit
            est_counts[y_pred, subset] += 1
            if reward:
                # correct prediction: y_pred == y_true, running-mean update
                est_means[y_pred, subset] += ((1.0 / est_counts[y_pred, subset]) * (x_obs - est_means[y_pred, subset]))
            else:
                # incorrect prediction (complementary-label update)
                eliminated = y_pred
                elim = np.zeros(nclasses)
                elim[eliminated] = 1.0
                l_grad = -2 * (x_obs[None, :] - means_sub)  # (nc, ns)
                grad = l_grad * (one_vec - (nclasses - 1) * elim)[:, None]
                est_means[:, subset] -= lr * grad

        # OMD dual update
        # ucb_argmax has no LP constraint to price and would buy the full
        # view set every round if lambda stayed at zero.
        if is_argmax:
            current_step_size = step_size_at_round(step_size, t)
            raw_lambda = (omd_lambda
                          + current_step_size * (inst_cost - spending_ratio))
            omd_lambda = max(0, min(lambda_max, raw_lambda))

    correct_vec = (record_pred == np.asarray(Y_train, dtype=int)).astype(float)
    train_acc = float(np.mean(correct_vec))
    cum_train_acc = np.cumsum(correct_vec) / np.arange(1, n_train + 1)
    train_f1 = _macro_f1(Y_train, record_pred)
    train_auroc = _macro_ovr_auroc(Y_train, record_scores, nclasses)

    return {
        "est_means": est_means,
        "est_counts": est_counts,
        "train_reward": train_acc,
        "cum_train_reward": cum_train_acc,
        "train_f1": train_f1,
        "train_auroc": train_auroc,
        "label_mapping": {k: k for k in range(nclasses)},  # identity -- see docstring
        "spent": total_spent,
        # ── acquisition diagnostics (extra keys; existing callers ignore) ──
        "acquisition": acquisition,
        "ucb_bound": ucb_bound,
        "reward_update": (reward_update if uses_empirical_arm_rewards else ""),
        "n_arms": 0 if combo_masks is None else int(combo_masks.shape[0]),
        "combo_rewards": r_hat,
        "combo_counts": combo_counts,
        "lambda_final": float(omd_lambda),
        # Final value of the automatic 1/sqrt(t+1) schedule; explicit
        # --step-size overrides remain constant and are returned unchanged.
        "step_size": step_size_at_round(step_size, n_train - 1),
        # Last value of the per-round LP allowance.
        "b_allowance_final": float(b_allowance),
        "oracle_probs": p_oracle,
        "avg_views_acquired": float(np.mean(views_trace)),
        "n_unique_masks": len(seen_masks),
        "selected_subsets": selected_subsets,
        "train_predictions": record_pred,
        "train_labels": np.asarray(Y_train, dtype=int).copy(),
        "train_scores": record_scores,
        "cost_trace": cost_trace,
        "budget_before_trace": budget_before_trace,
        "budget_remaining_trace": budget_remaining_trace,
    }


# ─────────────────────────────────────────────────────────────────────────
# Inference phase: LP-based inference policy + physical sampling
# ─────────────────────────────────────────────────────────────────────────
def run_inference_phase(masks, probs, combo_costs, costs, est_means,
                         inference_budget, X_inf, Y_inf, rng):
    n = len(X_inf)
    nclasses = est_means.shape[0]
    remaining_budget = inference_budget
    idxs = np.arange(len(masks))

    fallback_subset = np.zeros(len(costs), dtype=bool)
    fallback_subset[0] = True
    fallback_cost = costs[0]

    correct = 0
    spent = 0.0
    y_pred = np.zeros(n, dtype=int)
    score_mat = np.zeros((n, nclasses))
    for t in range(n):
        sel_i = rng.choice(idxs, p=probs)
        subset = masks[sel_i]
        cost = combo_costs[sel_i]

        if remaining_budget - cost < 0:
            subset = fallback_subset
            cost = fallback_cost

        remaining_budget -= cost
        spent += cost

        x_obs = X_inf[t, subset]
        means_sub = est_means[:, subset]
        pred = int(pred_linear_cla(x_obs, means_sub))
        correct += int(pred == Y_inf[t])
        y_pred[t] = pred
        score_mat[t] = class_posterior_scores(x_obs, means_sub)

    f1 = _macro_f1(Y_inf, y_pred)
    auroc = _macro_ovr_auroc(Y_inf, score_mat, nclasses)

    return {
        "inference_reward": correct / n,
        "spent": spent,
        "inference_f1": f1,
        "inference_auroc": auroc,
    }
