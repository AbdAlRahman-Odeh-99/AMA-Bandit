# -*- coding: utf-8 -*-
"""
core/multiclass_common.py

Shared MULTICLASS building blocks: Stage-1 supervised centre
initialisation, the two hard-decision prediction rules and their shared
posterior scores, the macro F1 / macro-OVR AUROC metrics, and the
LP-column-generation inference routine.

These helpers are shared by the current Two-stage runner and unified runner.

=== Note on the duplicated metric helpers ===
_macro_f1 / _macro_ovr_auroc here are still a small deliberate duplicate of
adaptive.py's, per this repo's "duplicate small helpers
between independent method families rather than cross-import" convention.
That convention is only broken where a SHARED implementation is what makes
a comparison valid (core/acquisition_policies.py's shared policies);
these two are not in that category, so the duplicate stands.

=== Instrumentation note ===
Three additions, none of which touch a number:
  * initialize_centers_multiclass carries @timed("t_init_centers"), so
    Stage 1 is separated from Stage 2 inside what used to be one
    train_time_sec;
  * the inference routine times its colgen solve (t_inference_solve) apart
    from its physical sampling loop (t_inference_sample) -- the solve is
    one call whose cost scales with nviews, the loop is n_test rounds whose
    cost scales with the test set, and lumping them hid which was which;
  * a NaN AUROC and an exhausted inference budget are now logged instead of
    silently producing a NaN cell and a depressed accuracy.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score, roc_auc_score

from core.logging_utils import get_logger, tick, timed
from core.lp_colgen import solve_lp_policy_colgen_multiclass
from core.two_stage_utils import match_cluster_labels

_log = get_logger("afa.multiclass")


# ─────────────────────────────────────────────────────────────────────────
# Metrics -- macro F1 / macro one-vs-rest AUROC, falling back to the
# ordinary binary forms at K == 2. Small local duplicate of
# adaptive.py's helpers (avoids a cross-import between
# independent method families -- same duplication convention already used
# between the two method families).
# ─────────────────────────────────────────────────────────────────────────
def _macro_f1(y_true, y_pred):
    return f1_score(y_true, y_pred, average="macro", zero_division=0)


def _macro_ovr_auroc(y_true, score_mat, nclasses):
    """score_mat: (n, nclasses) posterior-score matrix, columns indexed by
    TRUE label (i.e. already passed through any label_map). Falls back to
    NaN if a class is missing from y_true (roc_auc_score raises
    ValueError in that case).

    The ValueError is now LOGGED. A NaN AUROC cell has two very different
    causes -- a class genuinely absent from this split, or a scoring bug --
    and the workbook cannot tell them apart; the log can.
    """
    try:
        if nclasses == 2:
            return roc_auc_score(y_true, score_mat[:, 1])
        return roc_auc_score(y_true, score_mat, multi_class="ovr",
                              average="macro", labels=np.arange(nclasses))
    except ValueError as exc:
        present = np.unique(np.asarray(y_true))
        _log.warning("AUROC unavailable (-> NaN): %s | nclasses=%d, classes present "
                     "in this split: %s", exc, nclasses, present.tolist())
        return float("nan")


# ─────────────────────────────────────────────────────────────────────────
# Stage 1: supervised online center initialization -- fixes the original's
# k_clusters-ignoring bug (see module docstring, item 1). Body is
# otherwise IDENTICAL to two_stage_asymmetric.initialize_centers (which
# was already written generically over k_clusters everywhere except the
# one rng.normal call).
# ─────────────────────────────────────────────────────────────────────────
@timed("t_init_centers")
def initialize_centers_multiclass(x, y, n_init_samples, k_clusters, m_modalities, rng,
                                   pred_rule="nearest_center"):
    """K-class counterpart of two_stage_asymmetric.initialize_centers.

    Centers start as a random draw ~ N(0, 1) with shape (k_clusters,
    m_modalities) (the ORIGINAL function's own convention -- it already
    used rng.normal, just hardcoded to 2 rows instead of k_clusters).
    Every incoming sample in the loop over the first n_init_samples:
        1. Predicts using the CURRENT centers over ALL views (Stage 1 is
           fully observed), via the selected pred_rule.
        2. Records the prediction error.
        3. Updates the TRUE class's center via a supervised running mean
           (revealed true label yi selects which center updates -- counts
           start at 1, representing the rng.normal draw as a pseudo-
           observation, blended down via `1 - 1/(count+1)`).

    pred_rule: "nearest_center" (default) or "pairwise_vote" -- kept
        CONSISTENT with the rule used in Stage-2 training/inference so the
        init_error blended into two_stage_error reflects the same decision
        rule. (The centre UPDATE is supervised and identical either way;
        only the recorded prediction error depends on pred_rule.)

    Returns
    -------
    learned_centers : ndarray, shape (k_clusters, m_modalities)
    init_error : float
    """
    learned_centers = rng.normal(size=(k_clusters, m_modalities))
    counts = np.ones(k_clusters, dtype=int)

    mistakes = 0
    predictions = 0

    # Running error
    for i in range(n_init_samples):
        xi = x[i]
        yi = int(y[i])

        # Stage 1 observes ALL views -> predict on the full feature vector.
        if pred_rule == "nearest_center":
            pred = _pred_nearest_center(xi, learned_centers)
        elif pred_rule == "pairwise_vote":
            pred = _pred_pairwise_vote(xi, learned_centers)
        else:
            raise ValueError(f"pred_rule must be one of {PRED_RULES}, got {pred_rule!r}")

        predictions += 1
        if pred != yi:
            mistakes += 1

        scaler = 1 - 1 / (counts[yi] + 1)
        learned_centers[yi] = scaler * learned_centers[yi] + (1 - scaler) * xi
        counts[yi] += 1

    # Batch Error
    # # Pass 1: update centers only -- no scoring here, since scoring against
    # # a still-updating center is what made the old init_error an online/
    # # running mistake count rather than a read on the FINAL centers' quality.
    # for i in range(n_init_samples):
    #     xi = x[i]
    #     yi = int(y[i])
    #     scaler = 1 - 1 / (counts[yi] + 1)
    #     learned_centers[yi] = scaler * learned_centers[yi] + (1 - scaler) * xi
    #     counts[yi] += 1

    # # Pass 2 (BATCH): score all n_init_samples against the FINAL centers.
    # mistakes = 0
    # predictions = 0
    # for i in range(n_init_samples):
    #     xi = x[i]
    #     yi = int(y[i])
    #     if pred_rule == "nearest_center":
    #         pred = _pred_nearest_center(xi, learned_centers)
    #     elif pred_rule == "pairwise_vote":
    #         pred = _pred_pairwise_vote(xi, learned_centers)
    #     else:
    #         raise ValueError(f"pred_rule must be one of {PRED_RULES}, got {pred_rule!r}")
    #     predictions += 1
    #     if pred != yi:
    #         mistakes += 1

    init_error = mistakes / predictions if predictions > 0 else 0.0

    # A class that never appears in the first n_init_samples keeps its
    # random N(0,1) centre into Stage 2, where it is then never updated
    # (two_stage freezes the centres). That is a silently crippled run --
    # visible in the numbers only as an unexplained accuracy floor.
    if n_init_samples > 0:
        unseen = np.flatnonzero(counts[:k_clusters] <= 1)
        if unseen.size:
            _log.warning("Stage 1 saw NO samples for %d/%d classes %s in its first "
                         "%d rows -- their centres are still the random N(0,1) draw",
                         unseen.size, k_clusters, unseen.tolist(), n_init_samples)

    return learned_centers, init_error


# ─────────────────────────────────────────────────────────────────────────
# Prediction -- TWO selectable hard-decision rules over the observed
# (combo) views, plus a shared per-class posterior score vector for AUROC.
#
#   pred_rule="nearest_center" (DEFAULT, two_stage's original rule):
#       K-way argmin of squared distance to each class centre -- the
#       Bayes-optimal MAP rule under the shared-isotropic-variance GMM these
#       datasets use. This is what two_stage_asymmetric.py always did (its
#       predict_combo / predict_single_combination returned distances[0] -
#       distances[1], i.e. a 2-class argmin), generalized to K classes.
#
#   pred_rule="pairwise_vote" (Adaptive's rule, shared with
#       adaptive.pred_linear_cla): one-vs-one linear
#       discriminant voting -- for every class pair (i,j) cast one vote for
#       whichever side of the midpoint hyperplane x falls, predict the class
#       with the most pairwise wins.
#
# *** These two rules are MATHEMATICALLY EQUIVALENT (not just "usually
#     agree") given the same observed views. ***
#   Proof: pred_linear_cla's per-pair test <x - (mu_i+mu_j)/2, mu_i-mu_j> > 0
#   expands to |x - mu_i|^2 < |x - mu_j|^2, i.e. "class i beats class j" iff
#   mu_i is CLOSER to x than mu_j. Distance-to-x is a total order over the
#   centres, so the nearest centre beats every other class -> wins all K-1
#   of its pairwise matchups -> is the unique argmax of the vote count ->
#   equals nearest_center's argmin. This holds for ANY data and ANY (even
#   anisotropic) centre configuration, because pred_linear_cla hardcodes the
#   identity-covariance discriminant form (it never estimates a covariance).
#   The only way they can differ is an EXACT distance tie |x-mu_i| ==
#   |x-mu_j| (measure zero on real-valued features; verified 0/30000
#   disagreements across adversarial random K/scale/partial-view trials).
#
# So --pred-rule is, on this method pair, effectively a NO-OP for accuracy:
# it exists to let you assert/confirm Two-stage and Adaptive share a
# decision rule (ruling the prediction rule OUT as a source of any accuracy
# gap between them -- the gap is entirely acquisition-policy + training
# dynamics), NOT because switching it will change two_stage's numbers.
# It's kept as a real switch anyway so that if pred_linear_cla is ever
# changed to a covariance-weighted form (which would NO LONGER reduce to
# nearest-centroid), two_stage can follow it without further plumbing.
#
# Both rules share class_posterior_scores for AUROC (a continuous per-class
# score is required regardless of which hard rule is used), so switching
# pred_rule changes ONLY the hard argmax/vote, never the score column.
# ─────────────────────────────────────────────────────────────────────────
PRED_RULES = ("nearest_center", "pairwise_vote")


def _pred_nearest_center(x_obs, means_sub):
    """K-way argmin of squared distance to each class centre (restricted to
    the observed views). two_stage's original rule."""
    distances = np.sum((means_sub - x_obs) ** 2, axis=1)
    return int(np.argmin(distances))


def _pred_pairwise_vote(x_obs, means_sub):
    """One-vs-one pairwise linear-discriminant vote. VERBATIM port of
    adaptive.pred_linear_cla (transposed to match this
    file's (nclasses, nviews) centre layout): for each ordered pair the sign
    of <x - midpoint(mu_i,mu_j), mu_i - mu_j> casts a vote, then argmax over
    per-class vote counts."""
    mean_tr = means_sub.T  # (v, nc)
    diff_mean_sq = mean_tr[:, :, None] - mean_tr[:, None, :]          # (v, nc, nc)
    pairwise_mean_avg = 0.5 * (mean_tr[:, :, None] + mean_tr[:, None, :])
    inner_prod = np.sum(
        (x_obs[:, None, None] - pairwise_mean_avg) * diff_mean_sq, axis=0
    ) > 0  # bool (nc, nc)
    np.fill_diagonal(inner_prod, False)
    return int(np.argmax(np.sum(inner_prod, axis=1)))


def _class_posterior_scores(x_obs, means_sub):
    """softmax(-0.5 * ||x_obs - mu_k[obs]||^2) over classes k -- the exact
    class posterior under equal priors and unit shared variance restricted
    to the observed views. Shared by BOTH pred_rules for AUROC. Same
    convention as adaptive.class_posterior_scores. Returns
    (nclasses,), sums to 1."""
    d2 = np.sum((means_sub - x_obs) ** 2, axis=1)
    logits = -0.5 * d2
    logits = logits - logits.max()  # numerical stability
    p = np.exp(logits)
    return p / p.sum()


def predict_single_combination_multiclass(
    x_sample, centers, combo, return_score=False, pred_rule="nearest_center",
):
    """K-class prediction over the observed (combo) views.

    pred_rule: "nearest_center" (default; Two-stage's K-way argmin)
        or "pairwise_vote" (Adaptive's one-vs-one linear-discriminant
        vote). See the section comment above for when they differ.
    return_score: if True, also returns the (nclasses,) softmax posterior
        score vector (same for both rules -- used only by AUROC).
    """
    mask = np.zeros(len(x_sample), dtype=bool)
    mask[np.array(combo) - 1] = True
    x_obs = x_sample[mask]
    means_sub = centers[:, mask]

    if pred_rule == "nearest_center":
        pred = _pred_nearest_center(x_obs, means_sub)
    elif pred_rule == "pairwise_vote":
        pred = _pred_pairwise_vote(x_obs, means_sub)
    else:
        raise ValueError(f"pred_rule must be one of {PRED_RULES}, got {pred_rule!r}")

    if return_score:
        return pred, _class_posterior_scores(x_obs, means_sub)
    return pred


# ─────────────────────────────────────────────────────────────────────────
# Inference phase: multiclass LP column generation, then a policy that is
# physically sampled and scored -- mirrors adaptive.py's
# run_inference_phase.
# ─────────────────────────────────────────────────────────────────────────
def run_stage2_lp_inference_multiclass(
    X_inference, Y_inference, learned_centers, costs,
    inference_budget, rng, label_map=None, pred_rule="nearest_center",
):
    """Shared "solve + physically sample" routine for both the synthetic
    (Hungarian-matched) and real-data (identity-mapped) callers below --
    the only difference between them is how `label_map` is built.

    label_map: raw predicted cluster index -> true label. None = identity
    (the correct default for real data / any supervised-init caller --
    see initialize_centers_multiclass's docstring for why no Hungarian
    matching is needed there).
    pred_rule: "nearest_center" (default) or "pairwise_vote" -- see
    predict_single_combination_multiclass. Should match whatever rule
    Stage-2 training used.

    Returns
    -------
    dict with keys: masks, probabilities, inference_accuracy,
    inference_error, inference_f1, inference_auroc, actual_cost,
    num_masks_inference, n_budget_fallbacks.
    """
    n = len(X_inference)
    nclasses = learned_centers.shape[0]
    if label_map is None:
        label_map = {k: k for k in range(nclasses)}

    with tick("t_inference_solve"):
        masks, probs = solve_lp_policy_colgen_multiclass(
            est_means=learned_centers, costs=costs,
            inference_budget=inference_budget, n_inference=n,
        )
    combo_costs = np.array([float(np.sum(costs[m])) for m in masks])
    probs = np.clip(probs, 0, 1)
    probs = probs / probs.sum() if probs.sum() > 0 else np.ones(len(masks)) / len(masks)

    # fallback_subset = np.zeros(len(costs), dtype=bool)
    # fallback_subset[0] = True
    # fallback_cost = float(costs[0])
    cheapest_idx = int(np.argmin(costs))
    fallback_subset = np.zeros(len(costs), dtype=bool)
    fallback_subset[cheapest_idx] = True
    fallback_cost = float(costs[cheapest_idx])

    remaining_budget = inference_budget
    actual_cost = 0.0
    idxs = np.arange(len(masks))

    y_pred = np.zeros(n, dtype=int)
    score_mat = np.zeros((n, nclasses))  # columns indexed by RAW cluster index for now
    correct = 0
    n_fallbacks = 0

    with tick("t_inference_sample"):
        for i in range(n):
            sel = rng.choice(idxs, p=probs)
            subset = masks[sel]
            cost = combo_costs[sel]

            if remaining_budget - cost < 0:
                subset = fallback_subset
                cost = fallback_cost
                n_fallbacks += 1

            remaining_budget -= cost
            actual_cost += cost

            combo = tuple(j + 1 for j, flag in enumerate(subset) if flag)
            raw_pred, score = predict_single_combination_multiclass(
                X_inference[i], learned_centers, combo, return_score=True,
                pred_rule=pred_rule,
            )
            matched_pred = label_map.get(raw_pred, raw_pred)
            y_pred[i] = matched_pred
            score_mat[i] = score
            correct += int(matched_pred == Y_inference[i])

    if n_fallbacks:
        # Not an error -- the policy is budget-feasible only in expectation,
        # so a run of expensive draws can exhaust it early. But it means the
        # tail of the test set was classified on the cheapest view alone,
        # which is a real and otherwise-unrecorded reason for a low
        # inference_accuracy at a budget fraction that looks generous.
        _log.info("inference budget exhausted on %d/%d rounds (%.1f%%) -- those "
                  "rounds fell back to the cheapest view alone", n_fallbacks, n,
                  100.0 * n_fallbacks / max(1, n))

    # Permute score-matrix columns from raw-cluster-index order to
    # true-label order (identity in the real-data case, so this is a
    # no-op there) so macro-OVR AUROC's per-class columns line up with
    # Y_inference's actual labels.
    if any(label_map.get(k, k) != k for k in range(nclasses)):
        permuted = np.zeros_like(score_mat)
        for k in range(nclasses):
            permuted[:, label_map.get(k, k)] = score_mat[:, k]
        score_mat = permuted

    inference_accuracy = correct / n if n > 0 else 0.0
    inference_f1 = _macro_f1(Y_inference, y_pred)
    inference_auroc = _macro_ovr_auroc(np.asarray(Y_inference), score_mat, nclasses)

    _log.debug("inference: %d masks, accuracy=%.4f, spent=%.4f/%.4f, fallbacks=%d",
               len(masks), inference_accuracy, actual_cost, inference_budget,
               n_fallbacks)

    return {
        "masks": masks,
        "probabilities": probs,
        "inference_accuracy": inference_accuracy,
        "inference_error": float(1 - inference_accuracy),
        "inference_f1": inference_f1,
        "inference_auroc": inference_auroc,
        "actual_cost": actual_cost,
        "num_masks_inference": len(masks),
        "n_budget_fallbacks": n_fallbacks,
    }


def run_inference_lp_colgen_multiclass(
    X_inference, Y_inference, learned_centers, true_means,
    per_view_costs, inference_budget, rng, pred_rule="nearest_center",
):
    """SYNTHETIC-data counterpart -- Hungarian-matches learned_centers to
    the known generative true_means via match_cluster_labels (imported,
    UNCHANGED -- it already builds a general K x K cost matrix), same
    role as two_stage_asymmetric.run_inference_lp_colgen."""
    label_map = match_cluster_labels(learned_centers, true_means)
    return run_stage2_lp_inference_multiclass(
        X_inference=X_inference, Y_inference=Y_inference,
        learned_centers=learned_centers, costs=per_view_costs,
        inference_budget=inference_budget, rng=rng, label_map=label_map,
        pred_rule=pred_rule,
    )


def run_inference_lp_dataset_colgen_multiclass(
    X_inference, Y_inference, learned_centers, costs,
    inference_budget, rng, label_map=None, pred_rule="nearest_center",
):
    """REAL-data counterpart -- identity label_map by default (no
    ground-truth generative means available; not needed anyway since
    initialize_centers_multiclass is supervised -- see
    two_stage_multiclass_runner.py's module docstring)."""
    return run_stage2_lp_inference_multiclass(
        X_inference=X_inference, Y_inference=Y_inference,
        learned_centers=learned_centers, costs=costs,
        inference_budget=inference_budget, rng=rng, label_map=label_map,
        pred_rule=pred_rule,
    )
