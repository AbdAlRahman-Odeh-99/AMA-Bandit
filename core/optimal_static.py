# -*- coding: utf-8 -*-
"""
core/optimal_static.py

Optimal STATIC (fixed-distribution) acquisition policy over view
combinations, computed from the KNOWN generative means -- the oracle
benchmark for the Two-stage acquisition policy, generalized to K classes
and to this codebase's
heterogeneous per-view cost convention.

    max_D  sum_c D(c) * acc(c)
    s.t.   sum_c D(c) * cost(c) <= budget_per_sample
           sum_c D(c) = 1,  D(c) >= 0

acc(c) is the EMPIRICAL accuracy of nearest-centroid prediction using the
TRUE class means restricted to combination c, measured on the same rounds
the comparator is being compared against. Using empirical accuracy (rather
than core.lp_colgen.multiclass_reward's Bhattacharyya proxy) is deliberate:
the number this benchmark is compared to -- run_alg_multiclass's
`error_rate` -- is an empirical 0/1 error rate, so the benchmark has to be
on the same scale to be interpretable. multiclass_reward is a monotone
proxy used for PRICING inside column generation, and its values are not
accuracies.

=== SYNTHETIC DATA ONLY ===
This needs the true generative means, so it is only defined for the
synthetic GMM datasets (core.datasets.SYNTHETIC_GENERATORS). There is no
meaningful "true mean" for ckd/mnist/etc., and substituting the sample mean
would silently turn the benchmark into a plug-in estimate that the
algorithm can (and on a finite sample sometimes does) beat -- which would
make it useless as a bound. Callers must gate on the dataset name;
`synthetic_true_means` raises for anything else.

=== WHAT THIS IS AND IS NOT A BOUND ON ===
IS: a lower bound on the error of any NON-ADAPTIVE policy -- i.e. any
policy that draws the view subset from a fixed distribution that does not
depend on the sample x_t. two_stage's Stage 2 is exactly such a policy
(run_alg_multiclass's expert weights never look at x[t]), so this is the
right comparator for `error_rate` / `inference_error` there.

IS NOT: a bound for ADAPTIVE acquisition. A policy that chooses which
paid views to buy AFTER seeing the free view can beat any fixed
distribution, so Adaptive per-sample acquisition in this repo may
legitimately fall BELOW this number. Do not report it as a floor for them.

=== THE THREE COMPARATORS ARE NOT NESTED ===
The runner reports this benchmark at three (budget, horizon, row-set)
settings: the Stage-2 training rounds, the inference rounds, and the whole
horizon. They are measured on DIFFERENT sample sets, so they are not
ordered relative to each other -- opt_static_error_full can come out very
slightly BELOW or ABOVE the Stage-2 figure purely from which rows each
averages over, even though the whole-horizon budget is weakly larger.
Compare each only to its own algorithm-side number:
    opt_static_error_stage2    <-> run_alg_multiclass's error_rate
    opt_static_error_inference <-> inference_error
    opt_static_error_full      <-> a gamma-independent reference line
Only within a fixed row set is the error monotone in the budget.

Two further relaxations, both of which only make the benchmark more
permissive (so the direction of the bound is safe):
  - The hard running budget is relaxed to an EXPECTED per-round budget
    (b_ub = budget_per_sample), as in the standard BwK LP benchmark. The
    real algorithms face a hard budget with an absorbing free-view fallback
    once it is exhausted.
  - acc(c) is measured on the same samples the LP optimizes over, so it is
    an in-sample optimum with no estimation error whatsoever.
"""

from __future__ import annotations

import numpy as np
import scipy.optimize as opt

from core.datasets import (
    MULTICLASS_SYNTHETIC_DATASETS,
    SYNTHETIC_DATASETS,
    SYNTHETIC_MEAN_SCALE,
    SYNTHETIC_N_CLASSES,
    SYNTHETIC_SEED,
)

# Cap on the (n_samples x n_combos) boolean correctness matrix built by
# combo_correctness_matrix. 1000 samples x 2^15 combos is only ~32 MB, but
# the matrix is O(N * 2^(nviews-1)) bytes and blows up fast in BOTH
# directions -- at nviews=20 with N=10000 it is already 5 GB.
MAX_CORRECTNESS_MATRIX_BYTES = 2 * 1024 ** 3


def synthetic_true_means(
    dataset_name,
    synthetic_n_views,
    n_views_used=None,
    synthetic_seed=SYNTHETIC_SEED,
    mean_scale=SYNTHETIC_MEAN_SCALE,
    n_classes=SYNTHETIC_N_CLASSES,
):
    """Recover the EXACT (n_classes, n_views_used) generative means that
    core.datasets' synthetic generator drew for this configuration.

    This is not an approximation. load_binary_afa_dataset seeds a fresh
    `np.random.default_rng(synthetic_seed)` and the generator's very first
    consumption of it is `means = rng.random(size=(K, n_views)) * mean_scale`,
    so replaying that one draw here reproduces the means bit-for-bit
    (verified by direct comparison). make_blobs then assigns label k to
    centers[k], so row k is class k's mean and NO Hungarian matching is
    needed -- same reasoning as initialize_centers_multiclass's supervised
    init (see two_stage_multiclass.py's module docstring).

    synthetic_n_views: the value passed to the GENERATOR (--n-views). The
        draw MUST be made at this full width and truncated afterwards --
        numpy fills row-major, so rng.random((K, 16))[:, :10] is NOT
        rng.random((K, 10)). Getting this backwards yields plausible-looking
        but entirely wrong means, and the resulting "bound" would sit at the
        wrong level rather than visibly failing.
    n_views_used: post-truncation view count actually present in X (default:
        same as synthetic_n_views). load_dataset_as_numpy applies
        `features[:, :max_modalities]` to synthetic datasets too -- its
        docstring says max_modalities has "no effect" for synthetic, which is
        true only of the GENERATION step; the truncation below it is
        unconditional. So --n-views 16 --max-modalities 10 yields 10 views.
        Pass X.shape[1].

    Raises for any non-synthetic dataset -- see the module docstring.
    """
    if dataset_name not in SYNTHETIC_DATASETS:
        msg = (
            f"optimal-static benchmark needs the TRUE generative means, which only "
            f"exist for the synthetic datasets {SYNTHETIC_DATASETS}; got "
            f"{dataset_name!r}. There is no ground-truth mean for a real dataset, "
            f"and using the sample mean instead would produce a 'bound' the "
            f"algorithm can beat."
        )
        raise ValueError(msg)

    gen_views = int(synthetic_n_views)
    used = gen_views if n_views_used is None else int(n_views_used)
    if not (1 <= used <= gen_views):
        msg = f"n_views_used must be in [1, {gen_views}], got {used}"
        raise ValueError(msg)

    k = int(n_classes) if dataset_name in MULTICLASS_SYNTHETIC_DATASETS else 2
    rng = np.random.default_rng(synthetic_seed)

    if dataset_name == "synthetic_symmetric":
        # generate_synthetic_symmetric draws a single half-gap vector and
        # mirrors it: means = [+half_gap, -half_gap].
        half_gap = rng.random(size=(gen_views,)) * mean_scale
        means = np.stack([half_gap, -half_gap], axis=0)
    else:
        # asymmetric / multiclass: i.i.d. Uniform(0, mean_scale) per (class, view)
        means = rng.random(size=(k, gen_views)) * mean_scale

    return np.asarray(means[:, :used], dtype=np.float64)


def combo_correctness_matrix(X, Y, true_means, view_combinations, chunk=2048):
    """(n_samples, n_combos) BOOLEAN matrix: entry [i, j] is True iff
    nearest-centroid prediction using `true_means` restricted to
    view_combinations[j] classifies sample i correctly.

    Computed once per dataset and then sliced, because the accuracies the LP
    needs differ only in WHICH ROWS are averaged (Stage-2 rounds vs. inference
    rounds vs. all rounds) -- recomputing per sweep point would repeat the
    same 2^(nviews-1) x n_samples work hundreds of times.

    Vectorized rather than looped over combos: with
    D[i, k, v] = (X[i, v] - mu[k, v])**2 and a (n_combos, nviews) 0/1
    membership matrix Mb, the restricted distance for every (sample, class,
    combo) triple is one matmul D_flat @ Mb.T. Chunked over combos to bound
    peak memory.
    """
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y).astype(np.int64)
    true_means = np.asarray(true_means, dtype=np.float64)

    n, nviews = X.shape
    nclasses = true_means.shape[0]
    if true_means.shape[1] != nviews:
        msg = (
            f"true_means has {true_means.shape[1]} views but X has {nviews}. "
            f"Pass X.shape[1] to synthetic_true_means -- load_dataset_as_numpy "
            f"truncates synthetic features by max_modalities too."
        )
        raise ValueError(msg)

    n_combos = len(view_combinations)
    nbytes = n * n_combos
    if nbytes > MAX_CORRECTNESS_MATRIX_BYTES:
        msg = (
            f"correctness matrix would be {nbytes / 1024 ** 3:.1f} GB "
            f"({n} samples x {n_combos} combos). Reduce nviews or subsample "
            f"(--max-samples); the benchmark is O(N * 2^(nviews-1)) bytes."
        )
        raise MemoryError(msg)

    # (n, nclasses, nviews) per-view squared distances -- small (nviews is
    # bounded by MAX_RECOMMENDED_MODALITIES anyway).
    d_per_view = (X[:, None, :] - true_means[None, :, :]) ** 2
    d_flat = d_per_view.reshape(n * nclasses, nviews)

    membership = np.zeros((n_combos, nviews), dtype=np.float64)
    for j, combo in enumerate(view_combinations):
        membership[j, np.asarray(combo) - 1] = 1.0

    correct = np.empty((n, n_combos), dtype=bool)
    for s in range(0, n_combos, chunk):
        blk = membership[s:s + chunk]                       # (b, nviews)
        dist = d_flat @ blk.T                               # (n*nclasses, b)
        dist = dist.reshape(n, nclasses, blk.shape[0])
        pred = dist.argmin(axis=1)                          # (n, b)
        correct[:, s:s + chunk] = pred == Y[:, None]

    return correct


def solve_optimal_static(accuracies, combo_costs, budget_per_sample):
    """Solve the static-policy LP and return (error, weights).

    accuracies / combo_costs: parallel float arrays over the same combo
    ordering. budget_per_sample: the EXPECTED per-round cost cap.

    Always feasible: view_combinations[0] is the free view alone, whose cost
    is 0 under this codebase's normalized cost convention, so D = e_0 is a
    feasible point for any nonnegative budget.

    The LP has one inequality plus one simplex equality, so an optimal
    VERTEX is supported on at most 2 combinations -- the classic "optimal
    static policy mixes at most two arms" result. Useful sanity check when
    reading the returned weights.
    """
    accuracies = np.asarray(accuracies, dtype=np.float64)
    combo_costs = np.asarray(combo_costs, dtype=np.float64)
    n = len(accuracies)

    res = opt.linprog(
        -accuracies,
        A_ub=combo_costs.reshape(1, -1),
        b_ub=np.array([float(budget_per_sample)]),
        A_eq=np.ones((1, n)),
        b_eq=np.array([1.0]),
        bounds=(0, 1),
        method="highs",
    )

    if not res.success:
        # Should be unreachable (the free view is always feasible); fall back
        # to it explicitly rather than returning a silently wrong optimum.
        w = np.zeros(n)
        w[int(np.argmin(combo_costs))] = 1.0
        return float(1.0 - accuracies @ w), w

    w = np.clip(res.x, 0.0, None)
    w = w / w.sum() if w.sum() > 0 else w
    return float(1.0 - accuracies @ w), w


def optimal_static_error(correct, row_idx, combo_costs, budget, n_rounds):
    """Convenience wrapper: average `correct` over the given rows to get
    per-combo accuracies, then solve the LP at budget / n_rounds.

    correct: (n_samples, n_combos) bool from combo_correctness_matrix.
    row_idx: integer indices of the rounds this comparator covers.
    budget / n_rounds: total budget available over those rounds, and how
        many there are -- their ratio is the per-round cap.

    Returns dict with the error, the support size, the mixture's expected
    cost, and the support itself (`idx` / `weights`, sorted by descending
    weight) so callers can report WHICH combinations the optimal policy
    actually buys -- usually the most informative output here, since it says
    what a bandit over the same expert set ought to be converging to.
    """
    row_idx = np.asarray(row_idx, dtype=np.int64)
    if len(row_idx) == 0 or n_rounds <= 0:
        return {"error": float("nan"), "support": 0, "expected_cost": float("nan"),
                "budget_per_round": float("nan"), "idx": np.array([], dtype=int),
                "weights": np.array([])}

    acc = correct[row_idx].mean(axis=0)
    per_round = max(0.0, float(budget) / float(n_rounds))
    err, w = solve_optimal_static(acc, combo_costs, per_round)

    nz = np.flatnonzero(w > 1e-9)
    nz = nz[np.argsort(-w[nz])]
    return {
        "error": err,
        "support": int(len(nz)),
        "expected_cost": float(np.asarray(combo_costs)[nz] @ w[nz]) if len(nz) else 0.0,
        "budget_per_round": per_round,
        "idx": nz,
        "weights": w[nz],
    }
