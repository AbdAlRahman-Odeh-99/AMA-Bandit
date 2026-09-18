"""
Driver for adaptive.py:

  - Data via core.datasets.load_dataset_as_numpy -- same contract as every
    other runner. Works on ANY registered dataset: for the 5 real binary
    datasets and synthetic_asymmetric/synthetic_symmetric, nclasses is
    INFERRED from the labels (= 2, where the multiclass algorithm reduces
    to nearest-mean binary classification); for "synthetic",
    the synthetic_n_classes knob controls how many classes are GENERATED.
  - Costs: fixed per-dataset heterogeneous costs
    (generate_modality_costs_heterogeneous), view 0 free, normalized so
    sum(costs) == 1 -- identical convention to the other runners, NOT the
    notebook's per-trial cost redraw.
  - Split: selectable with split_mode. Use 80-20 for proposed-method
    comparisons, or 60-20-20 when comparing with EDDI/DIME.
    Each seed gets its own split, center init, and inference-phase sampling rng -- this
    REPLACES the notebook's buggy trial loop, where generate_data and
    simulation were re-seeded with the SAME args.seed every trial so all
    trials saw identical data (only costs varied).
  - est_means_init: nclasses distinct rows of X_train, drawn once per seed
    and reused across budget fractions (repo convention, generalizing the
    binary runners' 2-row init; replaces the notebook's rng.normal init).
  - Budget: total_budget = budget_fraction * n_total, train/inference
    split auto-derived as n_train/n_total -- identical to the other
    runners.
  - Inference phase: core.lp_colgen.solve_lp_policy_colgen_multiclass
    (multiclass column generation) -- tractable at any nviews, same as the
    binary colgen path.

`feedback` ("full" | "bandit") selects the training-phase update rule --
see adaptive.run_training_phase.

`acquisition` selects the training-phase policy orthogonally to `feedback`.
The supported policies are lp_chain, lp_full_opt, ucb_argmax, and hedge.
`reward_update` ("subsets" | "selected") controls whether a round scores
every arm contained in the played subset (counterfactual replay) or only
the played arm. It is live for learned policies and ignored by lp_full_opt. See
adaptive.py's ACQUISITION MODES docstring section. Results dicts have the
The per-fraction results use the common proposed-method schema (so
run_proposed_methods._normalize_frac_keyed_results applies unchanged);
feedback / n_classes are experiment-level settings the caller records.

=== Observability (see core/logging_utils.py) ===
Same treatment as the two_stage runner: logging instead of print, a
per-cell `guard` so one failure does not end the sweep, append-as-you-go
row checkpointing to results/{run_id}.rows.jsonl, a Progress heartbeat with
an ETA, and the fine timing decomposition alongside the untouched
train/inference/seed timings.

One structural note specific to THIS runner: its results object is a
dict-of-lists keyed by budget fraction, and every list must stay the same
length or the per-seed alignment breaks. So a failed cell appends a full
row of NaNs rather than being skipped -- CELL_FIELDS below is the single
list that keeps the success and failure paths in step.
"""

from __future__ import annotations

import sys
import time

import numpy as np
import pandas as pd

from core.datasets import (
    ALL_DATASETS,
    DEFAULT_SAMPLING_MODE,
    DEFAULT_IMAGE_POOL_SIDE,
    MULTICLASS_SYNTHETIC_DATASETS,
    SYNTHETIC_CLUSTER_STD,
    SYNTHETIC_DATASETS,
    SYNTHETIC_MEAN_SCALE,
    SYNTHETIC_N_CLASSES,
    SYNTHETIC_N_SAMPLES,
    SYNTHETIC_N_VIEWS,
    SYNTHETIC_SEED,
    SPLIT_MODES,
    SAMPLING_MODES,
    generate_modality_costs_heterogeneous,
    load_dataset_as_numpy,
    split_by_mode,
)
from core.excel_utils import serialize_selected_subsets, style_and_save
from core.logging_utils import (
    TIMING_COLUMNS,
    Progress,
    cell_timers,
    get_logger,
    get_run,
    guard,
    setup_run,
    tick,
    timing_row,
)
from core.lp_colgen import solve_lp_policy_colgen_multiclass
from core.optimal_static import synthetic_true_means

from adaptive.adaptive import (
    run_training_phase,
    run_inference_phase,
)
# Single definition of "does --reward-update do anything for these flags",
# replacing the three inline copies this file used to carry.
from core.acquisition_policies import (
    ACQUISITION_MODES,
    ORACLE_ACQUISITION_MODES,
    REWARD_UPDATE_SCOPES,
    UCB_BOUNDS,
    uses_empirical_arm_rewards as _uses_empirical_arm_rewards,
    validate_ucb_bound,
)
from core.training_state import (
    find_training_states,
    load_training_state,
    restored_rng,
    save_training_state,
    state_directory,
)

log = get_logger("afa.adaptive.runner")


#: Every per-cell series in the results dict, in one place.
#:
#: The success path fills these from the phase results; the failure path
#: fills them with NaN. Both append EXACTLY these keys, which is what keeps
#: results[frac]["train_reward"][i] and results[frac]["seed_time_sec"][i]
#: referring to the same seed after a cell has failed. Adding a metric means
#: adding it here and nowhere else.
CELL_FIELDS = (
    "train_reward", "train_f1", "train_auroc",
    "inference_reward", "inference_f1", "inference_auroc",
    "total_reward",
    "train_spent", "inference_spent", "num_masks_inference",
    "train_time_sec", "inference_time_sec",
    "split_mode", "n_train", "n_validation", "n_inference",
    "train_budget", "inference_budget",
    "n_arms", "avg_views_train", "selected_subsets",
    "status", "error_msg",
    "training_state_path",
) + TIMING_COLUMNS

#: seed_time_sec is appended once per SEED (after its budget-fraction loop),
#: not once per cell, so it is tracked separately from CELL_FIELDS.
ALL_FIELDS = CELL_FIELDS + ("seed_time_sec",)


def _ok_mask(r):
    """Boolean mask of the cells that succeeded, for the summary sheet."""
    status = r.get("status")
    if not status:
        return np.ones(len(r["train_reward"]), dtype=bool)
    return np.array([s == "ok" for s in status], dtype=bool)


def _agg(r, key, fn=np.mean):
    """Aggregate one metric over the SUCCESSFUL cells only.

    Guards the empty case explicitly: np.mean of an empty slice returns NaN
    with a RuntimeWarning, and a budget fraction where every seed failed is
    exactly the situation where a stray warning in a 10-hour log is least
    likely to be noticed. NaN is the right answer; the warning is not.
    """
    vals = np.asarray(r.get(key, []), dtype=float)
    mask = _ok_mask(r)
    if len(vals) == len(mask):
        vals = vals[mask]
    vals = vals[np.isfinite(vals)] if fn in (np.mean, np.std) else vals
    if vals.size == 0:
        return float("nan")
    return float(fn(vals))


def run_experiment(
    dataset_name,
    feedback="full",
    acquisition="lp_chain",
    ucb_bound="vc",
    reward_update="subsets",
    max_modalities=None,
    seeds=(42, 43, 44, 45, 46, 47, 48, 49, 50, 51),
    budget_fractions=(0.1, 0.3, 0.5, 0.7, 0.9),
    data_path=None,
    max_samples=None,
    sampling=DEFAULT_SAMPLING_MODE,
    synthetic_n_samples=SYNTHETIC_N_SAMPLES,
    synthetic_n_views=SYNTHETIC_N_VIEWS,
    synthetic_seed=SYNTHETIC_SEED,
    synthetic_mean_scale=SYNTHETIC_MEAN_SCALE,
    synthetic_cluster_std=SYNTHETIC_CLUSTER_STD,
    synthetic_n_classes=SYNTHETIC_N_CLASSES,
    alpha_ucb=1.0,
    lr=1e-2,
    step_size=None,
    lambda_max=10.0,
    split_mode="80-20",
    run_inference=True,
    state_dir=None,
    image_pool_side=DEFAULT_IMAGE_POOL_SIDE,
    image_data_home=None,
):
    """
    Run one adaptive multiclass experiment. The same
    return shape: {budget_fraction: {metric_name: [per-seed values]}}.

    feedback: "full" (y_true revealed every round) or "bandit" (one-bit
        reward only) -- selects the training-phase update rule.
    acquisition: "lp_chain" (default; per-round LP over a nested greedy chain),
        "lp_full_opt" (oracle full-action LP with exact true-means arm
        values, LP solved once before the loop, each round a draw from the
        frozen distribution). "lp_full_opt" needs the generative means and
        so is SYNTHETIC-ONLY; this function recovers them itself via
        core.optimal_static.synthetic_true_means and raises for any real
        dataset. It is also the one mode that ignores synthetic_seed drift:
        the means are re-derived from synthetic_seed / synthetic_mean_scale
        / synthetic_n_classes, so those must match what generated X_full --
        they do here by construction, since the same values feed
        load_dataset_as_numpy a few lines above.
    reward_update: "subsets" or "selected". Used by learned policies and
        ignored by lp_full_opt.
    synthetic_n_classes: "synthetic" only. For every other
        dataset, nclasses is inferred from the labels.
    alpha_ucb / lr: training-phase exploration bonus scale and (bandit-only)
        complementary-update learning rate -- notebook defaults.
    run_inference: if False, ONLY the training phase (adaptive online
        training) executes; the inference-phase column-generation LP +
        physical-sampling inference is skipped entirely. The train/inference
        split and budget pools are unchanged (the inference budget is just
        never spent); all inference_* metrics are set to NaN, inference
        spent/time to 0.0, num_masks_inference to NaN, and total_reward
        becomes the TRAIN-ONLY figure (== ph1 train_reward, already the
        accuracy over all n_train rounds) rather than the sample-weighted
        train+inference figure -- so
        a run_inference=False total_reward is NOT directly comparable to a
        full run's. Use it when you only care about training dynamics.
    image_data_home: mnist/fashion_mnist only -- where fetch_openml caches
        its download. See core.datasets.DEFAULT_OPENML_DATA_HOME's comment:
        defaults to a relative "data/openml_cache" folder rather than
        fetch_openml's own ~/scikit_learn_data, since $HOME's quota is tiny
        (and often already exhausted) on clusters like NCI Gadi.
    """
    if dataset_name not in ALL_DATASETS:
        msg = f"dataset_name must be one of {ALL_DATASETS}, got {dataset_name!r}"
        raise ValueError(msg)
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

    if (feedback == "bandit" and reward_update == "subsets"and uses_empirical_arm_rewards):
        raise ValueError(
            "feedback='bandit' with reward_update='subsets' is incoherent: "
            "counterfactual replay reads y_true, which bandit feedback does "
            "not reveal. Use reward_update='selected' with feedback='bandit', "
            "or feedback='full' with reward_update='subsets'."
        )

    if acquisition in ORACLE_ACQUISITION_MODES and dataset_name not in SYNTHETIC_DATASETS:
        raise ValueError(
            f"acquisition={acquisition!r} needs the TRUE generative means, which "
            f"exist only for the synthetic datasets {SYNTHETIC_DATASETS}; got "
            f"{dataset_name!r}. Substituting a sample mean would turn the oracle "
            f"into a plug-in estimate the learning modes can beat, which is worse "
            f"than no ceiling at all.")

    run = get_run()
    state_dir = state_directory(run=run, state_dir=state_dir, method="adaptive")

    with tick("t_data_load"):
        X_full, Y_full, feature_names = load_dataset_as_numpy(
            dataset_name, max_modalities=max_modalities, data_path=data_path,
            max_samples=max_samples, sampling=sampling,
            synthetic_n_samples=synthetic_n_samples,
            synthetic_n_views=synthetic_n_views, synthetic_seed=synthetic_seed,
            synthetic_mean_scale=synthetic_mean_scale,
            synthetic_cluster_std=synthetic_cluster_std,
            synthetic_n_classes=synthetic_n_classes,
            image_pool_side=image_pool_side,
            image_data_home=image_data_home,
        )
    n_samples, nviews = X_full.shape
    nclasses = int(Y_full.max()) + 1

    # ORACLE ACQUISITION SETUP
    true_means = None
    if acquisition in ORACLE_ACQUISITION_MODES:
        true_means = synthetic_true_means(
            dataset_name,
            synthetic_n_views=synthetic_n_views,
            n_views_used=nviews,
            synthetic_seed=synthetic_seed,
            mean_scale=synthetic_mean_scale,
            n_classes=synthetic_n_classes,
        )
        log.info("[oracle] recovered true means for acquisition=%s: shape %s",
                 acquisition, true_means.shape)

    _has_ru = _uses_empirical_arm_rewards(acquisition)
    ru_tag = f"/{reward_update}" if _has_ru else ""
    log.info("%s: %d samples, %d views, %d classes, feedback=%s, acquisition=%s%s "
             "(free: '%s', %d paid)", dataset_name, n_samples, nviews, nclasses,
             feedback, acquisition, ru_tag, feature_names[0], nviews - 1)
    class_counts = np.bincount(Y_full.astype(int), minlength=nclasses)
    log.info("Class sample counts (sampling=%s, max_samples=%s): %s",
             sampling, max_samples,
             {i: int(count) for i, count in enumerate(class_counts)})

    # Fixed per-dataset costs
    raw_feature_costs = np.array(
        generate_modality_costs_heterogeneous(n_features=nviews, dataset_name=dataset_name),
        dtype=np.float64,
    )
    costs = raw_feature_costs / raw_feature_costs.sum()
    paid_costs = costs[1:]
    log.info("feature_costs[0] (free) = %s, paid costs (normalized, sum(costs)==1): "
             "min=%.4f, max=%.4f, mean=%.4f, sum=%.4f",
             costs[0], paid_costs.min(), paid_costs.max(), paid_costs.mean(),
             paid_costs.sum())

    results = {frac: {k: [] for k in ALL_FIELDS} for frac in budget_fractions}

    total_cells = len(seeds) * len(budget_fractions)
    progress = Progress(total_cells, label="cells", logger=log)
    log.info("sweep: %d seeds x %d budget fractions = %d cells",
             len(seeds), len(budget_fractions), total_cells)

    for seed in seeds:
        seed_start = time.time()
        log.info("=" * 60)
        log.info("=== SEED %s (%s, %s feedback, %s acquisition) ===",
                 seed, dataset_name, feedback, acquisition)
        log.info("=" * 60)

        train_idx, val_idx, test_idx = split_by_mode(
            n_samples, split_mode=split_mode, seed=seed
        )
        X_train, Y_train = X_full[train_idx], Y_full[train_idx]
        X_inf, Y_inf = X_full[test_idx], Y_full[test_idx]
        n_train, n_inf = len(train_idx), len(test_idx)
        n_total = n_train + n_inf
        log.info("Split %s: train=%d, validation=%d, test=%d",
                 split_mode, n_train, len(val_idx), n_inf)

        rng = np.random.default_rng(seed=seed)  # warmup preds, inference-phase sampling, center init

        # nclasses distinct rows of X_train as initial centers -- repo
        # convention (generalizes the binary runners' 2-row init). Drawn
        # once per seed, reused across budget fractions.
        #n_seed_rows = min(nclasses, len(X_train))
        #seed_rows = rng.choice(len(X_train), size=n_seed_rows, replace=False)
        #est_means_init = X_train[seed_rows[np.arange(nclasses) % n_seed_rows]].astype(float).copy()
        est_means_init = rng.normal(size=(nclasses, nviews))

        for frac in budget_fractions:
            total_budget = frac * n_total
            train_inference_split = n_train / n_total
            training_budget = train_inference_split * total_budget
            inference_budget = total_budget - training_budget
            log.info("Budget fraction %.2f -> total_budget=%.4f (n_total=%d)",
                     frac, total_budget, n_total)
            log.info("  Train pool: %.4f | Inference pool: %.4f",
                     training_budget, inference_budget)

            cell = {"seed": int(seed), "budget_fraction": float(frac)}
            vals = None

            with cell_timers(inherit=("t_data_load",)) as timers, \
                    guard(cell, logger=log) as outcome:
                train_start = time.time()
                ph1 = run_training_phase(
                    nviews=nviews, nclasses=nclasses, costs=costs, n_train=n_train,
                    training_budget=training_budget, X_train=X_train, Y_train=Y_train,
                    est_means_init=est_means_init, feedback=feedback,
                    alpha_ucb=alpha_ucb, step_size=step_size, lambda_max=lambda_max, lr=lr, rng=rng,
                    acquisition=acquisition, reward_update=reward_update,
                    ucb_bound=ucb_bound,
                    true_means=true_means,
                )
                train_time = time.time() - train_start
                log.info("  [Training] Reward: %.3f | F1: %.3f | AUROC: %.3f "
                         "| Spent: %.4f / %.4f | Time: %.2fs",
                         ph1['train_reward'], ph1['train_f1'], ph1['train_auroc'],
                         ph1['spent'], training_budget, train_time)
                log.info("  [Training] Acquisition %s%s: %s arms "
                         "| avg views/round: %.2f | distinct masks played: %s",
                         acquisition, ru_tag, ph1['n_arms'],
                         ph1['avg_views_acquired'], ph1['n_unique_masks'])

                # Persist the exact post-training/pre-inference boundary for
                # every successful cell, including ordinary full runs.
                training_row = {
                    "train_reward": ph1["train_reward"],
                    "train_f1": ph1["train_f1"],
                    "train_auroc": ph1["train_auroc"],
                    "n_arms": ph1["n_arms"],
                    "avg_views_train": ph1["avg_views_acquired"],
                    "selected_subsets": ph1["selected_subsets"],
                    "train_spent": ph1["spent"],
                    "step_size": ph1["step_size"],
                    "train_time_sec": train_time,
                    "split_mode": split_mode,
                    "n_train": n_train,
                    "n_validation": len(val_idx),
                    "n_inference": n_inf,
                    "train_budget": training_budget,
                    "inference_budget": inference_budget,
                }
                state_path = save_training_state(
                    method="adaptive", cell=cell, run=run, state_dir=state_dir,
                    rng=rng,
                    arrays={
                        "learned_centers": ph1["est_means"],
                        "costs": costs,
                        "X_inference": X_inf,
                        "Y_inference": Y_inf,
                        "train_indices": train_idx,
                        "inference_indices": test_idx,
                    },
                    metadata={
                        "dataset": dataset_name,
                        "dataset_config": {
                            "max_modalities": max_modalities,
                            "data_path": data_path,
                            "max_samples": max_samples,
                            "sampling": sampling,
                            "synthetic_n_samples": synthetic_n_samples,
                            "synthetic_n_views": synthetic_n_views,
                            "synthetic_seed": synthetic_seed,
                            "synthetic_mean_scale": synthetic_mean_scale,
                            "synthetic_cluster_std": synthetic_cluster_std,
                            "synthetic_n_classes": synthetic_n_classes,
                            "image_pool_side": image_pool_side,
                            "image_data_home": image_data_home,
                        },
                        "nclasses": nclasses,
                        "nviews": nviews,
                        "feature_names": list(feature_names),
                        "run_config": {
                            "feedback": feedback,
                            "acquisition": acquisition,
                            "ucb_bound": ucb_bound,
                            "reward_update": reward_update,
                            "alpha_ucb": alpha_ucb,
                            "lr": lr,
                            "step_size": step_size,
                            "lambda_max": lambda_max,
                            "split_mode": split_mode,
                        },
                        "training_row": training_row,
                    },
                )
                log.info("  [Training state] %s", state_path)
                if run_inference:
                    inference_start = time.time()
                    with tick("t_inference_solve"):
                        masks, probs = solve_lp_policy_colgen_multiclass(
                            est_means=ph1["est_means"], costs=costs,
                            inference_budget=inference_budget, n_inference=n_inf,
                        )
                    combo_costs = np.array([np.sum(costs[m]) for m in masks])
                    log.info("  [Inference] Column generation discovered %d subsets "
                             "(power set would have been 2^%d-1 = %d)",
                             len(masks), nviews, 2 ** nviews - 1)
                    with tick("t_inference_sample"):
                        ph2 = run_inference_phase(
                            masks=masks, probs=probs, combo_costs=combo_costs, costs=costs,
                            est_means=ph1["est_means"],
                            inference_budget=inference_budget, X_inf=X_inf, Y_inf=Y_inf, rng=rng,
                        )
                    inference_time = time.time() - inference_start
                    log.info("  [Inference] Reward: %.3f | F1: %.3f | AUROC: %.3f "
                             "| Spent: %.4f / %.4f | Time: %.2fs",
                             ph2['inference_reward'], ph2['inference_f1'],
                             ph2['inference_auroc'], ph2['spent'], inference_budget,
                             inference_time)

                    total_reward = (n_train * ph1["train_reward"] + n_inf * ph2["inference_reward"]) / n_total
                    num_masks = len(masks)
                else:
                    # --skip-inference
                    ph2 = {
                        "inference_reward": float("nan"),
                        "inference_f1": float("nan"),
                        "inference_auroc": float("nan"),
                        "spent": 0.0,
                    }
                    inference_time = 0.0
                    num_masks = np.nan
                    total_reward = ph1["train_reward"]
                    log.info("  [Inference] SKIPPED (--skip-inference); "
                             "train-only total_reward")

                log.info("  [Total]   Reward: %.3f", total_reward)

                vals = {
                    "train_reward": ph1["train_reward"],
                    "train_f1": ph1["train_f1"],
                    "train_auroc": ph1["train_auroc"],
                    "inference_reward": ph2["inference_reward"],
                    "inference_f1": ph2["inference_f1"],
                    "inference_auroc": ph2["inference_auroc"],
                    "total_reward": total_reward,
                    "n_arms": ph1["n_arms"],
                    "avg_views_train": ph1["avg_views_acquired"],
                    "selected_subsets": ph1["selected_subsets"],
                    "train_spent": ph1["spent"],
                    "inference_spent": ph2["spent"],
                    "num_masks_inference": num_masks,
                    "train_time_sec": train_time,
                    "inference_time_sec": inference_time,
                    "split_mode": split_mode,
                    "n_train": n_train,
                    "n_validation": len(val_idx),
                    "n_inference": n_inf,
                    "train_budget": training_budget,
                    "inference_budget": inference_budget,
                    "training_state_path": str(state_path),
                }

            if vals is None:
                # Failed cell: a full-width NaN row, so every list in
                # results[frac] stays the same length and the i-th entry of
                # each still belongs to the i-th seed.
                vals = {k: float("nan") for k in CELL_FIELDS}
                vals["selected_subsets"] = []
                vals.update({"split_mode": split_mode, "n_train": n_train,
                             "n_validation": len(val_idx), "n_inference": n_inf,
                             "train_budget": training_budget,
                             "inference_budget": inference_budget})
            vals["status"] = outcome.status
            vals["error_msg"] = outcome.error
            vals.update(timing_row(timers))

            for k in CELL_FIELDS:
                results[frac][k].append(vals.get(k, float("nan")))

            if run is not None:
                run.emit_row({**cell, **{k: vals.get(k) for k in CELL_FIELDS
                                         if k != "selected_subsets"}})
                if run.trace_rounds:
                    # This method's algorithm module returns only the played
                    # subset per round (no per-round lambda/budget state), so
                    # the trace it can honestly produce is thinner than
                    # two_stage's -- subsets and their costs, nothing invented.
                    for i, subset in enumerate(vals.get("selected_subsets") or []):
                        run.emit_trace({**cell, "round": i, "subset": list(subset),
                                        "n_views": len(subset)})
            progress.step(note=f"seed {seed} bf {frac:g}")

        seed_elapsed = time.time() - seed_start
        log.info("[SEED %s] wall-clock time: %.1fs (%d budget fractions)",
                 seed, seed_elapsed, len(budget_fractions))
        for frac in budget_fractions:
            results[frac]["seed_time_sec"].append(seed_elapsed)

    n_failed = sum(sum(1 for s in results[f]["status"] if s != "ok")
                   for f in budget_fractions)
    if n_failed:
        log.warning("%d/%d cells FAILED and were recorded with status='error' -- see "
                    "the log for tracebacks and the manifest for the list",
                    n_failed, total_cells)

    return results


def run_inference_from_states(source):
    """Run adaptive inference from saved cells without rerunning training."""
    run = get_run()
    paths = find_training_states(source, method="adaptive")
    if not paths:
        raise ValueError(f"no adaptive training states found in {source}")
    rows = []
    progress = Progress(len(paths), label="states", logger=log)
    for path in paths:
        meta, arrays = load_training_state(path, expected_method="adaptive")
        cell = dict(meta["cell"])
        row = None
        with cell_timers() as timers, guard(cell, logger=log) as outcome:
            rng = restored_rng(meta)
            centers = arrays["learned_centers"]
            costs = arrays["costs"]
            X_inf = arrays["X_inference"]
            Y_inf = arrays["Y_inference"]
            inference_budget = float(meta["training_row"]["inference_budget"])
            inference_start = time.time()
            with tick("t_inference_solve"):
                masks, probs = solve_lp_policy_colgen_multiclass(
                    est_means=centers, costs=costs,
                    inference_budget=inference_budget, n_inference=len(Y_inf),
                )
            combo_costs = np.array([np.sum(costs[m]) for m in masks])
            with tick("t_inference_sample"):
                ph2 = run_inference_phase(
                    masks=masks, probs=probs, combo_costs=combo_costs,
                    costs=costs, est_means=centers,
                    inference_budget=inference_budget,
                    X_inf=X_inf, Y_inf=Y_inf, rng=rng,
                )
            row = dict(meta["training_row"])
            n_train = int(row["n_train"])
            n_inf = int(row["n_inference"])
            row.update({
                **cell,
                "inference_reward": ph2["inference_reward"],
                "inference_f1": ph2["inference_f1"],
                "inference_auroc": ph2["inference_auroc"],
                "inference_spent": ph2["spent"],
                "inference_time_sec": time.time() - inference_start,
                "num_masks_inference": len(masks),
                "total_reward": (
                    n_train * row["train_reward"]
                    + n_inf * ph2["inference_reward"]
                ) / max(1, n_train + n_inf),
                "training_state_path": str(path),
                "dataset": meta["dataset"],
                "nclasses": meta["nclasses"],
            })
        if row is None:
            row = {**cell, "training_state_path": str(path)}
        row["status"] = outcome.status
        row["error_msg"] = outcome.error
        row.update(timing_row(timers))
        rows.append(row)
        if run is not None:
            run.emit_row(row)
        progress.step(note=path.name)
    return rows


def save_results_to_excel(results, budget_fractions, dataset_name, feedback,
                          seeds=None, filename=None, acquisition="lp_chain",
                          reward_update="", info_rows=None):
    """Save the adaptive results, plus
    'Feedback' / 'Acquisition' / 'Reward Update' columns on every row
    (experiment-level settings, constant within one file, recorded per row
    so files concatenate cleanly).

    CHANGED, additively: Status / Error and the fine timing block are
    appended to the Detailed sheet, the Summary sheet aggregates over
    successful cells only (see _agg), and a "Run Info" sheet carries the
    provenance manifest when a run context is active.
    """
    if filename is None:
        filename = f"results_gmm_{feedback}_{acquisition}_{dataset_name}.xlsx"

    detailed_rows = []
    for frac in budget_fractions:
        r = results[frac]
        n = len(r["train_reward"])
        row_labels = seeds if seeds is not None and len(seeds) == n else range(n)
        for label, i in zip(row_labels, range(n)):
            row = {
                "Seed": label,
                "Feedback": feedback,
                "Acquisition": acquisition,
                "Reward Update": reward_update,
                "Budget Fraction": frac,
                "Train Reward": r["train_reward"][i],
                "Train F1": r["train_f1"][i],
                "Train AUROC": r["train_auroc"][i],
                "Inference Reward": r["inference_reward"][i],
                "Inference F1": r["inference_f1"][i],
                "Inference AUROC": r["inference_auroc"][i],
                "Total Reward": r["total_reward"][i],
                "Total Error": 1 - r["total_reward"][i],
                "Train Spent": r["train_spent"][i],
                "Inference Spent": r["inference_spent"][i],
                "Num Masks Inference": r["num_masks_inference"][i],
                "Train Time (s)": r["train_time_sec"][i],
                "Inference Time (s)": r["inference_time_sec"][i],
                "Seed Time (s)": r["seed_time_sec"][i],
                "Train Samples": r["n_train"][i],
                "Inference Samples": r["n_inference"][i],
                "Train Budget": r["train_budget"][i],
                "Inference Budget": r["inference_budget"][i],
                "Num Arms": r["n_arms"][i] if "n_arms" in r else np.nan,
                "Avg Views Train": (r["avg_views_train"][i]
                                    if "avg_views_train" in r else np.nan),
                "Selected Subsets": serialize_selected_subsets(
                    r["selected_subsets"][i]),
                "Status": r["status"][i],
                "Error": r["error_msg"][i],
            }
            for k in TIMING_COLUMNS:
                row[k] = r[k][i] if k in r else np.nan
            detailed_rows.append(row)
    df_detailed = pd.DataFrame(detailed_rows)

    summary_rows = []
    for frac in budget_fractions:
        r = results[frac]
        n_ok = int(_ok_mask(r).sum())
        summary = {
            "Budget Fraction": frac,
            "Feedback": feedback,
            "Acquisition": acquisition,
            "Reward Update": reward_update,
            "Seeds OK": n_ok,
            "Seeds Failed": len(r["train_reward"]) - n_ok,
            "Train Reward Mean": _agg(r, "train_reward"),
            "Train Reward Std": _agg(r, "train_reward", np.std),
            "Train F1 Mean": _agg(r, "train_f1"),
            "Train F1 Std": _agg(r, "train_f1", np.std),
            "Train AUROC Mean": _agg(r, "train_auroc"),
            "Train AUROC Std": _agg(r, "train_auroc", np.std),
            "Inference Reward Mean": _agg(r, "inference_reward"),
            "Inference Reward Std": _agg(r, "inference_reward", np.std),
            "Inference F1 Mean": _agg(r, "inference_f1"),
            "Inference F1 Std": _agg(r, "inference_f1", np.std),
            "Inference AUROC Mean": _agg(r, "inference_auroc"),
            "Inference AUROC Std": _agg(r, "inference_auroc", np.std),
            "Total Reward Mean": _agg(r, "total_reward"),
            "Total Reward Std": _agg(r, "total_reward", np.std),
            "Total Error Mean": 1 - _agg(r, "total_reward"),
            "Train Spent Mean": _agg(r, "train_spent"),
            "Inference Spent Mean": _agg(r, "inference_spent"),
            "Num Masks Inference Mean": _agg(r, "num_masks_inference"),
            "Train Time Mean (s)": _agg(r, "train_time_sec"),
            "Inference Time Mean (s)": _agg(r, "inference_time_sec"),
            "Seed Time Mean (s)": _agg(r, "seed_time_sec"),
            "Train Samples": r["n_train"][0] if r["n_train"] else np.nan,
            "Inference Samples": r["n_inference"][0] if r["n_inference"] else np.nan,
            "Train Budget Mean": _agg(r, "train_budget"),
            "Inference Budget Mean": _agg(r, "inference_budget"),
            "Num Arms": (r["n_arms"][0] if r.get("n_arms") else np.nan),
            "Avg Views Train Mean": _agg(r, "avg_views_train"),
        }
        for k in TIMING_COLUMNS:
            summary[f"{k} Mean"] = _agg(r, k)
        summary_rows.append(summary)
    df_summary = pd.DataFrame(summary_rows)

    with pd.ExcelWriter(filename, engine="openpyxl") as writer:
        df_detailed.to_excel(writer, sheet_name="Detailed Results", index=False)
        df_summary.to_excel(writer, sheet_name="Summary", index=False)

    style_and_save(filename, ["Detailed Results", "Summary"], info_rows=info_rows)
    log.info("Results saved to %s", filename)


if __name__ == "__main__":
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(
        description="Run the adaptive AFA algorithm (full or bandit feedback) "
                     "on a dataset, with the same training/inference-phase and budget conventions "
                     "as the other runners."
    )
    parser.add_argument("--dataset", choices=ALL_DATASETS, default="synthetic")
    parser.add_argument("--feedback", choices=("full", "bandit"), default="full")
    parser.add_argument("--acquisition", choices=ACQUISITION_MODES, default="lp_chain",
                         help="Training-phase acquisition policy. 'lp_chain' "
                              "keeps empirical UCB reward estimates over a nested greedy "
                              "chain and solves the per-round budgeted LP over them exactly "
                              "(default). "
                              "'ucb_argmax' is the NOTEBOOK rule "
                              "(multiclass_supervised_unbiased_adaptive.ipynb): a "
                              "deterministic Lagrangian argmax over the full empirical "
                              "arm table. 'hedge' uses multiplicative resource weights, "
                              "and 'lp_full_opt' is the synthetic-only oracle LP.")
    parser.add_argument("--reward-update", choices=REWARD_UPDATE_SCOPES, default="subsets",
                         help="How empirical arm rewards are updated. "
                                "'subsets' replays every arm contained in the played subset "
                                "(counterfactual; requires y_true); 'selected' updates only the "
                                "played arm from its 0/1 reward. Used by learned policies "
                                "and ignored by lp_full_opt."
                        )
    parser.add_argument("--data-path", type=str, default=None)
    parser.add_argument("--split-mode", choices=SPLIT_MODES, default="80-20",
                        help="80-20 for comparing adaptive with two-stage; "
                             "60-20-20 for comparison with EDDI/DIME.")
    parser.add_argument("--max-modalities", type=str, default="all",
                         help="Integer, or 'all' (default). Retained acquisition policies "
                              "are capped at MAX_REWARD_ESTIMATE_VIEWS.")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument(
        "--sampling", choices=SAMPLING_MODES, default=DEFAULT_SAMPLING_MODE,
        help="REAL datasets only: balanced (default, no replacement), "
             "stratified, or random.",
    )
    parser.add_argument("--budget-fractions", type=str, default="0.1,0.3,0.5,0.7,0.9")
    parser.add_argument("--seeds", type=str, default="42,43,44,45,46,47,48,49,50,51")
    parser.add_argument("--n-samples", type=int, default=SYNTHETIC_N_SAMPLES)
    parser.add_argument("--n-views", type=int, default=SYNTHETIC_N_VIEWS)
    parser.add_argument("--num-classes", type=int, default=SYNTHETIC_N_CLASSES,
                         help="synthetic only: how many classes to generate. "
                              "Every other dataset infers nclasses from its labels.")
    parser.add_argument("--synthetic-seed", type=int, default=SYNTHETIC_SEED)
    parser.add_argument("--mean-scale", type=float, default=SYNTHETIC_MEAN_SCALE,
                         help="Per-(class,view) means ~ Uniform(0, mean_scale). The multiclass "
                              "notebook's snrdb=9 corresponds to mean_scale=10**(9/20)~=2.818.")
    parser.add_argument("--alpha-ucb", type=float, default=1.0)
    parser.add_argument(
        "--ucb-bound", choices=UCB_BOUNDS, default="vc",
        help="Confidence bound for empirical arm rewards. 'vc' uses "
             "alpha_ucb*sqrt((K*d_arm^2+log(t+2))/N_arm); 'vector' uses "
             "alpha_ucb*sqrt((d_arm+log(t+2))/N_arm).")
    parser.add_argument("--lr", type=float, default=1e-2,
                         help="bandit feedback only: complementary-label update learning rate.")
    parser.add_argument("--skip-inference", action="store_true",
                         help="Run ONLY the training phase (adaptive online training); skip the "
                              "inference-phase column-generation LP inference. Split and budgets are unchanged; "
                              "all inference_* columns become NaN and total_reward becomes the "
                              "train-only figure. Output filename gets a _TR tag.")
    parser.add_argument("--image-pool-side", type=int, default=DEFAULT_IMAGE_POOL_SIDE,
                         help=f"mnist/fashion_mnist only: block-average each 28x28 image down "
                              f"to x*x features (x=this value; default {DEFAULT_IMAGE_POOL_SIDE} "
                              f"-> {DEFAULT_IMAGE_POOL_SIDE**2} features). Pass 28 to keep all "
                              f"784 raw pixels. Ignored for non-image datasets.")
    parser.add_argument("--image-cache-dir", type=str, default=None,
                         help="mnist/fashion_mnist only: directory fetch_openml caches its "
                              "download in. Default (None) resolves to core.datasets."
                              "DEFAULT_OPENML_DATA_HOME, a RELATIVE 'data/openml_cache' folder -- "
                              "deliberately NOT fetch_openml's own ~/scikit_learn_data default, "
                              "which exceeds $HOME's quota on clusters like NCI Gadi. Point this "
                              "at your project/scratch space if the default location itself lacks "
                              "quota. Ignored for non-image datasets.")
    parser.add_argument(
        "--step-size", type=float, default=None,
        help="OMD dual ascent step size for ucb_argmax. Default: "
             "the varying schedule 1/sqrt(t+1), with zero-based training "
             "round t. Pass a number to use a fixed step size instead.")
    parser.add_argument("--lambda-max", type=float, default=10.0,
                         help="OMD dual clipping ceiling for ucb_argmax.",)
    parser.add_argument(
        "--output-xlsx", type=str, default=None,
        help=(
            "Explicit output workbook path. By default, Adaptive single-pass "
            "results are saved under results/Adaptive/singlepass/<Dataset>."
        ),
    )
    # ── observability flags (see core/logging_utils.py) ──
    parser.add_argument("--log-level", default="INFO",
                        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
                        help="Console verbosity, and file verbosity with --file-log.")
    parser.add_argument("--file-log", action="store_true",
                        help="Write a separate logs/{run_id}.log file (off by default).")
    parser.add_argument("--json-sidecars", action="store_true",
                        help="Write .manifest.json and .rows.jsonl files (off by default).")
    parser.add_argument("--log-dir", default="logs")
    parser.add_argument("--trace-rounds", action="store_true",
                        help="Write one record per training round to "
                             "results/{run_id}.trace.jsonl.")
    parser.add_argument("--no-fine-timers", action="store_true",
                        help="Disable the fine-grained timing buckets (t_* columns "
                             "become NaN). train/inference/seed timings are unaffected.")
    args = parser.parse_args()

    budget_fractions = tuple(float(x) for x in args.budget_fractions.split(","))
    seeds = tuple(int(x) for x in args.seeds.split(","))
    max_modalities = None if args.max_modalities.lower() == "all" else int(args.max_modalities)

    _cli_has_ru = _uses_empirical_arm_rewards(args.acquisition)
    output_acquisition = args.acquisition
    acq_label = output_acquisition + (f"/{args.reward_update}" if _cli_has_ru else "")

    ti_tag = "_TR" if args.skip_inference else ""
    # max_modalities is None when --max-modalities all was passed (no
    # truncation) -- render that as "ALL" instead of the literal "maxNone".
    maxmod_label = "ALL" if max_modalities is None else str(max_modalities)

    classes_tag = ""
    if args.dataset in SYNTHETIC_DATASETS:
        n_classes = args.num_classes if args.dataset in MULTICLASS_SYNTHETIC_DATASETS else 2
        classes_tag = f"_K{n_classes}"

    # Same scheme as run_proposed_methods.py, so the two entry points write
    # matching names.
    acq_tag = f"_{output_acquisition}"
    if _cli_has_ru:
        acq_tag += f"-{args.reward_update}"
    bound_tag = f"_{args.ucb_bound}"
    split_tag = ("" if args.split_mode == "80-20"
                 else f"_split{args.split_mode.replace('-', '')}")

    output_filename = (
        f"results_adaptive_{args.feedback}{acq_tag}{bound_tag}_"
        f"{args.dataset}_V{maxmod_label}_T{len(seeds)}"
        f"{split_tag}{ti_tag}{classes_tag}.xlsx"
    )
    if args.output_xlsx:
        output_xlsx = args.output_xlsx
    else:
        dataset_directory = (
            "Synthetic" if args.dataset.startswith("synthetic") else args.dataset
        )
        output_directory = (
            Path("results") / "Adaptive" / "singlepass" / dataset_directory
        )
        output_directory.mkdir(parents=True, exist_ok=True)
        output_xlsx = str(output_directory / output_filename)

    run = setup_run(
        "adaptive_runner",
        args=args, argv=sys.argv,
        name_hint=Path(output_xlsx).stem,
        log_dir=args.log_dir,
        console_level=args.log_level,
        file_level=args.log_level,
        file_logging=args.file_log,
        json_sidecars=args.json_sidecars,
        trace_rounds=args.trace_rounds,
        timing=not args.no_fine_timers,
        extra={"resolved_max_modalities": max_modalities,
               "resolved_seeds": list(seeds),
               "resolved_budget_fractions": list(budget_fractions),
               "output_xlsx": output_xlsx},
    )

    t0 = time.time()
    try:
        results = run_experiment(
            args.dataset,
            feedback=args.feedback,
            acquisition=args.acquisition,
            ucb_bound=args.ucb_bound,
            reward_update=args.reward_update,
            max_modalities=max_modalities,
            seeds=seeds,
            budget_fractions=budget_fractions,
            data_path=args.data_path,
            max_samples=args.max_samples,
            sampling=args.sampling,
            synthetic_n_samples=args.n_samples,
            synthetic_n_views=args.n_views,
            synthetic_seed=args.synthetic_seed,
            synthetic_mean_scale=args.mean_scale,
            synthetic_n_classes=args.num_classes,
            alpha_ucb=args.alpha_ucb,
            lr=args.lr,
            step_size=args.step_size,
            lambda_max=args.lambda_max,
            split_mode=args.split_mode,
            run_inference=not args.skip_inference,
            image_pool_side=args.image_pool_side,
            image_data_home=args.image_cache_dir,
            state_dir=Path(output_xlsx).with_suffix(".training_states.sqlite3"),
        )
    except BaseException as exc:                          # noqa: BLE001
        if run.json_sidecars:
            log.exception("run aborted; rows completed so far are in %s",
                          run.rows_path)
        else:
            log.exception("run aborted; JSON checkpointing was disabled")
        run.finalize(status=f"failed: {type(exc).__name__}: {exc}")
        raise

    log.info("=" * 70)
    log.info("SUMMARY (mean +/- std across seeds) -- %s / %s / %s",
             args.dataset, args.feedback, acq_label)
    log.info("=" * 70)
    log.info("%-10s%16s%12s%14s%14s%10s%12s%16s%14s", 'Fraction', 'Train Reward',
             'Train F1', 'Train AUROC', 'Inf Reward', 'Inf F1', 'Inf AUROC',
             'Total Reward', 'Total Error')
    log.info("-" * 130)
    for frac in budget_fractions:
        r = results[frac]
        log.info(
            "%-10.1f%11.3f+/-%.3f%7.3f%14.3f%9.3f%10.3f%12.3f%11.3f+/-%.3f%10.3f",
            frac,
            _agg(r, 'train_reward'), _agg(r, 'train_reward', np.std),
            _agg(r, 'train_f1'), _agg(r, 'train_auroc'),
            _agg(r, 'inference_reward'), _agg(r, 'inference_f1'),
            _agg(r, 'inference_auroc'),
            _agg(r, 'total_reward'), _agg(r, 'total_reward', np.std),
            1 - _agg(r, 'total_reward'),
        )

    save_results_to_excel(results, budget_fractions, args.dataset, args.feedback,
                          seeds=seeds, filename=output_xlsx,
                          acquisition=output_acquisition,
                          reward_update=(args.reward_update if _cli_has_ru else ""),
                          info_rows=run.info_rows())
    log.info("Execution time: %.1f seconds", time.time() - t0)
    run.finalize(status="ok")
