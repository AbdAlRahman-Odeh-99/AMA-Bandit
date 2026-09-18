"""Run the causal online AFA baselines with a shared protocol.

All baselines use 60/20/20 train/validation/test. For comparable
adaptive or two-stage results, use ``run_proposed_methods.py`` with
``--split-mode 60-20-20`` and the same dataset, costs, seeds, and synthetic
configuration.
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch
from sklearn.metrics import f1_score, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

from core.datasets import (
    ALL_DATASETS,
    DEFAULT_SAMPLING_MODE,
    DEFAULT_IMAGE_POOL_SIDE,
    SYNTHETIC_DATASETS,
    SYNTHETIC_N_CLASSES,
    SYNTHETIC_N_VIEWS,
    SYNTHETIC_SEED,
    SAMPLING_MODES,
    generate_modality_costs_heterogeneous,
    load_dataset_as_numpy,
    split_dataset,
)
from core.utils import MaskLayerGrouped


def baseline_output_filename(method, dataset, n_modalities, n_seeds,
                             n_classes, linear_classifier=True):
    """Return the adaptive-style V/T/K filename for any baseline variant."""
    dimensions = {
        "n_modalities": n_modalities,
        "n_seeds": n_seeds,
        "n_classes": n_classes,
    }
    for name, value in dimensions.items():
        if int(value) < 1:
            raise ValueError(f"{name} must be positive, got {value!r}")
    classifier_tag = "" if linear_classifier else "_mlp"
    return (
        f"results_{method}_{dataset}_V{int(n_modalities)}_T{int(n_seeds)}"
        f"{classifier_tag}_K{int(n_classes)}.csv"
    )


def _budget_summary_fields(state, prefix):
    """
    Flatten a budget_state.BudgetState's summary() into prefixed fields
    for a results dict/CSV row -- e.g. prefix="cae_stage1" gives
    cae_stage1_lambda_final, cae_stage1_cumulative_spent,
    cae_stage1_total_budget, cae_stage1_remaining_budget_final,
    cae_stage1_spent_total.

    state=None (i.e. use_matched_budget_constraint=False, so this stage
    never had a BudgetState at all) fills every field with None, so the
    CSV always has the same columns whether or not the constraint was
    used for that trial.
    """
    if state is None:
        return {
            f"{prefix}_lambda_final": None,
            f"{prefix}_cumulative_spent": None,
            f"{prefix}_total_budget": None,
            f"{prefix}_remaining_budget_final": None,
            f"{prefix}_spent_total": None,
        }
    s = state.summary()
    return {
        f"{prefix}_lambda_final": s["lambda_final"],
        f"{prefix}_cumulative_spent": s["cumulative_spent"],
        f"{prefix}_total_budget": s["total_budget"],
        f"{prefix}_remaining_budget_final": s["remaining_budget"],
        f"{prefix}_spent_total": s["spent_total"],
    }


def f1_metric(pred, y):
    """
    F1 score wrapped as a torch tensor for the shared metric convention.

    Class-count is read from pred.shape[1] (the classifier's logit width =
    d_out): exactly 2 -> binary F1 on the positive class (index 1), matching
    the original binary behavior; >2 -> MACRO-averaged F1 (unweighted mean
    of per-class F1), the standard multiclass summary. zero_division=0
    avoids a warning/error when a class is never predicted.
    """
    y_pred = pred.argmax(dim=1).cpu().numpy()
    y_true = y.cpu().numpy()
    average = "binary" if pred.shape[1] == 2 else "macro"
    return torch.tensor(f1_score(y_true, y_pred, average=average, zero_division=0))


def accuracy_metric(pred, y):
    return (pred.argmax(dim=1) == y).float().mean()


def auroc_metric(pred, y):
    """
    AUROC from softmax probabilities (unlike accuracy/F1, this needs
    probabilities, not hard argmax predictions).

    Class-count is read from pred.shape[1] (= d_out): exactly 2 -> binary
    AUROC on the positive class (index 1), matching the original behavior;
    >2 -> one-vs-rest AUROC, MACRO-averaged over classes.

    roc_auc_score needs every class it scores to be present in y_true. If a
    batch/split has <2 classes present (binary), or is missing one of the K
    classes (multiclass OVR), there's no valid AUROC -- returns NaN rather
    than raising, so a sweep doesn't crash on an unlucky split.
    """
    probs = torch.softmax(pred, dim=1).detach().cpu().numpy()
    y_true = y.cpu().numpy()
    n_classes = pred.shape[1]
    if len(np.unique(y_true)) < 2:
        return torch.tensor(float("nan"))
    if n_classes == 2:
        return torch.tensor(roc_auc_score(y_true, probs[:, 1]))
    # Multiclass: OVR needs all K classes present in y_true; guard with nan.
    try:
        auroc = roc_auc_score(
            y_true, probs, multi_class="ovr", average="macro",
            labels=list(range(n_classes)),
        )
    except ValueError:
        auroc = float("nan")
    return torch.tensor(auroc)


# ============================================================
# Shared setup: data, split, mask layer, feature costs
# ============================================================

DEFAULT_SYNTHETIC_NVIEWS = 10


def build_experiment(dataset_name, device, data_path=None, max_modalities=None, split_seed=None,
                      synthetic_seed=SYNTHETIC_SEED,
                      nsamples=1000, n_views=None, max_samples=None,
                      sampling=DEFAULT_SAMPLING_MODE, image_pool_side=None,
                      image_data_home=None, num_classes=None):
    """
    Load the chosen dataset, split into train/val/test, and build the
    shared modality structure (mask layer + feature costs) used by the
    online baseline adapters.

    dataset_name: one of ALL_BINARY_DATASETS (real UCI datasets) OR one
    of SYNTHETIC_DATASETS (including multiclass "synthetic") -- the
    same synthetic data used by the proposed methods.
    For the synthetic datasets, data_path is ignored (nothing is read
    from disk -- see load_gmm_dataset) and nsamples controls the
    generated dataset's row count; for real datasets, nsamples is ignored.

    data_path: optional override for where to read the dataset's raw CSV
    from. Required for "miniboone"/"physionet" if your local file isn't
    at afa_tabular_datasets.DEFAULT_PATHS' default location. Ignored for
    synthetic GMM datasets.

    max_modalities: controls the number of modalities/views for BOTH
    dataset kinds, but via a different mechanism for each -- there used
    to be a separate `nviews` parameter for synthetic datasets; it's been
    folded into this one, since the two were doing the same conceptual
    job (how many modalities does this experiment use) and having both
    invited exactly the max_modalities < nviews truncation-without-
    renormalization bug this merge eliminates.
      - REAL datasets: keeps only the FIRST max_modalities columns of the
        dataset's feature matrix (in whatever column order
        load_afa_dataset returns -- i.e. the original CSV's column
        order, unchanged by preprocessing). Modality 0 (the free one) is
        always feature_names[0], so max_modalities=5 means "the free
        modality + the first 4 paid features", not 5 arbitrary/random ones.
      - SYNTHETIC datasets: GENERATES exactly max_modalities views in the
        first place (passed straight through to load_gmm_dataset's own
        nviews argument) -- there is no separate generate-then-truncate
        step, so feature_costs always sums to exactly 1 over whatever
        width was requested, with no renormalization needed. If left
        None, defaults to DEFAULT_SYNTHETIC_NVIEWS (5, matching the
        original scripts' NUM_VIEWS).

    split_seed: seed for the train/val/test split (afa_tabular_datasets.
    split_dataset's own SPLIT_SEED=42 default is used if left None).
    Vary this across a seed loop to get a genuinely different split per
    seed, not just different model initialization. For synthetic GMM
    datasets, this ALSO seeds data generation itself (see
    load_gmm_dataset) -- so unlike the real datasets (same underlying
    data, different split, across seeds), each seed here gets a
    genuinely fresh dataset AND costs, matching how the original GMM
    scripts' own per-trial loop redraws data every trial rather than
    reusing one fixed dataset. If left None, seed 42 is used (matching
    the original scripts' default SEED).

    Returns a dict with everything either runner needs.
    """
    is_synthetic = dataset_name in SYNTHETIC_DATASETS
    synthetic_n_views = n_views if n_views is not None else SYNTHETIC_N_VIEWS
    synthetic_n_classes = num_classes if num_classes is not None else SYNTHETIC_N_CLASSES
    pool = image_pool_side if image_pool_side is not None else DEFAULT_IMAGE_POOL_SIDE

    # Use the exact same loader contract and synthetic generator as the
    # aligned proposed-method runners.  The only intentional protocol
    # difference is the 60/20/20 split below, because the baselines need a
    # validation set.
    X_np, y_np, feature_names = load_dataset_as_numpy(
        dataset_name,
        max_modalities=None if is_synthetic else max_modalities,
        data_path=data_path,
        max_samples=max_samples,
        sampling=sampling,
        synthetic_n_samples=nsamples,
        synthetic_n_views=synthetic_n_views,
        synthetic_seed=synthetic_seed,
        synthetic_n_classes=synthetic_n_classes,
        image_pool_side=pool,
        image_data_home=image_data_home,
    )
    X = torch.as_tensor(X_np, dtype=torch.float32)
    y = torch.as_tensor(y_np, dtype=torch.long)
    num_modalities = X.shape[1]  # 1 free + (num_modalities - 1) paid
    d_in = num_modalities
    d_out = int(torch.unique(y).numel())
    print(f"{dataset_name}: {X.shape[0]} samples, {num_modalities} modalities "
          f"(free modality: '{feature_names[0]}', {num_modalities - 1} paid), {d_out} classes"
          + (f" [truncated to first {max_modalities}]" if (max_modalities is not None and not is_synthetic) else "")
          + (f" [generated at {num_modalities} views]" if is_synthetic else ""))

    split_kwargs = {} if split_seed is None else {"seed": split_seed}
    train_idx, val_idx, test_idx = split_dataset(X.shape[0], **split_kwargs)
    train_idx = torch.tensor(train_idx, dtype=torch.long)
    val_idx = torch.tensor(val_idx, dtype=torch.long)
    test_idx = torch.tensor(test_idx, dtype=torch.long)

    train_dataset = TensorDataset(X[train_idx], y[train_idx])
    val_dataset = TensorDataset(X[val_idx], y[val_idx])
    test_dataset = TensorDataset(X[test_idx], y[test_idx])

    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    group_matrix = torch.eye(num_modalities)
    mask_layer = MaskLayerGrouped(group_matrix).to(device)

    raw_feature_costs = np.asarray(
        generate_modality_costs_heterogeneous(
            n_features=num_modalities, dataset_name=dataset_name
        ),
        dtype=float,
    )
    feature_costs = (raw_feature_costs / raw_feature_costs.sum()).tolist()
    paid_costs = feature_costs[1:]
    print(f"feature_costs[0] (free) = {feature_costs[0]}, "
          f"paid costs: min={min(paid_costs):.4f}, max={max(paid_costs):.4f}, "
          f"mean={sum(paid_costs) / len(paid_costs):.4f}, total={sum(paid_costs):.4f}, "
          f"total (incl. free) = {sum(feature_costs):.4f}")

    return {
        "dataset_name": dataset_name,
        "split_mode": "60-20-20",
        "feature_names": feature_names,
        "num_modalities": num_modalities,
        "d_in": d_in,
        "d_out": d_out,
        "train_dataset": train_dataset,
        "val_dataset": val_dataset,
        "test_dataset": test_dataset,
        "train_dataloader": train_dataloader,
        "val_dataloader": val_dataloader,
        "test_dataloader": test_dataloader,
        "mask_layer": mask_layer,
        "feature_costs": feature_costs,
        "paid_costs": paid_costs,
    }


# ============================================================
# Causal online baseline variants
# ============================================================

def _online_classification_metrics(trace, nclasses):
    """Metrics for predictions recorded before each online update."""
    labels = np.asarray(trace["train_labels"], dtype=int)
    predictions = np.asarray(trace["train_predictions"], dtype=int)
    average = "binary" if nclasses == 2 else "macro"
    f1 = f1_score(labels, predictions, average=average, zero_division=0)
    logits = trace["train_logits"]
    auroc = auroc_metric(logits, torch.as_tensor(labels, dtype=torch.long)).item()
    return float(f1), float(auroc)


def _evaluate_online_acquisition(model, dataset, budget, device):
    """Evaluate a trained online model on held-out data with a fresh pool."""
    from core.budget_state import BudgetState

    X, y = dataset.tensors
    X = X.to(device)
    y = y.to(device)
    state = BudgetState(
        spending_ratio=budget,
        total_budget=budget * len(dataset),
    )
    model.predictor.eval()
    model.acquisition_model.eval()
    with torch.no_grad():
        x_masked, _, total_cost = model.acquisition_model.select_features(
            X, budget=budget, verbose=False, global_budget_state=state
        )
        logits = model.predictor(x_masked)
    return {
        "test_accuracy": float(accuracy_metric(logits, y).item()),
        "test_f1": float(f1_metric(logits, y).item()),
        "test_auroc": float(auroc_metric(logits, y).item()),
        "test_avg_cost_per_sample": float(total_cost / max(len(dataset), 1)),
        "state": state,
    }


def _run_online_baseline(
    ctx,
    device,
    method,
    *,
    train_budget,
    test_budget,
    learning_rate=1e-3,
    updates_per_sample=1,
    pvae_samples=128,
    cost_normalized=True,
    cmi_scaling="bounded",
    cae_entropy_weight=1e-2,
    cae_gamma=0.2,
    aaco_k_neighbors=5,
    aaco_acquisition_cost=0.05,
    aaco_max_candidates=100,
    aaco_exact_max_paid=12,
    gdfs_entropy_weight=1e-3,
    pt_importance_decay=0.9,
    jafa_embedding_size=16,
    jafa_hidden_size=32,
    jafa_memory_size=16,
    jafa_processing_steps=5,
    jafa_acquisition_cost_weight=0.05,
    jafa_epsilon_start=1.0,
    jafa_epsilon_end=0.1,
    jafa_max_candidates=512,
    jafa_exact_max_paid=12,
    ol_hidden_sizes=(64, 32, 16),
    ol_dropout=0.5,
    ol_use_feature_mask=False,
    ol_reward_method="Bayesian-L1",
    ol_mcdrop_samples=100,
    ol_epsilon_start=1.0,
    ol_epsilon_end=0.1,
    ol_max_candidates=512,
    ol_exact_max_paid=12,
    cwcf_hidden_size=128,
    cwcf_hidden_layers=3,
    cwcf_cost_weight=1.0,
    cwcf_epsilon_start=1.0,
    cwcf_epsilon_end=0.1,
    cwcf_max_candidates=512,
    cwcf_exact_max_paid=12,
    cwcf_grad_clip=1.0,
    lambda_max=10.0,
    step_size=1.0,
    linear_classifier=True,
    seed=42,
):
    """Shared runner for the explicitly named causal online variants."""
    from baselines.online import (
        build_online_aaco,
        build_online_cae,
        build_online_cwcf,
        build_online_dime,
        build_online_eddi,
        build_online_gdfs,
        build_online_jafa,
        build_online_ol,
        build_online_pt,
        run_prequential_stream,
    )
    from core.budget_state import BudgetState

    n_train = len(ctx["train_dataset"])
    nclasses = ctx["d_out"]
    X_train, y_train = ctx["train_dataset"].tensors
    X_train = X_train.to(device)
    y_train = y_train.to(device)

    if method == "online_aaco":
        online_model = build_online_aaco(
            ctx, device,
            train_budget=train_budget,
            learning_rate=learning_rate,
            updates_per_sample=updates_per_sample,
            k_neighbors=aaco_k_neighbors,
            acquisition_cost=aaco_acquisition_cost,
            max_candidates=aaco_max_candidates,
            exact_max_paid=aaco_exact_max_paid,
            linear_classifier=linear_classifier,
            seed=seed,
        )
    elif method == "online_cae":
        online_model = build_online_cae(
            ctx, device,
            train_budget=train_budget,
            learning_rate=learning_rate,
            updates_per_sample=updates_per_sample,
            entropy_weight=cae_entropy_weight,
            gamma=cae_gamma,
            linear_classifier=linear_classifier,
        )
    elif method == "online_gdfs":
        online_model = build_online_gdfs(
            ctx, device,
            train_budget=train_budget,
            learning_rate=learning_rate,
            updates_per_sample=updates_per_sample,
            entropy_weight=gdfs_entropy_weight,
            linear_classifier=linear_classifier,
        )
    elif method == "online_jafa":
        online_model = build_online_jafa(
            ctx, device,
            train_budget=train_budget,
            learning_rate=learning_rate,
            updates_per_sample=updates_per_sample,
            embedding_size=jafa_embedding_size,
            hidden_size=jafa_hidden_size,
            memory_size=jafa_memory_size,
            processing_steps=jafa_processing_steps,
            acquisition_cost_weight=jafa_acquisition_cost_weight,
            epsilon_start=jafa_epsilon_start,
            epsilon_end=jafa_epsilon_end,
            max_candidates=jafa_max_candidates,
            exact_max_paid=jafa_exact_max_paid,
            linear_classifier=linear_classifier,
            seed=seed,
        )
    elif method == "online_ol":
        online_model = build_online_ol(
            ctx, device,
            train_budget=train_budget,
            learning_rate=learning_rate,
            updates_per_sample=updates_per_sample,
            hidden_sizes=ol_hidden_sizes,
            dropout=ol_dropout,
            use_feature_mask=ol_use_feature_mask,
            reward_method=ol_reward_method,
            mcdrop_samples=ol_mcdrop_samples,
            epsilon_start=ol_epsilon_start,
            epsilon_end=ol_epsilon_end,
            max_candidates=ol_max_candidates,
            exact_max_paid=ol_exact_max_paid,
            seed=seed,
        )
    elif method == "online_cwcf":
        online_model = build_online_cwcf(
            ctx, device, train_budget=train_budget,
            learning_rate=learning_rate,
            updates_per_sample=updates_per_sample,
            hidden_size=cwcf_hidden_size,
            hidden_layers=cwcf_hidden_layers,
            cost_weight=cwcf_cost_weight,
            epsilon_start=cwcf_epsilon_start,
            epsilon_end=cwcf_epsilon_end,
            max_candidates=cwcf_max_candidates,
            exact_max_paid=cwcf_exact_max_paid,
            grad_clip=cwcf_grad_clip,
            seed=seed,
        )
    elif method == "online_pt":
        online_model = build_online_pt(
            ctx, device,
            train_budget=train_budget,
            learning_rate=learning_rate,
            updates_per_sample=updates_per_sample,
            importance_decay=pt_importance_decay,
            linear_classifier=linear_classifier,
            seed=seed,
        )
    elif method == "online_eddi":
        online_model = build_online_eddi(
            ctx, device,
            learning_rate=learning_rate,
            updates_per_sample=updates_per_sample,
            cost_normalized=cost_normalized,
            pvae_samples=pvae_samples,
            linear_classifier=linear_classifier,
        )
    elif method == "online_dime":
        online_model = build_online_dime(
            ctx, device,
            learning_rate=learning_rate,
            updates_per_sample=updates_per_sample,
            cmi_scaling=cmi_scaling,
            linear_classifier=linear_classifier,
        )
    else:
        raise ValueError(f"unsupported online method: {method!r}")

    train_state = BudgetState(
        spending_ratio=train_budget,
        total_budget=train_budget * n_train,
        lambda_max=lambda_max,
        step_size=step_size,
    )
    training_t0 = time.time()
    trace = run_prequential_stream(
        X_train, y_train,
        nclasses=nclasses,
        feature_costs=ctx["feature_costs"],
        budget_state=train_state,
        acquire_and_predict=online_model.acquire_and_predict,
        score_with_mask=online_model.score_with_mask,
        update=online_model.update,
        rng=np.random.default_rng(42 if seed is None else seed),
    )
    train_time_sec = time.time() - training_t0
    train_f1, train_auroc = _online_classification_metrics(trace, nclasses)

    inference_t0 = time.time()
    test = _evaluate_online_acquisition(
        online_model, ctx["test_dataset"], test_budget, device
    )
    inference_time_sec = time.time() - inference_t0

    return {
        "method": method,
        "online_protocol": "predict_then_update_one_pass",
        "train_budget": train_budget,
        "test_budget": test_budget,
        "train_reward": trace["train_reward"],
        "training_error": trace["training_error"],
        "training_regret": trace["training_error"],
        "train_f1": train_f1,
        "train_auroc": train_auroc,
        "cum_train_reward": trace["cum_train_reward"],
        "cumulative_mistakes": trace["cumulative_mistakes"],
        "train_predictions": trace["train_predictions"],
        "selected_modalities": trace["selected_modalities"],
        "train_cost_trace": trace["train_cost_trace"],
        "train_avg_cost_per_sample": trace["train_spent"] / max(n_train, 1),
        "test_accuracy": test["test_accuracy"],
        "test_f1": test["test_f1"],
        "test_auroc": test["test_auroc"],
        "test_avg_cost_per_sample": test["test_avg_cost_per_sample"],
        "online_learning_rate": learning_rate,
        "online_updates_per_sample": updates_per_sample,
        "pvae_samples": pvae_samples if method == "online_eddi" else None,
        "cae_entropy_weight": (cae_entropy_weight
                               if method == "online_cae" else None),
        "cae_gamma": cae_gamma if method == "online_cae" else None,
        "aaco_k_neighbors": (aaco_k_neighbors
                             if method == "online_aaco" else None),
        "aaco_acquisition_cost": (aaco_acquisition_cost
                                  if method == "online_aaco" else None),
        "aaco_max_candidates": (aaco_max_candidates
                                if method == "online_aaco" else None),
        "aaco_exact_max_paid": (aaco_exact_max_paid
                                if method == "online_aaco" else None),
        "reference_size": (len(online_model.acquisition_model.reference_y)
                           if method == "online_aaco" else None),
        "gdfs_entropy_weight": (gdfs_entropy_weight
                                if method == "online_gdfs" else None),
        "jafa_num_candidate_subsets": (
            len(online_model.acquisition_model.candidate_masks)
            if method == "online_jafa" else None
        ),
        "jafa_embedding_size": (
            jafa_embedding_size if method == "online_jafa" else None
        ),
        "jafa_hidden_size": (
            jafa_hidden_size if method == "online_jafa" else None
        ),
        "jafa_memory_size": (
            jafa_memory_size if method == "online_jafa" else None
        ),
        "jafa_processing_steps": (
            jafa_processing_steps if method == "online_jafa" else None
        ),
        "jafa_acquisition_cost_weight": (
            jafa_acquisition_cost_weight
            if method == "online_jafa" else None
        ),
        "jafa_epsilon_start": (
            jafa_epsilon_start if method == "online_jafa" else None
        ),
        "jafa_epsilon_end": (
            jafa_epsilon_end if method == "online_jafa" else None
        ),
        "ol_num_candidate_subsets": (
            len(online_model.acquisition_model.candidate_masks)
            if method == "online_ol" else None
        ),
        "ol_hidden_sizes": (
            list(ol_hidden_sizes) if method == "online_ol" else None
        ),
        "ol_dropout": ol_dropout if method == "online_ol" else None,
        "ol_use_feature_mask": (
            ol_use_feature_mask if method == "online_ol" else None
        ),
        "ol_reward_method": (
            ol_reward_method if method == "online_ol" else None
        ),
        "ol_mcdrop_samples": (
            ol_mcdrop_samples if method == "online_ol" else None
        ),
        "ol_epsilon_start": (
            ol_epsilon_start if method == "online_ol" else None
        ),
        "ol_epsilon_end": (
            ol_epsilon_end if method == "online_ol" else None
        ),
        "cwcf_num_candidate_subsets": (
            len(online_model.acquisition_model.candidate_masks)
            if method == "online_cwcf" else None
        ),
        "cwcf_hidden_size": cwcf_hidden_size if method == "online_cwcf" else None,
        "cwcf_hidden_layers": cwcf_hidden_layers if method == "online_cwcf" else None,
        "cwcf_cost_weight": cwcf_cost_weight if method == "online_cwcf" else None,
        "cwcf_epsilon_start": cwcf_epsilon_start if method == "online_cwcf" else None,
        "cwcf_epsilon_end": cwcf_epsilon_end if method == "online_cwcf" else None,
        "pt_importance_decay": (pt_importance_decay
                                if method == "online_pt" else None),
        "pt_feature_importance": (
            online_model.acquisition_model.feature_importance.detach().cpu().tolist()
            if method == "online_pt" else None
        ),
        "pt_importance_metric": (
            "causal_past_value_cross_entropy_increase"
            if method == "online_pt" else None
        ),
        "train_time_sec": train_time_sec,
        "inference_time_sec": inference_time_sec,
        "linear_classifier": (
            False if method in ("online_ol", "online_cwcf") else linear_classifier
        ),
        "classifier_architecture": (
            "ol_shared_p_net" if method == "online_ol" else
            ("cwcf_dueling_q_terminal_actions" if method == "online_cwcf" else
            ("linear" if linear_classifier else "mlp")
            )
        ),
        **_budget_summary_fields(train_state, "online_training"),
        **_budget_summary_fields(test["state"], "test_inference"),
    }


def run_online_aaco(ctx, device, train_budget=None, test_budget=None,
                    train_fraction=0.75, test_fraction=0.25,
                    learning_rate=1e-3, updates_per_sample=1,
                    k_neighbors=5, acquisition_cost=0.05,
                    max_candidates=100, exact_max_paid=12,
                    lambda_max=10.0, step_size=1.0,
                    linear_classifier=True, seed=42):
    """Run causal one-pass AACO with a growing acquired-only KNN bank."""
    total_paid_cost = sum(ctx["paid_costs"])
    train_budget = train_fraction * total_paid_cost if train_budget is None else train_budget
    test_budget = test_fraction * total_paid_cost if test_budget is None else test_budget
    return _run_online_baseline(
        ctx, device, "online_aaco", train_budget=train_budget,
        test_budget=test_budget, learning_rate=learning_rate,
        updates_per_sample=updates_per_sample,
        aaco_k_neighbors=k_neighbors,
        aaco_acquisition_cost=acquisition_cost,
        aaco_max_candidates=max_candidates,
        aaco_exact_max_paid=exact_max_paid,
        lambda_max=lambda_max, step_size=step_size,
        linear_classifier=linear_classifier, seed=seed,
    )


def run_online_gdfs(ctx, device, train_budget=None, test_budget=None,
                    train_fraction=0.75, test_fraction=0.25,
                    learning_rate=1e-3, updates_per_sample=1,
                    entropy_weight=1e-3, lambda_max=10.0,
                    step_size=1.0, linear_classifier=True, seed=42):
    """Run causal acquired-only GDFS with one-shot sampled subsets."""
    total_paid_cost = sum(ctx["paid_costs"])
    train_budget = train_fraction * total_paid_cost if train_budget is None else train_budget
    test_budget = test_fraction * total_paid_cost if test_budget is None else test_budget
    return _run_online_baseline(
        ctx, device, "online_gdfs", train_budget=train_budget,
        test_budget=test_budget, learning_rate=learning_rate,
        updates_per_sample=updates_per_sample,
        gdfs_entropy_weight=entropy_weight,
        lambda_max=lambda_max, step_size=step_size,
        linear_classifier=linear_classifier, seed=seed,
    )


def run_online_jafa(ctx, device, train_budget=None, test_budget=None,
                    train_fraction=0.75, test_fraction=0.25,
                    learning_rate=1e-3, updates_per_sample=1,
                    embedding_size=16, hidden_size=32, memory_size=16,
                    processing_steps=5, acquisition_cost_weight=0.05,
                    epsilon_start=1.0, epsilon_end=0.1,
                    max_candidates=512, exact_max_paid=12,
                    lambda_max=10.0, step_size=1.0,
                    linear_classifier=True, seed=42):
    """Run causal one-pass JAFA with one complete subset action."""
    total_paid_cost = sum(ctx["paid_costs"])
    train_budget = (
        train_fraction * total_paid_cost if train_budget is None
        else train_budget
    )
    test_budget = (
        test_fraction * total_paid_cost if test_budget is None
        else test_budget
    )
    return _run_online_baseline(
        ctx, device, "online_jafa", train_budget=train_budget,
        test_budget=test_budget, learning_rate=learning_rate,
        updates_per_sample=updates_per_sample,
        jafa_embedding_size=embedding_size,
        jafa_hidden_size=hidden_size,
        jafa_memory_size=memory_size,
        jafa_processing_steps=processing_steps,
        jafa_acquisition_cost_weight=acquisition_cost_weight,
        jafa_epsilon_start=epsilon_start,
        jafa_epsilon_end=epsilon_end,
        jafa_max_candidates=max_candidates,
        jafa_exact_max_paid=exact_max_paid,
        lambda_max=lambda_max, step_size=step_size,
        linear_classifier=linear_classifier, seed=seed,
    )


def run_online_ol(ctx, device, train_budget=None, test_budget=None,
                  train_fraction=0.75, test_fraction=0.25,
                  learning_rate=1e-3, updates_per_sample=1,
                  hidden_sizes=(64, 32, 16), dropout=0.5,
                  use_feature_mask=False, reward_method="Bayesian-L1",
                  mcdrop_samples=100, epsilon_start=1.0,
                  epsilon_end=0.1, max_candidates=512,
                  exact_max_paid=12, lambda_max=10.0,
                  step_size=1.0, seed=42):
    """Run causal OL with one confidence-reward subset action per row."""
    total_paid_cost = sum(ctx["paid_costs"])
    train_budget = (
        train_fraction * total_paid_cost if train_budget is None
        else train_budget
    )
    test_budget = (
        test_fraction * total_paid_cost if test_budget is None
        else test_budget
    )
    return _run_online_baseline(
        ctx, device, "online_ol", train_budget=train_budget,
        test_budget=test_budget, learning_rate=learning_rate,
        updates_per_sample=updates_per_sample,
        ol_hidden_sizes=hidden_sizes,
        ol_dropout=dropout,
        ol_use_feature_mask=use_feature_mask,
        ol_reward_method=reward_method,
        ol_mcdrop_samples=mcdrop_samples,
        ol_epsilon_start=epsilon_start,
        ol_epsilon_end=epsilon_end,
        ol_max_candidates=max_candidates,
        ol_exact_max_paid=exact_max_paid,
        lambda_max=lambda_max, step_size=step_size, seed=seed,
    )


def run_online_cwcf(ctx, device, train_budget=None, test_budget=None,
                    train_fraction=0.75, test_fraction=0.25,
                    learning_rate=5e-4, updates_per_sample=1,
                    hidden_size=128, hidden_layers=3, cost_weight=1.0,
                    epsilon_start=1.0, epsilon_end=0.1,
                    max_candidates=512, exact_max_paid=12, grad_clip=1.0,
                    lambda_max=10.0, step_size=1.0, seed=42):
    """Run causal one-pass CwCF with post-prediction Q updates."""
    total_paid_cost = sum(ctx["paid_costs"])
    train_budget = train_fraction * total_paid_cost if train_budget is None else train_budget
    test_budget = test_fraction * total_paid_cost if test_budget is None else test_budget
    return _run_online_baseline(
        ctx, device, "online_cwcf", train_budget=train_budget,
        test_budget=test_budget, learning_rate=learning_rate,
        updates_per_sample=updates_per_sample,
        cwcf_hidden_size=hidden_size, cwcf_hidden_layers=hidden_layers,
        cwcf_cost_weight=cost_weight, cwcf_epsilon_start=epsilon_start,
        cwcf_epsilon_end=epsilon_end, cwcf_max_candidates=max_candidates,
        cwcf_exact_max_paid=exact_max_paid, cwcf_grad_clip=grad_clip,
        lambda_max=lambda_max, step_size=step_size, seed=seed,
    )


def run_online_pt(ctx, device, train_budget=None, test_budget=None,
                  train_fraction=0.75, test_fraction=0.25,
                  learning_rate=1e-3, updates_per_sample=1,
                  importance_decay=0.9, lambda_max=10.0,
                  step_size=1.0, linear_classifier=True, seed=42):
    """Run causal PT with past-value streaming permutation importance."""
    total_paid_cost = sum(ctx["paid_costs"])
    train_budget = train_fraction * total_paid_cost if train_budget is None else train_budget
    test_budget = test_fraction * total_paid_cost if test_budget is None else test_budget
    return _run_online_baseline(
        ctx, device, "online_pt", train_budget=train_budget,
        test_budget=test_budget, learning_rate=learning_rate,
        updates_per_sample=updates_per_sample,
        pt_importance_decay=importance_decay,
        lambda_max=lambda_max, step_size=step_size,
        linear_classifier=linear_classifier, seed=seed,
    )


def run_online_eddi(ctx, device, train_budget=None, test_budget=None,
                    train_fraction=0.75, test_fraction=0.25,
                    learning_rate=1e-3, updates_per_sample=1,
                    pvae_samples=128, cost_normalized=True,
                    lambda_max=10.0, step_size=1.0,
                    linear_classifier=True, seed=42):
    """Run causal one-pass EDDI; the online EDDI adapter."""
    total_paid_cost = sum(ctx["paid_costs"])
    train_budget = train_fraction * total_paid_cost if train_budget is None else train_budget
    test_budget = test_fraction * total_paid_cost if test_budget is None else test_budget
    return _run_online_baseline(
        ctx, device, "online_eddi", train_budget=train_budget,
        test_budget=test_budget, learning_rate=learning_rate,
        updates_per_sample=updates_per_sample, pvae_samples=pvae_samples,
        cost_normalized=cost_normalized, lambda_max=lambda_max,
        step_size=step_size, linear_classifier=linear_classifier, seed=seed,
    )


def run_online_cae(ctx, device, train_budget=None, test_budget=None,
                   train_fraction=0.75, test_fraction=0.25,
                   learning_rate=1e-3, updates_per_sample=1,
                   entropy_weight=1e-2, gamma=0.2,
                   lambda_max=10.0, step_size=1.0,
                   linear_classifier=True, seed=42):
    """Run causal one-pass CAE; the online CAE adapter."""
    total_paid_cost = sum(ctx["paid_costs"])
    train_budget = train_fraction * total_paid_cost if train_budget is None else train_budget
    test_budget = test_fraction * total_paid_cost if test_budget is None else test_budget
    return _run_online_baseline(
        ctx, device, "online_cae", train_budget=train_budget,
        test_budget=test_budget, learning_rate=learning_rate,
        updates_per_sample=updates_per_sample,
        cae_entropy_weight=entropy_weight, cae_gamma=gamma,
        lambda_max=lambda_max, step_size=step_size,
        linear_classifier=linear_classifier, seed=seed,
    )


def run_online_dime(ctx, device, train_budget=None, test_budget=None,
                    train_fraction=0.75, test_fraction=0.25,
                    learning_rate=1e-3, updates_per_sample=1,
                    cmi_scaling="bounded", lambda_max=10.0, step_size=1.0,
                    linear_classifier=True, seed=42):
    """Run causal one-pass DIME; the online DIME adapter."""
    total_paid_cost = sum(ctx["paid_costs"])
    train_budget = train_fraction * total_paid_cost if train_budget is None else train_budget
    test_budget = test_fraction * total_paid_cost if test_budget is None else test_budget
    return _run_online_baseline(
        ctx, device, "online_dime", train_budget=train_budget,
        test_budget=test_budget, learning_rate=learning_rate,
        updates_per_sample=updates_per_sample, cmi_scaling=cmi_scaling,
        lambda_max=lambda_max, step_size=step_size,
        linear_classifier=linear_classifier, seed=seed,
    )


# ============================================================
# Online-only budget sweeps and CLI
# ============================================================

ONLINE_METHODS = (
    "online_aaco",
    "online_cae",
    "online_cwcf",
    "online_eddi",
    "online_dime",
    "online_gdfs",
    "online_jafa",
    "online_ol",
    "online_pt",
)


def run_budget_sweep(
    ctx,
    device,
    method,
    budget_fractions,
    *,
    seed=None,
    online_options=None,
):
    """Run one causal online baseline over the requested budget fractions."""
    if method not in ONLINE_METHODS:
        raise ValueError(
            f"method must be one of {ONLINE_METHODS}, got {method!r}"
        )
    online_options = dict(online_options or {})
    total_paid_cost = float(sum(ctx["paid_costs"]))
    n_train = len(ctx["train_dataset"])
    n_test = len(ctx["test_dataset"])
    n_total = n_train + n_test
    train_fraction = n_train / n_total

    rows = []
    for budget_fraction in budget_fractions:
        budget_fraction = float(budget_fraction)
        if not 0 <= budget_fraction <= 1:
            raise ValueError(
                f"budget fractions must lie in [0, 1], got {budget_fraction}"
            )
        total_budget = budget_fraction * n_total * total_paid_cost
        train_total = train_fraction * total_budget
        test_total = total_budget - train_total
        train_budget = train_total / n_train
        test_budget = test_total / n_test

        print(
            f"\n{'#' * 60}\n"
            f"# method={method}, budget_fraction={budget_fraction:.2f}, "
            f"train_budget={train_budget:.4f}, test_budget={test_budget:.4f}\n"
            f"{'#' * 60}"
        )
        started = time.time()
        result = _run_online_baseline(
            ctx,
            device,
            method,
            train_budget=train_budget,
            test_budget=test_budget,
            seed=seed,
            **online_options,
        )
        rows.append({
            "dataset": ctx["dataset_name"],
            "split_mode": ctx["split_mode"],
            "seed": seed,
            "num_modalities": int(ctx["num_modalities"]),
            "num_classes": int(ctx["d_out"]),
            "budget_fraction": budget_fraction,
            "total_budget": total_budget,
            "train_total": train_total,
            "test_total": test_total,
            "train_budget_per_sample": train_budget,
            "test_budget_per_sample": test_budget,
            "trial_wall_time_sec": time.time() - started,
            **result,
        })
    return rows


def run_multi_seed_sweep(
    dataset_name,
    device,
    method,
    budget_fractions,
    seeds,
    *,
    data_path=None,
    max_modalities=None,
    nsamples=1000,
    n_views=None,
    max_samples=None,
    sampling=DEFAULT_SAMPLING_MODE,
    image_pool_side=None,
    image_data_home=None,
    num_classes=None,
    synthetic_seed=SYNTHETIC_SEED,
    online_options=None,
):
    """Rebuild the split and model for every seed, then run the budget sweep."""
    rows = []
    for seed in seeds:
        print(f"\n{'=' * 70}\n=== SEED {seed} ===\n{'=' * 70}")
        torch.manual_seed(seed)
        ctx = build_experiment(
            dataset_name,
            device,
            data_path=data_path,
            max_modalities=max_modalities,
            split_seed=seed,
            synthetic_seed=synthetic_seed,
            nsamples=nsamples,
            n_views=n_views,
            max_samples=max_samples,
            sampling=sampling,
            image_pool_side=image_pool_side,
            image_data_home=image_data_home,
            num_classes=num_classes,
        )
        rows.extend(
            run_budget_sweep(
                ctx,
                device,
                method,
                budget_fractions,
                seed=seed,
                online_options=online_options,
            )
        )
    return rows


def _comma_separated_numbers(value, cast, label):
    try:
        values = [cast(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"{label} must be a comma-separated list"
        ) from exc
    if not values:
        raise argparse.ArgumentTypeError(f"{label} cannot be empty")
    return values


def _online_options(args, method, ol_hidden_sizes):
    learning_rate = (
        args.cwcf_learning_rate
        if method == "online_cwcf"
        else args.online_learning_rate
    )
    return {
        "learning_rate": learning_rate,
        "updates_per_sample": args.online_updates_per_sample,
        "pvae_samples": args.online_pvae_samples,
        "cost_normalized": not args.eddi_no_cost_normalization,
        "cae_entropy_weight": args.online_cae_entropy_weight,
        "cae_gamma": args.online_cae_gamma,
        "aaco_k_neighbors": args.aaco_k_neighbors,
        "aaco_acquisition_cost": args.aaco_acquisition_cost,
        "aaco_max_candidates": args.aaco_max_candidates,
        "aaco_exact_max_paid": args.aaco_exact_max_paid,
        "gdfs_entropy_weight": args.gdfs_entropy_weight,
        "pt_importance_decay": args.pt_importance_decay,
        "jafa_embedding_size": args.jafa_embedding_size,
        "jafa_hidden_size": args.jafa_hidden_size,
        "jafa_memory_size": args.jafa_memory_size,
        "jafa_processing_steps": args.jafa_processing_steps,
        "jafa_acquisition_cost_weight": args.jafa_acquisition_cost_weight,
        "jafa_epsilon_start": args.jafa_epsilon_start,
        "jafa_epsilon_end": args.jafa_epsilon_end,
        "jafa_max_candidates": args.jafa_max_candidates,
        "jafa_exact_max_paid": args.jafa_exact_max_paid,
        "ol_hidden_sizes": ol_hidden_sizes,
        "ol_dropout": args.ol_dropout,
        "ol_use_feature_mask": args.ol_use_feature_mask,
        "ol_reward_method": args.ol_reward_method,
        "ol_mcdrop_samples": args.ol_mcdrop_samples,
        "ol_epsilon_start": args.ol_epsilon_start,
        "ol_epsilon_end": args.ol_epsilon_end,
        "ol_max_candidates": args.ol_max_candidates,
        "ol_exact_max_paid": args.ol_exact_max_paid,
        "cwcf_hidden_size": args.cwcf_hidden_size,
        "cwcf_hidden_layers": args.cwcf_hidden_layers,
        "cwcf_cost_weight": args.cwcf_cost_weight,
        "cwcf_epsilon_start": args.cwcf_epsilon_start,
        "cwcf_epsilon_end": args.cwcf_epsilon_end,
        "cwcf_max_candidates": args.cwcf_max_candidates,
        "cwcf_exact_max_paid": args.cwcf_exact_max_paid,
        "cwcf_grad_clip": args.cwcf_grad_clip,
        "lambda_max": args.lambda_max,
        "step_size": args.step_size,
        "linear_classifier": args.linear_classifier,
    }


def _validate_args(parser, args, ol_hidden_sizes):
    positive = {
        "--online-learning-rate": args.online_learning_rate,
        "--online-updates-per-sample": args.online_updates_per_sample,
        "--online-pvae-samples": args.online_pvae_samples,
        "--online-cae-gamma": args.online_cae_gamma,
        "--aaco-k-neighbors": args.aaco_k_neighbors,
        "--aaco-max-candidates": args.aaco_max_candidates,
        "--aaco-exact-max-paid": args.aaco_exact_max_paid,
        "--jafa-embedding-size": args.jafa_embedding_size,
        "--jafa-hidden-size": args.jafa_hidden_size,
        "--jafa-memory-size": args.jafa_memory_size,
        "--jafa-processing-steps": args.jafa_processing_steps,
        "--jafa-max-candidates": args.jafa_max_candidates,
        "--jafa-exact-max-paid": args.jafa_exact_max_paid,
        "--ol-mcdrop-samples": args.ol_mcdrop_samples,
        "--ol-max-candidates": args.ol_max_candidates,
        "--ol-exact-max-paid": args.ol_exact_max_paid,
        "--cwcf-learning-rate": args.cwcf_learning_rate,
        "--cwcf-hidden-size": args.cwcf_hidden_size,
        "--cwcf-hidden-layers": args.cwcf_hidden_layers,
        "--cwcf-max-candidates": args.cwcf_max_candidates,
        "--cwcf-exact-max-paid": args.cwcf_exact_max_paid,
        "--cwcf-grad-clip": args.cwcf_grad_clip,
    }
    invalid = [name for name, value in positive.items() if value <= 0]
    if invalid:
        parser.error("these options must be positive: " + ", ".join(invalid))
    if not ol_hidden_sizes or min(ol_hidden_sizes) < 1:
        parser.error("--ol-hidden-sizes must contain positive integers")
    if args.online_cae_entropy_weight < 0:
        parser.error("--online-cae-entropy-weight must be nonnegative")
    if args.aaco_acquisition_cost < 0 or args.gdfs_entropy_weight < 0:
        parser.error("AACO/GDFS cost and entropy weights must be nonnegative")
    if args.jafa_acquisition_cost_weight < 0 or args.cwcf_cost_weight < 0:
        parser.error("JAFA/CwCF cost weights must be nonnegative")
    if not 0 <= args.ol_dropout < 1:
        parser.error("--ol-dropout must lie in [0, 1)")
    if not 0 <= args.pt_importance_decay < 1:
        parser.error("--pt-importance-decay must lie in [0, 1)")
    for label, start, end in (
        ("JAFA", args.jafa_epsilon_start, args.jafa_epsilon_end),
        ("OL", args.ol_epsilon_start, args.ol_epsilon_end),
        ("CwCF", args.cwcf_epsilon_start, args.cwcf_epsilon_end),
    ):
        if not 0 <= end <= start <= 1:
            parser.error(
                f"{label} epsilon values must satisfy 0 <= end <= start <= 1"
            )


def main():
    parser = argparse.ArgumentParser(
        description="Run causal predict-then-update AFA baselines."
    )
    parser.add_argument("--dataset", choices=ALL_DATASETS, default="ckd")
    parser.add_argument("--data-path", default=None)
    parser.add_argument(
        "--max-modalities",
        default="all",
        help="'all' or the number of real-data modalities to retain.",
    )
    parser.add_argument("--n-views", type=int, default=SYNTHETIC_N_VIEWS)
    parser.add_argument(
        "--n-samples", "--nsamples", dest="nsamples", type=int, default=1000
    )
    parser.add_argument("--synthetic-seed", type=int, default=SYNTHETIC_SEED)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument(
        "--sampling", choices=SAMPLING_MODES, default=DEFAULT_SAMPLING_MODE
    )
    parser.add_argument("--num-classes", type=int, default=None)
    parser.add_argument("--image-pool-side", type=int, default=None)
    parser.add_argument("--image-cache-dir", default=None)
    parser.add_argument(
        "--method",
        choices=(*ONLINE_METHODS, "online_all"),
        default="online_all",
    )
    parser.add_argument(
        "--budget-fractions", default="0.1,0.3,0.5,0.7,0.9"
    )
    parser.add_argument(
        "--seeds", default="42,43,44,45,46,47,48,49,50,51"
    )
    parser.add_argument("--output-csv", default=None)
    parser.add_argument("--lambda-max", type=float, default=10.0)
    parser.add_argument("--step-size", type=float, default=1.0)
    parser.add_argument("--online-learning-rate", type=float, default=1e-3)
    parser.add_argument("--online-updates-per-sample", type=int, default=1)
    parser.add_argument("--online-pvae-samples", type=int, default=128)
    parser.add_argument("--online-cae-entropy-weight", type=float, default=1e-2)
    parser.add_argument("--online-cae-gamma", type=float, default=0.2)
    parser.add_argument("--eddi-no-cost-normalization", action="store_true")
    parser.add_argument("--aaco-k-neighbors", type=int, default=5)
    parser.add_argument("--aaco-acquisition-cost", type=float, default=0.05)
    parser.add_argument("--aaco-max-candidates", type=int, default=100)
    parser.add_argument("--aaco-exact-max-paid", type=int, default=12)
    parser.add_argument("--gdfs-entropy-weight", type=float, default=1e-3)
    parser.add_argument("--jafa-embedding-size", type=int, default=16)
    parser.add_argument("--jafa-hidden-size", type=int, default=32)
    parser.add_argument("--jafa-memory-size", type=int, default=16)
    parser.add_argument("--jafa-processing-steps", type=int, default=5)
    parser.add_argument("--jafa-acquisition-cost-weight", type=float, default=0.05)
    parser.add_argument("--jafa-epsilon-start", type=float, default=1.0)
    parser.add_argument("--jafa-epsilon-end", type=float, default=0.1)
    parser.add_argument("--jafa-max-candidates", type=int, default=512)
    parser.add_argument("--jafa-exact-max-paid", type=int, default=12)
    parser.add_argument("--ol-hidden-sizes", default="64,32,16")
    parser.add_argument("--ol-dropout", type=float, default=0.5)
    parser.add_argument("--ol-use-feature-mask", action="store_true")
    parser.add_argument(
        "--ol-reward-method",
        choices=("softmax", "Bayesian-L1", "Bayesian-L2"),
        default="Bayesian-L1",
    )
    parser.add_argument("--ol-mcdrop-samples", type=int, default=100)
    parser.add_argument("--ol-epsilon-start", type=float, default=1.0)
    parser.add_argument("--ol-epsilon-end", type=float, default=0.1)
    parser.add_argument("--ol-max-candidates", type=int, default=512)
    parser.add_argument("--ol-exact-max-paid", type=int, default=12)
    parser.add_argument("--cwcf-hidden-size", type=int, default=128)
    parser.add_argument("--cwcf-hidden-layers", type=int, default=3)
    parser.add_argument("--cwcf-cost-weight", type=float, default=1.0)
    parser.add_argument("--cwcf-learning-rate", type=float, default=5e-4)
    parser.add_argument("--cwcf-epsilon-start", type=float, default=1.0)
    parser.add_argument("--cwcf-epsilon-end", type=float, default=0.1)
    parser.add_argument("--cwcf-max-candidates", type=int, default=512)
    parser.add_argument("--cwcf-exact-max-paid", type=int, default=12)
    parser.add_argument("--cwcf-grad-clip", type=float, default=1.0)
    parser.add_argument("--pt-importance-decay", type=float, default=0.9)
    parser.add_argument(
        "--mlp-classifier",
        action="store_false",
        dest="linear_classifier",
        default=True,
    )
    args = parser.parse_args()

    budget_fractions = _comma_separated_numbers(
        args.budget_fractions, float, "budget fractions"
    )
    seeds = _comma_separated_numbers(args.seeds, int, "seeds")
    try:
        ol_hidden_sizes = tuple(
            int(value.strip())
            for value in args.ol_hidden_sizes.split(",")
            if value.strip()
        )
    except ValueError:
        parser.error("--ol-hidden-sizes must be comma-separated integers")
    try:
        max_modalities = (
            None
            if str(args.max_modalities).lower() == "all"
            else int(args.max_modalities)
        )
    except ValueError:
        parser.error("--max-modalities must be 'all' or an integer")
    if any(not 0 <= value <= 1 for value in budget_fractions):
        parser.error("--budget-fractions values must lie in [0, 1]")
    _validate_args(parser, args, ol_hidden_sizes)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    methods = ONLINE_METHODS if args.method == "online_all" else (args.method,)
    all_results = []
    for method in methods:
        all_results.extend(
            run_multi_seed_sweep(
                args.dataset,
                device,
                method,
                budget_fractions,
                seeds,
                data_path=args.data_path,
                max_modalities=max_modalities,
                nsamples=args.nsamples,
                n_views=args.n_views,
                max_samples=args.max_samples,
                sampling=args.sampling,
                image_pool_side=args.image_pool_side,
                image_data_home=args.image_cache_dir,
                num_classes=args.num_classes,
                synthetic_seed=args.synthetic_seed,
                online_options=_online_options(args, method, ol_hidden_sizes),
            )
        )

    from pathlib import Path
    import pandas as pd

    if args.output_csv:
        output_path = Path(args.output_csv)
    else:
        dimensions = {
            (int(row["num_modalities"]), int(row["num_classes"]))
            for row in all_results
        }
        if len(dimensions) != 1:
            raise RuntimeError(
                f"result rows disagree on dimensions: {sorted(dimensions)}"
            )
        n_modalities, n_classes = dimensions.pop()
        output_path = Path("results") / baseline_output_filename(
            args.method,
            args.dataset,
            n_modalities,
            len(seeds),
            n_classes,
            linear_classifier=args.linear_classifier,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    flat_rows = [
        {
            key: str(value) if isinstance(value, (list, dict)) else value
            for key, value in row.items()
        }
        for row in all_results
    ]
    pd.DataFrame(flat_rows).to_csv(output_path, index=False)
    print(f"\nSaved {len(all_results)} result row(s) to {output_path}")


if __name__ == "__main__":
    main()
