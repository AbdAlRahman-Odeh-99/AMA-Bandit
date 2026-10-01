"""Dataset setup, metrics, and result naming shared by the OL runner."""

from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import f1_score, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

from core.datasets import (
    DEFAULT_SAMPLING_MODE, DEFAULT_IMAGE_POOL_SIDE, SYNTHETIC_DATASETS,
    SYNTHETIC_N_CLASSES, SYNTHETIC_N_VIEWS, SYNTHETIC_SEED,
    generate_modality_costs_heterogeneous, load_dataset_as_numpy, split_dataset,
)
from core.utils import MaskLayerGrouped


def baseline_output_filename(method, dataset, n_modalities, n_seeds,
                             n_classes, linear_classifier=True):
    """Return a V/T/K filename for an OL result file."""
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


def build_experiment(dataset_name, device, data_path=None, max_modalities=None, split_seed=None,
                      synthetic_seed=SYNTHETIC_SEED,
                      nsamples=1000, n_views=None, max_samples=None,
                      sampling=DEFAULT_SAMPLING_MODE, image_pool_side=None,
                      image_data_home=None, num_classes=None):
    """Build OL's train/validation/test data, mask layer, and modality costs.

    Real data can be limited to its first ``max_modalities`` features.
    Synthetic data is generated with ``n_views`` modalities instead.
    ``split_seed`` selects the split, while ``synthetic_seed`` selects the
    synthetic generator's means and samples.
    """
    is_synthetic = dataset_name in SYNTHETIC_DATASETS
    synthetic_n_views = n_views if n_views is not None else SYNTHETIC_N_VIEWS
    synthetic_n_classes = num_classes if num_classes is not None else SYNTHETIC_N_CLASSES
    pool = image_pool_side if image_pool_side is not None else DEFAULT_IMAGE_POOL_SIDE

    # Use the shared loader and a 60/20/20 split so OL has validation data.
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
