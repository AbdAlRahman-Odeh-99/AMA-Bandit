"""
Standalone loaders for AFA-Benchmark datasets, replicating
afabench/datasets/datasets.py's preprocessing without needing their Hydra /
Snakemake / bundle scaffolding.

BINARY (2-class) tabular:
  - "ckd"             CKDDataset            UCI id=336, auto-fetched
  - "bank_marketing"  BankMarketingDataset  UCI id=222, auto-fetched
  - "actg175"         ACTG175Dataset        UCI id=890, auto-fetched
  - "miniboone"       MiniBooNEDataset      local CSV required (no auto-fetch
                                            in the original code either)
  - "physionet"       PhysionetDataset      local CSV required (data-use
                                            agreement)

MULTICLASS (K>2) -- runnable on Adaptive and Two-stage:
  - "diabetes"        3-class NHANES-derived diabetes status. LOCAL CSV
                      required (3-class target as the LAST column) -- the
                      exact AFA preprocessing isn't publicly auto-fetchable.
                      Goes through the SAME tabular pipeline as the UCI
                      datasets (now class-count-inferring, not 2-hardcoded).
  - "mnist"           10-class handwritten digits, fetched via OpenML
                      (sklearn.datasets.fetch_openml, id=554).
  - "fashion_mnist"   10-class clothing items, OpenML id=40996.
                      Both images are treated as TABULAR pixels ([0,1]-scaled)
                      with OPTIONAL block-average downsampling from 28x28 to
                      x*x (see average_pool_images / the image_pool_side arg).

SYNTHETIC GMM: "synthetic_asymmetric" / "synthetic_symmetric" (2-class),
"synthetic" (K-class).

The tabular datasets share the original code's pipeline: label-encode
categoricals (non-missing entries only), coerce to numeric, mean-impute,
z-normalize (population stats, ddof=0), one-hot the (label-encoded) target.
Implemented once as _preprocess_features, reused by all tabular loaders.

Requires: pip install ucimlrepo pandas torch scikit-learn numpy
(MNIST/FashionMNIST use scikit-learn's fetch_openml, which caches the
download under ~/scikit_learn_data.)
"""

from __future__ import annotations

import random
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.datasets import make_blobs
from sklearn.preprocessing import LabelEncoder
from ucimlrepo import fetch_ucirepo

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------
# Split config
SPLIT_RATIO = {"train": 0.6, "val": 0.2, "test": 0.2}
SPLIT_SEED = 42
SPLIT_MODES = ("80-20", "60-20-20")
SAMPLING_MODES = ("balanced", "stratified", "random")
DEFAULT_SAMPLING_MODE = "balanced"
# Cost config 
DISTRIBUTION = "lognormal"  # or "uniform"
LOGNORMAL_MEAN = 0.0
LOGNORMAL_SIGMA = 1.0
UNIFORM_LOW = 0.5
UNIFORM_HIGH = 1.5
# Synthetic data config
SYNTHETIC_N_SAMPLES = 1000
SYNTHETIC_N_VIEWS = 10
SYNTHETIC_SEED = 42
SYNTHETIC_MEAN_SCALE = 2.0  # asymmetric/symmetric means ~ Uniform(0, mean_scale) per view
SYNTHETIC_CLUSTER_STD = 1.0
SYNTHETIC_N_CLASSES = 4
# Image dataset config (MNIST / FashionMNIST -- native 28x28 = 784 pixels)
IMAGE_SIDE = 28
DEFAULT_IMAGE_POOL_SIDE = 7  # default pooled side; 7x7 = 49 features (block-avg of the 28x28)
# fetch_openml's own default cache dir is ~/scikit_learn_data -- on clusters
# like NCI Gadi, $HOME has a tiny quota (a few GB) that's often already
# exhausted, while the project's actual data lives on gdata/scratch (much
# larger quota). Default the cache to a RELATIVE "data/openml_cache" folder
# instead (same convention as DEFAULT_PATHS' "data/*.csv" -- relative to
# wherever the process's cwd is, which on NCI jobs is the gdata project dir
# after `cd`, per run_single.pbs) so the very first MNIST/FashionMNIST run
# doesn't blow the home quota. Override via the data_home arg/CLI flag, or
# by pre-setting the sklearn-native $SCIKIT_LEARN_DATA env var yourself.
DEFAULT_OPENML_DATA_HOME = "data/openml_cache"


# --------------------------------------------------------------------------
# Preprocessing helper (afabench/datasets/datasets.py::_z_normalize)
# --------------------------------------------------------------------------
def _z_normalize(features_df: pd.DataFrame) -> pd.DataFrame:
    """Feature-wise Z-normalization using population statistics (ddof=0).

    Zero std is replaced with 1.0 to avoid division by zero.
    """
    means = features_df.mean()
    stds = features_df.std(ddof=0).replace(0, 1.0)
    return (features_df - means) / stds


# --------------------------------------------------------------------------
# Train/val/test split (scripts/dataset_generation/generate_dataset.py)
# --------------------------------------------------------------------------
def split_dataset(
    n_samples: int,
    split_ratio: dict[str, float] = SPLIT_RATIO,
    seed: int = SPLIT_SEED,
) -> tuple[list[int], list[int], list[int]]:
    """Shuffled train/val/test index split using Python's random.Random(seed)."""
    train_size = int(split_ratio["train"] * n_samples)
    val_size = int(split_ratio["val"] * n_samples)

    all_indices = list(range(n_samples))
    random.Random(seed).shuffle(all_indices)

    train_indices = all_indices[:train_size]
    val_indices = all_indices[train_size : train_size + val_size]
    test_indices = all_indices[train_size + val_size :]
    return train_indices, val_indices, test_indices


def split_by_mode(
    n_samples: int,
    split_mode: str = "80-20",
    seed: int = SPLIT_SEED,
) -> tuple[list[int], list[int], list[int]]:
    """Return train, validation, and test indices for one comparison mode.

    ``80-20`` is for comparing adaptive and two-stage directly. Its
    validation list is empty. ``60-20-20`` is for comparisons against EDDI
    or DIME and uses the same held-out test indices as those baselines.
    Both modes use the same deterministic ``random.Random(seed)`` shuffle.
    """
    if split_mode not in SPLIT_MODES:
        raise ValueError(f"split_mode must be one of {SPLIT_MODES}, got {split_mode!r}")
    if split_mode == "60-20-20":
        return split_dataset(n_samples, seed=seed)

    train_size = int(0.8 * n_samples)
    all_indices = list(range(n_samples))
    random.Random(seed).shuffle(all_indices)
    return all_indices[:train_size], [], all_indices[train_size:]


# --------------------------------------------------------------------------
# Feature acquisition costs (port of scripts/misc/generate_feature_costs.py)
# --------------------------------------------------------------------------


def generate_feature_costs(
    n_features: int,
    distribution: str = DISTRIBUTION,
    lognormal_mean: float = LOGNORMAL_MEAN,
    lognormal_sigma: float = LOGNORMAL_SIGMA,
    uniform_low: float = UNIFORM_LOW,
    uniform_high: float = UNIFORM_HIGH,
    seed: int | None = None,
    dataset_name: str = "ckd",
) -> np.ndarray:
    """
    Port of scripts/misc/generate_feature_costs.py::_generate_costs.

    Generate synthetic per-feature acquisition costs, rescaled to mean 1.
    Seed defaults to crc32(dataset_name) if not given, matching the original
    script's reproducible-by-name behavior.
    """
    if seed is None:
        seed = zlib.crc32(dataset_name.encode("utf-8")) & 0xFFFFFFFF
    rng = np.random.default_rng(seed)

    if distribution == "lognormal":
        costs = rng.lognormal(mean=lognormal_mean, sigma=lognormal_sigma, size=n_features)
    else:
        costs = rng.uniform(low=uniform_low, high=uniform_high, size=n_features)

    return costs / costs.mean()


def generate_modality_costs_heterogeneous(
    n_features: int = 24,
    distribution: str = DISTRIBUTION,
    lognormal_mean: float = LOGNORMAL_MEAN,
    lognormal_sigma: float = LOGNORMAL_SIGMA,
    uniform_low: float = UNIFORM_LOW,
    uniform_high: float = UNIFORM_HIGH,
    seed: int | None = None,
    dataset_name: str = "ckd",
) -> list[float]:
    """
    Build a feature_costs list over any dataset's real features (no
    synthetic column added) with the FIRST feature forced free (cost 0)
    and genuinely heterogeneous per-feature costs for the rest, drawn with
    ``generate_feature_costs``. The online CAE selector supports these
    heterogeneous paid costs directly.

    dataset_name only affects the crc32-derived seed (so different
    datasets get different, reproducible cost draws); n_features should
    be set to that dataset's actual feature count. AFA-Benchmark itself
    has no built-in cost model for any of the 5 binary datasets here
    (only CubeNonUniformCosts/CubeNM do) -- this heterogeneous cost
    generation is a choice layered on top for all of them equally.
    """
    n_paid = n_features - 1
    paid_costs = generate_feature_costs(
        n_features=n_paid,
        distribution=distribution,
        lognormal_mean=lognormal_mean,
        lognormal_sigma=lognormal_sigma,
        uniform_low=uniform_low,
        uniform_high=uniform_high,
        seed=seed,
        dataset_name=dataset_name,
    )
    return [0.0] + paid_costs.tolist()

# --------------------------------------------------------------------------
# Synthetic 2-class GMM generation (symmetric / asymmetric)
# --------------------------------------------------------------------------


def generate_synthetic_asymmetric(
    rng: np.random.Generator,
    n_samples: int = SYNTHETIC_N_SAMPLES,
    n_views: int = SYNTHETIC_N_VIEWS,
    seed: int = SYNTHETIC_SEED,
    mean_scale: float = SYNTHETIC_MEAN_SCALE,
    cluster_std: float = SYNTHETIC_CLUSTER_STD,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Heterogeneous-mean 2-class GMM used by both proposed methods:
    each (class, view) mean is drawn i.i.d. Uniform(0, mean_scale), with
    no relationship between the two classes' per-view means (hence
    "asymmetric" -- unlike generate_synthetic_symmetric below, class 1's
    means are NOT a mirror of class 0's).

    Returns:
        X: float64 ndarray [n_samples, n_views]
        y: int64 ndarray [n_samples], values in {0, 1}
        means: float64 ndarray [2, n_views], the TRUE generating means
               (kept for reference/plotting -- not part of the standard
               load_binary_afa_dataset return signature).
    """
    means = rng.random(size=(2, n_views)) * mean_scale
    X, y = make_blobs(
        n_samples, n_features=n_views, centers=means,
        cluster_std=cluster_std, random_state=seed,
    )
    return X.astype(np.float64), y.astype(np.int64), means


def generate_synthetic_symmetric(
    rng: np.random.Generator,
    n_samples: int = SYNTHETIC_N_SAMPLES,
    n_views: int = SYNTHETIC_N_VIEWS,
    seed: int = SYNTHETIC_SEED,
    mean_scale: float = SYNTHETIC_MEAN_SCALE,
    cluster_std: float = SYNTHETIC_CLUSTER_STD,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Symmetric binary GMM: a single per-view mean-magnitude vector is drawn
    i.i.d. Uniform(0, mean_scale), class 0 is centered at +that vector and
    class 1 at the MIRRORED -that vector. Matches the "symmetric binary
    GMM" convention from the earlier theoretical work (unbiased mean
    estimation for symmetric binary GMMs, sandwich bounds) and the
    historical gmm_2class_*_symmetric.py scripts.

    Returns the same (X, y, means) shape as generate_synthetic_asymmetric.
    """
    half_gap = rng.random(size=(n_views,)) * mean_scale
    means = np.stack([half_gap, -half_gap], axis=0)
    X, y = make_blobs(
        n_samples, n_features=n_views, centers=means,
        cluster_std=cluster_std, random_state=seed,
    )
    return X.astype(np.float64), y.astype(np.int64), means


def generate_synthetic_multiclass(
    rng: np.random.Generator,
    n_samples: int = SYNTHETIC_N_SAMPLES,
    n_views: int = SYNTHETIC_N_VIEWS,
    seed: int = SYNTHETIC_SEED,
    mean_scale: float = SYNTHETIC_MEAN_SCALE,
    cluster_std: float = SYNTHETIC_CLUSTER_STD,
    n_classes: int = SYNTHETIC_N_CLASSES,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    K-class generalization of generate_synthetic_asymmetric: each
    (class, view) mean is drawn i.i.d. Uniform(0, mean_scale) for
    n_classes classes (n_classes=2 reduces exactly to the asymmetric
    binary generator). Ported from the multiclass Colab notebook's
    generate_data() (gmm_afa_supervised_full_vs_bandit_feedback), with
    its snrdb knob replaced by this codebase's mean_scale convention --
    mean_scale = 10**(snrdb/20), see SYNTHETIC_N_CLASSES's comment.

    Returns the same (X, y, means) contract as the binary generators,
    with y in {0, ..., n_classes-1} and means shaped [n_classes, n_views].
    """
    means = rng.random(size=(n_classes, n_views)) * mean_scale
    X, y = make_blobs(
        n_samples, n_features=n_views, centers=means,
        cluster_std=cluster_std, random_state=seed,
    )
    return X.astype(np.float64), y.astype(np.int64), means


# Registry name -> generator, used by load_binary_afa_dataset and by
# ALL_DATASETS below so synthetic datasets are selectable everywhere a
# real dataset name would be (e.g. --dataset synthetic_asymmetric).
SYNTHETIC_GENERATORS = {
    "synthetic_asymmetric": generate_synthetic_asymmetric,
    "synthetic_symmetric": generate_synthetic_symmetric,
    "synthetic": generate_synthetic_multiclass,
}
SYNTHETIC_DATASETS = tuple(SYNTHETIC_GENERATORS.keys())

# The subset of SYNTHETIC_DATASETS whose generator accepts an n_classes
# kwarg (and can therefore produce labels outside {0, 1}). The BINARY
# runners use this to decide whether to pass an explicit class count.
MULTICLASS_SYNTHETIC_DATASETS = ("synthetic",)


# --------------------------------------------------------------------------
# Image datasets (MNIST / FashionMNIST) -- treated as TABULAR, one feature
# per (optionally block-averaged) pixel, exactly as AFA-Benchmark does.
# Fetched via sklearn.datasets.fetch_openml (sklearn caches the download in
# ~/scikit_learn_data, so there's no per-dataset CSV to manage the way the
# UCI datasets have). Both are 10-class.
# --------------------------------------------------------------------------

# OpenML data_ids: mnist_784 == 554, Fashion-MNIST == 40996.
IMAGE_OPENML_IDS = {"mnist": 554, "fashion_mnist": 40996}
IMAGE_DATASETS = tuple(IMAGE_OPENML_IDS.keys())


def average_pool_images(
    X: np.ndarray, orig_side: int = IMAGE_SIDE, target_side: int = DEFAULT_IMAGE_POOL_SIDE,
) -> np.ndarray:
    """Block-average dimensionality reduction for flattened square images.

    Reshapes each flat row of X (shape (N, orig_side**2), row-major
    r*orig_side + c) back to an (orig_side, orig_side) image, partitions
    BOTH axes into `target_side` contiguous NON-OVERLAPPING groups via
    np.array_split (group sizes differ by at most 1, and every pixel
    belongs to exactly one group), and averages each resulting block --
    yielding a (target_side, target_side) image re-flattened to
    (N, target_side**2).

    Works for ANY 1 <= target_side <= orig_side. When target_side divides
    orig_side (for 28: target_side in {1, 2, 4, 7, 14, 28}) the blocks are
    perfectly uniform (e.g. 28->14 is exact 2x2 averaging, 28->7 exact
    4x4); otherwise the split is as even as possible (e.g. 28->5 gives row/
    column group sizes [6, 6, 6, 5, 5]). target_side == orig_side returns X
    unchanged (each block is a single pixel).

    This is the function to pick your reduced side `x` with: X of 28x28
    pixels -> x*x features.
    """
    X = np.asarray(X, dtype=np.float64)
    n = X.shape[0]
    if X.shape[1] != orig_side * orig_side:
        raise ValueError(
            f"average_pool_images expected {orig_side * orig_side} columns "
            f"({orig_side}x{orig_side}), got {X.shape[1]}"
        )
    if not (1 <= target_side <= orig_side):
        raise ValueError(
            f"target_side must be in [1, {orig_side}], got {target_side}"
        )
    imgs = X.reshape(n, orig_side, orig_side)
    groups = np.array_split(np.arange(orig_side), target_side)
    out = np.empty((n, target_side, target_side), dtype=np.float64)
    for i, rg in enumerate(groups):
        for j, cg in enumerate(groups):
            block = imgs[:, rg[0]:rg[-1] + 1, cg[0]:cg[-1] + 1]
            out[:, i, j] = block.mean(axis=(1, 2))
    return out.reshape(n, target_side * target_side)


def _load_image_dataset(
    name: str, image_pool_side: int | None = DEFAULT_IMAGE_POOL_SIDE,
    data_home: str | Path | None = None,
) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    """Load MNIST / FashionMNIST as a tabular pixel dataset via OpenML.

    Pixels are scaled to [0, 1] (divide by 255 -- the canonical "normalized
    pixel features" convention; NOT z-normalized, so the many all-zero
    border pixels stay 0 instead of becoming NaN/spurious). If
    image_pool_side is set (and != 28), the 28x28 image is block-averaged
    down to image_pool_side x image_pool_side via average_pool_images
    BEFORE flattening, so the feature count is image_pool_side**2 (e.g. 49
    for the default 7x7) instead of the full 784. Pass image_pool_side=None
    or 28 to keep all 784 raw pixels.

    data_home: where fetch_openml caches its download. None (default) means
        "use DEFAULT_OPENML_DATA_HOME" -- see that constant's comment for
        why this deliberately does NOT fall back to fetch_openml's own
        default (~/scikit_learn_data), which exceeds $HOME's quota on
        clusters like NCI Gadi. The directory is created if missing (both
        fetch_openml and the explicit mkdir here would otherwise raise on
        a not-yet-existing path).

    Returns the same (features, one-hot labels, feature_names) contract as
    the tabular loaders; labels are 10-class one-hot.
    """
    try:
        from sklearn.datasets import fetch_openml
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "MNIST/FashionMNIST loading needs scikit-learn's fetch_openml "
            "(pip install scikit-learn)."
        ) from e

    resolved_data_home = Path(data_home) if data_home is not None else Path(DEFAULT_OPENML_DATA_HOME)
    resolved_data_home.mkdir(parents=True, exist_ok=True)

    data_id = IMAGE_OPENML_IDS[name]
    bunch = fetch_openml(data_id=data_id, as_frame=False, data_home=str(resolved_data_home))
    X = np.asarray(bunch.data, dtype=np.float64) / 255.0  # (N, 784), -> [0, 1]
    y = np.asarray(bunch.target).astype(np.int64)

    side = IMAGE_SIDE
    if image_pool_side is not None and int(image_pool_side) != IMAGE_SIDE:
        side = int(image_pool_side)
        X = average_pool_images(X, orig_side=IMAGE_SIDE, target_side=side)

    n_classes = int(y.max()) + 1
    features = torch.tensor(X, dtype=torch.float32)
    labels = torch.nn.functional.one_hot(
        torch.tensor(y, dtype=torch.long), num_classes=n_classes,
    ).float()
    feature_names = [f"px_{r}_{c}" for r in range(side) for c in range(side)]
    return features, labels, feature_names

# --------------------------------------------------------------------------
# Per-dataset config
# --------------------------------------------------------------------------

DEFAULT_PATHS = {
    "ckd": Path("data/chronic_kidney_disease.csv"),
    "bank_marketing": Path("data/bank_marketing.csv"),
    "actg175": Path("data/actg175.csv"),
    "miniboone": Path("data/miniboone.csv"),
    "physionet": Path("data/physionet.csv"),
    # Diabetes: NHANES-derived 3-class task (normal / pre-diabetes /
    # diabetes) used by AFA-Benchmark. There is no clean public auto-fetch
    # for the exact preprocessed artifact (it comes from the Opportunistic
    # Learning benchmark), so it's LOCAL-ONLY like miniboone/physionet --
    # place the CSV yourself with the 3-class target as the LAST column.
    "diabetes": Path("data/diabetes.csv"),
}

# Datasets AFA-Benchmark auto-fetches via ucimlrepo (has its own
# _fetch_and_save in the original code).
UCI_AUTO_FETCH_IDS = {
    "ckd": 336,
    "bank_marketing": 222,
    "actg175": 890,
}

# Datasets with NO fetch method here -- local CSV required. Diabetes joins
# miniboone/physionet: its exact AFA preprocessing isn't auto-fetchable.
LOCAL_ONLY_DATASETS = {"miniboone", "physionet", "diabetes"}

# Real MULTICLASS (K>2) datasets. Diabetes is 3-class; MNIST/FashionMNIST
# are 10-class. These run on both proposed methods.
MULTICLASS_REAL_DATASETS = ("diabetes",) + IMAGE_DATASETS

# Known fixed class counts (used for reporting / the run_proposed_methods
# "Num Classes" column, which otherwise can't know K without loading). The
# 5 UCI tabular datasets and both synthetic-binary datasets are 2-class.
DATASET_N_CLASSES = {
    "ckd": 2, "bank_marketing": 2, "actg175": 2, "miniboone": 2, "physionet": 2,
    "diabetes": 3, "mnist": 10, "fashion_mnist": 10,
}

ALL_BINARY_DATASETS = ["ckd", "bank_marketing", "actg175", "miniboone", "physionet"]

# Real + synthetic, for CLI --dataset choices etc. Keep ALL_BINARY_DATASETS
# unchanged (the 5 real BINARY UCI datasets) since some validation logic
# specifically means "one of the 5 real binary AFA-Benchmark datasets".
ALL_DATASETS = (
    ALL_BINARY_DATASETS
    + list(MULTICLASS_REAL_DATASETS)
    + list(SYNTHETIC_DATASETS)
)


# --------------------------------------------------------------------------
# Fetch logic (per-dataset, matching each class's own _fetch_and_save)
# --------------------------------------------------------------------------


def _fetch_and_save_ckd(path: Path) -> None:
    """Matches CKDDataset._fetch_and_save exactly."""
    path.parent.mkdir(parents=True, exist_ok=True)
    ckd_data = fetch_ucirepo(id=336)
    assert ckd_data is not None
    assert ckd_data.data is not None
    features_df = ckd_data.data.features.copy()
    target_df = ckd_data.data.targets.copy()
    target_series = target_df.iloc[:, 0].astype(str).str.strip().str.lower()
    target_series = target_series.map({"ckd": 1, "notckd": 0})
    df_data = features_df.copy()
    df_data["target"] = target_series.to_numpy()
    df_data.to_csv(path, index=False)


def _fetch_and_save_bank_marketing(path: Path) -> None:
    """Matches BankMarketingDataset._fetch_and_save exactly (sep=';')."""
    path.parent.mkdir(parents=True, exist_ok=True)
    bank_data = fetch_ucirepo(id=222)
    assert bank_data is not None
    assert bank_data.data is not None
    df_data = pd.concat([bank_data.data.features, bank_data.data.targets], axis=1)
    df_data.to_csv(path, sep=";", index=False)


def _fetch_and_save_actg175(path: Path) -> None:
    """Matches ACTG175Dataset._fetch_and_save exactly."""
    path.parent.mkdir(parents=True, exist_ok=True)
    actg_data = fetch_ucirepo(id=890)
    assert actg_data is not None
    assert actg_data.data is not None
    features_df = actg_data.data.features.copy()
    target_df = actg_data.data.targets.copy()
    target_series = target_df.iloc[:, 0].astype(int)
    df_data = features_df.copy()
    df_data["target"] = target_series.to_numpy()
    df_data.to_csv(path, index=False)


_FETCH_FUNCS = {
    "ckd": _fetch_and_save_ckd,
    "bank_marketing": _fetch_and_save_bank_marketing,
    "actg175": _fetch_and_save_actg175,
}


# --------------------------------------------------------------------------
# Shared preprocessing (matches CKDDataset/ACTG175Dataset/BankMarketingDataset
# __init__ bodies exactly, aside from CSV-reading quirks handled by the caller)
# --------------------------------------------------------------------------


def _preprocess_features(df_data: pd.DataFrame) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    """
    Shared preprocessing pipeline used by CKD / BankMarketing / ACTG175 in
    the original code: label-encode categoricals (non-missing entries
    only), coerce to numeric, mean-impute, z-normalize.

    Assumes the last column of df_data is the target and everything else is
    a feature. The target is label-encoded to contiguous integer labels
    {0..K-1} and one-hot width K is INFERRED from it, so this handles the
    binary datasets (K=2) AND multiclass ones (e.g. Diabetes, K=3)
    identically -- the only change vs. the original 2-class-hardcoded
    version is that num_classes is inferred rather than fixed at 2.
    """
    features_df = df_data.iloc[:, :-1].copy()
    target_series = df_data.iloc[:, -1]

    # Label-encode categorical columns. Checking dtype == "object" alone
    # misses pandas' newer StringDtype (pandas >= 2.x with infer_string,
    # or pandas 3.x default) -- see is_string_dtype fallback, same fix
    # applied in ckd_dataset.py's load_ckd_dataset.
    for col in features_df.columns:
        is_categorical = features_df[col].dtype == "object" or pd.api.types.is_string_dtype(
            features_df[col]
        )
        if is_categorical:
            le = LabelEncoder()
            mask = features_df[col].notna()
            if mask.any():
                encoded = le.fit_transform(features_df.loc[mask, col].astype(str))
                features_df[col] = features_df[col].astype(object)
                features_df.loc[mask, col] = encoded

    for col in features_df.columns:
        features_df[col] = pd.to_numeric(features_df[col], errors="coerce")
    features_df = features_df.fillna(features_df.mean())
    features_df = _z_normalize(features_df)

    # Label-encode the TARGET too, so arbitrary label sets (e.g. {1,2,3} or
    # string classes) map to contiguous {0..K-1} and one-hot width == K.
    target_encoded = LabelEncoder().fit_transform(target_series.astype(str))
    n_classes = int(target_encoded.max()) + 1

    features = torch.tensor(features_df.values, dtype=torch.float32)
    labels = torch.nn.functional.one_hot(
        torch.tensor(target_encoded, dtype=torch.long),
        num_classes=n_classes,
    ).float()
    feature_names = features_df.columns.tolist()
    return features, labels, feature_names


def _resolve_bank_marketing_target(df_data: pd.DataFrame) -> pd.DataFrame:
    """
    BankMarketingDataset uses column "y" (bank-full.csv variant) or
    "deposit" (bank-additional variant) as target, mapped yes/no -> 1/0,
    and moves it to the LAST column so _preprocess_features can assume
    that convention uniformly (matching CKD/ACTG175's own CSV layout,
    where the fetch step already writes target as the last column --
    BankMarketingDataset's raw fetch doesn't guarantee that, so we do it
    here instead of duplicating a separate preprocessing path for it).
    """
    target_col = "y" if "y" in df_data.columns else "deposit"
    features_df = df_data.drop(columns=[target_col])
    pd.set_option("future.no_silent_downcasting", True)
    target_series = df_data[target_col].replace({"yes": 1, "no": 0}).astype("int64")
    out = features_df.copy()
    out["target"] = target_series.to_numpy()
    return out


# --------------------------------------------------------------------------
# Public loader
# --------------------------------------------------------------------------


def _subsample_max_samples(
    features: torch.Tensor, labels: torch.Tensor, max_samples: int | None,
    sampling: str = DEFAULT_SAMPLING_MODE,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Cap a REAL dataset via reproducible sampling without replacement.

    ``balanced`` (default) makes class counts as equal as the available rows
    permit, redistributing unused minority-class quota to other classes.
    ``stratified`` preserves the original class proportions as closely as
    integer quotas permit. ``random`` retains the original uniform sampling.

    If max_samples is None, or the dataset already has <= max_samples rows,
    this is a no-op. A FIXED seed is independent of the train/test split seed,
    so every experiment seed sees the same capped pool.

    Distinct from the synthetic generators' n_samples parameter (which
    controls how many rows are GENERATED for synthetic_symmetric /
    synthetic_asymmetric) -- max_samples only ever caps something that was
    already loaded from disk.
    """
    if sampling not in SAMPLING_MODES:
        raise ValueError(
            f"sampling must be one of {SAMPLING_MODES}, got {sampling!r}"
        )
    if max_samples is not None and max_samples <= 0:
        raise ValueError(f"max_samples must be positive, got {max_samples}")

    n = features.shape[0]
    if max_samples is None or n <= max_samples:
        return features, labels

    subsample_rng = np.random.default_rng(0)
    if sampling == "random":
        keep_idx = subsample_rng.choice(n, size=max_samples, replace=False)
    else:
        if labels.ndim == 1:
            class_ids = labels.detach().cpu().numpy()
        elif labels.shape[1] == 1:
            class_ids = labels[:, 0].detach().cpu().numpy()
        else:
            class_ids = labels.argmax(dim=1).detach().cpu().numpy()
        classes, counts = np.unique(class_ids, return_counts=True)

        if sampling == "balanced":
            quotas = np.zeros(len(classes), dtype=int)
            remaining = max_samples
            while remaining:
                active = np.flatnonzero(quotas < counts)
                if len(active) == 0:
                    raise RuntimeError("Balanced sampling quota allocation failed.")
                share = remaining // len(active)
                if share == 0:
                    quotas[active[:remaining]] += 1
                    remaining = 0
                else:
                    additions = np.minimum(share, counts[active] - quotas[active])
                    quotas[active] += additions
                    remaining -= int(additions.sum())
        else:
            exact = counts.astype(float) * max_samples / n
            quotas = np.floor(exact).astype(int)
            remaining = max_samples - int(quotas.sum())
            order = np.argsort(-(exact - quotas), kind="stable")
            quotas[order[:remaining]] += 1

        selected = []
        for class_value, quota in zip(classes, quotas):
            candidates = np.flatnonzero(class_ids == class_value)
            selected.append(
                subsample_rng.choice(candidates, size=int(quota), replace=False)
            )
        keep_idx = np.concatenate(selected)

    keep_idx = np.sort(keep_idx)
    keep_idx_t = torch.as_tensor(keep_idx, dtype=torch.long)
    return features[keep_idx_t], labels[keep_idx_t]


def load_binary_afa_dataset(
    name: str,
    path: Path | None = None,
    max_samples: int | None = None,
    sampling: str = DEFAULT_SAMPLING_MODE,
    synthetic_n_samples: int = SYNTHETIC_N_SAMPLES,
    synthetic_n_views: int = SYNTHETIC_N_VIEWS,
    synthetic_seed: int = SYNTHETIC_SEED,
    synthetic_mean_scale: float = SYNTHETIC_MEAN_SCALE,
    synthetic_cluster_std: float = SYNTHETIC_CLUSTER_STD,
    synthetic_n_classes: int = SYNTHETIC_N_CLASSES,
    image_pool_side: int | None = DEFAULT_IMAGE_POOL_SIDE,
    image_data_home: str | Path | None = None,
) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    """
    Load any registered AFA dataset -- same return contract regardless of
    kind, so downstream code doesn't need to care which it got:
      - 5 REAL BINARY tabular datasets: "ckd", "bank_marketing", "actg175",
        "miniboone", "physionet" (2-class).
      - REAL MULTICLASS datasets: "diabetes" (3-class, local CSV),
        "mnist" / "fashion_mnist" (10-class, fetched via OpenML, treated as
        tabular pixels with optional block-average downsampling).
      - SYNTHETIC GMM datasets: "synthetic_asymmetric" / "synthetic_symmetric"
        (2-class) and "synthetic" (K-class).

    max_samples: caps REAL datasets to at most this many rows (reproducible
        subsample using ``sampling`` -- see _subsample_max_samples). If the dataset
        already has fewer rows than max_samples, all of them are used.
        Ignored for synthetic datasets -- see synthetic_n_samples below
        instead, which is a DIFFERENT knob (how many rows to generate, not
        how many to keep from something already loaded).
    synthetic_*: only used when name is one of SYNTHETIC_DATASETS.
    image_pool_side: MNIST/FashionMNIST only -- reduce each 28x28 image to
        image_pool_side x image_pool_side via block-averaging
        (average_pool_images), giving image_pool_side**2 features. Default
        7 (-> 49 features); pass None or 28 to keep all 784 raw pixels.
        Ignored by every non-image dataset.
    image_data_home: MNIST/FashionMNIST only -- where fetch_openml caches
        its download. None (default) means DEFAULT_OPENML_DATA_HOME (a
        RELATIVE "data/openml_cache" folder, NOT fetch_openml's own
        ~/scikit_learn_data default -- see _load_image_dataset's docstring
        for why: on clusters like NCI Gadi, $HOME's quota is tiny and
        already exhausted, so the untouched sklearn default reliably
        fails there). Ignored by every non-image dataset.

    Returns:
        features: FloatTensor [N, n_features]. Z-normalized for real tabular
            datasets; [0,1]-scaled pixels for images; raw make_blobs output
            for synthetic datasets.
        labels:   FloatTensor [N, K], one-hot (K inferred per dataset).
        feature_names: list of feature column names / pixel names / view names.
    """
    if name in IMAGE_DATASETS:
        features, labels, feature_names = _load_image_dataset(
            name, image_pool_side, data_home=image_data_home,
        )
        features, labels = _subsample_max_samples(
            features, labels, max_samples, sampling=sampling
        )
        return features, labels, feature_names

    if name in SYNTHETIC_DATASETS:
        rng = np.random.default_rng(synthetic_seed)
        gen_kwargs = dict(
            n_samples=synthetic_n_samples, n_views=synthetic_n_views,
            seed=synthetic_seed, mean_scale=synthetic_mean_scale,
            cluster_std=synthetic_cluster_std,
        )
        # Only the multiclass generator accepts n_classes; the binary
        # generators are hard-wired to 2 classes and would reject the
        # kwarg (deliberately -- their generative conventions are 2-class
        # by definition, not truncations of a K-class family).
        n_classes = 2
        if name in MULTICLASS_SYNTHETIC_DATASETS:
            n_classes = int(synthetic_n_classes)
            if n_classes < 2:
                msg = f"synthetic_n_classes must be >= 2, got {n_classes}"
                raise ValueError(msg)
            gen_kwargs["n_classes"] = n_classes
        X, y, _means = SYNTHETIC_GENERATORS[name](rng, **gen_kwargs)
        features = torch.tensor(X, dtype=torch.float32)
        labels = torch.nn.functional.one_hot(
            torch.tensor(y, dtype=torch.long), num_classes=n_classes,
        ).float()
        feature_names = [f"view_{i}" for i in range(synthetic_n_views)]
        return features, labels, feature_names

    if name not in DEFAULT_PATHS:
        msg = f"Unknown dataset '{name}'. Choose from: {ALL_DATASETS}"
        raise ValueError(msg)

    resolved_path = path or DEFAULT_PATHS[name]

    if not resolved_path.exists():
        if name in LOCAL_ONLY_DATASETS:
            msg = (
                f"'{name}' has no auto-fetch in AFA-Benchmark's own code either -- "
                f"place the raw CSV at {resolved_path} yourself before loading. "
                f"({'PhysioNet data requires its own data-use agreement.' if name == 'physionet' else 'See the UCI MiniBooNE page for the raw file.'})"
            )
            raise FileNotFoundError(msg)
        _FETCH_FUNCS[name](resolved_path)

    if name == "bank_marketing":
        df_data = pd.read_csv(resolved_path, sep=";")
        df_data = _resolve_bank_marketing_target(df_data)
    else:
        df_data = pd.read_csv(resolved_path)

    features, labels, feature_names = _preprocess_features(df_data)
    features, labels = _subsample_max_samples(
        features, labels, max_samples, sampling=sampling
    )
    return features, labels, feature_names


# --------------------------------------------------------------------------
# Shared numpy-conversion wrapper, used by all three method runners
# (gmm_bandit.gmm_bandit_runner, adaptive.adaptive_runner,
# two_stage.two_stage_runner) so none of them need to depend on each other
# just to load data. Previously this function was duplicated near-verbatim
# in gmm_bandit_runner.py and adaptive_runner.py (and two_stage_runner.py
# imported the bandit copy directly) -- consolidated here instead.
# --------------------------------------------------------------------------

# 2^15 = 32,767 subsets -- the shared tractability threshold for any
# exhaustive-subset-enumeration step. Each runner does its own tailored warning
# print using this constant (the failure mode differs per method), rather
# than load_dataset_as_numpy printing one generic warning itself.
MAX_RECOMMENDED_MODALITIES = 15


def load_dataset_as_numpy(
    dataset_name, max_modalities=None, data_path=None, max_samples=None,
    sampling=DEFAULT_SAMPLING_MODE,
    synthetic_n_samples=SYNTHETIC_N_SAMPLES, synthetic_n_views=SYNTHETIC_N_VIEWS,
    synthetic_seed=SYNTHETIC_SEED, synthetic_mean_scale=SYNTHETIC_MEAN_SCALE,
    synthetic_cluster_std=SYNTHETIC_CLUSTER_STD,
    synthetic_n_classes=SYNTHETIC_N_CLASSES,
    image_pool_side=DEFAULT_IMAGE_POOL_SIDE,
    image_data_home=None,
):
    """
    Load one of the 5 real binary AFA-Benchmark datasets, OR one of the 2
    synthetic 2-class GMM datasets ("synthetic_asymmetric",
    "synthetic_symmetric"), and convert to the plain numpy arrays every
    method's training/inference functions expect. Dataset-kind-agnostic --
    same return contract either way.

    data_path may be either:
      - None: use DEFAULT_PATHS[dataset_name] (relative to cwd)
      - a full path to the dataset's CSV file
      - a directory, in which case DEFAULT_PATHS[dataset_name]'s filename
        (e.g. "physionet.csv") is joined onto it -- this is what lets
        --data-path point at a shared data folder for any dataset.
    Ignored for synthetic datasets (nothing is read from disk).

    max_modalities: for REAL datasets, keeps only the first N feature
        columns. For SYNTHETIC datasets, has NO effect here -- pass the
        desired view count as synthetic_n_views instead (callers that want
        one "--max-modalities" knob to drive both, e.g.
        run_proposed_methods.py, do that mapping themselves before calling
        this function).

    max_samples: caps a REAL dataset to at most this many rows (see
        _subsample_max_samples). ``sampling`` selects balanced (default),
        stratified, or random sampling. Ignored for synthetic datasets -- use
        synthetic_n_samples instead, which controls how many rows get
        GENERATED in the first place (a different knob).
    synthetic_*: only used when dataset_name is in SYNTHETIC_DATASETS.

    synthetic_n_classes: "synthetic" only -- how many classes to
        generate (labels {0..K-1}, one-hot width K). Ignored by every other
        dataset (real datasets are all binary; the binary synthetic
        generators are 2-class by definition).
    image_data_home: MNIST/FashionMNIST only -- see
        load_binary_afa_dataset's docstring / DEFAULT_OPENML_DATA_HOME's
        comment (defaults to a relative "data/openml_cache" folder instead
        of fetch_openml's own ~/scikit_learn_data, which blows $HOME's
        quota on clusters like NCI Gadi).

    Returns:
        X: float64 ndarray [N, nviews]
        Y: int64 ndarray [N], values in {0, 1} (or {0..K-1} for
           "synthetic")
        feature_names: list of str, length nviews

    Does NOT print any "nviews too large" warning -- callers know their
    own tractability threshold and failure mode (get_subset vs
    enumerate_subsets vs generate_view_combinations) and print their own
    message using MAX_RECOMMENDED_MODALITIES above.
    """
    resolved_path = None
    if (dataset_name not in SYNTHETIC_DATASETS
            and dataset_name not in IMAGE_DATASETS
            and data_path is not None):
        resolved_path = Path(data_path)
        if resolved_path.suffix.lower() != ".csv":
            resolved_path = resolved_path / DEFAULT_PATHS[dataset_name].name

    features, labels, feature_names = load_binary_afa_dataset(
        dataset_name, path=resolved_path, max_samples=max_samples, sampling=sampling,
        synthetic_n_samples=synthetic_n_samples, synthetic_n_views=synthetic_n_views,
        synthetic_seed=synthetic_seed, synthetic_mean_scale=synthetic_mean_scale,
        synthetic_cluster_std=synthetic_cluster_std,
        synthetic_n_classes=synthetic_n_classes,
        image_pool_side=image_pool_side,
        image_data_home=image_data_home,
    )

    if max_modalities is not None:
        features = features[:, :max_modalities]
        feature_names = feature_names[:max_modalities]

    X = features.numpy().astype(np.float64)
    Y = labels.argmax(dim=1).numpy().astype(np.int64)
    return X, Y, feature_names


if __name__ == "__main__":
    for dataset_name in ["ckd", "bank_marketing", "actg175"]:
        print(f"\n=== {dataset_name} ===")
        try:
            features, labels, feature_names = load_binary_afa_dataset(dataset_name)
            print(f"features: {tuple(features.shape)}, labels: {tuple(labels.shape)}, "
                  f"{len(feature_names)} feature names")
        except Exception as e:  # noqa: BLE001
            print(f"FAILED: {e}")

    for dataset_name in LOCAL_ONLY_DATASETS:
        print(f"\n=== {dataset_name} (local-only, expected to fail without a local CSV) ===")
        try:
            load_binary_afa_dataset(dataset_name)
        except FileNotFoundError as e:
            print(f"Expected: {e}")
