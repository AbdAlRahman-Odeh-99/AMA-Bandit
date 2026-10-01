"""Run adaptive AFA experiments and write an Excel workbook.

The driver also supports checkpoint recovery and inference from saved states.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# This file is launched by path from scripts/local; make repository packages
# importable while keeping results and logs relative to the launch directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd

from core.datasets import (
    ALL_DATASETS,
    DATASET_N_CLASSES,
    DEFAULT_SAMPLING_MODE,
    DEFAULT_IMAGE_POOL_SIDE,
    MULTICLASS_SYNTHETIC_DATASETS,
    SYNTHETIC_DATASETS,
    SYNTHETIC_MEAN_SCALE,
    SYNTHETIC_N_CLASSES,
    SYNTHETIC_N_SAMPLES,
    SYNTHETIC_N_VIEWS,
    SYNTHETIC_SEED,
    SPLIT_MODES,
    SAMPLING_MODES,
)
from core.excel_utils import serialize_selected_subsets, style_and_save
from core.logging_utils import (
    EXCEL_TIMING_COLUMNS,
    TIMING_COLUMNS,
    excel_timing_columns,
    get_logger,
    read_manifest,
    read_rows_jsonl,
    setup_run,
)
from core.training_state import find_training_states, peek_training_state

import adaptive.adaptive_runner as multiclass_runner

# The driver reads acquisition and reward-update choices from core so their
# CLI options stay in sync with the adaptive implementation.
from core.acquisition_policies import (
    ACQUISITION_MODES,
    ORACLE_ACQUISITION_MODES,
    UCB_BOUNDS,
    uses_empirical_arm_rewards as _uses_empirical_arm_rewards,
    REWARD_UPDATE_SCOPES,
)

log = get_logger("afa.driver")


METHODS = ("adaptive",)
DEFAULT_MAX_MODALITIES = {"adaptive": None}


def _nan(d, key):
    """d[key] if present, else NaN.

    Used throughout the normalizers below. Before per-cell failure
    isolation existed, every row was complete and direct indexing was safe;
    now a failed cell contributes a row carrying only its coordinates and
    its status, and a KeyError there would defeat the entire point of
    surviving the failure.
    """
    v = d.get(key, np.nan)
    return np.nan if v is None else v


# ─────────────────────────────────────────────────────────────────────────
# Normalization: reshape adaptive results into one common
# row schema (a list of flat dicts), ready for pd.DataFrame(rows).
# ─────────────────────────────────────────────────────────────────────────
def _timing_fragment(source, indexer=None):
    """{Excel column name: value} for the timing block.

    `source` is a flat checkpoint row or a per-fraction dict of lists;
    for the latter, `indexer` is the seed index.
    """
    out = {}
    for key in TIMING_COLUMNS:
        col = EXCEL_TIMING_COLUMNS[key]
        if indexer is None:
            out[col] = _nan(source, key)
        else:
            series = source.get(key)
            out[col] = (series[indexer]
                        if series is not None and indexer < len(series) else np.nan)
    return out


def _normalize_frac_keyed_results(results, budget_fractions, seeds, dataset_name, method_name,
                                   feedback=np.nan, n_classes=2,
                                   acquisition=np.nan, alpha_ucb=np.nan):
    """Normalizer for adaptive's results dict, which has the shape
    {budget_fraction: {metric_name: [per-seed values]}}.

    feedback / n_classes / acquisition / alpha_ucb: experiment-level
    settings recorded per row. alpha_ucb is taken from the caller because
    adaptive_runner does not put it in its results dict
    so the caller records the same value for every row of a run."""
    rows = []
    for frac in budget_fractions:
        r = results[frac]
        n = len(r["train_reward"])
        row_labels = seeds if len(seeds) == n else range(n)
        for label, i in zip(row_labels, range(n)):
            row = {
                "Method": method_name,
                "Feedback": feedback,
                "Acquisition": acquisition,
                "Alpha UCB": alpha_ucb,
                "Num Classes": n_classes,
                "Dataset": dataset_name,
                "Split Mode": r["split_mode"][i],
                "Seed": label,
                "Budget Fraction": frac,
                "Train Reward": r["train_reward"][i],
                "Train F1": r["train_f1"][i],
                "Train AUROC": r["train_auroc"][i],
                "Inference Reward": r["inference_reward"][i],
                "Inference F1": r["inference_f1"][i],
                "Inference AUROC": r["inference_auroc"][i],
                "Total Reward": r["total_reward"][i],
                "Train Spent": r["train_spent"][i],
                "Inference Spent": r["inference_spent"][i],
                "Train Time (s)": r["train_time_sec"][i],
                "Inference Time (s)": r["inference_time_sec"][i],
                "Seed Time (s)": r["seed_time_sec"][i],
                "Num Masks Inference": r["num_masks_inference"][i] if "num_masks_inference" in r else np.nan,
                "Train Samples": r["n_train"][i],
                "Validation Samples": r["n_validation"][i],
                "Inference Samples": r["n_inference"][i],
                "Train Budget": r["train_budget"][i],
                "Inference Budget": r["inference_budget"][i],
                "Avg Views Acquired": r["avg_views_train"][i],
                "Num Arms": r["n_arms"][i] if "n_arms" in r else np.nan,
                "Selected Subsets": serialize_selected_subsets(r["selected_subsets"][i]),
                "Status": (r["status"][i] if "status" in r else "ok"),
                "Error": (r["error_msg"][i] if "error_msg" in r else ""),
            }
            row.update(_timing_fragment(r, i))
            rows.append(row)
    return rows


def normalize_adaptive_flat_rows(rows, dataset_name, feedback=np.nan, n_classes=np.nan,
                                 acquisition=np.nan, alpha_ucb=np.nan):
    """Normalizer for adaptive rows read back from a .rows.jsonl checkpoint.

    The checkpoint stores adaptive's cells FLAT (one dict per cell with its
    seed and budget fraction) rather than in the dict-of-lists shape
    run_experiment returns, because a checkpoint is written one cell at a
    time -- that is the whole point of it. So rebuilding needs this
    counterpart to _normalize_frac_keyed_results. Used only by
    --rebuild-from.
    """
    out = []
    for d in rows:
        row = {
            "Method": "adaptive",
            "Feedback": feedback,
            "Acquisition": acquisition,
            "Alpha UCB": alpha_ucb,
            "Num Classes": n_classes,
            "Dataset": dataset_name,
            "Seed": _nan(d, "seed"),
            "Budget Fraction": _nan(d, "budget_fraction"),
            "Train Reward": _nan(d, "train_reward"),
            "Train F1": _nan(d, "train_f1"),
            "Train AUROC": _nan(d, "train_auroc"),
            "Inference Reward": _nan(d, "inference_reward"),
            "Inference F1": _nan(d, "inference_f1"),
            "Inference AUROC": _nan(d, "inference_auroc"),
            "Total Reward": _nan(d, "total_reward"),
            "Train Spent": _nan(d, "train_spent"),
            "Inference Spent": _nan(d, "inference_spent"),
            "Train Time (s)": _nan(d, "train_time_sec"),
            "Inference Time (s)": _nan(d, "inference_time_sec"),
            # seed_time_sec is only known once a seed's whole budget loop
            # finishes, so a checkpoint written mid-seed does not have it.
            "Seed Time (s)": _nan(d, "seed_time_sec"),
            "Num Masks Inference": _nan(d, "num_masks_inference"),
            "Train Samples": _nan(d, "n_train"),
            "Inference Samples": _nan(d, "n_inference"),
            "Train Budget": _nan(d, "train_budget"),
            "Inference Budget": _nan(d, "inference_budget"),
            "Num Arms": _nan(d, "n_arms"),
            "Selected Subsets": "",   # not checkpointed -- see emit_row
            "Avg Views Acquired": _nan(d, "avg_views_train"),
            "Status": d.get("status", "ok"),
            "Error": d.get("error_msg", ""),
        }
        row.update(_timing_fragment(d))
        out.append(row)
    return out


#: The unified schema. The first block is unchanged from before -- same
#: names, same order -- and the observability columns are APPENDED, so an
#: older workbook and a newer one still concatenate (pandas fills the
#: missing new columns with NaN).
UNIFIED_COLUMNS = [
    "Method", "Feedback", "Num Classes", "Dataset", "Split Mode", "Seed",
    "Budget Fraction",
    "Train Reward", "Train F1", "Train AUROC",
    "Inference Reward", "Inference F1", "Inference AUROC",
    "Total Reward",
    "Train Samples", "Validation Samples", "Inference Samples",
    "Train Budget", "Inference Budget",
    "Train Spent", "Inference Spent",
    "Train Time (s)", "Inference Time (s)", "Seed Time (s)",
    "Num Masks Inference",
    "Acquisition", "Avg Views Acquired", "Num Arms", "Alpha UCB",
    "Selected Subsets",
] + excel_timing_columns() + ["Status", "Error"]

#: Columns that must never enter the numeric aggregation on the Summary
#: sheet, beyond the grouping keys. Named once so save_unified_results_to_excel
#: cannot drift from the column list above.
NON_NUMERIC_COLUMNS = ["Seed", "Split Mode", "Selected Subsets", "Status", "Error"]


def _acquisition_label(acquisition, reward_update):
    uses_reward_update = _uses_empirical_arm_rewards(acquisition)
    if not uses_reward_update:
        return acquisition
    return f"{acquisition}+{reward_update}"


# ─────────────────────────────────────────────────────────────────────────
# Dispatch
# ─────────────────────────────────────────────────────────────────────────
def run_method(method, dataset, max_modalities, seeds, budget_fractions,
                data_path, max_samples, sampling,
                synthetic_n_samples, synthetic_seed, synthetic_mean_scale,
                feedback="full",
                synthetic_n_classes=SYNTHETIC_N_CLASSES,
                step_size=None, lambda_max=10.0,
                run_inference=True, image_pool_side=DEFAULT_IMAGE_POOL_SIDE,
                image_data_home=None,
                reward_update="subsets",
                alpha_ucb=1.0, lr=1e-2,
                acquisition="lp_chain",
                ucb_bound="vc",
                split_mode="80-20",
                state_file=None):

    synthetic_n_views = max_modalities if max_modalities is not None else SYNTHETIC_N_VIEWS

    common_kwargs = dict(
        max_modalities=max_modalities, seeds=seeds, budget_fractions=budget_fractions,
        data_path=data_path, max_samples=max_samples, sampling=sampling,
        synthetic_n_samples=synthetic_n_samples, synthetic_n_views=synthetic_n_views,
        synthetic_seed=synthetic_seed, synthetic_mean_scale=synthetic_mean_scale,
        split_mode=split_mode,
    )

    if (acquisition in ORACLE_ACQUISITION_MODES and reward_update != "subsets"):
        log.warning(
            "--reward-update has no effect under --acquisition %s (its arm values "
            "come from the true means and are never scored); ignored.", acquisition)
    if acquisition in ORACLE_ACQUISITION_MODES and method == "adaptive" and feedback != "full":
        log.info("--acquisition %s gives the ACQUISITION policy the true means; the "
                 "classifier still learns under --feedback %s. This is not an "
                 "oracle-classifier run.", acquisition, feedback)
    if method == "adaptive":
        results = multiclass_runner.run_experiment(
            dataset, feedback=feedback,
            acquisition=acquisition, reward_update=reward_update,
            ucb_bound=ucb_bound,
            alpha_ucb=alpha_ucb, lr=lr, step_size=step_size, lambda_max=lambda_max,
            synthetic_n_classes=synthetic_n_classes,
            run_inference=run_inference, image_pool_side=image_pool_side,
            image_data_home=image_data_home,
            state_dir=state_file,
            **common_kwargs
        )

        if dataset in MULTICLASS_SYNTHETIC_DATASETS:
            n_classes = synthetic_n_classes
        else:
            n_classes = DATASET_N_CLASSES.get(dataset, 2)
        return _normalize_frac_keyed_results(
            results, budget_fractions, seeds, dataset, "adaptive",
            feedback=feedback, n_classes=n_classes,
            acquisition=_acquisition_label(acquisition, reward_update),
            alpha_ucb=alpha_ucb,
        )

    raise ValueError(f"Unknown method {method!r}, choose from {METHODS}")


def run_inference_from_saved_states(source):
    """Dispatch an inference-only run using the method/config in saved states."""
    paths = find_training_states(source)
    metadata = [peek_training_state(p) for p in paths]
    methods = {m["method"] for m in metadata}
    datasets = {m["dataset"] for m in metadata}
    if len(methods) != 1:
        raise ValueError(f"state source mixes methods: {sorted(methods)}")
    if len(datasets) != 1:
        raise ValueError(f"state source mixes datasets: {sorted(datasets)}")
    method = methods.pop()
    dataset = datasets.pop()
    config = metadata[0].get("run_config", {})
    # Reject accidental directory-level mixtures that happen to share a method.
    for item in metadata[1:]:
        if item.get("run_config", {}) != config:
            raise ValueError("state source mixes incompatible run configurations")

    if method == "adaptive":
        native_rows = multiclass_runner.run_inference_from_states(source)
        return normalize_adaptive_flat_rows(
            native_rows, dataset,
            feedback=config.get("feedback", np.nan),
            n_classes=metadata[0].get("nclasses", np.nan),
            acquisition=_acquisition_label(
                config.get("acquisition", "lp_chain"),
                config.get("reward_update", "subsets")),
            alpha_ucb=config.get("alpha_ucb", np.nan),
        ), method, dataset
    raise ValueError(f"unsupported method in training state: {method!r}")


def save_unified_results_to_excel(rows, filename, info_rows=None):
    df = pd.DataFrame(rows, columns=UNIFIED_COLUMNS)
    group_cols = ["Method", "Feedback", "Acquisition", "Alpha UCB",
                  "Num Classes", "Dataset", "Budget Fraction"]
    numeric_cols = [
        c for c in UNIFIED_COLUMNS
        if c not in group_cols + NON_NUMERIC_COLUMNS
    ]
    # Aggregate over SUCCESSFUL cells only. An isolated failure now
    # contributes a NaN row rather than killing the sweep, and mean/std
    # over a group containing it would otherwise be NaN for every metric --
    # which would turn one lost cell back into one lost group.
    ok = df[df["Status"] == "ok"] if "Status" in df.columns else df
    summary = (
        ok.groupby(group_cols, dropna=False)[numeric_cols]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary.columns = [
        col[0] if col[1] == "" else f"{col[0]} ({col[1]})" for col in summary.columns
    ]

    with pd.ExcelWriter(filename, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="Detailed Results", index=False)
        summary.to_excel(writer, sheet_name="Summary", index=False)

    style_and_save(filename, ["Detailed Results", "Summary"], info_rows=info_rows)
    log.info("Results saved to %s", filename)


# ─────────────────────────────────────────────────────────────────────────
# Salvage: rebuild a workbook from a killed run's checkpoint
# ─────────────────────────────────────────────────────────────────────────
def rebuild_from_checkpoint(source, output_xlsx=None):
    """Turn results/{run_id}.rows.jsonl back into a workbook.

    This is what the row checkpoint is FOR. A job killed at seed 8 of 10
    used to leave nothing; now it leaves every completed cell on disk, and
    this reads them back, normalizes them through exactly the same
    functions a live run uses, and writes the same two sheets.

    The run's arguments come from the sibling manifest, so the Method /
    Dataset / Acquisition columns and the output filename are the real ones
    rather than guesses. The output gets a _partial tag: these rows are a
    prefix of the intended sweep, and a file that does not say so will
    eventually be compared against a complete one as though it were.
    """
    p = Path(source)
    if p.is_dir():
        raise ValueError(f"{source!r} is a directory; pass the .rows.jsonl file "
                         f"or the run_id")
    if not p.exists():
        # Accept a bare run_id as well as a path.
        cand = Path("results") / f"{source}.rows.jsonl"
        if not cand.exists():
            raise FileNotFoundError(f"no checkpoint at {p} or {cand}")
        p = cand

    manifest_path = Path(str(p).replace(".rows.jsonl", ".manifest.json"))
    manifest = read_manifest(manifest_path) if manifest_path.exists() else {}
    if not manifest:
        log.warning("no manifest beside %s -- rebuilding with unknown run settings; "
                    "the Method/Dataset/Acquisition columns will be blank", p)
    a = manifest.get("args", {}) or {}

    rows_native = read_rows_jsonl(p)
    if not rows_native:
        raise ValueError(f"{p} contains no readable rows")
    log.info("rebuilding from %s: %d rows, entry_point=%s",
             p, len(rows_native), manifest.get("entry_point"))

    method = a.get("method")
    if method is None:
        method = "adaptive"
    dataset = a.get("dataset", "")
    acquisition = _acquisition_label(
        a.get("acquisition", "lp_chain"),
        a.get("reward_update", "subsets"))

    if method != "adaptive":
        raise ValueError(f"unsupported checkpoint method: {method!r}")
    if dataset in MULTICLASS_SYNTHETIC_DATASETS:
        n_classes = a.get("num_classes", np.nan)
    else:
        n_classes = DATASET_N_CLASSES.get(dataset, np.nan)
    rows = normalize_adaptive_flat_rows(
        rows_native, dataset, feedback=a.get("feedback", np.nan),
        n_classes=n_classes, acquisition=acquisition,
        alpha_ucb=a.get("alpha_ucb", np.nan))

    if output_xlsx is None:
        stem = a.get("output_xlsx") or manifest.get("output_xlsx")
        if stem:
            stem = Path(stem)
            output_xlsx = str(stem.with_name(stem.stem + "_partial" + stem.suffix))
        else:
            output_xlsx = str(p).replace(".rows.jsonl", "_partial.xlsx")

    info = [(k, v) for k, v in manifest.items()
            if not isinstance(v, (dict, list))]
    info.append(("rebuilt_from", str(p)))
    info.append(("rebuilt_rows", len(rows)))
    save_unified_results_to_excel(rows, output_xlsx, info_rows=info)
    return output_xlsx


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run adaptive MULTICLASS AFA on a "
                     "real AFA-Benchmark or synthetic dataset, and write results in one unified "
                     "schema."
    )
    parser.add_argument("--method", choices=METHODS,
                        help="Required for a normal run; omitted only with --rebuild-from.")
    parser.add_argument("--dataset", choices=ALL_DATASETS, default="synthetic")
    parser.add_argument("--data-path", type=str, default=None)
    parser.add_argument("--split-mode", choices=SPLIT_MODES, default="80-20", help="Train/test split for adaptive runs; use 60-20-20 for OL comparison.")
    parser.add_argument("--max-modalities", type=str, default=None, help="Number of modalities; all uses the dataset width or default synthetic view count.")
    parser.add_argument("--max-samples", type=int, default=None,
                         help="Cap a REAL dataset to at most this many rows (reproducible "
                              "subsample; if the dataset already has fewer rows, all of them "
                              "are used). Ignored for synthetic datasets -- use --n-samples "
                               "instead.")
    parser.add_argument(
        "--sampling",
        choices=SAMPLING_MODES,
        default=DEFAULT_SAMPLING_MODE,
        help="REAL datasets only: balanced (default, no replacement), "
             "stratified (preserve class proportions), or random.",
    )
    parser.add_argument("--budget-fractions", type=str, default="0.1,0.3,0.5,0.7,0.9")
    parser.add_argument("--seeds", type=str, default="42,43,44,45,46,47,48,49,50,51")
    parser.add_argument("--n-samples", type=int, default=SYNTHETIC_N_SAMPLES,
                         help="synthetic datasets only: how many rows to generate. Ignored for "
                              "real datasets.")
    parser.add_argument("--synthetic-seed", type=int, default=SYNTHETIC_SEED,
                         help="synthetic datasets only: seed for the generative means draw "
                              "(independent of --seeds, which controls the train/inference split).")
    parser.add_argument("--mean-scale", type=float, default=SYNTHETIC_MEAN_SCALE,
                         help="synthetic datasets only: per-(class,view) means drawn ~ "
                              "Uniform(0, mean_scale).")
    parser.add_argument("--output-xlsx", type=str, default=None, help="Workbook path; defaults under results/Adaptive/singlepass/.")
    parser.add_argument("--print-output-stem", action="store_true",
                         help="Resolve the output workbook name from these arguments, print "
                              "only its filename stem, and exit without starting a run. "
                              "Intended for submission wrappers that need the PBS log to "
                              "match Python's Excel filename exactly.")
    parser.add_argument("--feedback", choices=("full", "bandit"), default="full", help="Adaptive centre update feedback mode.")
    parser.add_argument("--acquisition", choices=ACQUISITION_MODES, default="lp_chain", help="Adaptive acquisition policy: lp_chain, lp_full_opt, ucb_argmax, or hedge.")
    parser.add_argument("--ucb-bound", choices=UCB_BOUNDS, default="vc", help="Confidence bound for empirical arm rewards.")
    parser.add_argument("--num-classes", type=int, default=SYNTHETIC_N_CLASSES,
                         help="synthetic only: how many classes to generate "
                              "(labels {0..K-1}). Every other dataset infers its class "
                              "count from the labels.")
    parser.add_argument("--step-size", type=float, default=None, help="OMD dual step size for ucb_argmax; default varies by round.")
    parser.add_argument("--lambda-max", type=float, default=10.0, help="Upper bound for the ucb_argmax OMD dual variable.")
    parser.add_argument("--reward-update", choices=REWARD_UPDATE_SCOPES, default="subsets",
                         help="'subsets': exact contained-arm rewards; 'selected': "
                              "played arm only.")
    parser.add_argument("--alpha-ucb", type=float, default=1.0, help="Optimism scale for vc/vector bounds.")
    parser.add_argument("--lr", type=float, default=1e-2, help="Complementary-label learning rate under bandit feedback.")
    parser.add_argument("--skip-inference", action="store_true", help="Run training only and save its state for later inference.")
    parser.add_argument("--inference-only", action="store_true",
                         help="Skip training and automatically load the state file whose "
                              "name would have been produced by the same CLI with "
                              "--skip-inference. Reuse the original method/dataset/config "
                              "arguments. With a custom --output-xlsx, that path is treated "
                              "as the original training-only workbook and the completed "
                              "workbook receives an _INF suffix.")
    parser.add_argument("--inference-from-state", type=str, default=None,
                         help="Inference-only: load the run's single "
                              ".training_states.sqlite3 file and run every saved cell "
                              "without loading the "
                              "dataset or rerunning training. Method, dataset, split, costs, "
                              "budgets and RNG state come from the saved files.")
    parser.add_argument("--image-pool-side", type=int, default=DEFAULT_IMAGE_POOL_SIDE, help="Pooled side length for MNIST and Fashion-MNIST images.")
    parser.add_argument("--image-cache-dir", type=str, default=None,
                         help="mnist/fashion_mnist only: directory fetch_openml caches its "
                              "download in. Default (None) resolves to core.datasets."
                              "DEFAULT_OPENML_DATA_HOME, a RELATIVE 'data/openml_cache' folder -- "
                              "deliberately NOT fetch_openml's own ~/scikit_learn_data default, "
                              "which exceeds $HOME's quota on clusters like NCI Gadi. Point this "
                              "at your project/scratch space if the default location itself lacks "
                              "quota. Ignored for non-image datasets.")
    # ── observability flags (see core/logging_utils.py) ──
    parser.add_argument("--log-level", default="INFO",
                         choices=("DEBUG", "INFO", "WARNING", "ERROR"),
                         help="Console verbosity, and file-log verbosity when --file-log "
                              "is enabled.")
    parser.add_argument("--file-log", action="store_true",
                         help="Also write logs/{run_id}.log. Off by default because batch "
                              "systems such as NCI/PBS already capture console output.")
    parser.add_argument("--json-sidecars", action="store_true",
                         help="Write the .manifest.json provenance file and .rows.jsonl "
                              "crash checkpoint. Off by default; enabling it is required "
                              "for --rebuild-from recovery after an interrupted run.")
    parser.add_argument("--log-dir", default="logs",
                         help="Directory for {run_id}.log (default: logs/, created if "
                              "missing). On PBS point this at the same directory as the "
                              "#PBS -o path so a job's two logs sit together.")
    parser.add_argument("--trace-rounds", action="store_true",
                         help="Record one row PER TRAINING ROUND (subset, cost, lambda, "
                              "reward, remaining budget) to results/{run_id}.trace.jsonl, "
                              "and drop the 'Selected Subsets' Excel cell, which is a "
                              "strictly poorer encoding of the same thing. Off by default: "
                              "it is O(n_train) records per sweep cell.")
    parser.add_argument("--no-fine-timers", action="store_true",
                         help="Disable the fine-grained timing buckets; the t_* columns "
                              "become NaN. The original Train/Inference/Seed timings are "
                              "unaffected either way.")
    parser.add_argument("--rebuild-from", type=str, default=None,
                         help="Skip the sweep entirely: rebuild a workbook from a previous "
                              "run's results/{run_id}.rows.jsonl checkpoint (pass the path "
                              "or just the run_id). Use this after a walltime kill or an OOM "
                              "to recover every cell that completed. The output is tagged "
                              "_partial, because it is a prefix of the intended sweep.")
    args = parser.parse_args()

    inference_modes = int(args.skip_inference) + int(args.inference_only) + int(
        bool(args.inference_from_state))
    if inference_modes > 1:
        parser.error("--skip-inference, --inference-only and "
                     "--inference-from-state are mutually exclusive")
    if args.rebuild_from and (args.inference_only or args.inference_from_state):
        parser.error("--rebuild-from cannot be combined with an inference-only mode")

    if args.rebuild_from:
        # No sweep, no run context -- this is a pure file-to-file operation
        # and should not create a new run_id or a new log.
        import logging
        logging.basicConfig(level=getattr(logging, args.log_level),
                            format="%(levelname).1s %(message)s")
        out = rebuild_from_checkpoint(args.rebuild_from, args.output_xlsx)
        log.info("rebuilt %s", out)
        sys.exit(0)

    resume_meta = None
    if args.inference_from_state:
        state_paths = find_training_states(args.inference_from_state)
        resume_meta = peek_training_state(state_paths[0])
        args.method = resume_meta["method"]
        if args.method != "adaptive":
            parser.error(f"unsupported saved-state method: {args.method}")
        args.dataset = resume_meta["dataset"]
        resume_config = resume_meta.get("run_config", {})
        for key in ("feedback", "acquisition", "ucb_bound", "reward_update",
                    "alpha_ucb", "lr", "step_size", "lambda_max",
                    "split_mode"):
            if key in resume_config:
                setattr(args, key, resume_config[key])
    elif args.method is None:
        parser.error("--method is required (omit it only with --rebuild-from or "
                     "--inference-from-state)")

    uses_empirical_arm_rewards = _uses_empirical_arm_rewards(args.acquisition)
    budget_fractions = tuple(float(x) for x in args.budget_fractions.split(","))
    seeds = tuple(int(x) for x in args.seeds.split(","))
    if args.max_modalities is None:
        max_modalities = DEFAULT_MAX_MODALITIES[args.method]
    else:
        max_modalities = None if args.max_modalities.lower() == "all" else int(args.max_modalities)

    # Resolve the normal auto-name once so --inference-only can reproduce
    # the exact state path from the same arguments used with --skip-inference.
    dataset_directory = ("Synthetic" if args.dataset.startswith("synthetic") else args.dataset)
    results_dir = Path("results") / "Adaptive" / "singlepass" / dataset_directory
    results_dir.mkdir(parents=True, exist_ok=True)

    def _auto_output_path(mode=None):
        mode_tag = f"_{mode}" if mode else ""
        split_tag = ("" if args.split_mode == "80-20"
                     else f"_split{args.split_mode.replace('-', '')}")
        fb_tag = f"_{args.feedback}"
        output_acquisition = args.acquisition
        acq_tag = f"_{output_acquisition}"
        if uses_empirical_arm_rewards:
            acq_tag += f"-{args.reward_update}"
        bound_tag = f"_{args.ucb_bound}"
        alpha_tag = (f"_alpha{args.alpha_ucb:g}" if (args.alpha_ucb != 1.0 and args.acquisition not in ORACLE_ACQUISITION_MODES) else "")
        maxmod_label = "ALL" if max_modalities is None else str(max_modalities)
        classes_tag = ""
        if args.dataset in SYNTHETIC_DATASETS:
            n_classes = args.num_classes if args.dataset in MULTICLASS_SYNTHETIC_DATASETS else 2
            classes_tag = f"_K{n_classes}"
        return str(results_dir / (
            f"results_{args.method}"
            f"{fb_tag}{acq_tag}{bound_tag}{alpha_tag}_"
            f"{args.dataset}_V{maxmod_label}_T{len(seeds)}{split_tag}"
            f"{mode_tag}{classes_tag}.xlsx"))

    inference_state_source = args.inference_from_state
    if args.inference_from_state:
        if args.output_xlsx:
            output_xlsx = args.output_xlsx
        else:
            source_tag = Path(args.inference_from_state).stem
            source_tag = "".join(c if (c.isalnum() or c in "._-") else "_"
                                 for c in source_tag)
            output_xlsx = str(results_dir / (
                f"results_{args.method}_INF_{source_tag}.xlsx"))
    elif args.inference_only:
        training_output_xlsx = args.output_xlsx or _auto_output_path(mode="TR")
        inference_state_source = str(
            Path(training_output_xlsx).with_suffix(".training_states.sqlite3"))
        if not Path(inference_state_source).is_file():
            parser.error(
                "--inference-only could not find the state file derived from "
                f"these arguments: {inference_state_source}")
        if args.output_xlsx:
            original = Path(args.output_xlsx)
            output_xlsx = str(original.with_name(original.stem + "_INF.xlsx"))
        else:
            output_xlsx = _auto_output_path(mode="INF")
    else:
        output_xlsx = args.output_xlsx or _auto_output_path(
            mode="TR" if args.skip_inference else None)

    state_file = None if inference_state_source else str(
        Path(output_xlsx).with_suffix(".training_states.sqlite3"))

    if args.print_output_stem:
        print(Path(output_xlsx).stem)
        sys.exit(0)

    run = setup_run(
        "run_proposed_methods",
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
               "output_xlsx": output_xlsx,
               "training_state_file": state_file,
               "inference_state_source": inference_state_source},
    )

    t0 = time.time()
    try:
        if inference_state_source:
            rows, resumed_method, resumed_dataset = run_inference_from_saved_states(
                inference_state_source)
            args.method = resumed_method
            args.dataset = resumed_dataset
        else:
            rows = run_method(
                args.method, args.dataset, max_modalities, seeds, budget_fractions,
                args.data_path, args.max_samples,
                args.sampling,
                args.n_samples, args.synthetic_seed, args.mean_scale,
                feedback=args.feedback,
                synthetic_n_classes=args.num_classes,
                step_size=args.step_size,
                lambda_max=args.lambda_max, run_inference=not args.skip_inference,
                image_pool_side=args.image_pool_side,
                image_data_home=args.image_cache_dir,
                reward_update=args.reward_update,
                alpha_ucb=args.alpha_ucb, lr=args.lr,
                acquisition=args.acquisition,
                ucb_bound=args.ucb_bound,
                split_mode=args.split_mode,
                state_file=state_file,
            )
    except BaseException as exc:                          # noqa: BLE001
        if run.json_sidecars:
            log.exception("run aborted -- recover completed cells with "
                          "--rebuild-from %s", run.rows_path)
        else:
            log.exception("run aborted; JSON checkpointing was disabled")
        run.finalize(status=f"failed: {type(exc).__name__}: {exc}")
        raise

    df = pd.DataFrame(rows, columns=UNIFIED_COLUMNS)
    ok = df[df["Status"] == "ok"] if "Status" in df.columns else df
    log.info("=" * 70)
    log.info("SUMMARY (mean +/- std across seeds) -- %s / %s / %s",
             args.method, args.dataset, args.acquisition)
    log.info("=" * 70)
    log.info("%-10s%12s%10s%13s%10s%9s%11s%13s%12s", 'Fraction', 'Train Rew',
             'Train F1', 'Train AUROC', 'Inf Rew', 'Inf F1', 'Inf AUROC',
             'Train Spent', 'Inf Spent')
    log.info("-" * 100)
    for frac, sub in ok.groupby("Budget Fraction"):
        log.info(
            "%-10.2f%9.3f %8.3f %11.3f %8.3f %7.3f %9.3f %11.4f %10.4f",
            frac,
            sub['Train Reward'].mean(), sub['Train F1'].mean(),
            sub['Train AUROC'].mean(), sub['Inference Reward'].mean(),
            sub['Inference F1'].mean(), sub['Inference AUROC'].mean(),
            sub['Train Spent'].mean(), sub['Inference Spent'].mean(),
        )

    save_unified_results_to_excel(rows, output_xlsx, info_rows=run.info_rows())
    log.info("Execution time: %.1f seconds", time.time() - t0)
    run.finalize(status="ok")
