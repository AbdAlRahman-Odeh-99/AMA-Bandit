# AMA-Bandit

**Cost-Aware Online Multi-Modal Classification**

AMA-Bandit is a research codebase for budget-constrained online active feature
acquisition (AFA). At each round, a learner chooses which modalities to acquire,
predicts a class label, observes feedback, and updates its predictor while
respecting a global acquisition budget.

This repository accompanies the manuscript **“AMA-Bandit: Cost-Aware Online
Multi-Modal Classification.”** The proposed method, **LP-Chain**, formulates
online AFA as a predictor-coupled combinatorial Bandits with Knapsacks problem.
It combines online predictor learning, subset feedback, optimistic reward
estimation, and budgeted acquisition over a nested chain of modality subsets.

## Why LP-Chain?

With `V` modalities and one always-available modality, direct Full-Space
optimization considers up to `2^(V-1)` feasible subsets. LP-Chain restricts the
search to at most `V` nested subsets. This substantially improves scalability
while retaining performance close to Full Space in our experiments.

The implementation provides:

- globally budgeted, one-pass online learning;
- subset feedback: acquiring a set also reveals rewards for its feasible
  subsets;
- subset-dependent confidence bounds;
- synthetic multiclass experiments with heterogeneous modality costs;
- the Full-Space, HEDGE, and synthetic oracle comparisons used in the paper;
- Online AFA baselines that predict before updating on each sample; and
- reproducible seeded data splits and experiment outputs.

## Repository layout

```text
adaptive/   LP-Chain, Full-Space, HEDGE, oracle acquisition, and experiment runner
baselines/  Online AFA baseline implementations and shared baseline runner
core/       Datasets, budget accounting, acquisition policies, LP routines,
            logging, metrics, and training-state utilities
```

Run all commands from the repository root so Python can resolve the package
imports correctly.

## Installation

Python 3.10 or newer is recommended.

```bash
git clone https://github.com/AbdAlRahman-Odeh-99/AMA-Bandit.git
cd AMA-Bandit

python -m venv .venv
```

Activate the environment:

```bash
# Linux/macOS
source .venv/bin/activate

# Windows PowerShell
.venv\Scripts\Activate.ps1
```

Install the required packages:

```bash
python -m pip install --upgrade pip
python -m pip install numpy pandas scipy scikit-learn torch numba tqdm openpyxl ucimlrepo
```

## Quick start

Run LP-Chain on synthetic data with 1,000 samples, 10 modalities, 2 classes,
and the default budget sweep:

```bash
python -m adaptive.adaptive_runner \
  --dataset synthetic \
  --n-samples 1000 \
  --n-views 10 \
  --num-classes 2 \
  --acquisition lp_chain
```

The default budget fractions are `0.1,0.3,0.5,0.7,0.9`. Adaptive results are
written under `results/Adaptive/singlepass/` unless `--output-xlsx` is supplied.

To see every available option:

```bash
python -m adaptive.adaptive_runner --help
python -m baselines.run_baselines --help
```

## Reproducing the synthetic protocol

The paper evaluates synthetic Gaussian-mixture data with 10 modalities,
heterogeneous normalized costs, one free modality, class counts `K in {2, 4}`,
five budget fractions, and 50 trials (seeds 42–91).

The command below runs LP-Chain for the two-class setting:

```bash
python -m adaptive.adaptive_runner \
  --dataset synthetic \
  --split-mode 60-20-20 \
  --n-samples 1000 \
  --n-views 10 \
  --num-classes 2 \
  --budget-fractions 0.1,0.3,0.5,0.7,0.9 \
  --seeds 42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91 \
  --acquisition lp_chain
```

Use `--num-classes 4` for the four-class experiment.

### Acquisition comparisons

Replace the final acquisition argument to run the corresponding comparison:

| Comparison | Argument | Description |
|---|---|---|
| LP-Chain | `--acquisition lp_chain` | Proposed chain-restricted LP policy |
| Full Space | `--acquisition ucb_argmax` | UCB argmax over the full subset space |
| HEDGE | `--acquisition hedge` | HEDGE-based Bandits with Knapsacks policy |
| Oracle | `--acquisition lp_full_opt` | Synthetic-only full-action static oracle |

Full Space and the oracle enumerate the subset action space and therefore scale
exponentially with the number of modalities. They are intended for controlled
comparisons at modest values of `V`, not large-scale runs.

## Online AFA baselines

The baseline runner implements a one-pass protocol: each method acquires
modalities and predicts before seeing the current label, then updates using only
the information available after that prediction. Acquisition costs are charged
to one global training budget.

Run all implemented online baselines:

```bash
python -m baselines.run_baselines \
  --method online_all \
  --dataset synthetic \
  --n-samples 1000 \
  --n-views 10 \
  --num-classes 2 \
  --seeds 42,43,44,45,46
```

Run one method by replacing `online_all` with one of:

- `online_aaco`
- `online_cae`
- `online_cwcf`
- `online_dime`
- `online_eddi`
- `online_gdfs`
- `online_jafa`
- `online_ol`
- `online_pt`

The paper reports EDDI, CwCF, and PT as representative information-theoretic,
policy-learning, and static-selection baselines. The remaining implementations
are included for broader evaluation.

Baseline results are saved as CSV files. Use `--output-csv` to choose an
explicit destination.

## Datasets

The synthetic dataset requires no download. The shared data loader also
supports:

- `ckd`, `bank_marketing`, and `actg175` through `ucimlrepo`;
- `mnist` and `fashion_mnist` through OpenML, with optional image pooling; and
- `diabetes`, `physionet`, and `miniboone` when the required local data files
  are provided.

Use `--data-path` for datasets that require a local file. See
[`core/datasets.py`](core/datasets.py) for preprocessing details, expected file
formats, and dataset-specific defaults.

## Outputs and reproducibility

Experiment outputs include reward and error metrics, acquisition statistics,
timings, selected subsets, and run configuration. Adaptive experiments can also
persist training states for later inference and write detailed diagnostic data.

Useful options include:

```text
--skip-inference   run and save only the online training phase
--trace-rounds     record one entry per training encounter
--json-sidecars    write manifest and crash-recovery sidecar files
--file-log         save a dedicated experiment log
```

For a quick source check:

```bash
python -m compileall -q adaptive baselines core
python -m adaptive.adaptive_runner --help
python -m baselines.run_baselines --help
```

## Citation

The manuscript is currently under submission. Citation details will be updated
when a public preprint or final publication is available. In the meantime,
please cite the repository as:

```bibtex
@misc{ama_bandit_2026,
  title        = {AMA-Bandit: Cost-Aware Online Multi-Modal Classification},
  year         = {2026},
  howpublished = {GitHub repository},
  url          = {https://github.com/AbdAlRahman-Odeh-99/AMA-Bandit}
}
```

## Acknowledgements

The online AFA comparisons build on ideas and implementations from the active
feature acquisition literature, including methods represented in AFABench.
Please consult the accompanying paper for the complete discussion and original
method citations.
