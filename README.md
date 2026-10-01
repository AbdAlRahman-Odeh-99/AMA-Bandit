# AFA-Bandit

**Provably Near-Optimal Online Multi-Feature Classification under Budget Constraints**

This repository accompanies the submitted manuscript **“AFA-Bandit: Provably
Near-Optimal Online Multi-Feature Classification under Budget Constraints”** by
AbdAlRahman Odeh, Teng-Hui Huang, and Hesham El Gamal. It studies online active
feature acquisition (AFA): a learner buys features, predicts a label, then
updates from the revealed label while respecting a global acquisition budget.
The code calls these features *modalities* or *views* in command options.

The paper formulates this as a predictor-coupled combinatorial Bandits with
Knapsacks problem. Its regret bound applies to the **full combinatorial
method**. **LP-Chain** makes the acquisition search scalable by considering a
cost-aware chain of at most `V` nested subsets when one of `V` features is free.
The paper reports that LP-Chain performs comparably to full subset search and
outperforms HEDGE and Opportunistic Learning (OL) on the synthetic comparisons.

## Run locally

Run these commands from the repository root after [installing the dependencies](#installation).
The examples use synthetic data with 10 features, 1,000 samples, a 60/20/20
train/validation/test split, budget fractions `0.1,0.3,0.5,0.7,0.9`, and ten
seeds (42–51). The manuscript's main synthetic results average **50 trials**;
these shorter commands are a convenient starting point. Change
`--num-classes 2` to `4` for the four-class setting.

**LP-Chain:**

```bash
python -u scripts/local/run_proposed_methods.py --method adaptive --dataset synthetic --split-mode 60-20-20 --max-modalities 10 --n-samples 1000 --seeds 42,43,44,45,46,47,48,49,50,51 --feedback full --acquisition lp_chain --reward-update subsets --num-classes 2 --ucb-bound theo
```

**HEDGE-based BwK comparison:**

```bash
python -u scripts/local/run_proposed_methods.py --method adaptive --dataset synthetic --split-mode 60-20-20 --max-modalities 10 --n-samples 1000 --seeds 42,43,44,45,46,47,48,49,50,51 --feedback full --acquisition hedge --reward-update subsets --num-classes 2 --ucb-bound theo
```

**Synthetic full-action oracle:**

```bash
python -u scripts/local/run_proposed_methods.py --method adaptive --dataset synthetic --split-mode 60-20-20 --max-modalities 10 --n-samples 1000 --seeds 42,43,44,45,46,47,48,49,50,51 --feedback full --acquisition lp_full_opt --reward-update subsets --num-classes 2 --ucb-bound theo
```

**Sequential OL baseline:**

```bash
python -m baselines.ol --dataset synthetic --n-samples 1000 --n-views 10 --num-classes 2 --budget-fractions 0.1,0.3,0.5,0.7,0.9 --seeds 42,43,44,45,46,47,48,49,50,51
```

For all local command variants, see
[`scripts/local/Synthetic_Local_CMD.txt`](scripts/local/Synthetic_Local_CMD.txt)
and [`scripts/local/Real_Local_CMD.txt`](scripts/local/Real_Local_CMD.txt).
Additional OL notes are in
[`scripts/local/Baselines_Local_CMD.txt`](scripts/local/Baselines_Local_CMD.txt).

## Repository layout

```text
adaptive/   LP-Chain, Combinatorial, HEDGE, oracle acquisition, and experiment runner
baselines/  Sequential Opportunistic Learning (OL) baseline and its helpers
core/       Datasets, budget accounting, acquisition policies, LP routines,
            logging, metrics, and training-state utilities
scripts/local/  Adaptive Python driver and local command lists
```

Run commands from the repository root so data, logs, and results use the
expected paths. Additional local commands are listed in `scripts/local/`.

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

## Methods and paper experiments

| Paper method | Command option | Role |
|---|---|---|
| LP-Chain | `--acquisition lp_chain` | Cost-aware nested chain with a budgeted LP over at most `V` actions |
| Combinatorial | `--acquisition ucb_argmax` | Full-action primal–dual policy used for the regret analysis and chain ablation |
| HEDGE-based BwK | `--acquisition hedge` | Full-action budgeted-bandit comparison |
| Full-action oracle | `--acquisition lp_full_opt` | Synthetic-only fixed distribution using the true synthetic means |
| Opportunistic Learning | `python -m baselines.ol` | Sequential neural acquisition and prediction baseline |

For LP-Chain and HEDGE, `--reward-update subsets` reuses the observed label and
acquired features to score every feasible subset of the selected action. The
appendix's Naive Update ablation uses `--reward-update selected` instead. The
oracle's `--reward-update` and `--ucb-bound` arguments do not change its policy.
Full-action methods enumerate `2^(V-1)` actions when one feature is free, so
large `V` can be expensive.

The main synthetic comparison uses Gaussian-mixture data with `V=10`,
`K∈{2,4}`, heterogeneous normalized costs, one free feature, and five budget
fractions. It averages 50 independent trials. For a local reproduction, use
the commands above with seeds 42–91 for **each** method. The Gaussian model
generates the data; the learning methods do not receive its true means. Only
the synthetic-only oracle uses them.

The appendix compares LP-Chain, HEDGE, and OL on CKD, Bank Marketing,
PhysioNet, Diabetes, MNIST, and Fashion-MNIST. These runs use 16 modalities,
five seeds (42–46), the 60/20/20 split, and the same five budget fractions.
MNIST and Fashion-MNIST are pooled to 4×4 features; datasets other than CKD
are capped at 10,000 rows. The local real-data commands are in
[`scripts/local/Real_Local_CMD.txt`](scripts/local/Real_Local_CMD.txt).

OL starts each encounter with the free modality, buys affordable modalities
sequentially, predicts, and then updates its P/Q networks from replay. It
respects the global budget and a per-encounter cap. Its default number of
encounters equals the training-set size, but its class-balanced stream can
repeat rows, so this does not guarantee one visit per row. OL uses a 60/20/20
split automatically.

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

Adaptive runs write Excel workbooks under `results/Adaptive/singlepass/` by
default; use `--output-xlsx` to choose another path. OL writes CSV files
directly under `results/` by default; use `--output-csv` to place them elsewhere.
Results include online training metrics recorded **before** the current sample
updates the model, along with held-out validation and test metrics. LP-Chain
and HEDGE use the same nearest-class-mean predictor, while OL trains its own
neural predictor and acquisition policy. Adaptive runs can also save training
states and diagnostic traces.

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
python scripts/local/run_proposed_methods.py --help
python -m baselines.ol --help
```

## Citation

The manuscript is under submission. Until a public preprint or final version
is available, cite it as:

```bibtex
@unpublished{odeh2026afabandit,
  author = {Odeh, AbdAlRahman and Huang, Teng-Hui and El Gamal, Hesham},
  title  = {AFA-Bandit: Provably Near-Optimal Online Multi-Feature Classification under Budget Constraints},
  year   = {2026},
  note   = {Manuscript under submission}
}
```

## Acknowledgements

The OL comparison builds on work from the active feature acquisition literature.
Please consult the accompanying paper for the complete discussion and original
method citations.
