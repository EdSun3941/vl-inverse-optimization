# Real-data experiments for Virtual Layer inverse optimization

This repository contains the reproducible experiment package for the manuscript "Virtual Layer Inverse Optimization for Engineering Input Parameter Recommendation."
The scripts reproduce the real-data experiments, the robotic-arm benchmark, and
the sequential-update ablation reported in the paper.

## Archived release

The reproducibility package is archived as GitHub release `v1.0.3`.

Release page:
https://github.com/EdSun3941/vl-inverse-optimization/releases/tag/v1.0.3

Zenodo concept DOI:
https://doi.org/10.5281/zenodo.20680511

## Citation and license

Citation metadata is provided in `CITATION.cff`, and Zenodo metadata is provided
in `.zenodo.json`. Unless otherwise noted, this repository is released under
the Creative Commons Attribution 4.0 International License (CC BY 4.0). The
included UCI datasets are redistributed under their source CC BY 4.0 licenses.

## Directory contents

| File | Description |
|---|---|
| `concrete.csv` | UCI Concrete Compressive Strength dataset (Yeh 1998, ID 165). It contains 1,030 records with eight mix-design inputs in kg/m^3 or days, plus compressive strength in MPa. |
| `energy.csv` | UCI Energy Efficiency ENB2012 dataset (Tsanas and Xifara 2012, ID 242). It contains 768 simulated buildings with eight design inputs, heating load Y1, and cooling load Y2 in kWh/m^2. |
| `nnlite.py` | Pure-NumPy MLP and Adam optimizer with input-gradient support for projected VL inverse optimization. |
| `prepare_data.py` | Downloads the two CSV files from the UCI Machine Learning Repository and validates them against SHA-256 checksums of the copies used in the reported experiments. |
| `vl_concrete_np.py` | Constrained concrete mixture recommendation, five scenarios with 20 restarts each. |
| `vl_concrete_gap.py` | Sparse-region constrained scenario (age = 3 days, target 45 MPa). No measured age-3 record exceeds 41.6 MPa, so constrained retrieval is capped below target while projected VL reaches it with verifier support. Reuses the same surrogate and verifier. |
| `vl_energy_np.py` | Multi-objective building design with mixed continuous and discrete feasible sets, four scenarios with 20 restarts each. |
| `vl_robotic_benchmark.py` | Robotic-arm inverse-kinematics benchmark over 200 targets. |
| `vl_sequential_ablation.py` | Sequential, joint, and stable-broadcast update-order ablation on a controlled VARX system. |
| `concrete_results.json` | Pre-computed results for the concrete experiment. |
| `concrete_gap_results.json` | Pre-computed results for the sparse-region scenario: unconstrained k-NN, the constrained k-NN cap, and projected VL with restart success rates. |
| `energy_results.json` | Pre-computed results for the energy experiment. |
| `robotic_benchmark_results.json` | Pre-computed robotic benchmark results, including per-target re-simulation errors, exact failure counts, worst-case errors, and out-of-domain counts. |
| `sequential_ablation_results.json` | Pre-computed sequential ablation results. |

## Environment

The reported scripts were tested with:

| Requirement | Version tested |
|---|---|
| Python | 3.9 or later |
| NumPy | 1.24 |
| pandas | 1.5 |
| ucimlrepo | 0.0.7 or later, only required for re-downloading UCI data |

Install the required packages:

```bash
pip install numpy pandas ucimlrepo
```

No deep-learning framework is required. The neural networks and Adam optimizer
used for these reproducibility scripts are implemented in `nnlite.py`.

## Data sources

The datasets are publicly available from the UCI Machine Learning Repository:

- Concrete Compressive Strength:
  https://archive.ics.uci.edu/dataset/165/concrete+compressive+strength

  Citation: Yeh, I.-C. (1998). Modeling of strength of high-performance
  concrete using artificial neural networks. Cement and Concrete Research,
  28(12), 1797-1808.

- Energy Efficiency:
  https://archive.ics.uci.edu/dataset/242/energy+efficiency

  Citation: Tsanas, A., and Xifara, A. (2012). Accurate quantitative estimation
  of energy performance of residential buildings using statistical machine
  learning tools. Energy and Buildings, 49, 560-567.

The CSV files in this package were downloaded from UCI and are included to make
the reported results directly reproducible. To re-download and validate the
files against the reported checksums, run:

```bash
python prepare_data.py
```

## Reproducing the results

Run each script from this directory:

```bash
cd real_data_experiments

python vl_concrete_np.py
python vl_concrete_gap.py
python vl_energy_np.py
python vl_robotic_benchmark.py
python vl_sequential_ablation.py
```

Approximate runtimes on a single CPU core are:

| Script | Approximate runtime |
|---|---:|
| `vl_concrete_np.py` | 8 seconds |
| `vl_concrete_gap.py` | 5 seconds |
| `vl_energy_np.py` | 12 seconds |
| `vl_robotic_benchmark.py` | 60 seconds |
| `vl_sequential_ablation.py` | 5 seconds |

Results are deterministic. Re-running the scripts with the same environment
should reproduce the JSON outputs.

## Random seeds

| Purpose | Seed value |
|---|---|
| Train, validation, and test split | 0 |
| Surrogate network initialization and training | 1 for surrogate 1; 2 for surrogate 2 where applicable |
| Verifier network initialization and training | 7 |
| Per-restart VL initialization | 100 through 119 for 20-restart scenarios |

## Output file structure

Each real-data JSON result file contains a `meta` object with dataset details,
model specifications, hyperparameters, and runtime, plus a `results` list with
one entry per scenario. Scenario entries report restart-level surrogate and
verifier statistics and a `best` object for the restart selected by surrogate
loss.

The robotic benchmark JSON file keys `results` by method name. Each method
reports `median`, `mean`, `std`, `p95`, `failures`, `max_error`,
`out_of_domain`, `time_s`, and `per_target_error`. The failure count is the
number of the 200 targets whose re-simulation error exceeds the threshold
stored in the file metadata.

## Notes on the NumPy implementation

The implementation in `nnlite.py` is intentionally small and dependency-light.
It implements forward passes, backpropagation, and Adam updates in NumPy. The
virtual layer is represented by the optimized input vector itself, and the
projected update is implemented as a gradient step followed by projection onto
the feasible set. This is mathematically equivalent to the layer-weight
formulation described in the manuscript.
