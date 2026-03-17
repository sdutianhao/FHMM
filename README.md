# Joint Classification of Haze and Dust via FHMM

This repository contains the reference Python implementation for a Factorial Hidden Markov Model (FHMM) framework that jointly classifies haze and dust events from multivariate meteorological and air-quality observations.

The repository includes:

- seven standalone model scripts: `M0.py`, `M1a.py`, `M1b.py`, `M2a.py`, `M2b.py`, `M2c.py`, and `M2d.py`
- the two data files used by the scripts: `Data.xls` and `Data_history.xls`
- precomputed posterior probability files: `M0_gammas.npy`, `M1_gammas.npy`, and `M2_gammas.npy`
- precomputed grid-search points for the final model: `data_of_3d_macro_micro_F1.txt`
- a lightweight launcher script: `run_model.py`

## Model Overview

MI denotes mutual information.

| Code | Correlation structure | Observation weight | Global optimization | Description |
| --- | --- | --- | --- | --- |
| `M0` | None | None | No | Baseline FHMM without dependence modelling. |
| `M1a` | Joint log-normal | None | No | Adds a covariance-based dependence structure. |
| `M1b` | Joint log-normal | Normalized MI | No | Adds MI-based weighting on top of `M1a`. |
| `M2a` | Gaussian copula | None | No | Uses a Gaussian copula to model nonlinear dependence. |
| `M2b` | Gaussian copula | Raw MI | No | Adds raw MI weighting on top of `M2a`. |
| `M2c` | Gaussian copula | Normalized MI | No | Adds normalized MI weighting on top of `M2a`. |
| `M2d` | Gaussian copula | Normalized MI | Yes | Final optimized model that jointly tunes the weight scale and global weight `v`. |

## Environment

The original experiments were developed with Python 3.9. A practical starting environment is:

```bash
python -m pip install -r requirements.txt
```

`requirements.txt` lists the core runtime dependencies used by the scripts:

- `numpy`
- `pandas`
- `scipy`
- `scikit-learn`
- `matplotlib`
- `xlrd`

## Repository Layout

```text
FHMM/
├── Data.xls
├── Data_history.xls
├── M0.py
├── M1a.py
├── M1b.py
├── M2a.py
├── M2b.py
├── M2c.py
├── M2d.py
├── M0_gammas.npy
├── M1_gammas.npy
├── M2_gammas.npy
├── data_of_3d_macro_micro_F1.txt
├── repo_paths.py
├── run_model.py
└── requirements.txt
```

## Quick Start

List available models:

```bash
python run_model.py --list-models
```

Check that the files required by a model are present:

```bash
python run_model.py M2d --check-only
```

Run a model:

```bash
python run_model.py M2c
```

The launcher always runs the selected script from the repository root, so the model scripts can find `Data.xls`, `Data_history.xls`, and the corresponding `.npy` assets even when you launch them from another directory.

## Required Assets by Model

- `M0`: `Data.xls`, `Data_history.xls`, `M0_gammas.npy`
- `M1a`, `M1b`: `Data.xls`, `Data_history.xls`, `M1_gammas.npy`
- `M2a`, `M2b`, `M2c`: `Data.xls`, `Data_history.xls`, `M2_gammas.npy`
- `M2d`: `Data.xls`, `Data_history.xls`, `M2_gammas.npy`, `data_of_3d_macro_micro_F1.txt`

## Reproducibility Notes

- The scripts are research scripts rather than a packaged Python library.
- By default, they skip the full EM optimization and reuse the stored jump / posterior artifacts required for evaluation and plotting.
- Full EM runs are computationally expensive on a laptop and may take several hours.
- The scripts generate figures with `matplotlib`, so running them in a desktop Python environment is recommended.

## Notes on the Current Codebase

- The model scripts remain intentionally close to the original research code.
- `run_model.py` is provided as a thin reproducibility wrapper rather than a behavioral rewrite of the original experiments.
- Paths to the bundled data and gamma files are now resolved relative to the repository itself, which makes the scripts more robust when launched outside the repo root.

## Associated Research

This repository accompanies the FHMM-based haze and dust classification study described in the paper:

`Joint Classification of Haze and Dust Events Using Factorial Hidden Markov Model Framework`

## Contact

Tianhao Zhang  
Email: `Tianhao_Zhang@outlook.sg`
