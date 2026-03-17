# FHMM for Joint Haze and Dust Classification

This repository contains the code used to classify haze and dust events from meteorological and air-quality observations with a Factorial Hidden Markov Model (FHMM).

If you only want to understand or run the final method, start with `M2d.py`. That is the integrated final script used for the full pipeline.

## Start Here

The main script is:

```bash
python M2d.py
```

You can also use the helper launcher:

```bash
python run_model.py
```

By default, `run_model.py` runs `M2d.py`.

## Quick Start

1. Install the dependencies.

```bash
python -m pip install -r requirements.txt
```

2. Keep these files in the same repository folder:

- `M2d.py`
- `Data.xls`
- `Data_history.xls`
- `M2_gammas.npy`
- `data_of_3d_macro_micro_F1.txt`

3. Run the final model.

```bash
python M2d.py
```

## What `M2d.py` Does

`M2d.py` is the final integrated version of the project. It combines:

- FHMM inference
- Gaussian-copula observation modelling
- mutual-information-based observation weighting
- global weight tuning in Viterbi decoding
- ROC and F1 evaluation
- 3D performance-surface plotting
- result visualization utilities

If you are reading the repository for the first time, this is the only script you really need.

## What the Other Scripts Are For

The files `M0.py` to `M2c.py` are older comparison versions kept for reproducibility and paper ablation.

- `M0.py`: baseline FHMM
- `M1a.py`, `M1b.py`: joint log-normal comparison versions
- `M2a.py`, `M2b.py`, `M2c.py`: Gaussian-copula comparison versions

They are useful if you want to reproduce the model-development path in the paper, but they are not the main entry of the repository.

## Main Files

- `M2d.py`: final integrated script
- `Data.xls`: main evaluation data
- `Data_history.xls`: historical data used for initialization / comparison
- `M0_gammas.npy`, `M1_gammas.npy`, `M2_gammas.npy`: stored posterior probabilities used for ROC / evaluation plots
- `data_of_3d_macro_micro_F1.txt`: precomputed grid data for the 3D figure in the final model
- `run_model.py`: small launcher for running the scripts from the repository root
- `repo_paths.py`: helper for repository-relative file loading

## Environment

The code was originally developed with Python 3.9.

Dependencies:

- `numpy`
- `pandas`
- `scipy`
- `scikit-learn`
- `matplotlib`
- `xlrd`

## Contact

Tianhao Zhang  
Email: `Tianhao_Zhang@outlook.sg`
