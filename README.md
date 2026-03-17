# FHMM Haze-Dust Classifier

This repository contains the final integrated implementation of an FHMM-based classifier for joint haze and dust event recognition from meteorological and air-quality observations.

The repository is intentionally centered on a single runnable script, `fhmm_haze_dust.py`, together with the data and precomputed assets that the script expects.

## Overview

`fhmm_haze_dust.py` bundles the full analysis pipeline used in the final version of the project:

- data loading from the current and historical observation tables
- FHMM parameter initialization
- loading the final stored parameter configuration through `jumpEM_710C`
- weighted Viterbi decoding
- ROC computation from stored posterior probabilities
- classification evaluation and confusion-matrix-based metrics
- visualization of hidden-state differentiation
- 3D Macro-F1 / Micro-F1 surface plotting

The current script entry uses the stored final parameter set and the bundled posterior file `M2_gammas.npy` rather than rerunning the full EM workflow from scratch every time.

## Repository Contents

The main branch now keeps only the files required by the final integrated program:

```text
fhmm_haze_dust.py
Data.xls
Data_history.xls
M2_gammas.npy
data_of_3d_macro_micro_F1.txt
requirements.txt
README.md
```

## Required Files

Place the following files in the repository root before running the program:

- `fhmm_haze_dust.py`
- `Data.xls`
- `Data_history.xls`
- `M2_gammas.npy`
- `data_of_3d_macro_micro_F1.txt`
- `requirements.txt`

## Environment

The code was developed against a standard scientific Python stack. Install the dependencies with:

```bash
python -m pip install -r requirements.txt
```

Current dependency file:

- `numpy`
- `pandas`
- `scipy`
- `scikit-learn`
- `matplotlib`
- `xlrd`

## Run

Execute the final program from the repository root:

```bash
python fhmm_haze_dust.py
```

## Input Data

- `Data.xls`: main observation table used for evaluation
- `Data_history.xls`: historical table used for initialization and transition estimation
- `M2_gammas.npy`: stored posterior probabilities used for ROC analysis
- `data_of_3d_macro_micro_F1.txt`: precomputed grid data used by the 3D performance-surface routine

## What the Script Produces

Running `fhmm_haze_dust.py` triggers the final evaluation workflow and includes:

- terminal output for shapes, parameters, and evaluation summaries
- multiclass ROC plotting
- hidden-state differentiation plotting
- F1 / confusion-matrix evaluation
- 3D Macro-F1 / Micro-F1 surface visualization

## Notes

- This repository is not packaged as a reusable Python library; it is a project-specific research code release centered on one script.
- The final branch focuses on the integrated end result instead of keeping older comparison scripts in the main directory.
- File paths are resolved relative to the script location, so the project should be run as a self-contained folder.
