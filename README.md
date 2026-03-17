# FHMM Haze-Dust Classifier

This repository contains the final integrated FHMM workflow for joint haze and dust event classification.

The main program is:

```bash
python fhmm_haze_dust.py
```

## What This Branch Contains

The current `main` branch is focused on the final integrated workflow only. It keeps the final script and the files required to run it:

```text
fhmm_haze_dust.py
Data.xls
Data_history.xls
M2_gammas.npy
data_of_3d_macro_micro_F1.txt
requirements.txt
README.md
```

This branch does not expose the paper's older comparison scripts as separate runnable entries anymore. It is the final `M2d` workflow in cleaned form.

## Install

Install dependencies with:

```bash
python -m pip install -r requirements.txt
```

## How To Run

### 1. Default mode: skip EM

By default, the script skips the expensive EM optimization and directly loads the stored final parameter preset inside the code. It also loads `M2_gammas.npy` for ROC analysis.

Run:

```bash
python fhmm_haze_dust.py
```

This is the recommended way to reproduce the final integrated workflow quickly.

### 2. Full EM mode

If you want to run EM instead of using the stored preset:

```bash
python fhmm_haze_dust.py --run-em
```

By default, the EM posteriors are written back to:

```text
M2_gammas.npy
```

You can save them to another file with:

```bash
python fhmm_haze_dust.py --run-em --gamma-file M2_gammas_em.npy
```

## Input Files

- `Data.xls`: main evaluation data
- `Data_history.xls`: historical data used for initialization and transition estimation
- `M2_gammas.npy`: stored posterior probabilities used for ROC plotting in default mode
- `data_of_3d_macro_micro_F1.txt`: precomputed data used by the 3D Macro-F1 / Micro-F1 routine

## What The Script Does

`fhmm_haze_dust.py` includes the full final workflow:

- load current and historical observations
- initialize FHMM parameters
- either run EM or skip EM and load the final stored preset
- run weighted Viterbi decoding
- compute ROC curves
- evaluate confusion-matrix / F1 metrics
- draw hidden-state differentiation plots
- draw the 3D Macro-F1 / Micro-F1 surface

## About Paper Model Switching

The original paper discussed multiple model variants such as `M0`, `M1a`, `M1b`, `M2a`, `M2b`, `M2c`, and `M2d`.

The current `main` branch does not provide those variants as separate top-level runnable files anymore. It keeps only the final integrated workflow corresponding to the final model line.

So:

- if you want the final integrated workflow, use the current branch
- if you want a full model-by-model comparison interface, that is not currently exposed in this trimmed main branch

## Notes

- Full EM can be slow.
- Default execution is designed to be the practical reproduction path.
- File paths are resolved relative to the script location, so the repository should be run as a self-contained folder.
