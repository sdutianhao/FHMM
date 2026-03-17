# FHMM Haze-Dust Classifier

This repository contains one main program: `fhmm_haze_dust.py`.

`fhmm_haze_dust.py` is the final integrated version of the project. It runs the FHMM-based haze-dust classification pipeline, including weighting, evaluation, plotting, and the final 3D performance analysis used in the paper.

## Run

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Run the main program:

```bash
python fhmm_haze_dust.py
```

## Required Files

Keep these files in the repository root:

- `fhmm_haze_dust.py`
- `Data.xls`
- `Data_history.xls`
- `M2_gammas.npy`
- `data_of_3d_macro_micro_F1.txt`
- `requirements.txt`

## What It Produces

`fhmm_haze_dust.py` includes the code for:

- FHMM inference
- ROC and F1 evaluation
- result plots
- 3D Macro-F1 / Micro-F1 surface analysis

## Contact

Tianhao Zhang  
Email: `Tianhao_Zhang@outlook.sg`
