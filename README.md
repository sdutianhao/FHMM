# FHMM Haze-Dust Project

This repository is centered on one integrated final script: `M2d.py`.

`M2d.py` is the actual main project entry and bundles the full final workflow used in the paper, including:

- FHMM inference
- Gaussian-copula observation modelling
- mutual-information-based observation weighting
- global weight tuning for Viterbi decoding
- ROC / F1 evaluation
- 3D performance-surface generation
- plotting and analysis utilities

The other scripts, `M0.py` to `M2c.py`, are preserved mainly as historical comparison / ablation versions used to show how the method evolves. They are not the main integrated project entry.

## Main Entry

Run the integrated final version directly:

```bash
python M2d.py
```

If you prefer the helper launcher:

```bash
python run_model.py
```

By default, `run_model.py` launches `M2d.py`.

## Files

- `M2d.py`: final integrated script
- `M0.py` to `M2c.py`: historical baselines / ablation scripts
- `Data.xls`: main evaluation data
- `Data_history.xls`: historical data used for initialization / comparison
- `M0_gammas.npy`, `M1_gammas.npy`, `M2_gammas.npy`: stored posterior probabilities for plotting / evaluation
- `data_of_3d_macro_micro_F1.txt`: precomputed grid points used by `M2d.py` for the 3D figure
- `repo_paths.py`: repository-relative path helper
- `run_model.py`: lightweight launcher
- `requirements.txt`: Python dependencies

## Environment

The original code was developed with Python 3.9. A practical setup command is:

```bash
python -m pip install -r requirements.txt
```

Core dependencies:

- `numpy`
- `pandas`
- `scipy`
- `scikit-learn`
- `matplotlib`
- `xlrd`

## Notes

- `M2d.py` already integrates the final design. If you only want the main project, this is the file to read and run first.
- `M0.py` to `M2c.py` are kept for reproducibility and paper comparison, not because the repository has seven equal main programs.
- The added path helper makes the scripts less fragile when launched outside the repository root.

## Contact

Tianhao Zhang  
Email: `Tianhao_Zhang@outlook.sg`
