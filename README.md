# Joint Haze-and-Dust Classification via FHMM

This repository provides Python implementations and a sample dataset to demonstrate how a Factorial Hidden Markov Model (FHMM) jointly classifies haze and dust events from multi-channel observational sequences.

Repository contents
- Model scripts: M0, M1a, M1b, M2a, M2b, M2c, M2d
- Sample data files: sample_data.xls, sample_history.xls  
  These two datasets are real-world measurements collected at Beijing Capital International Airport between 2022 and 2024, including hourly meteorological and air-quality observations.
- Stored posterior probability files: M0_gammas.npy, M1_gammas.npy, M2_gammas.npy
- 3D sampling points for M2d plotting: data_of_3d_macro_micro_F1.txt


---


## Requirements

| Dependency | Version |
| --- | --- |
| Python | 3.9.19 |
| NumPy | 1.21.6 |
| scikit-learn | 0.24.2 |
| Others | pandas, scipy, matplotlib |


---

## Models

The repository includes seven model configurations. MI denotes mutual information. Differences are summarised below.

| Code | Correlation structure | Obs. weight w | Global opt. | Key description |
| --- | --- | --- | --- | --- |
| M0  | None (independence) | None | No | Baseline FHMM that ignores inter-dimensional correlation. |
| M1a | Joint log-normal | None | No | Uses a covariance matrix to capture linear correlation. |
| M1b | Joint log-normal | Normalised MI | No | Adds mutual-information weights on top of M1a. |
| M2a | Gaussian copula | None | No | Uses a copula to capture non-linear correlation. |
| M2b | Gaussian copula | Raw MI | No | Adds unnormalised MI weights on top of M2a. |
| M2c | Gaussian copula | Normalised MI | No | Adds normalised MI weights on top of M2a. |
| M2d | Gaussian copula | Normalised MI | Yes | Final optimised model: jointly tunes Σw and the global weight v |

---

## Runtime and reproducibility

- EM iterations are computationally expensive on a laptop, typically 3–6 hours.
- Training is deterministic. With the same inputs and settings, repeated runs produce identical results.
- All scripts default to skipping EM and loading precomputed EM results for subsequent steps.
- To run the full EM procedure, comment out lines 3308–3310 and uncomment line 3313 in the selected model script.

ROC curves when skipping EM
- ROC plotting requires posterior probabilities (gammas) from EM. If you skip EM, download the corresponding stored gamma file:
  - M0_gammas.npy for M0
  - M1_gammas.npy for M1a and M1b
  - M2_gammas.npy for M2a, M2b, M2c, and M2d

---

## Quick start

- Choose a model code from the table above.
  
- Ensure the required support files are present.
  
- - M0 required files:  
    sample_data.xls, sample_history.xls, M0_gammas.npy

- - M1a, M1b required files:  
    sample_data.xls, sample_history.xls, M1_gammas.npy

- - M2a, M2b, M2c required files:  
    sample_data.xls, sample_history.xls, M2_gammas.npy

- - M2d required files:  
    sample_data.xls, sample_history.xls, M2_gammas.npy, data_of_3d_macro_micro_F1.txt
  
- Run



