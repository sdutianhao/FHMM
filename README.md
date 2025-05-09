# Joint Haze-and-Dust Classification via FHMM

This repository contains a **Python implementation** together with a **sample dataset** that demonstrates how to use a Factorial Hidden Markov Model (FHMM) to jointly classify haze and dust events from multi-channel observational sequences.

- `5.9.2025.py`  Main script: data loading → parameter initialization → EM iterations → Viterbi decoding → result output  
- `合并数据前1.xls` Sample data: four observation channels (PM10, wind speed, visibility, relative humidity) with timestamps

> ⚠️  If you replace or add data files, ensure that the column order matches the script logic.

---

## Requirements

| Dependency     | Version |
| -------------- | ------- |
| Python         | **3.9.19** |
| NumPy          | **1.21.6** |
| scikit-learn   | **0.24.2** |
| Others (auto-installed) | pandas, scipy, matplotlib, etc. |

```bash
# It is recommended to install dependencies in a virtual environment
pip install -r requirements.txt
