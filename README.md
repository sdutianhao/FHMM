# Joint Haze-and-Dust Classification via FHMM

This repository contains a **Python implementation** together with a **sample dataset** that demonstrates how to use a Factorial Hidden Markov Model (FHMM) to jointly classify haze and dust events from multi-channel observational sequences.

- `run_fhmm.py`  Main script: data loading → parameter initialization → EM iterations → Viterbi decoding → result output  
- `sample_data.xls`, `sample_history.xls`  Sample data: four observation channels (PM10, wind speed, visibility, relative humidity) with timestamps  
  > These two datasets are real-world measurements collected from **Beijing Capital International Airport** between **2022 and 2024**, including hourly meteorological and air quality data.
- `Output.txt`  
  Sample output log from running the FHMM model, showcasing how the most probable hidden state is inferred over time.  
  > Output includes emission probabilities, transition scores, and selected time steps (e.g., `t = 3`, `t = 8475`); intermediate steps are omitted for brevity.


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
