# 📈 Recursive Criticality of AI Self-Improvement: Replication & Data Mining Extension

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-latest-orange.svg)](https://scikit-learn.org/)

This repository contains the source code, synthetic datasets, and experimental results for our **Data Mining** course project. Our work focuses on analyzing, replicating, and extending the findings of the paper *"Recursive Criticality of AI Self-Improvement"* by Mikhail Burtsev (arXiv:2609.00137, 2026).

Beyond independently implementing the Delay Differential Equation (DDE) dynamical model to simulate AI capability trajectories, this project applies Machine Learning techniques (**Random Forest** and **K-Means clustering**) to diagnose recursive self-amplifying regimes from early observation windows.

---

## 🎯 Project Objectives

1. **Independent Replication (Author's Scenarios):** Re-implement the delayed dynamical model $\dot{x}(t)$ and multi-actor coupled networks using 2nd-order Heun and 4th-order Runge-Kutta integration methods to verify AGI and ASI threshold-crossing times.
2. **Synthetic Dataset Generation:** Generate a large-scale synthetic dataset of **4,000 capability trajectories** by randomly sampling model parameters (recursive gain $a$, frontier-hardening exponent $\beta$, operational-closure steepness $k$, feedback delay $\tau$, and research effort multiplier $m_r$).
3. **Data Mining & Regime Diagnosis (Extended Scenarios):** Extract 10 time-series features (slope, milestone timings, and trajectory shape) and evaluate whether supervised classification and unsupervised clustering can distinguish between **subcritical** ($\mathcal{R}_{\mathrm{AI}} \le 1$) and **supercritical** ($\mathcal{R}_{\mathrm{AI}} > 1$) regimes using only early observation windows (first 2–12 years).

---

## 📂 Repository Structure

```text
📦 recursive-criticality-data-mining
 ┣ 📂 code                  # Main Python source code
 ┃ ┣ 📜 rsi_model.py        # Single-actor DDE model (2nd-order Heun), R_AI(x), threshold crossings
 ┃ ┣ 📜 rsi_network.py      # Multi-actor coupled network model and time-dependent R_net(t)
 ┃ ┣ 📜 rsi_dataset.py      # Vectorized batch generation of 4,000 trajectories & R_peak labeling
 ┃ ┣ 📜 run_all.py          # Master script to run all experiments, export CSVs, and plot figures
 ┃ ┗ 📜 requirements.txt    # Required Python dependencies
 ┣ 📂 data                  # Generated datasets and evaluation results (CSV, JSON)
 ┃ ┣ 📜 tham_so.csv         # Sampled input parameters for 4,000 trajectories
 ┃ ┗ 📜 dac_trung.csv       # 10 extracted trajectory features and criticality labels
 ┣ 📂 figures               # Generated plots and visualizations (PDF, PNG)
 ┣ 📜 main.tex              # LaTeX source file for the project report (compiled with XeLaTeX)
 ┣ 📜 latexmkrc             # LaTeX build configuration
 ┗ 📜 README.md             # Project documentation
