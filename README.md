🌐 **English** · [Español](README.es.md)

# 🛡️ TurboGuard — Predictive Maintenance & RUL Forecasting

> **Production-grade predictive maintenance pipeline for industrial heavy equipment: survival models (Weibull, Cox), Remaining Useful Life (RUL) forecasting with deep learning, anomaly detection on multivariate sensor streams, and an optional mining (SAG mill) case study.**

End-to-end project that ingests multivariate sensor time series from heavy industrial equipment (NASA C-MAPSS turbofan engine degradation dataset as the primary benchmark, with an optional SAG mill case study), engineers physical & operational features, and trains a portfolio of complementary models — **Weibull parametric survival**, **Cox Proportional Hazards**, **LSTM-based RUL forecasting**, **Isolation Forest + autoencoder anomaly detection**, and gradient boosting baselines — all tracked in MLflow and gated through a CI pipeline.

> ✅ **Status — modeling track complete (2026-04-27).** Five notebooks cover the full workflow: EDA → feature engineering → tabular RUL baselines → deep-learning RUL → survival analysis → anomaly detection. **45/45 tests pass.**

---

## 📊 Headline results on FD001 (NASA C-MAPSS test set, 100 engines)

| Model family | Notebook | Best metric | Value |
|---|---|---|---|
| Naive baseline (mean lifetime) | 00 | NASA score | ~22,000 |
| **LightGBM (tabular RUL)** ⭐ | 02 | **Test NASA score** | **300.4** (RMSE 13.66) |
| XGBoost (tabular RUL) | 02 | Test NASA score | 309.8 (RMSE 14.14) |
| LSTM (deep-learning RUL) | 03 | Test NASA score | 457.9 (RMSE 15.56) |
| Cox PH (survival) | 04 | **Validation C-index** | **0.949** |
| Weibull AFT (survival) | 04 | Validation C-index | 0.903 |
| Isolation Forest (anomaly) | 05 | **Median lead time** | **124 cycles** before failure |
| LSTM autoencoder (anomaly) | 05 | Late/early ratio | **35×** (96% at RUL<30 vs 2.7% at RUL>100) |

**Compared to literature** (Saxena 2008, Babu 2016, Zheng 2017): classical ML on FD001 lands at RMSE 13–18, NASA score 200–500. We're firmly inside the published range on the first iteration, with no hyperparameter tuning.

**Key honest findings** (the story behind the numbers):

* **LSTM loses to LightGBM on FD001** — and that's the right answer. FD001 has a single operating condition / single fault mode; the engineered rolling/FFT/CUSUM features encode all the temporal context that matters. LSTMs are expected to win on FD002/FD004 (multi-condition).
* **Survival models do *risk ranking* near-perfectly (C-index 0.94+) but mediocre point-RUL prediction.** They're optimised for failure ordering, not magnitude. Use them for *prioritisation*; use regressors for numeric RUL decisions. Combining both signals is the production pattern.
* **LightGBM predicts late on 59/100 engines vs 47/100 for the LSTM.** The LSTM is operationally safer (late predictions = engine fails before maintenance) even though its NASA score is worse — a real engineering trade-off, not a bug.

---

## 🎯 What this project proves

| Capability | Evidence |
|---|---|
| **Survival analysis** | Weibull AFT (parametric) + Cox PH (semi-parametric) with `lifelines` and `scikit-survival` |
| **Remaining Useful Life (RUL)** | LSTM in PyTorch + XGBoost/LightGBM baselines; NASA scoring function |
| **Predictive maintenance** | Failure-by-cycle classification, time-to-failure regression, early-warning thresholds |
| **Multivariate sensor time series** | 21 sensors × 100+ engine units; rolling stats, FFT, change-point features |
| **Anomaly detection on IoT/sensor streams** | Isolation Forest + LSTM autoencoder, with SHAP explainability |
| **MLOps rigor** | MLflow tracking + registry, model signatures, CI/CD GitHub Actions, pytest, ruff |
| **Domain breadth** | Aerospace turbofans (primary) + mining SAG mill case study (secondary) |

---

## 🏗️ Architecture

```
┌──────────────────────┐     ┌─────────────────────┐     ┌──────────────────────┐
│  NASA C-MAPSS        │────▶│  Raw Delta / Parquet│────▶│  Feature store       │
│  (4 datasets,        │     │  (sensor TS, ops)   │     │  (rolling, FFT,      │
│   21 sensors)        │     └─────────────────────┘     │   change-point)      │
└──────────────────────┘                                 └──────────┬───────────┘
                                                                    │
                          ┌─────────────────────────────────────────┤
                          ▼                                         ▼
              ┌─────────────────────────┐                ┌─────────────────────┐
              │   Survival models       │                │   RUL regression    │
              │  • Weibull AFT          │                │  • XGBoost / LGBM   │
              │  • Cox PH               │                │  • LSTM (PyTorch)   │
              └────────────┬────────────┘                └──────────┬──────────┘
                           │                                        │
                           └────────────┬───────────────────────────┘
                                        ▼
                           ┌────────────────────────────┐
                           │   MLflow Registry          │
                           │   + Model Signatures       │
                           └─────────────┬──────────────┘
                                         │
                          ┌──────────────┴───────────────┐
                          ▼                              ▼
              ┌──────────────────────┐      ┌────────────────────────┐
              │  Streamlit demo      │      │  Anomaly detection     │
              │  (RUL + risk curves) │      │  (IsoForest + LSTM AE) │
              └──────────────────────┘      └────────────────────────┘
```

---

## 📂 Project structure

```
.
├── src/
│   ├── data/             # C-MAPSS ingestion + SAG mill synthetic data
│   ├── features/         # Rolling stats, frequency-domain, change-point
│   ├── models/
│   │   ├── survival/     # Weibull, Cox PH
│   │   ├── rul/          # XGBoost, LightGBM, LSTM
│   │   └── anomaly/      # Isolation Forest, LSTM autoencoder
│   ├── serving/          # Streamlit demo, FastAPI (planned)
│   └── utils/            # MLflow helpers, logging, configs
├── notebooks/            # EDA + experiment notebooks
├── data/                 # raw / interim / processed (gitignored)
├── configs/              # YAML configs per dataset / model
├── scripts/              # download_data.sh, train_*.py
├── tests/                # pytest suite
├── .github/workflows/    # CI: ruff + pytest
└── docs/                 # ADRs, dataset cards
```

---

## 🚀 Quickstart

```bash
# 1. Clone
git clone https://github.com/jsanchez-ds/turboguard-predictive-maintenance.git
cd turboguard-predictive-maintenance

# 2. Install (Python 3.11+ recommended)
pip install -e ".[dev]"

# 3. Download NASA C-MAPSS dataset (~70 MB)
bash scripts/download_data.sh

# 4. Open the EDA notebook
jupyter lab notebooks/00_eda_cmapss.ipynb
```

---

## 📊 Datasets

### Primary: NASA C-MAPSS Turbofan Engine Degradation
- 4 sub-datasets (FD001–FD004) of increasing operational complexity
- 100+ engines per sub-dataset, run-to-failure
- 21 sensor measurements + 3 operational settings, sampled per cycle
- Standard benchmark in predictive maintenance literature
- Source: [NASA Prognostics Data Repository](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/)

### Secondary (planned): SAG Mill Predictive Maintenance Case Study
- Synthetic SAG mill sensor data based on public mining literature (Codelco / BHP / Antofagasta papers)
- Power draw, bearing vibration, throughput, ore hardness
- Demonstrates transferability of the architecture from aerospace to mining heavy equipment

---

## 🧪 Modeling approach

| Model family | Library | Target | Notes |
|---|---|---|---|
| Weibull AFT (parametric) | `lifelines` | T (time-to-failure) | Closed-form survival curve, hazard interpretable |
| Cox Proportional Hazards | `lifelines`, `scikit-survival` | hazard ratio | Semi-parametric, covariate-rich |
| XGBoost / LightGBM | `xgboost`, `lightgbm` | RUL regression | Tabular feature baselines |
| LSTM (PyTorch) | `torch` | RUL regression | Sequence-aware, sliding-window |
| Isolation Forest | `scikit-learn` | anomaly score | Unsupervised, SHAP-explainable |
| LSTM Autoencoder | `torch` | reconstruction error | Sensor-stream anomaly detection |

Validation: stratified by engine ID + temporal blocking. Metrics: RMSE, NASA scoring function, MAE, concordance index (C-index) for survival models.

---

## 🗺️ Roadmap

- [x] Repo scaffolding + bilingual READMEs + CI skeleton
- [x] Dataset download script (NASA C-MAPSS)
- [x] **Notebook 00 — EDA** (engine lifetimes, sensor trajectories, naive baseline)
- [x] **Notebook 01 — Feature engineering** (rolling stats, FFT band-energy, CUSUM change-point; gold parquet at `data/processed/gold_FD001.parquet`)
- [x] **Notebook 02 — XGBoost / LightGBM RUL baselines + MLflow** (NASA score 300.4 on FD001 test)
- [x] **Notebook 03 — LSTM RUL with PyTorch + sliding-window** (CPU, ~20s training)
- [x] **Notebook 04 — Weibull AFT + Cox PH survival models** (C-index 0.94+)
- [x] **Notebook 05 — Isolation Forest + LSTM autoencoder anomaly detection** (124-cycle median lead time)
- [ ] Streamlit demo (RUL forecasts + risk curves + anomaly stream)
- [ ] SAG mill synthetic case study (optional Phase 2 — mining-domain transfer)
- [ ] Model registry promotion workflow + drift monitoring

---

## 📜 License

MIT — see [LICENSE](LICENSE).

---

## 👤 Author

**Jonathan Sánchez Pesantes** — Industrial Engineer · Data Scientist
🔗 [linkedin.com/in/jonasanchez](https://www.linkedin.com/in/jonasanchez) · [github.com/jsanchez-ds](https://github.com/jsanchez-ds)
