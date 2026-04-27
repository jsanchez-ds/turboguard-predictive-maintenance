🌐 **English** · [Español](README.es.md)

# 🛡️ TurboGuard — Predictive Maintenance & RUL Forecasting

> **Production-grade predictive maintenance pipeline for industrial heavy equipment: survival models (Weibull, Cox), Remaining Useful Life (RUL) forecasting with deep learning, anomaly detection on multivariate sensor streams, and an optional mining (SAG mill) case study.**

End-to-end project that ingests multivariate sensor time series from heavy industrial equipment (NASA C-MAPSS turbofan engine degradation dataset as the primary benchmark, with an optional SAG mill case study), engineers physical & operational features, and trains a portfolio of complementary models — **Weibull parametric survival**, **Cox Proportional Hazards**, **LSTM-based RUL forecasting**, **Isolation Forest + autoencoder anomaly detection**, and gradient boosting baselines — all tracked in MLflow and gated through a CI pipeline.

> ⚠️ **Status — work in progress (started 2026-04-26).** Scaffolding + dataset ingestion are live; modeling notebooks and MLflow registry integration are being added iteratively. See the [Roadmap](#-roadmap) for milestone tracking.

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
- [ ] EDA notebook (`00_eda_cmapss.ipynb`)
- [ ] Feature engineering pipeline (rolling, FFT, change-point)
- [ ] XGBoost / LightGBM RUL baselines + MLflow tracking
- [ ] LSTM RUL with PyTorch + sliding-window data loader
- [ ] Weibull AFT + Cox PH survival models
- [ ] Isolation Forest + LSTM autoencoder anomaly detection
- [ ] Streamlit demo (RUL forecasts + risk curves)
- [ ] SAG mill synthetic case study (optional Phase 2)
- [ ] Model registry promotion workflow + drift monitoring

---

## 📜 License

MIT — see [LICENSE](LICENSE).

---

## 👤 Author

**Jonathan Sánchez Pesantes** — Industrial Engineer · Data Scientist
🔗 [linkedin.com/in/jonasanchez](https://www.linkedin.com/in/jonasanchez) · [github.com/jsanchez-ds](https://github.com/jsanchez-ds)
