[English](README.md) · 🌐 **Español**

# 🛡️ TurboGuard — Mantenimiento Predictivo y Pronóstico de RUL

> **Pipeline production-grade de mantenimiento predictivo para equipos industriales pesados: modelos de supervivencia (Weibull, Cox), pronóstico de Vida Útil Remanente (RUL) con deep learning, detección de anomalías sobre series multivariadas de sensores, más un caso de estudio opcional aplicado a minería (molinos SAG).**

Proyecto end-to-end que ingiere series temporales multivariadas de sensores de equipos industriales pesados (dataset NASA C-MAPSS de degradación de turbinas como benchmark principal, con un caso de estudio opcional de molino SAG), construye features físicos y operacionales, y entrena un portafolio de modelos complementarios — **Weibull paramétrico**, **Cox PH semi-paramétrico**, **LSTM para pronóstico de RUL**, **Isolation Forest + autoencoder LSTM para anomalías**, y baselines de gradient boosting — todo trackeado en MLflow y validado en un pipeline de CI.

> ⚠️ **Estado — work in progress (iniciado 2026-04-26).** El scaffolding y la ingesta de datos ya están operativos; los notebooks de modelado y la integración con MLflow Registry se incorporan iterativamente. Ver el [Roadmap](#-roadmap) para el seguimiento de hitos.

---

## 🎯 Lo que demuestra este proyecto

| Capacidad | Evidencia |
|---|---|
| **Análisis de supervivencia** | Weibull AFT (paramétrico) + Cox PH (semi-paramétrico) con `lifelines` y `scikit-survival` |
| **Vida Útil Remanente (RUL)** | LSTM en PyTorch + baselines XGBoost/LightGBM; función de scoring NASA |
| **Mantenimiento predictivo** | Clasificación de falla por ciclo, regresión time-to-failure, umbrales de alerta temprana |
| **Series multivariadas de sensores** | 21 sensores × 100+ unidades; rolling stats, FFT, features de change-point |
| **Detección de anomalías sobre IoT/sensores** | Isolation Forest + autoencoder LSTM, con explicabilidad SHAP |
| **Rigor MLOps** | MLflow tracking + registry, model signatures, CI/CD GitHub Actions, pytest, ruff |
| **Amplitud de dominio** | Turbinas aeroespaciales (principal) + caso de estudio molino SAG en minería (secundario) |

---

## 🏗️ Arquitectura

```
┌──────────────────────┐     ┌─────────────────────┐     ┌──────────────────────┐
│  NASA C-MAPSS        │────▶│  Raw Delta / Parquet│────▶│  Feature store       │
│  (4 datasets,        │     │  (sensor TS, ops)   │     │  (rolling, FFT,      │
│   21 sensores)       │     └─────────────────────┘     │   change-point)      │
└──────────────────────┘                                 └──────────┬───────────┘
                                                                    │
                          ┌─────────────────────────────────────────┤
                          ▼                                         ▼
              ┌─────────────────────────┐                ┌─────────────────────┐
              │   Modelos supervivencia │                │   Regresión RUL     │
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
              │  Demo Streamlit      │      │  Detección de anomalías│
              │  (RUL + curvas riesgo)│      │  (IsoForest + LSTM AE)│
              └──────────────────────┘      └────────────────────────┘
```

---

## 📂 Estructura del proyecto

```
.
├── src/
│   ├── data/             # Ingesta C-MAPSS + datos sintéticos SAG mill
│   ├── features/         # Rolling stats, frequency-domain, change-point
│   ├── models/
│   │   ├── survival/     # Weibull, Cox PH
│   │   ├── rul/          # XGBoost, LightGBM, LSTM
│   │   └── anomaly/      # Isolation Forest, autoencoder LSTM
│   ├── serving/          # Demo Streamlit, FastAPI (planificado)
│   └── utils/            # Helpers MLflow, logging, configs
├── notebooks/            # Notebooks de EDA y experimentos
├── data/                 # raw / interim / processed (gitignored)
├── configs/              # Configs YAML por dataset / modelo
├── scripts/              # download_data.sh, train_*.py
├── tests/                # Suite pytest
├── .github/workflows/    # CI: ruff + pytest
└── docs/                 # ADRs, dataset cards
```

---

## 🚀 Inicio rápido

```bash
# 1. Clonar
git clone https://github.com/jsanchez-ds/turboguard-predictive-maintenance.git
cd turboguard-predictive-maintenance

# 2. Instalar (Python 3.11+ recomendado)
pip install -e ".[dev]"

# 3. Descargar dataset NASA C-MAPSS (~70 MB)
bash scripts/download_data.sh

# 4. Abrir el notebook de EDA
jupyter lab notebooks/00_eda_cmapss.ipynb
```

---

## 📊 Datasets

### Principal: NASA C-MAPSS Turbofan Engine Degradation
- 4 sub-datasets (FD001–FD004) de complejidad operacional creciente
- 100+ turbinas por sub-dataset, run-to-failure
- 21 mediciones de sensores + 3 settings operacionales, muestreados por ciclo
- Benchmark estándar en literatura de mantenimiento predictivo
- Fuente: [NASA Prognostics Data Repository](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/)

### Secundario (planificado): Caso de estudio molino SAG
- Datos sintéticos de sensores de molino SAG basados en literatura pública minera (papers de Codelco / BHP / Antofagasta)
- Potencia, vibración de descansos, throughput, dureza del mineral
- Demuestra transferibilidad de la arquitectura desde aeroespacial a equipos pesados en minería

---

## 🧪 Enfoque de modelado

| Familia de modelos | Librería | Objetivo | Notas |
|---|---|---|---|
| Weibull AFT (paramétrico) | `lifelines` | T (tiempo a falla) | Curva de supervivencia cerrada, hazard interpretable |
| Cox Proportional Hazards | `lifelines`, `scikit-survival` | hazard ratio | Semi-paramétrico, rico en covariables |
| XGBoost / LightGBM | `xgboost`, `lightgbm` | regresión RUL | Baselines tabulares con features |
| LSTM (PyTorch) | `torch` | regresión RUL | Sensible a secuencia, sliding-window |
| Isolation Forest | `scikit-learn` | score de anomalía | No supervisado, explicable con SHAP |
| Autoencoder LSTM | `torch` | error de reconstrucción | Anomalías sobre flujos de sensores |

Validación: estratificada por ID de turbina + bloqueo temporal. Métricas: RMSE, función de scoring NASA, MAE, índice de concordancia (C-index) para modelos de supervivencia.

---

## 🗺️ Roadmap

- [x] Scaffolding del repo + READMEs bilingües + esqueleto de CI
- [x] Script de descarga de dataset (NASA C-MAPSS)
- [ ] Notebook de EDA (`00_eda_cmapss.ipynb`)
- [ ] Pipeline de feature engineering (rolling, FFT, change-point)
- [ ] Baselines XGBoost / LightGBM para RUL + tracking MLflow
- [ ] LSTM RUL con PyTorch + data loader sliding-window
- [ ] Modelos de supervivencia Weibull AFT + Cox PH
- [ ] Detección de anomalías Isolation Forest + autoencoder LSTM
- [ ] Demo Streamlit (RUL + curvas de riesgo)
- [ ] Caso sintético de molino SAG (Fase 2 opcional)
- [ ] Workflow de promoción en model registry + monitoreo de drift

---

## 📜 Licencia

MIT — ver [LICENSE](LICENSE).

---

## 👤 Autor

**Jonathan Sánchez Pesantes** — Ingeniero Civil Industrial · Data Scientist
🔗 [linkedin.com/in/jonasanchez](https://www.linkedin.com/in/jonasanchez) · [github.com/jsanchez-ds](https://github.com/jsanchez-ds)
