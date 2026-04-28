[English](README.md) · 🌐 **Español**

# 🛡️ TurboGuard — Mantenimiento Predictivo y Pronóstico de RUL

> **Pipeline production-grade de mantenimiento predictivo para equipos industriales pesados: modelos de supervivencia (Weibull, Cox), pronóstico de Vida Útil Remanente (RUL) con deep learning, detección de anomalías sobre series multivariadas de sensores, más un caso de estudio opcional aplicado a minería (molinos SAG).**

Proyecto end-to-end que ingiere series temporales multivariadas de sensores de equipos industriales pesados (dataset NASA C-MAPSS de degradación de turbinas como benchmark principal, con un caso de estudio opcional de molino SAG), construye features físicos y operacionales, y entrena un portafolio de modelos complementarios — **Weibull paramétrico**, **Cox PH semi-paramétrico**, **LSTM para pronóstico de RUL**, **Isolation Forest + autoencoder LSTM para anomalías**, y baselines de gradient boosting — todo trackeado en MLflow y validado en un pipeline de CI.

> ✅ **Estado — track de modelado completo (2026-04-27).** Cinco notebooks cubren el flujo completo: EDA → feature engineering → baselines tabulares de RUL → RUL con deep learning → análisis de supervivencia → detección de anomalías. **45/45 tests pasan.**

---

## 📊 Resultados principales en FD001 (test set NASA C-MAPSS, 100 turbinas)

| Familia de modelo | Notebook | Mejor métrica | Valor |
|---|---|---|---|
| Baseline ingenuo (lifetime promedio) | 00 | NASA score | ~22.000 |
| **LightGBM (RUL tabular)** ⭐ | 02 | **NASA score test** | **300.4** (RMSE 13.66) |
| XGBoost (RUL tabular) | 02 | NASA score test | 309.8 (RMSE 14.14) |
| LSTM (RUL deep learning) | 03 | NASA score test | 457.9 (RMSE 15.56) |
| Cox PH (supervivencia) | 04 | **C-index validación** | **0.949** |
| Weibull AFT (supervivencia) | 04 | C-index validación | 0.903 |
| Isolation Forest (anomalía) | 05 | **Lead time mediano** | **124 ciclos** antes de falla |
| Autoencoder LSTM (anomalía) | 05 | Ratio late/early | **35×** (96% en RUL<30 vs 2.7% en RUL>100) |

**Comparado con literatura** (Saxena 2008, Babu 2016, Zheng 2017): ML clásico en FD001 reporta RMSE 13–18, NASA score 200–500. Estamos en el rango publicado en la primera iteración, sin hyperparameter tuning.

**Hallazgos honestos clave** (la historia detrás de los números):

* **LSTM pierde contra LightGBM en FD001** — y esa es la respuesta correcta. FD001 tiene una sola condición operacional / un solo modo de falla; las features de rolling/FFT/CUSUM ya encapsulan todo el contexto temporal que importa. Los LSTMs deberían ganar en FD002/FD004 (multi-condición).
* **Los modelos de supervivencia hacen *ranking de riesgo* casi perfecto (C-index 0.94+) pero predicción de RUL puntual mediocre.** Están optimizados para ordenar fallas, no magnitudes. Úsalos para *priorización*; usa regresores para decisiones numéricas de RUL. Combinar ambas señales es el patrón productivo.
* **LightGBM predice tarde en 59/100 turbinas vs 47/100 del LSTM.** El LSTM es operacionalmente más seguro (predicción tardía = turbina falla antes del mantenimiento) aunque su NASA score sea peor — un trade-off real de ingeniería, no un bug.

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
- [x] **Notebook 00 — EDA** (lifetimes de turbinas, trayectorias de sensores, baseline ingenuo)
- [x] **Notebook 01 — Feature engineering** (rolling stats, FFT band-energy, change-point CUSUM; gold parquet en `data/processed/gold_FD001.parquet`)
- [x] **Notebook 02 — Baselines XGBoost / LightGBM + MLflow** (NASA score 300.4 en FD001 test)
- [x] **Notebook 03 — LSTM RUL con PyTorch + sliding-window** (CPU, ~20s entrenamiento)
- [x] **Notebook 04 — Weibull AFT + Cox PH supervivencia** (C-index 0.94+)
- [x] **Notebook 05 — Isolation Forest + autoencoder LSTM detección de anomalías** (lead time mediano 124 ciclos)
- [ ] Demo Streamlit (RUL + curvas de riesgo + stream de anomalías)
- [ ] Caso sintético de molino SAG (Fase 2 opcional — transferencia a dominio minero)
- [ ] Workflow de promoción en model registry + monitoreo de drift

---

## 📜 Licencia

MIT — ver [LICENSE](LICENSE).

---

## 👤 Autor

**Jonathan Sánchez Pesantes** — Ingeniero Civil Industrial · Data Scientist
🔗 [linkedin.com/in/jonasanchez](https://www.linkedin.com/in/jonasanchez) · [github.com/jsanchez-ds](https://github.com/jsanchez-ds)
