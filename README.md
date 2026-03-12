# 🌲 Universal Indian Forest Phenology Predictor — v6

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/Streamlit-1.32%2B-FF4B4B?logo=streamlit&logoColor=white" />
  <img src="https://img.shields.io/badge/scikit--learn-1.4%2B-F7931E?logo=scikit-learn&logoColor=white" />
  <img src="https://img.shields.io/badge/Design-Monsoon--Aware-brightgreen" />
  <img src="https://img.shields.io/badge/License-MIT-lightgrey" />
</p>

---

<p align="center">

  <!-- ▶ LIVE APP — one click to open the deployed Streamlit app -->
  <a href="https://indian-forest-phenology-v6.streamlit.app/" target="_blank">
    <img src="https://img.shields.io/badge/▶%20Open%20Live%20App-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Open Live App" />
  </a>

  &nbsp;&nbsp;

  <!-- 🚀 DEPLOY YOUR OWN — one click to deploy a fresh copy on Streamlit Cloud -->
  <a href="https://streamlit.io/deploy?repository=https://github.com/shreejisharma/Indian-forest-phenology&branch=main&mainModule=app/universal_Indian_forest_phenology_v6.py" target="_blank">
    <img src="https://img.shields.io/badge/🚀%20Deploy%20Your%20Own-0068C8?style=for-the-badge&logo=streamlit&logoColor=white" alt="Deploy your own" />
  </a>

</p>

---

A **monsoon-aware, causally correct Streamlit application** for extracting and predicting phenological events — **Start of Season (SOS)**, **Peak of Season (POS)**, **End of Season (EOS)**, and **Length of Season (LOS)** — across all Indian forest types.

> **Key upgrade from v5:** v5 always selected T2M as the dominant driver for every metric. This is **ecologically incorrect** for Indian monsoon forests where T2M varies by less than 1°C across seasons. v6 fixes this with monsoon-aware feature engineering, moisture protection, and variance-weighted feature ranking.

---

## What's Wrong with v5 (and Fixed in v6)

| Problem | v5 Behaviour | v6 Fix |
|---------|-------------|--------|
| T2M always wins | Every metric: `= a + b×T2M` | Variance-weighted ranking (CV<2% → down-ranked) |
| Moisture features dropped | PRECTOTCORR collinear with T2M → dropped | Moisture protection flag — keeps PRECTOTCORR over T2M |
| GDD_cum data leakage | Used as predictor | Excluded (leakage guard) |
| 30d window misses monsoon | SOS trigger is missed | 60d default for SOS, user-adjustable |
| Mean precip | Physical signal lost | `PRECTOTCORR_sum` = accumulated rain |

### The Root Cause
```
T2M range in sample data:  24.15 → 24.82°C  (Δ = 0.67°C)
PRECTOTCORR range:         1071 → 1559 mm   (Δ = 488 mm, CV=15%)

T2M cannot drive phenological differences of 17–36 DOY with <1°C variation.
PRECTOTCORR (monsoon onset) is the true causal driver.
```

---

## Correct Causal Drivers for Indian Monsoon Forests

### SOS (Green-up)
- ✅ **PRIMARY:** `PRECTOTCORR_sum` (60–90d window) — monsoon onset rainfall
- ✅ **SECONDARY:** `SPEI_proxy` — moisture surplus/deficit
- ✅ **SECONDARY:** `RH2M` — humidity confirming monsoon arrival  
- ❌ **NOT T2M** — < 1°C variation = noise, not signal

### POS (Peak NDVI)
- ✅ **PRIMARY:** Seasonal cumulative precipitation
- ✅ **PRIMARY:** `RH2M` during growing season
- ✅ **SECONDARY:** `ALLSKY_SFC_SW_DWN` — radiation for photosynthesis

### EOS (Senescence)
- ✅ **PRIMARY:** `WS2M` — dry post-monsoon winds
- ✅ **PRIMARY:** `VPD` — vapour pressure deficit
- ✅ **SECONDARY:** Precipitation cessation signal

---

## Features

### 📊 Data Overview Tab
- Auto-characterizes uploaded data: cadence, NDVI range, season detection
- **NEW: Variance diagnostic table** — shows CV% for each met variable
- Flags low-variation features that cannot meaningfully drive phenology

### 🔬 Training Tab
- Monsoon-aware feature engineering (PRECTOTCORR sum, 60d window default)
- **NEW: Causal diagnosis panel** — explains WHY each feature was selected or dropped
- Moisture protection: PRECTOTCORR/RH2M survive collinearity filter against T2M
- GDD_cum leakage guard — excluded from selection
- Feature role table with CV% column
- LOO cross-validated Ridge regression

### 📈 Correlations Tab
- Seasonal mean correlations with phenology metrics
- Year-by-year met variable plots

### 🔮 Predict Tab
- Inputs pre-filled with training data means
- Ecological order enforcement (SOS < POS < EOS)
- Predictions with MAE uncertainty

### 📖 Technical Guide
- Full explanation of v6 improvements
- Causal driver table for Indian monsoon forests
- Methodology documentation

---

## Local Installation

```bash
git clone https://github.com/shreejisharma/Indian-forest-phenology.git
cd Indian-forest-phenology
pip install -r requirements.txt
streamlit run app/universal_Indian_forest_phenology_v6.py
```

**One-click launchers:**
- 🪟 Windows: `run_app.bat`
- 🍎 macOS: `Run_Phenology_App.command`
- 🐧 Linux: `run_app.sh`

---

## Data Requirements

### NDVI CSV
```
Date,NDVI
2003-02-24,0.458
2003-03-01,0.437
```

### Meteorology CSV (NASA POWER)
Required columns: `T2M`, `T2M_MAX`, `T2M_MIN`, `PRECTOTCORR`, `RH2M`, `WS2M`, `ALLSKY_SFC_SW_DWN`, `VPD`

Auto-derived: `GDD_5`, `GDD_10`, `DTR`, `log_precip`, `SPEI_proxy`, `MSI`, `PRECTOTCORR_sum`

---

## Repository Structure

```
Indian-forest-phenology/
├── app/
│   ├── universal_Indian_forest_phenology_v5.py   ← v5 (original)
│   └── universal_Indian_forest_phenology_v6.py   ← v6 (monsoon-aware, THIS VERSION)
├── data/
│   ├── ndvi/combined_NDVI_2003_2007.csv
│   └── combined_MET_2003_2007.csv
├── docs/
│   └── v6_driver_analysis.md
├── run_app.bat
├── run_app.sh
├── requirements.txt
└── README.md
```

---

## Citation

```
Sharma, S. (2025). Universal Indian Forest Phenology Predictor v6 [Software].
GitHub. https://github.com/shreejisharma/Indian-forest-phenology
```

## License

MIT License — see [LICENSE](LICENSE) for details.
