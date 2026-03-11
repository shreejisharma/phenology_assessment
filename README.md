# 🌲 Universal Indian Forest Phenology Predictor

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/Streamlit-1.32%2B-FF4B4B?logo=streamlit&logoColor=white" />
  <img src="https://img.shields.io/badge/scikit--learn-1.4%2B-F7931E?logo=scikit-learn&logoColor=white" />
  <img src="https://img.shields.io/badge/Forest%20Types-11-brightgreen" />
  <img src="https://img.shields.io/badge/Satellite-MODIS%20%7C%20Sentinel--2-blue" />
  <img src="https://img.shields.io/badge/License-MIT-lightgrey" />
</p>

A **data-driven Streamlit application** for extracting and predicting phenological events — **Start of Season (SOS)**, **Peak of Season (POS)**, and **End of Season (EOS)** — across 11 Indian forest types. Upload your own NDVI time series and NASA POWER meteorological data; the app derives site-specific regression equations with no hard-coded coefficients.

**[▶ Open Live App](https://indian-forest-phenology-pnlas9tfyhyoft2vmglxpm.streamlit.app/)**

---

## Overview

| Item | Detail |
|---|---|
| **Forest types** | 11 — Tropical Moist/Dry Deciduous, Evergreen, Shola, Thorn, Mangrove, NE India, Himalayan, Alpine/Subalpine |
| **Phenological events** | SOS · POS · EOS · LOS (Length of Season) |
| **NDVI input** | Any CSV with `Date` + `NDVI` columns (MODIS MOD13Q1, Sentinel-2, or other) |
| **Meteorological input** | NASA POWER daily export (headers auto-detected and skipped) |
| **Model** | Ridge Regression with RidgeCV auto-tuned α, Leave-One-Out cross-validation |
| **Feature selection** | Pearson \|r\| ≥ 0.40 filter → collinearity removal → incremental LOO R² check |
| **Minimum data** | 3 growing seasons (≥ 5 recommended for reliable R²) |

---

## Features

### 🔬 Training Tab
- Automatic NDVI smoothing (Savitzky-Golay) and phenology extraction
- 11 configurable forest-type season windows with ecologically justified defaults
- Optional IMD monsoon onset date input (strengthens SOS model for monsoon forests)
- Model performance cards (LOO R², MAE, number of seasons)
- Clean fitted equations displayed in tabbed layout (SOS / POS / EOS)
- Feature role table — colour-coded: IN MODEL · Excluded (spurious) · Not selected · Below threshold
- Observed vs Predicted scatter plots

### 📊 Correlations Tab
- Grouped bar chart: Pearson r for all features across SOS / POS / EOS
- Pearson r heatmap with significance stars (** p < 0.05, * p < 0.10)
- Scatter plots: best model driver vs observed event timing per event
- Detailed per-event correlation tables — r, |r|, significance (consistent with heatmap)

### 🔮 Predict Tab
- Event-scoped input fields — each event uses its own 15-day pre-event meteorological values
- Ecological order enforcement (SOS < POS < EOS with automatic correction)
- LOS, green-up lag, and senescence lag computed from predictions
- Download predictions as CSV

### 🌳 Forest Guide Tab
- Reference guide for all 11 Indian forest types
- Decision tree for forest type selection by rainfall, deciduousness, and elevation
- Species lists and phenology seasonality descriptions

---

## Data Requirements

### NDVI CSV
```
date,NDVI
2017-01-09,0.48
2017-02-08,0.39
```
Column names are auto-detected. Supports MODIS 8-day, Sentinel-2 10-day, or irregularly spaced composites.

### NASA POWER Meteorological CSV
Download from [NASA POWER Data Access](https://power.larc.nasa.gov/data-access-viewer/).  
Select **Daily** temporal resolution and **point** geometry. Recommended parameters:

| Parameter | Variable | Role |
|---|---|---|
| Mean temperature 2m | `T2M` | GDD, VPD |
| Min temperature 2m | `T2M_MIN` | SOS trigger |
| Max temperature 2m | `T2M_MAX` | Heat stress |
| Precipitation | `PRECTOTCORR` | Monsoon trigger |
| Relative humidity | `RH2M` | Moisture proxy |
| Surface soil wetness | `GWETTOP` | Leaf flush |
| Root zone soil wetness | `GWETROOT` | Drought resistance |
| Wind speed 2m | `WS2M` | Senescence (EOS) |
| Incoming solar radiation | `ALLSKY_SFC_SW_DWN` | POS timing |

The app **automatically derives**: `GDD_5`, `GDD_10`, `GDD_cum`, `DTR`, `VPD`, `SPEI_proxy`, `log_precip`, `MSI`, `T2M_RANGE`.

---

## Installation

```bash
git clone https://github.com/shreejisharma/Indian-forest-phenology.git
cd Indian-forest-phenology
pip install -r requirements.txt
streamlit run app/universal_Indian_forest_phenology_assesment.py
```

---

## Repository Structure

```
Indian-forest-phenology/
├── app/
│   └── universal_Indian_forest_phenology_assesment.py   ← main app (single file)
├── data/
│   └── sample/                                           ← example input files
├── docs/                                                 ← screenshots
├── requirements.txt
└── README.md
```

---

## Methodology

### Phenology Extraction
1. NDVI resampled to daily resolution by linear interpolation
2. Savitzky-Golay smoothing (adaptive window ≈ 5% of series length)
3. Threshold-based extraction at a configurable fraction of seasonal amplitude
4. Optional derivative-based or drought-triggered EOS detection

### Regression Model
1. 15-day pre-event meteorological windows computed per season per event
2. Pearson r filter: |r| ≥ 0.40 required
3. Collinearity filter: |r| > 0.85 between candidate features → weaker one dropped
4. Incremental LOO R² check: feature added only if it improves LOO R² by ≥ 0.03
5. Ridge Regression with RidgeCV (Leave-One-Out) for final fitting

### Consistency Guarantee
Pearson r and significance shown in the **Training tab** and the **Correlations tab heatmap** are computed from the **same `corr_tables` object** — they are always identical.

---

## Citation

```
Sharma, S. (2025). Universal Indian Forest Phenology Predictor [Software].
GitHub. https://github.com/shreejisharma/Indian-forest-phenology
```

---

## License

MIT License — see [LICENSE](LICENSE) for details.
