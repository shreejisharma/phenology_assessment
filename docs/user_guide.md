# User Guide — Indian Forest Phenology Predictor

Complete reference for running the app, loading data, and interpreting results.  
For quick site-by-site settings, see the [App Settings table in README](../README.md#app-settings-quick-reference).

---

## Step 1 — Upload Your Data

In the **sidebar**:

1. **NDVI CSV** — Upload any of the files in `data/ndvi/`
2. **NASA POWER Met CSV** — Upload the matching file from `data/meteorology/`

The app auto-detects all columns. No manual configuration needed.

---

## Step 2 — Select Forest Type

Choose the forest type matching your site from the dropdown.  
This sets the season window, default SOS threshold, and SOS detection method automatically.

### Which forest type to choose for each site?

| Site | Select this forest type |
|------|------------------------|
| Tirupati, Mudumalai | Tropical Dry Deciduous — Monsoon (Jun-May) |
| Simlipal, Bastar | Tropical Moist Deciduous — Monsoon (Jun-May) |
| Agumbe | Tropical Wet Evergreen / Semi-Evergreen (Jan-Dec) |
| Mukurthi | Shola Forest — Southern Montane (Jan-Dec) |
| Bhitarkanika | Mangrove Forest (Jan-Dec) |
| Warangal | Tropical Dry Deciduous or Kharif Crop (Jun-Oct) |
| Spiti, Valley of Flowers | Alpine / Subalpine Forest and Meadow (May-Oct) |
| Jaisalmer | Tropical Thorn Forest / Scrub (Jun-May) |

---

## Step 3 — Configure Detection Methods

### SOS Method

| Method | When to use |
|--------|-------------|
| 🌧️ First Sustained Rainfall | Monsoon dry/moist deciduous forests, thorn scrub |
| 📈 NDVI Threshold | Evergreen, mangrove, shola, NE India |
| 📉 Max NDVI Rate | Alpine, temperate, subtropical hill |

### SOS Threshold (% of NDVI amplitude)

- **20–25%** — standard for deciduous forests (default)
- **15–18%** — evergreen and near-evergreen (low amplitude sites)
- **10–12%** — desert / thorn scrub only (Jaisalmer)
- **25–30%** — alpine meadows (high noise, want clear green flush only)

---

## Step 4 — Run Analysis

After uploading both files and selecting settings, the app automatically:

- Smooths NDVI with Savitzky-Golay filter
- Extracts SOS, POS, EOS, LOS for each year
- Computes all meteorological features
- Selects best features (Pearson |r| ≥ 0.40)
- Fits Ridge regression with LOO cross-validation
- Shows results across 6 tabs

---

## Tabs Explained

### Tab 1 — NDVI & Phenology
Shows the smoothed NDVI curve for all years with detected SOS (green ▲), POS (blue ●), EOS (red ▼) markers.

**What to look for:**
- SOS and EOS should fall on the rising/falling limb of the curve
- POS should be at or near the annual NDVI peak
- If events are in wrong positions → adjust threshold or season window

### Tab 2 — Phenology Trends
Bar/line charts of SOS, POS, EOS, LOS across years.

**What to look for:**
- Year-to-year variation — if all years show same DOY, there is no signal for modelling
- Long-term trends (warming → earlier SOS?)
- Outlier years (drought years → very late SOS?)

### Tab 3 — Models & Equations
Shows the fitted Ridge regression equations and LOO R² for each event.

**Example equation:**
```
SOS_days = 47.3 − 2.18 × T2M_MIN
R²(LOO) = 0.64,  MAE = 7.2 days
```

This means: every 1°C warmer minimum temperature → SOS 2.18 days earlier.

### Tab 4 — Correlations
Heatmap and bar chart of Pearson r between all meteorological features and each phenology event.

**Useful for:**
- Identifying which climate drivers matter most at your site
- Checking whether the app selected the most ecologically meaningful feature

### Tab 5 — Predict
Enter climate data for a new year → get predicted SOS/POS/EOS dates.

**How to use:**
1. Enter the pre-event window mean values for the selected predictor
2. Click Predict
3. The app returns predicted DOY and confidence interval (±MAE)

### Tab 6 — Methods
Full technical documentation, R² interpretation guide, and improvement tips.

---

## Interpreting R² Values

| R² (LOO) | Interpretation |
|----------|----------------|
| > 0.70 | Strong — reliable for prediction |
| 0.40–0.70 | Moderate — useful for trend analysis |
| 0.10–0.40 | Weak — relative comparisons only |
| 0.0 | No feature met threshold — returns mean DOY |
| < 0 | Model worse than mean — check settings |

### Why is my R² negative?

Most common causes:
1. **Wrong season window** — SOS detected same date every year (no variation)
2. **Cloud-contaminated NDVI** — baseline wrong, events extracted incorrectly
3. **Wrong forest type** — e.g. using evergreen window for a deciduous site
4. **Too few seasons** — n < 6 is unreliable

### How to improve R²

1. Add more seasons (most impactful)
2. Add `ALLSKY_SFC_SW_DWN` (solar radiation) to NASA POWER — critical for POS
3. Add `GWETROOT` (root zone soil moisture) — key EOS driver
4. Adjust threshold ±5% and compare
5. Enter IMD monsoon onset dates manually — raises SOS R² from ~0.3 to 0.7+ for monsoon forests

---

## Pre-Filtering Required Files

### Spiti Valley and Valley of Flowers (alpine sites)

Before loading these files, open in Python or Excel and remove:
- Rows where `NDVI < 0.01` (35 rows in Spiti — snow/ice covered)
- Rows for year 2016 and 2017 (no Jun–Sep data)
- Rows where `month` is outside 5–10 (May–October)

Then keep only `date` and `NDVI` columns.

### Jaisalmer (desert thorn scrub)

**Do NOT filter** — the low NDVI values (0.02–0.10) are real bare desert signal.  
Instead, set SOS threshold to 10–12% in the app.

---

## Getting More NDVI Data

Use the GEE scripts in `scripts/` to extract NDVI for any new site:

- `gee_extract_modis_ndvi.js` — for all monsoon forest types
- `gee_extract_sentinel2_ndvi.js` — for alpine/thorn scrub sites

Edit the CONFIGURATION section at the top of each script (coordinates, buffer, date range) and run in [code.earthengine.google.com](https://code.earthengine.google.com).
