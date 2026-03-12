# v6 Driver Analysis — Why T2M Was Wrong

## The Problem

In v5, every phenological metric was modelled as a linear function of T2M:

```
SOS_days_from_Jun1 = -28.194 + 2.01408 × T2M - 2.07850 × RH2M   [R²=0.796]
POS_days_from_Jun1 = 981.958 - 35.81610 × T2M                    [R²=0.722]
EOS_days_from_Jun1 = 204.228 + 3.21661 × T2M                     [R²=0.930]
```

## Why This Is Ecologically Incorrect

### 1. T2M Has Almost No Variation

| Season | T2M (mean) | PRECTOTCORR (total) |
|--------|-----------|---------------------|
| 2003   | 24.22°C   | 1542 mm             |
| 2005   | 24.82°C   | 1560 mm             |
| 2007   | 24.15°C   | 1071 mm             |
| **Range** | **0.67°C** | **489 mm (CV=15%)** |

A 0.67°C temperature range cannot drive SOS differences of 19 days (DOY 165 vs 184).

### 2. T2M Is a Proxy for PRECTOTCORR

In Indian monsoon climates, T2M and PRECTOTCORR are anti-correlated:
- High T2M = dry pre-monsoon = low precipitation
- Low T2M  = wet monsoon = high precipitation

The app's collinearity filter (|r| > 0.85) drops PRECTOTCORR because it's "redundant" with T2M. But PRECTOTCORR is the **causal** driver — T2M is just a correlated proxy.

### 3. Pre-SOS Window Was Too Short

The 30-day window before SOS captures post-monsoon conditions:
- **2003:** only 20mm precip in 30d before SOS → late SOS (DOY 165)
- **2005:** 174mm precip in 30d before SOS → later SOS (DOY 184)  
- **2007:** 223mm precip in 30d before SOS → later SOS (DOY 182)

Wait — precip is HIGHER for later SOS? That's because the monsoon arrives late in 2005/2007, so precip accumulates just before SOS. The correct window to use is 60–90 days, which captures the **onset of monsoon accumulation**.

### 4. GDD_cum Is a Data Leakage Feature

GDD_cum = cumulative GDD from season start up to the event date.
This is computed from data within the same season → it perfectly predicts event timing by construction, but provides no predictive power for future years.

## v6 Fix Summary

```python
# BEFORE (v5): T2M collinearity filter drops PRECTOTCORR
if abs(r_prectot_vs_t2m) > 0.85:
    drop(PRECTOTCORR)  # WRONG for monsoon sites

# AFTER (v6): moisture features are protected
if abs(r_prectot_vs_t2m) > 0.85 and protect_moisture:
    if PRECTOTCORR in moisture_keys and T2M in low_variation_features:
        drop(T2M)  # CORRECT — keep the causal driver
        keep(PRECTOTCORR)
```

```python
# BEFORE (v5): 30-day window
window_sos = 30

# AFTER (v6): 60-day window (captures monsoon onset)
window_sos = 60  # user-adjustable

# BEFORE (v5): mean precipitation
feats["PRECTOTCORR"] = sub["PRECTOTCORR"].mean()

# AFTER (v6): accumulated precipitation
feats["PRECTOTCORR_sum"] = sub["PRECTOTCORR"].sum()  # physically meaningful
```
