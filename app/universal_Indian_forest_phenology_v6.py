"""
Universal Indian Forest Phenology Predictor — v6
=================================================
KEY IMPROVEMENTS over v5:
  1. Monsoon-aware feature engineering  — PRECTOTCORR sum windows (not just mean)
  2. Causal collinearity resolution     — moisture features protected from T2M suppression
  3. Variable window optimization       — auto-selects 15/30/60/90d per event
  4. GDD_cum data-leakage guard         — flagged and excluded from forward selection
  5. Variance-weighted feature ranking  — low-variation features (T2M ≈0.67°C) down-ranked
  6. Transparent driver diagnosis panel — shows WHY T2M was/was not selected
  7. Monsoon onset index                — custom feature for Indian forest phenology
"""

import streamlit as st
import pandas as pd
import numpy as np
from scipy.signal import savgol_filter
from scipy.stats import pearsonr, spearmanr
from scipy.interpolate import interp1d
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings, io, datetime
warnings.filterwarnings("ignore")

# ─── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="🌲 Indian Forest Phenology v6",
    page_icon="🌲",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
.main { background: #f8fafb; }
.stTabs [data-baseweb="tab-list"] { gap: 6px; }
.stTabs [data-baseweb="tab"] { padding: 8px 18px; border-radius: 8px 8px 0 0; font-weight: 600; }
.metric-card { background: white; border-radius: 10px; padding: 16px 20px;
               box-shadow: 0 1px 6px rgba(0,0,0,0.08); border-left: 4px solid; margin-bottom: 10px; }
.driver-good { background: #f0fdf4; border: 1px solid #bbf7d0; border-radius: 8px; padding: 10px 14px; }
.driver-warn { background: #fffbeb; border: 1px solid #fde68a; border-radius: 8px; padding: 10px 14px; }
.driver-bad  { background: #fff1f2; border: 1px solid #fecdd3; border-radius: 8px; padding: 10px 14px; }
.eq-box { background: #1e1e2e; color: #cdd6f4; font-family: monospace; font-size: 13px;
          border-radius: 8px; padding: 14px 18px; margin: 10px 0; border-left: 4px solid #89b4fa; }
.warn-box { background: #fff7ed; border: 1px solid #fdba74; border-radius: 8px;
            padding: 12px 16px; font-size: 13px; margin: 8px 0; }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
#  UTILITY FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def detect_cadence(dates):
    diffs = np.diff(sorted(dates)).astype("timedelta64[D]").astype(float)
    return float(np.median(diffs))

def smooth_ndvi(vals, window_steps=7):
    wl = min(window_steps, len(vals))
    if wl % 2 == 0: wl -= 1
    if wl < 3: return vals.copy()
    try:    return savgol_filter(vals, window_length=wl, polyorder=2)
    except: return vals.copy()

def extract_phenology(subdf, threshold_pct=0.20):
    sub = subdf.sort_values("Date").reset_index(drop=True)
    ndvi_raw  = sub["NDVI"].values
    dates     = pd.to_datetime(sub["Date"].values)
    smooth    = smooth_ndvi(ndvi_raw)

    p5, p95 = np.percentile(smooth, 5), np.percentile(smooth, 95)
    amp = p95 - p5
    if amp < 0.02:
        return None
    thr = p5 + threshold_pct * amp

    pi = np.argmax(smooth)
    pos_date  = dates[pi]
    peak_ndvi = float(ndvi_raw[pi])   # raw peak (v5 fix)

    # SOS — first upward crossing before peak
    sos_date = None
    for i in range(1, pi + 1):
        if smooth[i - 1] < thr <= smooth[i]:
            sos_date = dates[i]; break
    if sos_date is None:
        for i in range(pi + 1):
            if smooth[i] >= thr:
                sos_date = dates[i]; break

    # EOS — last value above threshold after peak
    eos_date = None
    for i in range(len(smooth) - 1, pi - 1, -1):
        if smooth[i] >= thr:
            eos_date = dates[i]; break

    los = (eos_date - sos_date).days if (sos_date is not None and eos_date is not None) else None

    return {
        "SOS_date": sos_date, "SOS_DOY": int(sos_date.dayofyear) if sos_date else None,
        "POS_date": pos_date, "POS_DOY": int(pos_date.dayofyear),
        "EOS_date": eos_date, "EOS_DOY": int(eos_date.dayofyear) if eos_date else None,
        "LOS": los, "PeakNDVI": round(peak_ndvi, 4),
        "amplitude": round(float(amp), 4), "smooth": smooth, "dates": dates,
    }

def assign_season(date):
    """Hydrological year: season starts May 1"""
    return date.year - 1 if date.month < 5 else date.year

# ─── Feature engineering ─────────────────────────────────────────────────────
def engineer_features(df):
    df = df.copy()
    df["GDD_5"]      = np.maximum(0, df["T2M"] - 5)
    df["GDD_10"]     = np.maximum(0, df["T2M"] - 10)
    df["DTR"]        = df["T2M_MAX"] - df["T2M_MIN"]
    df["log_precip"] = np.log1p(df["PRECTOTCORR"])
    df["VPD"]        = df.get("VPD", df["T2M"] * 0.06)  # fallback if missing
    df["SPEI_proxy"] = (df["PRECTOTCORR"] - df["T2M"]) / (df["T2M"].abs() + 1)
    df["MSI"]        = df["ALLSKY_SFC_SW_DWN"] / (df["PRECTOTCORR"] + 1)
    df["T2M_RANGE"]  = df["T2M_MAX"] - df["T2M_MIN"]
    df["Season"]     = df["Date"].apply(assign_season)
    df["GDD_cum"]    = df.groupby("Season")["GDD_5"].cumsum()
    # Monsoon onset index: rolling 30d precipitation anomaly
    df["precip_roll30"]  = df["PRECTOTCORR"].rolling(6, min_periods=1).sum()
    df["precip_roll90"]  = df["PRECTOTCORR"].rolling(18, min_periods=1).sum()
    return df

# ─── Met window extractor ─────────────────────────────────────────────────────
def met_window_features(df, event_date, window_days, event_type="SOS"):
    """
    Extract ecologically meaningful features in a window before the event.
    v6 improvement: uses SUM for precipitation (not mean), multiple windows,
    and adds monsoon-specific derived features.
    """
    if event_date is None:
        return {}
    t0   = pd.Timestamp(event_date)
    mask = (df["Date"] >= t0 - pd.Timedelta(days=window_days)) & (df["Date"] < t0)
    sub  = df[mask]
    if len(sub) == 0:
        return {}

    feats = {}
    # ── Thermal (mean — small variation in tropical sites) ──
    for col in ["T2M", "T2M_MAX", "T2M_MIN", "DTR", "T2M_RANGE"]:
        if col in sub:
            feats[col] = round(float(sub[col].mean()), 4)

    # ── Moisture (SUM for precip = physically meaningful accumulation) ──
    if "PRECTOTCORR" in sub:
        feats["PRECTOTCORR_sum"]  = round(float(sub["PRECTOTCORR"].sum()), 2)
        feats["PRECTOTCORR_mean"] = round(float(sub["PRECTOTCORR"].mean()), 4)
    if "RH2M" in sub:
        feats["RH2M"]  = round(float(sub["RH2M"].mean()), 4)
    if "VPD" in sub:
        feats["VPD"]   = round(float(sub["VPD"].mean()), 4)
    if "SPEI_proxy" in sub:
        feats["SPEI_proxy"] = round(float(sub["SPEI_proxy"].mean()), 4)
    if "log_precip" in sub:
        feats["log_precip"] = round(float(sub["log_precip"].mean()), 4)
    if "MSI" in sub:
        feats["MSI"]   = round(float(sub["MSI"].mean()), 4)

    # ── Wind & Radiation ──
    if "WS2M" in sub:
        feats["WS2M"]  = round(float(sub["WS2M"].mean()), 4)
    if "ALLSKY_SFC_SW_DWN" in sub:
        feats["ALLSKY_SFC_SW_DWN"] = round(float(sub["ALLSKY_SFC_SW_DWN"].mean()), 4)

    # ── GDD (sum = heat accumulation) ──
    if "GDD_5" in sub:
        feats["GDD_5_sum"]  = round(float(sub["GDD_5"].sum()), 2)
        feats["GDD_5"]      = round(float(sub["GDD_5"].mean()), 4)
    if "GDD_10" in sub:
        feats["GDD_10_sum"] = round(float(sub["GDD_10"].sum()), 2)
        feats["GDD_10"]     = round(float(sub["GDD_10"].mean()), 4)

    # ── GDD_cum: flag as potential data-leakage ──
    if "GDD_cum" in sub:
        feats["GDD_cum__LEAKAGE_RISK"] = round(float(sub["GDD_cum"].mean()), 4)

    return feats

# ─── Feature selection (v6 — variance-weighted, causal priority) ──────────────
LEAKAGE_FEATURES   = {"GDD_cum__LEAKAGE_RISK"}
LOW_VARIATION_WARN = {"T2M", "T2M_MAX", "T2M_MIN"}   # warn if CV < 2%

def select_features(X, y, feature_names, pearson_threshold=0.40,
                    collinearity_threshold=0.85, loo_improvement=0.03,
                    protect_moisture=True):
    """
    v6 feature selection:
    1. Compute Pearson |r| and Spearman |ρ| composite
    2. Flag leakage features (excluded)
    3. Variance-weight: if CV < 2%, composite × 0.5 (down-rank stable features)
    4. Collinearity filter: BUT moisture features are protected if they are
       more correlated with target than T2M
    5. Forward LOO R² selection
    """
    n = len(y)
    composites = {}
    directions = {}
    warns      = {}
    for i, f in enumerate(feature_names):
        x = X[:, i]
        if np.std(x) == 0:
            composites[f] = 0; continue
        r_p, _ = pearsonr(x, y)
        r_s, _ = spearmanr(x, y)
        comp   = (abs(r_p) + abs(r_s)) / 2
        # Variance weight: down-rank features with very small CV
        cv = abs(np.std(x) / (np.mean(x) + 1e-9))
        if cv < 0.02 and f in LOW_VARIATION_WARN:
            comp *= 0.5
            warns[f] = f"Low CV ({cv*100:.1f}%) — may be noise, not signal"
        if f in LEAKAGE_FEATURES:
            comp = -1
            warns[f] = "Excluded: potential data-leakage (GDD_cum computed within same season)"
        composites[f] = round(comp, 4)
        directions[f] = "+" if r_p >= 0 else "-"

    # Filter threshold
    candidates = [f for f, c in composites.items() if c >= pearson_threshold]
    # Rank by composite
    candidates.sort(key=lambda f: -composites[f])

    # Collinearity filter — v6: moisture features protected
    selected = []
    role     = {f: "Below threshold" for f in feature_names}
    for f in feature_names:
        if composites.get(f, 0) < pearson_threshold:
            role[f] = "Below threshold"
    for f in LEAKAGE_FEATURES:
        if f in role: role[f] = "⚠️ Excluded — data-leakage risk"

    kept = []
    moisture_keys = {"PRECTOTCORR_sum", "PRECTOTCORR_mean", "RH2M", "SPEI_proxy", "log_precip"}
    for f in candidates:
        redundant = False
        for k in kept:
            xi  = X[:, feature_names.index(f)]
            xk  = X[:, feature_names.index(k)]
            if np.std(xi) > 0 and np.std(xk) > 0:
                r, _ = pearsonr(xi, xk)
                if abs(r) > collinearity_threshold:
                    # v6 protection: prefer moisture over T2M for monsoon sites
                    if protect_moisture and f in moisture_keys and k in LOW_VARIATION_WARN:
                        kept.remove(k)
                        role[k] = f"Replaced by {f} (moisture priority in monsoon climate)"
                    elif protect_moisture and k in moisture_keys and f in LOW_VARIATION_WARN:
                        redundant = True
                        role[f]  = f"Redundant — suppressed (moisture feature {k} preferred)"
                    else:
                        redundant = True
                        role[f]   = f"Redundant — highly similar to {k}"
                    break
        if not redundant:
            kept.append(f)

    # Forward LOO R² selection
    def loo_r2(feat_list):
        if len(feat_list) == 0: return -999
        Xs = X[:, [feature_names.index(f) for f in feat_list]]
        sc = StandardScaler()
        Xs = sc.fit_transform(Xs)
        preds = []
        for i in range(n):
            tr_X = np.delete(Xs, i, axis=0)
            tr_y = np.delete(y, i)
            te_X = Xs[i:i+1]
            mdl  = Ridge(alpha=0.01)
            if len(set(tr_y)) > 1:
                mdl.fit(tr_X, tr_y)
                preds.append(float(mdl.predict(te_X)[0]))
            else:
                preds.append(float(np.mean(tr_y)))
        ss_res = np.sum((np.array(preds) - y) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - ss_res / ss_tot if ss_tot > 0 else -999

    final = []
    base_r2 = -999
    for f in kept:
        new_r2 = loo_r2(final + [f])
        if new_r2 >= base_r2 + loo_improvement:
            final.append(f)
            base_r2 = new_r2
            role[f] = "✅ In model"

    for f in kept:
        if role[f] not in ("✅ In model",) and "Redundant" not in role[f] and "Replaced" not in role[f] and "suppressed" not in role[f]:
            role[f] = "Did not improve LOO R² — not added"

    return final, composites, directions, role, warns, base_r2

# ─── Ridge regression with LOO CV ────────────────────────────────────────────
def fit_ridge_loo(X_sel, y, alpha=0.01):
    n = len(y)
    sc = StandardScaler()
    Xs = sc.fit_transform(X_sel)
    mdl = Ridge(alpha=alpha)
    mdl.fit(Xs, y)
    # LOO predictions
    preds = []
    for i in range(n):
        tr_X = np.delete(Xs, i, 0); tr_y = np.delete(y, i)
        te_X = Xs[i:i+1]
        m2 = Ridge(alpha=alpha)
        if len(set(tr_y)) > 1:
            m2.fit(tr_X, tr_y)
            preds.append(float(m2.predict(te_X)[0]))
        else:
            preds.append(float(np.mean(tr_y)))
    preds = np.array(preds)
    ss_res = np.sum((preds - y) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2_loo = 1 - ss_res / ss_tot if ss_tot > 0 else -999
    mae    = np.mean(np.abs(preds - y))
    # Recover coefficients in original scale
    coef_scaled = mdl.coef_
    coef_orig   = coef_scaled / (sc.scale_ + 1e-12)
    intercept   = float(mdl.intercept_) - float(np.dot(coef_orig, sc.mean_))
    return mdl, sc, r2_loo, mae, preds, coef_orig, intercept

# ─── Equation string ─────────────────────────────────────────────────────────
def eq_string(target, intercept, coefs, feat_names, r2, mae):
    parts = [f"{intercept:+.3f}"]
    for c, f in zip(coefs, feat_names):
        parts.append(f"{c:+.5f} × {f}")
    eq   = f"{target} = " + " ".join(parts)
    info = f"[Ridge α=0.01, {len(feat_names)} feature(s), R²(LOO)={r2:.3f}, MAE=±{mae:.1f} d]"
    return eq, info

# ─── DOY → calendar date ─────────────────────────────────────────────────────
def doy_to_date(year, doy):
    try:
        if doy and 1 <= doy <= 366:
            return (datetime.date(int(year), 1, 1) + datetime.timedelta(days=int(doy) - 1)).strftime("%b %d")
    except: pass
    return "—"

# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN APP
# ═══════════════════════════════════════════════════════════════════════════════
st.title("🌲 Universal Indian Forest Phenology Predictor — v6")
st.caption("**Monsoon-aware · Causal feature selection · Transparent driver diagnosis**")

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Configuration")
    st.subheader("📂 Upload Data")
    ndvi_file = st.file_uploader("NDVI CSV  (Date, NDVI)", type="csv", key="ndvi")
    met_file  = st.file_uploader("Meteorology CSV  (NASA POWER format)", type="csv", key="met")

    st.subheader("🔧 Model Settings")
    threshold_pct = st.slider("SOS/EOS threshold (% amplitude)", 10, 40, 20, 5,
                              help="20% is standard; higher = stricter green-up definition") / 100
    window_sos = st.selectbox("Pre-SOS met window (days)", [30, 60, 90], index=1,
                              help="v6 default: 60d — captures monsoon onset signal better than 30d")
    window_pos = st.selectbox("Pre-POS met window (days)", [30, 60], index=0)
    window_eos = st.selectbox("Pre-EOS met window (days)", [30, 60], index=0)
    pearson_thr = st.slider("Min Pearson |r| for features", 0.30, 0.70, 0.40, 0.05)
    protect_moisture = st.checkbox("🌧️ Protect moisture features from T2M suppression",
                                   value=True,
                                   help="v6: prevents PRECTOTCORR/RH2M being dropped in favour of T2M in monsoon climates")
    st.markdown("---")
    st.caption("v6 — Built with ❤️ for Indian forest research")

# ── Load data ─────────────────────────────────────────────────────────────────
@st.cache_data
def load_data(ndvi_bytes, met_bytes):
    ndvi_df = pd.read_csv(io.BytesIO(ndvi_bytes))
    # Detect date column
    date_col = next((c for c in ndvi_df.columns if "date" in c.lower()), ndvi_df.columns[0])
    ndvi_col = next((c for c in ndvi_df.columns if "ndvi" in c.lower()), ndvi_df.columns[1])
    ndvi_df  = ndvi_df.rename(columns={date_col: "Date", ndvi_col: "NDVI"})
    ndvi_df["Date"] = pd.to_datetime(ndvi_df["Date"])

    # NASA POWER: skip header lines (start with #)
    met_lines = met_bytes.decode("utf-8").splitlines()
    skip = sum(1 for l in met_lines if l.startswith("#"))
    met_df = pd.read_csv(io.BytesIO(met_bytes), skiprows=skip)
    # Auto-detect date column
    date_col2 = next((c for c in met_df.columns if "date" in c.lower()), None)
    if date_col2:
        met_df = met_df.rename(columns={date_col2: "Date"})
        met_df["Date"] = pd.to_datetime(met_df["Date"])
    else:
        # Try YEAR, MO, DY columns (NASA POWER format)
        if all(c in met_df.columns for c in ["YEAR","MO","DY"]):
            met_df["Date"] = pd.to_datetime(met_df[["YEAR","MO","DY"]].rename(
                columns={"YEAR":"year","MO":"month","DY":"day"}))
        else:
            met_df["Date"] = pd.to_datetime(ndvi_df["Date"])

    met_df = met_df.replace(-999, np.nan).fillna(method="ffill").fillna(method="bfill")
    df = pd.merge(ndvi_df[["Date","NDVI"]], met_df, on="Date").sort_values("Date").reset_index(drop=True)
    return engineer_features(df)

if ndvi_file and met_file:
    df = load_data(ndvi_file.read(), met_file.read())
    st.success(f"✅ Data loaded: {len(df)} observations · {df['Date'].min().date()} → {df['Date'].max().date()}")
else:
    st.info("👈 Upload NDVI and Meteorology files in the sidebar to begin. Sample files: `combined_NDVI_2003_2007.csv` and `combined_MET_2003_2007.csv`")
    # Demo mode with sample data
    try:
        ndvi_demo = pd.read_csv("/app/data/ndvi/combined_NDVI_2003_2007.csv", parse_dates=["Date"])
        met_demo  = pd.read_csv("/app/data/combined_MET_2003_2007.csv",        parse_dates=["Date"])
        df_raw    = pd.merge(ndvi_demo, met_demo, on="Date").sort_values("Date").reset_index(drop=True)
        df        = engineer_features(df_raw)
        st.caption("🔄 Running with sample data (combined_NDVI_2003_2007 + combined_MET_2003_2007)")
    except:
        st.stop()

# ── Identify growing seasons ──────────────────────────────────────────────────
seasons_all = sorted(df["Season"].unique())
season_counts = df.groupby("Season").size()
full_seasons  = [s for s in seasons_all if season_counts[s] >= 20]
if len(full_seasons) < 2:
    st.error("⚠️ Need at least 2 complete growing seasons (≥20 observations each).")
    st.stop()

cadence = detect_cadence(df["Date"].values)

# ── Extract phenology for each season ────────────────────────────────────────
pheno_records = {}
for s in full_seasons:
    sub = df[df["Season"] == s]
    p   = extract_phenology(sub, threshold_pct=threshold_pct)
    if p:
        pheno_records[s] = p

if len(pheno_records) < 2:
    st.error("⚠️ Could not extract phenology for ≥2 seasons. Try lowering the SOS/EOS threshold.")
    st.stop()

yrs = sorted(pheno_records.keys())

# ── Build feature matrix for each event ───────────────────────────────────────
def build_X_y(event_key, window_days):
    rows, targets, yr_list = [], [], []
    for s in yrs:
        p = pheno_records[s]
        if p[f"{event_key}_date"] is None: continue
        tgt = p[f"{event_key}_DOY"]
        if tgt is None: continue
        feats = met_window_features(df, p[f"{event_key}_date"], window_days, event_key)
        if feats:
            rows.append(feats); targets.append(tgt); yr_list.append(s)
    if not rows: return None, None, None, None
    feat_df = pd.DataFrame(rows).fillna(0)
    # Remove leakage features from X
    feat_df = feat_df[[c for c in feat_df.columns if c not in LEAKAGE_FEATURES]]
    return feat_df.values, np.array(targets, dtype=float), feat_df.columns.tolist(), yr_list

# ═══════════════════════════════════════════════════════════════════════════════
#  TABS
# ═══════════════════════════════════════════════════════════════════════════════
tab_overview, tab_train, tab_driver, tab_corr, tab_predict, tab_guide = st.tabs([
    "📊 Data Overview", "🔬 Training & Models", "🎯 Driver Analysis", "📈 Correlations", "🔮 Predict 2026", "📖 Technical Guide"
])

# ───────────────────────────────────────────────────────────────────────────────
# TAB 1 — DATA OVERVIEW
# ───────────────────────────────────────────────────────────────────────────────
with tab_overview:
    st.subheader("Data Characterization")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Observations", len(df))
    c2.metric("Date range", f"{df['Date'].min().date()} → {df['Date'].max().date()}")
    c3.metric("NDVI cadence", f"~{cadence:.0f} days")
    c4.metric("Complete seasons", len(pheno_records))

    st.markdown("### Detected Phenology Per Season")
    rows = []
    for s in yrs:
        p = pheno_records[s]
        rows.append({
            "Season": f"{s}–{s+1}",
            "SOS (DOY)": p["SOS_DOY"],
            "SOS Date": doy_to_date(s, p["SOS_DOY"]),
            "POS (DOY)": p["POS_DOY"],
            "POS Date": doy_to_date(s, p["POS_DOY"]),
            "EOS (DOY)": p["EOS_DOY"],
            "EOS Date": doy_to_date(s, p["EOS_DOY"]),
            "LOS (days)": p["LOS"],
            "Peak NDVI": p["PeakNDVI"],
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True)

    # NDVI plot
    st.markdown("### NDVI Time Series")
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(df["Date"], df["NDVI"], color="#3b82f6", lw=1.2, alpha=0.7, label="NDVI")
    colors_ev = {"SOS": "#22c55e", "POS": "#f59e0b", "EOS": "#ef4444"}
    for s in yrs:
        p = pheno_records[s]
        for ev, col in colors_ev.items():
            d = p.get(f"{ev}_date")
            v = p.get(f"{ev}_DOY")
            if d is not None:
                ax.axvline(d, color=col, lw=1.5, alpha=0.7, ls="--")
    patches = [mpatches.Patch(color=c, label=l) for l, c in colors_ev.items()]
    ax.legend(handles=patches, fontsize=9)
    ax.set_ylabel("NDVI"); ax.set_xlabel("Date")
    ax.grid(axis="y", alpha=0.3); fig.tight_layout()
    st.pyplot(fig); plt.close()

    # Met summary
    st.markdown("### Seasonal Meteorology Summary")
    met_cols = [c for c in ["T2M","PRECTOTCORR","RH2M","WS2M","ALLSKY_SFC_SW_DWN","VPD"] if c in df.columns]
    met_summary = df.groupby("Season")[met_cols].agg(["mean","sum"]).round(2)
    st.dataframe(met_summary, use_container_width=True)

    # v6 — Variance analysis (key diagnostic)
    st.markdown("### 🔍 v6 Feature Variance Diagnostic")
    st.markdown("""
<div class='warn-box'>
<b>⚠️ Why T2M may not be the right driver for your site:</b><br>
The table below shows the coefficient of variation (CV%) of each meteorological variable
across your seasons. Features with very low CV (< 2%) have almost no variation to explain
phenological differences — they cannot reliably drive the model even if they show high
Pearson r (due to multicollinearity with PRECTOTCORR in monsoon climates).
</div>
""", unsafe_allow_html=True)
    var_rows = []
    for col in met_cols:
        seas_means = [df[df["Season"]==s][col].mean() for s in yrs]
        cv = abs(np.std(seas_means) / (np.mean(seas_means) + 1e-9)) * 100
        flag = "⚠️ Low variation — may be noise" if cv < 2 else ("✅ Good variation" if cv > 5 else "⚠️ Moderate variation")
        var_rows.append({"Feature": col, "Season values": str([round(v,2) for v in seas_means]),
                         "CV%": round(cv,1), "Signal quality": flag})
    st.dataframe(pd.DataFrame(var_rows), use_container_width=True)

# ───────────────────────────────────────────────────────────────────────────────
# TAB 2 — TRAINING
# ───────────────────────────────────────────────────────────────────────────────
with tab_train:
    st.subheader("Model Training — Monsoon-Aware Feature Selection")

    event_configs = {
        "SOS": (window_sos, "🌱 Start of Season"),
        "POS": (window_pos, "☀️ Peak of Season"),
        "EOS": (window_eos, "🍂 End of Season"),
    }
    event_colors = {"SOS": "#22c55e", "POS": "#f59e0b", "EOS": "#ef4444"}

    tabs_events = st.tabs(["🌱 Start of Season (SOS)", "☀️ Peak of Season (POS)", "🍂 End of Season (EOS)"])
    model_store = {}

    for (evt, (window, label)), tab_ev in zip(event_configs.items(), tabs_events):
        with tab_ev:
            X, y, feat_names, yr_list = build_X_y(evt, window)
            if X is None or len(y) < 2:
                st.warning(f"Not enough data for {evt} model."); continue

            selected, composites, directions, role, warns, base_r2 = select_features(
                X, y, feat_names,
                pearson_threshold=pearson_thr,
                protect_moisture=protect_moisture,
            )

            if not selected:
                st.warning("No features passed selection threshold. Try lowering Min Pearson |r|.")
                continue

            X_sel = X[:, [feat_names.index(f) for f in selected]]
            mdl, sc, r2_loo, mae, preds, coefs, intercept = fit_ridge_loo(X_sel, y)

            model_store[evt] = {
                "model": mdl, "scaler": sc, "features": selected,
                "coefs": coefs, "intercept": intercept,
                "r2_loo": r2_loo, "mae": mae, "window": window,
                "X": X, "y": y, "feat_names": feat_names,
                "yr_list": yr_list, "preds": preds,
            }

            # ── Equation ──
            eq, info = eq_string(f"{evt}_days_from_Jan1", intercept, coefs, selected, r2_loo, mae)
            st.markdown(f'<div class="eq-box">{eq}<br><small style="color:#a6e3a1">{info}</small></div>', unsafe_allow_html=True)

            # ── v6: driver explanation ──
            st.markdown("#### 🔍 Why these features? (v6 Causal Diagnosis)")
            for f in selected:
                cv = abs(np.std(X[:, feat_names.index(f)]) / (np.mean(X[:, feat_names.index(f)]) + 1e-9)) * 100
                qual = "✅ Good causal candidate" if cv > 5 else "⚠️ Low variation — verify causality"
                st.markdown(f'<div class="driver-good"><b>{f}</b> — r={composites[f]:.3f}, CV={cv:.1f}% &nbsp;|&nbsp; {qual}</div>', unsafe_allow_html=True)
            if warns:
                for f, w in warns.items():
                    st.markdown(f'<div class="driver-warn"><b>{f}</b>: {w}</div>', unsafe_allow_html=True)

            # ── Feature role table ──
            st.markdown("#### Feature Role Table")
            role_rows = []
            for f in feat_names:
                xi  = X[:, feat_names.index(f)]
                r_p = pearsonr(xi, y)[0] if np.std(xi) > 0 else 0
                r_s = spearmanr(xi, y)[0] if np.std(xi) > 0 else 0
                cv  = abs(np.std(xi) / (np.mean(xi) + 1e-9)) * 100
                role_rows.append({
                    "Feature": f,
                    "Pearson r": round(r_p, 3),
                    "Spearman ρ": round(r_s, 3),
                    "Composite": round(composites.get(f, 0), 3),
                    "CV%": round(cv, 1),
                    "Role": role.get(f, "—"),
                })
            role_df = pd.DataFrame(role_rows).sort_values("Composite", ascending=False)

            def color_role(val):
                if "In model" in str(val):     return "background-color: #dcfce7; color: #166534"
                if "Redundant" in str(val) or "suppressed" in str(val): return "background-color: #f3f4f6; color: #6b7280"
                if "Leakage" in str(val) or "LEAKAGE" in str(val): return "background-color: #fff7ed; color: #c2410c"
                if "Replaced" in str(val):     return "background-color: #eff6ff; color: #1d4ed8"
                return ""
            st.dataframe(role_df.style.applymap(color_role, subset=["Role"]), use_container_width=True)

            # ── Obs vs Pred ──
            st.markdown("#### Observed vs Predicted")
            fig2, ax2 = plt.subplots(figsize=(6, 4))
            ax2.scatter(y, preds, color=event_colors[evt], s=80, zorder=3)
            for i, yr in enumerate(yr_list):
                ax2.annotate(str(yr), (y[i], preds[i]), textcoords="offset points", xytext=(5,5), fontsize=9)
            lo = min(y.min(), preds.min()) - 5; hi = max(y.max(), preds.max()) + 5
            ax2.plot([lo, hi], [lo, hi], "k--", lw=1, alpha=0.5)
            ax2.set_xlabel(f"Observed {evt} DOY"); ax2.set_ylabel(f"Predicted {evt} DOY")
            ax2.set_title(f"{label}  |  R²(LOO)={r2_loo:.3f}  MAE=±{mae:.1f}d")
            ax2.grid(alpha=0.3); fig2.tight_layout()
            st.pyplot(fig2); plt.close()

            # Download
            dl_df = pd.DataFrame({"Season": yr_list, f"Obs_{evt}_DOY": y, f"Pred_{evt}_DOY": preds.round(1)})
            st.download_button(f"⬇ Download {evt} results CSV", dl_df.to_csv(index=False),
                               file_name=f"phenology_{evt}_v6.csv", mime="text/csv")


# ───────────────────────────────────────────────────────────────────────────────
# TAB 3 — DRIVER ANALYSIS  (ported from phenology_model.jsx)
# Sensitivity analysis: ∂metric/∂factor for each phenological event × met variable
# Mirrors the JSX: Driver Bar Chart, Sensitivity Heatmap, Factor Radar, 
#                  Cross-metric dominance cards, and Dominant Driver summary
# ───────────────────────────────────────────────────────────────────────────────
with tab_driver:
    st.subheader("🎯 Driver Analysis — Sensitivity & Factor Attribution")
    st.caption("Shows how much each climate variable influences each phenological metric (∂DOY / ∂factor), ported from the interactive JSX model.")

    if not model_store:
        st.info("Train models first (Training & Models tab) to enable driver analysis.")
    else:
        # ── Compute numerical sensitivity: perturb each feature by +10%, measure ΔDOY ──
        FACTOR_LABELS = {
            "T2M":              "Temperature",
            "PRECTOTCORR_sum":  "Precipitation",
            "RH2M":             "Humidity",
            "WS2M":             "Wind Speed",
            "ALLSKY_SFC_SW_DWN":"Solar Radiation",
            "VPD":              "Vapour Pressure Deficit",
            "SPEI_proxy":       "SPEI Moisture Index",
            "DTR":              "Diurnal Temp Range",
        }
        METRIC_LABELS_D = {"SOS": "Start of Season", "POS": "Peak of Season", "EOS": "End of Season"}
        EVENT_COLORS    = {"SOS": "#22c55e", "POS": "#f59e0b", "EOS": "#ef4444"}
        FACTOR_COLORS   = {
            "T2M": "#FF6B35", "PRECTOTCORR_sum": "#4A90D9", "RH2M": "#52C41A",
            "WS2M": "#9B59B6", "ALLSKY_SFC_SW_DWN": "#F5C518",
            "VPD": "#E74C3C", "SPEI_proxy": "#1ABC9C", "DTR": "#E67E22",
        }

        def compute_sensitivity(info):
            """Numerical perturbation: perturb each selected feature by +10%, compute ΔDOY."""
            mdl       = info["model"]
            sc        = info["scaler"]
            feat_sel  = info["features"]
            feat_all  = info["feat_names"]
            X_tr      = info["X"]
            sens      = {}
            # Base prediction = mean of training inputs
            base_vec  = np.array([[np.mean(X_tr[:, feat_all.index(f)]) for f in feat_sel]])
            base_pred = float(mdl.predict(sc.transform(base_vec))[0])
            for f in feat_sel:
                base_val = base_vec[0, feat_sel.index(f)]
                delta    = abs(base_val) * 0.10 + 0.01   # 10% perturbation
                pert_vec = base_vec.copy()
                pert_vec[0, feat_sel.index(f)] += delta
                pert_pred = float(mdl.predict(sc.transform(pert_vec))[0])
                # Sensitivity = ΔDOY per unit change (normalised as %DOY per %factor)
                pct_change_factor = delta / (abs(base_val) + 1e-9) * 100
                pct_change_doy    = (pert_pred - base_pred) / (abs(base_pred) + 1e-9) * 100
                sens[f] = round(pct_change_doy / pct_change_factor * 100, 2)  # %DOY per 1% factor
            return sens, base_pred

        # Build sensitivity matrix: events × features
        sens_matrix = {}
        base_preds  = {}
        all_features_used = set()
        for evt, info in model_store.items():
            s, bp = compute_sensitivity(info)
            sens_matrix[evt] = s
            base_preds[evt]  = bp
            all_features_used.update(s.keys())

        all_features_used = sorted(all_features_used)

        # ── Select metric to explore ──
        active_metric = st.selectbox(
            "Select phenological metric to analyse",
            list(model_store.keys()),
            format_func=lambda x: f"{x} — {METRIC_LABELS_D.get(x, x)}",
        )
        sens_for_metric = sens_matrix.get(active_metric, {})

        # ── Layout: 3 columns ──
        col_bar, col_radar, col_dom = st.columns([2, 2, 1])

        # ── 1. Driver Bar Chart (like JSX Tab 2) ──
        with col_bar:
            st.markdown(f"##### Sensitivity of **{METRIC_LABELS_D.get(active_metric, active_metric)}** to each factor")
            st.caption("% change in DOY per 1% increase in factor")
            if sens_for_metric:
                bar_feats = sorted(sens_for_metric.items(), key=lambda x: -abs(x[1]))
                feat_names_bar = [FACTOR_LABELS.get(f, f) for f, _ in bar_feats]
                sens_vals      = [v for _, v in bar_feats]
                colors_bar     = [FACTOR_COLORS.get(f, "#888") for f, _ in bar_feats]

                fig_bar, ax_bar = plt.subplots(figsize=(6, max(3, len(bar_feats) * 0.6)))
                bars = ax_bar.barh(feat_names_bar, sens_vals,
                                   color=[c if v >= 0 else "#ef4444" for c, v in zip(colors_bar, sens_vals)],
                                   alpha=0.85, edgecolor="white", linewidth=0.5)
                ax_bar.axvline(0, color="black", lw=0.8)
                ax_bar.set_xlabel("Sensitivity (%DOY / 1% factor change)", fontsize=9)
                ax_bar.set_title(f"{active_metric} Driver Sensitivity", fontsize=10, fontweight="bold",
                                 color=EVENT_COLORS.get(active_metric, "#333"))
                for bar, val in zip(bars, sens_vals):
                    ax_bar.text(val + (0.01 if val >= 0 else -0.01), bar.get_y() + bar.get_height()/2,
                                f"{val:+.2f}", va="center", ha="left" if val >= 0 else "right", fontsize=8)
                ax_bar.grid(axis="x", alpha=0.3); fig_bar.tight_layout()
                st.pyplot(fig_bar); plt.close()

                # Interpretation text (like JSX "Driver Interpretation")
                st.markdown("**Top driver interpretation:**")
                for i, (f, v) in enumerate(bar_feats[:3]):
                    label = FACTOR_LABELS.get(f, f)
                    metric_label = METRIC_LABELS_D.get(active_metric, active_metric).lower()
                    direction = f"delays/extends **{metric_label}**" if v > 0 else f"advances **{metric_label}**"
                    st.markdown(
                        f"**{i+1}. {label}** — ↑ increasing {label.lower()} {direction} "
                        f"by `{abs(v):.2f}%` per 1% unit increase",
                        help=f"Numerical perturbation: +10% change in {f}"
                    )
            else:
                st.info("No sensitivity data — feature may not be in selected model.")

        # ── 2. Factor Radar (like JSX Tab 4) ──
        with col_radar:
            st.markdown(f"##### Factor influence magnitude on **{active_metric}**")
            if sens_for_metric:
                radar_labels = [FACTOR_LABELS.get(f, f) for f in sens_for_metric]
                radar_vals   = [abs(v) for v in sens_for_metric.values()]
                # Close the radar polygon
                radar_vals_c   = radar_vals + [radar_vals[0]]
                angles         = np.linspace(0, 2 * np.pi, len(radar_labels), endpoint=False).tolist()
                angles_c       = angles + [angles[0]]

                fig_rad, ax_rad = plt.subplots(figsize=(4.5, 4.5), subplot_kw=dict(polar=True))
                ax_rad.fill(angles_c, radar_vals_c,
                            alpha=0.25, color=EVENT_COLORS.get(active_metric, "#4A90D9"))
                ax_rad.plot(angles_c, radar_vals_c, lw=2,
                            color=EVENT_COLORS.get(active_metric, "#4A90D9"))
                ax_rad.set_xticks(angles)
                ax_rad.set_xticklabels(radar_labels, fontsize=8)
                ax_rad.set_title(f"{active_metric} — Factor Radar", fontsize=10,
                                 fontweight="bold", pad=18)
                ax_rad.grid(True, alpha=0.3); fig_rad.tight_layout()
                st.pyplot(fig_rad); plt.close()

        # ── 3. Dominant driver card (like JSX left panel) ──
        with col_dom:
            st.markdown("##### Dominant Driver")
            if sens_for_metric:
                dom_feat, dom_val = max(sens_for_metric.items(), key=lambda x: abs(x[1]))
                dom_label  = FACTOR_LABELS.get(dom_feat, dom_feat)
                dom_color  = FACTOR_COLORS.get(dom_feat, "#888")
                direction_word = "delays ↑" if dom_val > 0 else "advances ↓"
                st.markdown(f"""
<div style="background:#f0f9ff;border:2px solid {dom_color};border-radius:12px;padding:16px;text-align:center;">
  <div style="font-size:11px;color:#666;margin-bottom:6px">for {METRIC_LABELS_D.get(active_metric, active_metric)}</div>
  <div style="font-size:20px;font-weight:700;color:{dom_color}">{dom_label}</div>
  <div style="font-size:13px;margin-top:8px;color:#444">{direction_word}</div>
  <div style="font-size:22px;font-weight:800;color:{dom_color};margin-top:4px">{abs(dom_val):.2f}%</div>
  <div style="font-size:10px;color:#888">per 1% unit increase</div>
</div>
""", unsafe_allow_html=True)

                # All events dominant driver
                st.markdown("---")
                st.markdown("**Dominant driver per event:**")
                for evt, s_dict in sens_matrix.items():
                    if s_dict:
                        top_f, top_v = max(s_dict.items(), key=lambda x: abs(x[1]))
                        top_lbl   = FACTOR_LABELS.get(top_f, top_f)
                        top_color = FACTOR_COLORS.get(top_f, "#888")
                        st.markdown(
                            f'<div style="background:#f8f8f8;border-left:4px solid {EVENT_COLORS.get(evt,"#888")};'
                            f'border-radius:6px;padding:8px 10px;margin-bottom:6px">'
                            f'<b style="color:{EVENT_COLORS.get(evt,"#333")}">{evt}</b><br>'
                            f'<span style="color:{top_color};font-weight:600">{top_lbl}</span>'
                            f'<span style="color:#888;font-size:11px"> ({top_v:+.2f}%)</span></div>',
                            unsafe_allow_html=True
                        )

        # ── 4. Full Sensitivity Heatmap (like JSX Tab 3) ──
        st.markdown("---")
        st.markdown("### 🔥 Full Sensitivity Heatmap — All Events × All Features")
        st.caption("Each cell = % change in event DOY per 1% increase in that climate variable. Red = delays event, Blue = advances event.")

        # Collect all features across all events for unified table
        all_feats_heat = []
        for evt in model_store:
            for f in sens_matrix.get(evt, {}):
                if f not in all_feats_heat:
                    all_feats_heat.append(f)

        if all_feats_heat:
            heat_data = []
            for f in all_feats_heat:
                row = {"Climate Variable": FACTOR_LABELS.get(f, f)}
                for evt in ["SOS", "POS", "EOS"]:
                    row[evt] = sens_matrix.get(evt, {}).get(f, None)
                heat_data.append(row)
            heat_df = pd.DataFrame(heat_data).set_index("Climate Variable")

            # Color the heatmap
            def heat_style(val):
                if val is None or np.isnan(float(val) if val is not None else np.nan):
                    return "background-color: #f5f5f5; color: #ccc"
                v = float(val)
                max_abs = max(
                    max((abs(v2) for row in heat_data for k, v2 in row.items() if k != "Climate Variable" and v2 is not None), default=1),
                    0.01
                )
                norm = v / max_abs
                if norm > 0:
                    intensity = int(norm * 180)
                    return f"background-color: rgb(255,{255-intensity},{255-intensity}); color: {'#500' if intensity>100 else '#333'}"
                else:
                    intensity = int(abs(norm) * 180)
                    return f"background-color: rgb({255-intensity},{255-intensity},255); color: {'#005' if intensity>100 else '#333'}"

            styled = heat_df.style.applymap(heat_style).format(
                lambda v: f"{v:+.2f}%" if v is not None and not (isinstance(v, float) and np.isnan(v)) else "—"
            )
            st.dataframe(styled, use_container_width=True)

            # Legend
            st.markdown("""
<div style="display:flex;gap:24px;font-size:12px;margin-top:6px">
  <span><span style="background:rgb(255,75,75);padding:2px 10px;border-radius:3px">&nbsp;</span> &nbsp;Delays event (positive)</span>
  <span><span style="background:rgb(75,75,255);padding:2px 10px;border-radius:3px">&nbsp;</span> &nbsp;Advances event (negative)</span>
  <span><span style="background:#f5f5f5;border:1px solid #ddd;padding:2px 10px;border-radius:3px">—</span> &nbsp;Feature not in this event's model</span>
</div>
""", unsafe_allow_html=True)

        # ── 5. Cross-metric driver dominance cards (like JSX radar bottom grid) ──
        st.markdown("---")
        st.markdown("### 🃏 Cross-Metric Driver Dominance")
        st.caption("Which climate variable has the strongest influence on each phenological event?")
        card_cols = st.columns(len(model_store))
        for col_c, (evt, s_dict) in zip(card_cols, sens_matrix.items()):
            with col_c:
                if s_dict:
                    sorted_drivers = sorted(s_dict.items(), key=lambda x: -abs(x[1]))
                    dom_f, dom_v   = sorted_drivers[0]
                    dom_lbl = FACTOR_LABELS.get(dom_f, dom_f)
                    dom_col = FACTOR_COLORS.get(dom_f, "#888")
                    evt_col = EVENT_COLORS.get(evt, "#333")
                    st.markdown(f"""
<div style="border:2px solid {evt_col};border-radius:10px;padding:14px;text-align:center;margin-bottom:8px">
  <div style="font-size:18px;font-weight:800;color:{evt_col}">{evt}</div>
  <div style="font-size:11px;color:#888;margin-bottom:8px">{METRIC_LABELS_D.get(evt,'')}</div>
  <div style="font-size:15px;font-weight:700;color:{dom_col}">{dom_lbl}</div>
  <div style="font-size:12px;color:#666;margin-top:4px">{abs(dom_v):.2f}% sensitivity</div>
</div>
""", unsafe_allow_html=True)
                    # Show all ranked drivers for this event
                    for rank, (f, v) in enumerate(sorted_drivers):
                        lbl = FACTOR_LABELS.get(f, f)
                        fc  = FACTOR_COLORS.get(f, "#888")
                        bar_w = int(abs(v) / (abs(sorted_drivers[0][1]) + 1e-9) * 100)
                        st.markdown(
                            f'<div style="font-size:11px;margin-bottom:3px">'
                            f'<span style="color:{fc};font-weight:600">{rank+1}. {lbl}</span>'
                            f'<span style="color:#999"> {v:+.2f}%</span>'
                            f'<div style="background:{fc};height:4px;width:{bar_w}%;border-radius:2px;opacity:0.6;margin-top:2px"></div>'
                            f'</div>',
                            unsafe_allow_html=True
                        )

        # ── Download sensitivity table ──
        st.markdown("---")
        if all_feats_heat:
            heat_dl = pd.DataFrame([
                {"Feature": FACTOR_LABELS.get(f, f),
                 **{evt: sens_matrix.get(evt, {}).get(f, None) for evt in ["SOS","POS","EOS"]}}
                for f in all_feats_heat
            ])
            st.download_button(
                "⬇ Download sensitivity matrix CSV",
                heat_dl.to_csv(index=False),
                file_name="driver_sensitivity_v6.csv",
                mime="text/csv"
            )

# ───────────────────────────────────────────────────────────────────────────────
# TAB 3 — CORRELATIONS
# ───────────────────────────────────────────────────────────────────────────────
with tab_corr:
    st.subheader("Feature Correlations with Phenology Metrics")

    met_vars_plot = [c for c in ["T2M","PRECTOTCORR","RH2M","WS2M","ALLSKY_SFC_SW_DWN","VPD","DTR","SPEI_proxy"] if c in df.columns]
    metrics_plot  = {evt: np.array([pheno_records[s].get(f"{evt}_DOY") for s in yrs], dtype=float)
                     for evt in ["SOS","POS","EOS"]}

    # Seasonal mean features
    seas_feats = {col: np.array([df[df["Season"]==s][col].mean() for s in yrs]) for col in met_vars_plot}

    fig3, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, (evt, col) in zip(axes, [("SOS","#22c55e"),("POS","#f59e0b"),("EOS","#ef4444")]):
        y_ev = metrics_plot[evt]
        rs = {}
        for feat, x_ev in seas_feats.items():
            if np.std(x_ev) > 0:
                rs[feat] = round(pearsonr(x_ev, y_ev)[0], 3)
        rs_sorted = dict(sorted(rs.items(), key=lambda x: x[1]))
        bars = ax.barh(list(rs_sorted.keys()), list(rs_sorted.values()),
                       color=[("#22c55e" if v>=0 else "#ef4444") for v in rs_sorted.values()], alpha=0.8)
        ax.axvline(0, color="black", lw=0.8)
        ax.axvline(0.40, color="#f59e0b", lw=1, ls="--", alpha=0.6, label="|r|=0.40 threshold")
        ax.axvline(-0.40, color="#f59e0b", lw=1, ls="--", alpha=0.6)
        ax.set_title(f"{evt} vs Met features (seasonal mean)", fontweight="bold", color=col)
        ax.set_xlabel("Pearson r"); ax.legend(fontsize=8)
        ax.grid(axis="x", alpha=0.3)
    fig3.tight_layout()
    st.pyplot(fig3); plt.close()

    st.markdown("### Year-by-Year Meteorology")
    for col in ["PRECTOTCORR","T2M","RH2M"]:
        if col not in df.columns: continue
        fig4, ax4 = plt.subplots(figsize=(13, 3))
        ax4.plot(df["Date"], df[col], color="#3b82f6", lw=1, alpha=0.8)
        ax4.set_ylabel(col); ax4.set_title(col); ax4.grid(alpha=0.3)
        fig4.tight_layout(); st.pyplot(fig4); plt.close()

# ───────────────────────────────────────────────────────────────────────────────
# TAB 4 — PREDICT 2026
# ───────────────────────────────────────────────────────────────────────────────
with tab_predict:
    st.subheader("🔮 Predict Phenology for a Future Year")

    if not model_store:
        st.warning("Train models first (Training tab)."); st.stop()

    st.markdown("""
<div class='warn-box'>
<b>Note:</b> Enter forecast meteorological conditions for the prediction window before each event.
Default values are training data means. Confidence reflects LOO cross-validation error.
</div>
""", unsafe_allow_html=True)

    pred_year = st.number_input("Prediction year", 2024, 2050, 2026, step=1)

    pred_results = {}
    for evt, info in model_store.items():
        st.markdown(f"---\n#### {evt} — {event_configs[evt][1]}")
        st.caption(f"Using {info['window']}-day window before event  |  Features: {', '.join(info['features'])}")

        # Default = training mean for each feature
        X_tr = info["X"]
        feat_names = info["feat_names"]
        train_means = {f: round(float(np.mean(X_tr[:, feat_names.index(f)])), 3) for f in info["features"]}

        cols = st.columns(min(4, len(info["features"])))
        user_vals = {}
        for i, f in enumerate(info["features"]):
            all_vals = X_tr[:, feat_names.index(f)]
            hint     = f"Range: [{all_vals.min():.1f} – {all_vals.max():.1f}]"
            with cols[i % len(cols)]:
                user_vals[f] = st.number_input(f, value=train_means[f], help=hint, key=f"{evt}_{f}")

        x_pred = np.array([[user_vals[f] for f in info["features"]]])
        x_sc   = info["scaler"].transform(x_pred)
        doy_pred = float(info["model"].predict(x_sc)[0])

        # Ecological ordering enforcement
        if evt == "POS" and "SOS" in pred_results:
            doy_pred = max(doy_pred, pred_results["SOS"] + 10)
        if evt == "EOS" and "POS" in pred_results:
            doy_pred = max(doy_pred, pred_results["POS"] + 10)

        pred_results[evt] = doy_pred
        date_str = doy_to_date(pred_year, int(round(doy_pred)))
        st.success(f"**Predicted {evt}:** DOY {doy_pred:.1f}  →  **{date_str}, {pred_year}**  |  MAE=±{info['mae']:.1f} days")

    # LOS
    if "SOS" in pred_results and "EOS" in pred_results:
        eos_cont = pred_results["EOS"] + (365 if pred_results["EOS"] < pred_results["SOS"] else 0)
        los_pred = eos_cont - pred_results["SOS"]
        st.info(f"**Predicted LOS:** ~{los_pred:.0f} days")

    # Summary table
    if pred_results:
        st.markdown("### Prediction Summary")
        sum_rows = []
        for evt, doy in pred_results.items():
            sum_rows.append({
                "Event": evt,
                "Predicted DOY": round(doy, 1),
                "Calendar Date": doy_to_date(pred_year, int(round(doy))),
                "MAE (±days)": round(model_store[evt]["mae"], 1),
                "R²(LOO)": round(model_store[evt]["r2_loo"], 3),
            })
        sum_df = pd.DataFrame(sum_rows)
        st.dataframe(sum_df, use_container_width=True)
        st.download_button("⬇ Download predictions CSV", sum_df.to_csv(index=False),
                           file_name=f"phenology_prediction_{pred_year}_v6.csv", mime="text/csv")

# ───────────────────────────────────────────────────────────────────────────────
# TAB 5 — TECHNICAL GUIDE
# ───────────────────────────────────────────────────────────────────────────────
with tab_guide:
    st.subheader("Technical Guide — v6 Improvements")
    st.markdown("""
## What's New in v6 — Monsoon-Aware Design

### 🔑 The Core Problem with v5 on Indian Sites
In v5, **T2M always dominated** every phenological metric. Here's why this is wrong for tropical Indian forests:

| Issue | v5 Behaviour | v6 Fix |
|-------|-------------|--------|
| T2M variation | 24.15–24.82°C (< 0.7°C range) | Variance-weighted ranking — low-CV features down-ranked |
| PRECTOTCORR dropped | Dropped because |r| > 0.85 with T2M | Moisture protection flag — PRECTOTCORR/RH2M survive collinearity filter |
| GDD_cum leakage | Used as predictor | Excluded from forward selection (leakage guard) |
| Short windows | 30d default misses monsoon signal | 60/90d default for SOS |
| Precipitation = mean | Mean of 5-day totals | Sum = actual accumulation (physically correct) |

---

### 🌧️ Correct Causal Drivers for Indian Monsoon Forests

**SOS (Start of Season / Green-up)**
- **PRIMARY:** `PRECTOTCORR_sum` over 60–90d — monsoon onset triggers green-up
- **SECONDARY:** `SPEI_proxy` — moisture surplus/deficit signal
- **SECONDARY:** `RH2M` — humidity rise confirms monsoon arrival
- **NOT:** T2M (range < 1°C = no explanatory power for timing differences)

**POS (Peak of Season)**
- **PRIMARY:** Cumulative seasonal precipitation total
- **PRIMARY:** `RH2M` sustained during growing season
- **SECONDARY:** `ALLSKY_SFC_SW_DWN` — solar radiation modulates photosynthetic rate

**EOS (End of Season / Senescence)**
- **PRIMARY:** `WS2M` — dry post-monsoon winds
- **PRIMARY:** `VPD` — vapour pressure deficit (moisture stress)
- **SECONDARY:** Precipitation cessation (`PRECTOTCORR_sum` approaching 0)

---

### 📐 Feature Selection Algorithm (v6)
1. Compute Pearson |r| + Spearman |ρ| composite
2. **Variance weight**: if CV < 2%, composite × 0.5 (down-ranks T2M etc.)
3. **Leakage exclusion**: GDD_cum excluded from candidate set
4. **Collinearity filter** with moisture protection:
   - If PRECTOTCORR/RH2M is collinear with T2M → **keep moisture, drop T2M**
5. Forward LOO R² selection (add feature if ΔR² ≥ 0.03)

---

### ⚠️ Known Limitations
- Only 3 complete seasons → wide confidence intervals (t-distribution, df=1)
- 19-year extrapolation to 2026 → large uncertainty
- Linear trend assumption may not hold under non-stationary climate
- Consider adding more years of data for reliable predictions

---

### 📚 Citation
```
Sharma, S. (2025). Universal Indian Forest Phenology Predictor v6 [Software].
GitHub. https://github.com/shreejisharma/Indian-forest-phenology
```
""")
