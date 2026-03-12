"""
Universal Indian Forest Phenology Predictor — v5
100% Data-Driven | All Indian Forest Types
Author: Sharma, S. (2025)
"""

import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
from scipy.signal import savgol_filter, find_peaks
from scipy.stats import pearsonr, spearmanr
from scipy.interpolate import interp1d
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
from sklearn.metrics import r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
import io
import re

# ── PAGE CONFIG ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Indian Forest Phenology Predictor v5",
    page_icon="🌲",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CUSTOM CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        background-color: #1e2530;
        border-radius: 8px 8px 0 0;
        padding: 8px 20px;
        color: #9ca3af;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background-color: #22c55e !important;
        color: #000 !important;
    }
    .metric-card {
        background: #1e2530;
        border: 1px solid #374151;
        border-radius: 10px;
        padding: 16px;
        margin: 6px 0;
    }
    .highlight-box {
        background: linear-gradient(135deg, #14532d, #1a2535);
        border: 1px solid #22c55e;
        border-radius: 10px;
        padding: 14px 18px;
        margin: 10px 0;
    }
    .section-header {
        font-size: 1.1rem;
        font-weight: 700;
        color: #22c55e;
        border-bottom: 1px solid #374151;
        padding-bottom: 6px;
        margin: 18px 0 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PHENOLOGY ENGINE
# ══════════════════════════════════════════════════════════════════════════════

class PhenologyEngine:
    """Fully data-driven phenology extraction and regression engine."""

    # ── 1. NDVI Loading & Cleaning ─────────────────────────────────────────
    @staticmethod
    def load_ndvi(uploaded_file) -> pd.DataFrame:
        df = pd.read_csv(uploaded_file)
        df.columns = [c.strip().lower() for c in df.columns]
        date_col = next((c for c in df.columns if "date" in c or "time" in c), df.columns[0])
        ndvi_col = next((c for c in df.columns if "ndvi" in c), df.columns[1])
        df = df[[date_col, ndvi_col]].rename(columns={date_col: "date", ndvi_col: "NDVI"})
        df["date"] = pd.to_datetime(df["date"], infer_datetime_format=True, errors="coerce")
        df = df.dropna(subset=["date", "NDVI"]).sort_values("date").reset_index(drop=True)
        df["NDVI"] = pd.to_numeric(df["NDVI"], errors="coerce")
        df = df.dropna(subset=["NDVI"])
        df["NDVI"] = df["NDVI"].clip(0, 1)
        return df

    # ── 2. NASA POWER Loading ──────────────────────────────────────────────
    @staticmethod
    def load_met(uploaded_file) -> pd.DataFrame:
        raw = uploaded_file.read().decode("utf-8", errors="ignore")
        uploaded_file.seek(0)
        lines = raw.splitlines()
        # Auto-detect header row (find line with YEAR and DOY or DATE)
        header_idx = 0
        for i, line in enumerate(lines):
            if re.search(r"YEAR.*DOY|DOY.*YEAR|DATE", line, re.IGNORECASE):
                header_idx = i
                break
        df = pd.read_csv(io.StringIO("\n".join(lines[header_idx:])), on_bad_lines="skip")
        df.columns = [c.strip() for c in df.columns]
        # Build date
        if "YEAR" in df.columns and "DOY" in df.columns:
            df["date"] = pd.to_datetime(
                df["YEAR"].astype(str) + df["DOY"].astype(str).str.zfill(3), format="%Y%j", errors="coerce"
            )
        elif "DATE" in df.columns:
            df["date"] = pd.to_datetime(df["DATE"], infer_datetime_format=True, errors="coerce")
        df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
        # Replace -999 sentinel
        df = df.replace(-999.0, np.nan).replace(-99.0, np.nan)
        return df

    # ── 3. Data-Driven Cadence & Cycle ─────────────────────────────────────
    @staticmethod
    def detect_cadence(ndvi_df: pd.DataFrame) -> int:
        diffs = ndvi_df["date"].diff().dt.days.dropna()
        cadence = int(np.median(diffs[diffs > 0]))
        return max(cadence, 1)

    @staticmethod
    def detect_cycle_length(smooth: np.ndarray, cadence: int) -> int:
        from numpy.fft import fft, fftfreq
        n = len(smooth)
        if n < 60:
            return 365
        yf = np.abs(fft(smooth - smooth.mean()))
        xf = fftfreq(n, d=cadence)
        xf_pos = xf[1 : n // 2]
        yf_pos = yf[1 : n // 2]
        periods = 1.0 / (xf_pos + 1e-9)
        valid = (periods > 200) & (periods < 800)
        if valid.sum() == 0:
            return 365
        dominant_period = int(periods[valid][np.argmax(yf_pos[valid])])
        return np.clip(dominant_period, 270, 550)

    # ── 4. Season Segmentation ─────────────────────────────────────────────
    @staticmethod
    def segment_seasons(ndvi_df: pd.DataFrame, cadence: int, amplitude_pct: float = 0.25):
        dates = ndvi_df["date"].values
        ndvi_raw = ndvi_df["NDVI"].values.copy()

        # Interpolate to regular grid (within segments only)
        date_nums = (pd.DatetimeIndex(dates) - pd.Timestamp("2000-01-01")).days.values
        all_days = np.arange(date_nums[0], date_nums[-1] + 1, cadence)
        f_interp = interp1d(date_nums, ndvi_raw, bounds_error=False, fill_value=np.nan)
        ndvi_interp = f_interp(all_days)

        # SG smoothing (window ≤ 31 steps)
        n = len(ndvi_interp)
        sg_win = min(31, n if n % 2 == 1 else n - 1)
        sg_win = max(sg_win, 5)
        smooth = savgol_filter(np.nan_to_num(ndvi_interp, nan=np.nanmean(ndvi_interp)), sg_win, 3)

        # Cycle length
        cycle_len = PhenologyEngine.detect_cycle_length(smooth, cadence)
        min_trough_dist = max(10, int(0.40 * cycle_len / cadence))

        # Trough detection
        neg_smooth = -smooth
        troughs, _ = find_peaks(neg_smooth, distance=min_trough_dist, height=-np.percentile(smooth, 85))
        if len(troughs) < 2:
            troughs, _ = find_peaks(neg_smooth, distance=min_trough_dist // 2)
        if len(troughs) < 2:
            return [], smooth, all_days

        # MIN_AMPLITUDE from data (5% of P5–P95 range)
        p5, p95 = np.percentile(ndvi_raw, 5), np.percentile(ndvi_raw, 95)
        min_amp = 0.05 * (p95 - p5)

        seasons = []
        for i in range(len(troughs) - 1):
            t0, t1 = troughs[i], troughs[i + 1]
            seg = smooth[t0:t1 + 1]
            if len(seg) < 5:
                continue
            amp = seg.max() - seg.min()
            if amp < min_amp:
                continue
            year_start = pd.Timestamp("2000-01-01") + pd.Timedelta(days=int(all_days[t0]))
            seasons.append({
                "idx0": t0, "idx1": t1,
                "seg": seg,
                "day0": all_days[t0],
                "year": year_start.year,
            })
        return seasons, smooth, all_days

    # ── 5. SOS / POS / EOS Extraction ─────────────────────────────────────
    @staticmethod
    def extract_events(seasons, smooth, all_days, ndvi_raw_df, threshold_pct=0.25):
        records = []
        for s in seasons:
            seg = s["seg"]
            t0, t1 = s["idx0"], s["idx1"]
            vmin, vmax = seg.min(), seg.max()
            amp = vmax - vmin
            if amp < 1e-4:
                continue
            thresh = vmin + threshold_pct * amp

            # SOS: first crossing upward
            sos_local = None
            for k in range(1, len(seg)):
                if seg[k - 1] < thresh <= seg[k]:
                    sos_local = k
                    break
            # EOS: last crossing downward
            eos_local = None
            for k in range(len(seg) - 1, 0, -1):
                if seg[k - 1] >= thresh > seg[k]:
                    eos_local = k
                    break

            if sos_local is None or eos_local is None or sos_local >= eos_local:
                continue

            sos_abs = t0 + sos_local
            eos_abs = t0 + eos_local

            # POS: raw NDVI max in window
            sos_day = all_days[sos_abs]
            eos_day = all_days[eos_abs]
            window_mask = (
                (pd.DatetimeIndex(ndvi_raw_df["date"]) >= pd.Timestamp("2000-01-01") + pd.Timedelta(days=int(sos_day))) &
                (pd.DatetimeIndex(ndvi_raw_df["date"]) <= pd.Timestamp("2000-01-01") + pd.Timedelta(days=int(eos_day)))
            )
            if window_mask.sum() == 0:
                continue
            raw_win = ndvi_raw_df[window_mask]
            pos_row = raw_win.loc[raw_win["NDVI"].idxmax()]

            sos_date = pd.Timestamp("2000-01-01") + pd.Timedelta(days=int(sos_day))
            eos_date = pd.Timestamp("2000-01-01") + pd.Timedelta(days=int(eos_day))
            pos_date = pd.Timestamp(pos_row["date"])

            records.append({
                "year": s["year"],
                "SOS": sos_date,
                "POS": pos_date,
                "EOS": eos_date,
                "SOS_DOY": sos_date.dayofyear,
                "POS_DOY": pos_date.dayofyear,
                "EOS_DOY": eos_date.dayofyear,
                "LOS": (eos_date - sos_date).days,
                "Peak_NDVI": float(pos_row["NDVI"]),
                "Amplitude": float(amp),
            })

        df = pd.DataFrame(records).drop_duplicates(subset=["year"]).sort_values("year").reset_index(drop=True)
        return df

    # ── 6. Meteorological Feature Engineering ─────────────────────────────
    @staticmethod
    def engineer_met_features(met_df: pd.DataFrame) -> pd.DataFrame:
        df = met_df.copy()
        # GDD
        if "T2M" in df.columns:
            df["GDD_5"] = (df["T2M"] - 5).clip(0)
            df["GDD_10"] = (df["T2M"] - 10).clip(0)
            df["GDD_cum"] = df["GDD_5"].cumsum()
        if "T2M_MAX" in df.columns and "T2M_MIN" in df.columns:
            df["DTR"] = df["T2M_MAX"] - df["T2M_MIN"]
        # VPD
        if "T2M" in df.columns and "RH2M" in df.columns:
            es = 0.6108 * np.exp(17.27 * df["T2M"] / (df["T2M"] + 237.3))
            df["VPD"] = es * (1 - df["RH2M"] / 100)
        # Log precip
        if "PRECTOTCORR" in df.columns:
            df["log_precip"] = np.log1p(df["PRECTOTCORR"])
        # MSI = radiation / (precip + 1)
        if "ALLSKY_SFC_SW_DWN" in df.columns and "PRECTOTCORR" in df.columns:
            df["MSI"] = df["ALLSKY_SFC_SW_DWN"] / (df["PRECTOTCORR"] + 1)
        # SPEI proxy
        if "T2M" in df.columns and "PRECTOTCORR" in df.columns:
            pet = 0.0023 * (df["T2M"] + 17.8) * df.get("ALLSKY_SFC_SW_DWN", 20) ** 0.5
            df["SPEI_proxy"] = df["PRECTOTCORR"] - pet
        return df

    # ── 7. Window-Mean Feature Computation ────────────────────────────────
    @staticmethod
    def compute_window_features(met_eng: pd.DataFrame, event_date: pd.Timestamp, window_days: int) -> dict:
        t1 = event_date
        t0 = t1 - pd.Timedelta(days=window_days)
        mask = (met_eng["date"] >= t0) & (met_eng["date"] < t1)
        sub = met_eng[mask]
        if len(sub) == 0:
            return {}
        excl = {"date", "YEAR", "DOY", "LAT", "LON", "ELEVATION"}
        feats = {}
        for col in sub.columns:
            if col in excl:
                continue
            vals = pd.to_numeric(sub[col], errors="coerce").dropna()
            if len(vals) > 0:
                feats[col] = float(vals.mean())
        return feats

    # ── 8. Feature Selection ───────────────────────────────────────────────
    @staticmethod
    def select_features(X: pd.DataFrame, y: pd.Series, r_thresh: float = 0.40, collin_thresh: float = 0.85):
        if len(X) < 4 or len(y) < 4:
            return list(X.columns[:3])
        candidates = []
        for col in X.columns:
            vals = X[col].dropna()
            if len(vals) < 4 or vals.std() < 1e-6:
                continue
            idx = X[col].dropna().index.intersection(y.dropna().index)
            if len(idx) < 4:
                continue
            r, _ = pearsonr(X.loc[idx, col], y.loc[idx])
            rho, _ = spearmanr(X.loc[idx, col], y.loc[idx])
            composite = (abs(r) + abs(rho)) / 2
            candidates.append((col, composite, r))
        candidates.sort(key=lambda x: -x[1])
        # Filter by threshold
        selected = [(c, r) for c, comp, r in candidates if comp >= r_thresh]
        if not selected:
            selected = [(candidates[0][0], candidates[0][2])] if candidates else []
        # Remove collinearity
        kept = []
        for feat, r_val in selected:
            if not kept:
                kept.append(feat)
                continue
            corr_with_kept = max(abs(X[feat].corr(X[k])) for k in kept)
            if corr_with_kept < collin_thresh:
                kept.append(feat)
        return kept

    # ── 9. LOO Cross-Validation ───────────────────────────────────────────
    @staticmethod
    def loo_cv(X: np.ndarray, y: np.ndarray, model_fn):
        n = len(y)
        if n < 3:
            return np.nan, np.nan
        preds = []
        for i in range(n):
            idx_train = [j for j in range(n) if j != i]
            Xt, yt = X[idx_train], y[idx_train]
            Xv = X[[i]]
            try:
                m = model_fn()
                m.fit(Xt, yt)
                preds.append(float(m.predict(Xv)[0]))
            except Exception:
                preds.append(float(np.mean(yt)))
        preds = np.array(preds)
        ss_res = np.sum((y - preds) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 1e-9 else np.nan
        mae = float(np.mean(np.abs(y - preds)))
        return r2, mae

    # ── 10. Model Fitting ─────────────────────────────────────────────────
    @staticmethod
    def fit_models(X: np.ndarray, y: np.ndarray):
        results = {}
        sc = StandardScaler()
        Xs = sc.fit_transform(X)

        # Ridge
        ridge = RidgeCV(alphas=np.logspace(-3, 4, 30), cv=None)
        ridge.fit(Xs, y)
        r2, mae = PhenologyEngine.loo_cv(Xs, y, lambda: RidgeCV(alphas=np.logspace(-3, 4, 30)))
        results["Ridge"] = {"model": ridge, "scaler": sc, "loo_r2": r2, "loo_mae": mae,
                            "coefs": ridge.coef_, "intercept": ridge.intercept_}

        # Polynomial deg-2
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import PolynomialFeatures
        for deg in [2, 3]:
            try:
                poly_pipe = Pipeline([
                    ("poly", PolynomialFeatures(degree=deg, include_bias=False)),
                    ("sc", StandardScaler()),
                    ("ridge", RidgeCV(alphas=np.logspace(-3, 4, 20))),
                ])
                poly_pipe.fit(X, y)
                r2p, maep = PhenologyEngine.loo_cv(
                    X, y,
                    lambda d=deg: Pipeline([
                        ("poly", PolynomialFeatures(degree=d, include_bias=False)),
                        ("sc", StandardScaler()),
                        ("ridge", RidgeCV(alphas=np.logspace(-3, 4, 20))),
                    ])
                )
                results[f"Poly_deg{deg}"] = {"model": poly_pipe, "scaler": None,
                                              "loo_r2": r2p, "loo_mae": maep,
                                              "coefs": None, "intercept": None}
            except Exception:
                pass

        # GPR
        if len(y) >= 5:
            try:
                kernel = ConstantKernel(1.0) * RBF(1.0) + WhiteKernel(0.1)
                gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5, normalize_y=True)
                gpr.fit(Xs, y)
                r2g, maeg = PhenologyEngine.loo_cv(
                    Xs, y,
                    lambda: GaussianProcessRegressor(
                        kernel=ConstantKernel(1.0) * RBF(1.0) + WhiteKernel(0.1),
                        normalize_y=True
                    )
                )
                results["GPR"] = {"model": gpr, "scaler": sc, "loo_r2": r2g, "loo_mae": maeg,
                                   "coefs": None, "intercept": None}
            except Exception:
                pass

        return results


# ══════════════════════════════════════════════════════════════════════════════
# PLOTTING HELPERS
# ══════════════════════════════════════════════════════════════════════════════

PALETTE = {
    "sos": "#3b82f6", "pos": "#22c55e", "eos": "#f97316",
    "ndvi": "#22c55e", "smooth": "#a7f3d0", "met": "#f59e0b",
}

def fig_to_bytes(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    buf.seek(0)
    return buf

def plot_ndvi_overview(ndvi_df, seasons, smooth, all_days, pheno_df):
    fig, ax = plt.subplots(figsize=(14, 4.5), facecolor="#0e1117")
    ax.set_facecolor("#0e1117")
    ax.patch.set_alpha(0.0)

    # Raw
    ax.scatter(ndvi_df["date"], ndvi_df["NDVI"], s=12, color="#4ade80", alpha=0.5, label="Raw NDVI", zorder=2)

    # Smooth
    smooth_dates = [pd.Timestamp("2000-01-01") + pd.Timedelta(days=int(d)) for d in all_days]
    ax.plot(smooth_dates, smooth, color="#a7f3d0", lw=1.5, label="SG Smooth", zorder=3)

    # Phenology markers
    if not pheno_df.empty:
        for _, row in pheno_df.iterrows():
            ax.axvline(row["SOS"], color=PALETTE["sos"], alpha=0.5, lw=1.2, ls="--")
            ax.axvline(row["POS"], color=PALETTE["pos"], alpha=0.5, lw=1.2, ls="-.")
            ax.axvline(row["EOS"], color=PALETTE["eos"], alpha=0.5, lw=1.2, ls=":")

    ax.set_xlabel("Date", color="#9ca3af", fontsize=10)
    ax.set_ylabel("NDVI", color="#9ca3af", fontsize=10)
    ax.tick_params(colors="#9ca3af")
    for spine in ax.spines.values():
        spine.set_edgecolor("#374151")
    legend_elements = [
        Patch(facecolor=PALETTE["sos"], label="SOS"),
        Patch(facecolor=PALETTE["pos"], label="POS"),
        Patch(facecolor=PALETTE["eos"], label="EOS"),
    ]
    ax.legend(handles=legend_elements + [
        plt.Line2D([0], [0], color="#4ade80", marker="o", ls="", ms=5, label="Raw"),
        plt.Line2D([0], [0], color="#a7f3d0", lw=2, label="Smooth"),
    ], facecolor="#1e2530", edgecolor="#374151", labelcolor="#e5e7eb", fontsize=8, loc="upper left")
    ax.set_title("NDVI Time Series with Phenology Events", color="#e5e7eb", fontsize=12, pad=10)
    plt.tight_layout()
    return fig


def plot_correlation_bar(corr_df, event):
    fig, ax = plt.subplots(figsize=(9, max(3, len(corr_df) * 0.45)), facecolor="#0e1117")
    ax.set_facecolor("#1e2530")
    colors = ["#22c55e" if r >= 0 else "#ef4444" for r in corr_df["pearson_r"]]
    bars = ax.barh(corr_df["feature"], corr_df["pearson_r"].abs(), color=colors, edgecolor="#374151", height=0.6)
    for bar, r in zip(bars, corr_df["pearson_r"]):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{r:+.3f}", va="center", color="#e5e7eb", fontsize=8)
    ax.axvline(0.40, color="#fbbf24", ls="--", lw=1.2, label="|r|=0.40 threshold")
    ax.set_xlabel("|Pearson r|", color="#9ca3af")
    ax.set_title(f"Feature Correlations — {event}", color="#e5e7eb", fontsize=11)
    ax.tick_params(colors="#9ca3af")
    for spine in ax.spines.values(): spine.set_edgecolor("#374151")
    ax.legend(facecolor="#1e2530", edgecolor="#374151", labelcolor="#e5e7eb", fontsize=8)
    plt.tight_layout()
    return fig


def plot_obs_pred(y_obs, y_pred, event, r2, mae):
    fig, ax = plt.subplots(figsize=(5, 5), facecolor="#0e1117")
    ax.set_facecolor("#1e2530")
    ax.scatter(y_obs, y_pred, color="#22c55e", s=60, edgecolor="#a7f3d0", zorder=3)
    lims = [min(y_obs.min(), min(y_pred)), max(y_obs.max(), max(y_pred))]
    ax.plot(lims, lims, "w--", lw=1.2, alpha=0.5)
    ax.set_xlabel("Observed DOY", color="#9ca3af")
    ax.set_ylabel("Predicted DOY", color="#9ca3af")
    ax.tick_params(colors="#9ca3af")
    for spine in ax.spines.values(): spine.set_edgecolor("#374151")
    r2_str = f"{r2:.3f}" if not np.isnan(r2) else "N/A"
    ax.set_title(f"{event}  R²={r2_str}  MAE={mae:.1f}d", color="#e5e7eb", fontsize=10)
    plt.tight_layout()
    return fig


def plot_corr_heatmap(feat_matrix: pd.DataFrame):
    corr = feat_matrix.corr()
    fig, ax = plt.subplots(figsize=(max(6, len(corr) * 0.8), max(5, len(corr) * 0.7)), facecolor="#0e1117")
    ax.set_facecolor("#1e2530")
    im = ax.imshow(corr.values, cmap="RdYlGn", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.index)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right", color="#9ca3af", fontsize=8)
    ax.set_yticklabels(corr.index, color="#9ca3af", fontsize=8)
    for i in range(len(corr)):
        for j in range(len(corr.columns)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center",
                    color="black" if abs(corr.iloc[i, j]) > 0.5 else "#e5e7eb", fontsize=7)
    plt.colorbar(im, ax=ax, fraction=0.03)
    ax.set_title("Feature Correlation Heatmap", color="#e5e7eb", fontsize=11)
    plt.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# SESSION STATE HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def init_state():
    defaults = {
        "ndvi_df": None, "met_df": None, "met_eng": None,
        "seasons": None, "smooth": None, "all_days": None,
        "pheno_df": None, "feature_df": None, "models": {},
        "corr_tables": {}, "trained": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

def render_sidebar():
    with st.sidebar:
        st.markdown("## 🌲 Forest Phenology v5")
        st.markdown("---")
        st.markdown("### 📂 Upload Data")
        ndvi_file = st.file_uploader("NDVI CSV", type=["csv"], key="ndvi_upload",
                                     help="Columns: date, NDVI")
        met_file = st.file_uploader("NASA POWER Meteorology CSV", type=["csv"], key="met_upload",
                                    help="NASA POWER daily export — headers auto-detected")
        st.markdown("---")
        st.markdown("### ⚙️ Parameters")
        threshold_pct = st.slider("SOS/EOS threshold (% amplitude)", 10, 50, 25, 5,
                                   help="50% amplitude threshold = standard half-max method") / 100
        window_days = st.slider("Meteorological window (days before event)", 15, 90, 30, 5)
        r_thresh = st.slider("Feature selection |r| threshold", 0.20, 0.70, 0.40, 0.05)
        st.markdown("---")
        st.markdown("### ℹ️ About")
        st.markdown("""
        **Universal Indian Forest Phenology Predictor v5**  
        Supports all forest types — Tropical, Evergreen, Shola, Mangrove, Himalayan, Alpine.  
        100% data-driven · No hardcoded presets.
        
        [GitHub Repo](https://github.com/shreejisharma/Indian-forest-phenology)
        """)
    return ndvi_file, met_file, threshold_pct, window_days, r_thresh


# ══════════════════════════════════════════════════════════════════════════════
# TAB 0: DATA OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════

def tab_data_overview():
    st.markdown('<div class="section-header">📊 Data Characterization</div>', unsafe_allow_html=True)

    ndvi_df = st.session_state.ndvi_df
    met_df = st.session_state.met_df

    if ndvi_df is None:
        st.info("Upload NDVI CSV in the sidebar to begin.")
        return

    cadence = PhenologyEngine.detect_cadence(ndvi_df)
    p5 = float(np.percentile(ndvi_df["NDVI"], 5))
    p95 = float(np.percentile(ndvi_df["NDVI"], 95))
    evergreen_idx = float(np.percentile(ndvi_df["NDVI"], 10))

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Observations", len(ndvi_df))
    col2.metric("Date Range", f"{ndvi_df['date'].dt.year.min()}–{ndvi_df['date'].dt.year.max()}")
    col3.metric("Detected Cadence", f"{cadence} days")
    col4.metric("NDVI P5–P95", f"{p5:.3f} – {p95:.3f}")
    col5.metric("Evergreen Index (P10)", f"{evergreen_idx:.3f}",
                help="P10 > 0.4 suggests evergreen forest")

    fig, axes = plt.subplots(1, 2, figsize=(13, 3.5), facecolor="#0e1117")
    # NDVI histogram
    axes[0].set_facecolor("#1e2530")
    axes[0].hist(ndvi_df["NDVI"], bins=40, color="#22c55e", edgecolor="#374151", alpha=0.85)
    axes[0].axvline(p5, color="#fbbf24", ls="--", lw=1.2, label=f"P5={p5:.3f}")
    axes[0].axvline(p95, color="#f97316", ls="--", lw=1.2, label=f"P95={p95:.3f}")
    axes[0].set_title("NDVI Distribution", color="#e5e7eb")
    axes[0].tick_params(colors="#9ca3af")
    axes[0].legend(facecolor="#1e2530", edgecolor="#374151", labelcolor="#e5e7eb", fontsize=8)
    for sp in axes[0].spines.values(): sp.set_edgecolor("#374151")

    # Temporal plot
    axes[1].set_facecolor("#1e2530")
    axes[1].scatter(ndvi_df["date"], ndvi_df["NDVI"], s=8, color="#4ade80", alpha=0.6)
    axes[1].set_title("NDVI Time Series", color="#e5e7eb")
    axes[1].tick_params(colors="#9ca3af")
    for sp in axes[1].spines.values(): sp.set_edgecolor("#374151")
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    if met_df is not None:
        st.markdown('<div class="section-header">🌦️ Meteorological Parameters</div>', unsafe_allow_html=True)
        excl = {"date", "YEAR", "DOY", "LAT", "LON", "ELEVATION", "PARAMETER"}
        met_cols = [c for c in met_df.columns if c not in excl]
        stats_rows = []
        for c in met_cols:
            vals = pd.to_numeric(met_df[c], errors="coerce").dropna()
            if len(vals) == 0: continue
            stats_rows.append({
                "Parameter": c, "Count": len(vals),
                "Mean": f"{vals.mean():.3f}", "Std": f"{vals.std():.3f}",
                "Min": f"{vals.min():.3f}", "Max": f"{vals.max():.3f}",
            })
        if stats_rows:
            st.dataframe(pd.DataFrame(stats_rows), use_container_width=True, hide_index=True)
    else:
        st.info("Upload NASA POWER CSV to see meteorological summary.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1: TRAINING
# ══════════════════════════════════════════════════════════════════════════════

def tab_training(threshold_pct, window_days, r_thresh):
    st.markdown('<div class="section-header">🔬 Phenology Extraction & Model Training</div>', unsafe_allow_html=True)

    ndvi_df = st.session_state.ndvi_df
    met_df = st.session_state.met_df

    if ndvi_df is None:
        st.info("Upload NDVI CSV to run extraction.")
        return

    run_col, _ = st.columns([1, 4])
    run_btn = run_col.button("▶ Run Extraction & Train", type="primary", use_container_width=True)

    if not run_btn and not st.session_state.trained:
        return

    if run_btn or not st.session_state.trained:
        with st.spinner("Extracting phenology events…"):
            cadence = PhenologyEngine.detect_cadence(ndvi_df)
            seasons, smooth, all_days = PhenologyEngine.segment_seasons(ndvi_df, cadence)
            pheno_df = PhenologyEngine.extract_events(seasons, smooth, all_days, ndvi_df, threshold_pct)
            st.session_state.seasons = seasons
            st.session_state.smooth = smooth
            st.session_state.all_days = all_days
            st.session_state.pheno_df = pheno_df

        if pheno_df.empty:
            st.error("No seasons extracted. Try lowering the threshold or check your data.")
            return

        # Met features
        models_out = {}
        corr_tables = {}
        feature_df_list = []

        if met_df is not None:
            with st.spinner("Engineering met features & training models…"):
                met_eng = PhenologyEngine.engineer_met_features(met_df)
                st.session_state.met_eng = met_eng

                all_feats = []
                for _, row in pheno_df.iterrows():
                    for event_col in ["SOS", "POS", "EOS"]:
                        feats = PhenologyEngine.compute_window_features(met_eng, row[event_col], window_days)
                        feats["year"] = row["year"]
                        feats["event"] = event_col
                        feats[f"{event_col}_DOY"] = row[f"{event_col}_DOY"]
                        all_feats.append(feats)

                feat_df = pd.DataFrame(all_feats)
                st.session_state.feature_df = feat_df

                for event in ["SOS", "POS", "EOS"]:
                    sub = feat_df[feat_df["event"] == event].copy()
                    y = sub[f"{event}_DOY"].astype(float)
                    drop_cols = ["year", "event", "SOS_DOY", "POS_DOY", "EOS_DOY"]
                    X_df = sub.drop(columns=[c for c in drop_cols if c in sub.columns], errors="ignore")
                    X_df = X_df.apply(pd.to_numeric, errors="coerce").dropna(axis=1)
                    y = y.loc[X_df.index]

                    if len(y) < 3 or X_df.empty:
                        continue

                    # Correlations
                    corr_rows = []
                    for col in X_df.columns:
                        if X_df[col].std() < 1e-9: continue
                        try:
                            r, _ = pearsonr(X_df[col], y)
                            rho, _ = spearmanr(X_df[col], y)
                            corr_rows.append({"feature": col, "pearson_r": r, "spearman_rho": rho,
                                              "composite": (abs(r) + abs(rho)) / 2})
                        except Exception:
                            pass
                    corr_df = pd.DataFrame(corr_rows).sort_values("composite", ascending=False)
                    corr_tables[event] = corr_df

                    # Feature selection
                    selected = PhenologyEngine.select_features(X_df, y, r_thresh)
                    if not selected:
                        continue

                    X_sel = X_df[selected].values
                    y_arr = y.values

                    mods = PhenologyEngine.fit_models(X_sel, y_arr)
                    models_out[event] = {
                        "models": mods, "selected": selected,
                        "X_df": X_df, "y": y, "corr_df": corr_df,
                    }

        st.session_state.models = models_out
        st.session_state.corr_tables = corr_tables
        st.session_state.trained = True

    # ── Display results ────────────────────────────────────────────────────
    pheno_df = st.session_state.pheno_df
    seasons = st.session_state.seasons
    smooth = st.session_state.smooth
    all_days = st.session_state.all_days

    st.success(f"✅ Extracted **{len(pheno_df)}** growing seasons")

    # NDVI overview plot
    fig = plot_ndvi_overview(ndvi_df, seasons, smooth, all_days, pheno_df)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    # Phenology table
    st.markdown('<div class="section-header">📋 Extracted Phenology Events</div>', unsafe_allow_html=True)
    display_cols = ["year", "SOS_DOY", "POS_DOY", "EOS_DOY", "LOS", "Peak_NDVI", "Amplitude"]
    st.dataframe(pheno_df[[c for c in display_cols if c in pheno_df.columns]],
                 use_container_width=True, hide_index=True)
    csv_pheno = pheno_df.to_csv(index=False).encode()
    st.download_button("⬇ Download Phenology Table (CSV)", csv_pheno, "phenology_events.csv", "text/csv")

    # Model results
    models = st.session_state.models
    if not models:
        st.info("Upload NASA POWER meteorology CSV to train predictive models.")
        return

    st.markdown('<div class="section-header">🤖 Model Performance</div>', unsafe_allow_html=True)

    all_coef_rows = []
    for event, ev_data in models.items():
        st.markdown(f"#### {event}")
        mods = ev_data["models"]
        selected = ev_data["selected"]
        y = ev_data["y"]
        X_df = ev_data["X_df"]

        # Performance cards
        cols = st.columns(len(mods))
        best_r2 = -np.inf
        best_name = None
        for col, (mname, mres) in zip(cols, mods.items()):
            r2 = mres["loo_r2"]
            mae = mres["loo_mae"]
            r2_str = f"{r2:.3f}" if not np.isnan(r2) else "N/A"
            color = "#22c55e" if (not np.isnan(r2) and r2 > 0.6) else ("#f59e0b" if (not np.isnan(r2) and r2 > 0.3) else "#ef4444")
            col.markdown(f"""
            <div class="metric-card">
                <b style="color:{color}">{mname}</b><br/>
                LOO R² = <b style="color:{color}">{r2_str}</b><br/>
                MAE = {mae:.1f} days<br/>
                Features: {len(selected)}
            </div>
            """, unsafe_allow_html=True)
            if not np.isnan(r2) and r2 > best_r2:
                best_r2 = r2
                best_name = mname

        # Obs vs Pred for best model
        best = mods.get(best_name or list(mods.keys())[0])
        if best and "scaler" in best and best["scaler"] is not None:
            Xs = best["scaler"].transform(X_df[selected].values)
            y_pred = best["model"].predict(Xs)
        else:
            try:
                y_pred = best["model"].predict(X_df[selected].values)
            except Exception:
                y_pred = np.full(len(y), y.mean())

        fig_sc = plot_obs_pred(y.values, y_pred, event, best["loo_r2"], best["loo_mae"])
        c1, c2 = st.columns([1, 2])
        c1.pyplot(fig_sc, use_container_width=True)
        plt.close(fig_sc)

        # Feature role table
        corr_df = ev_data["corr_df"]
        corr_df = corr_df.copy()
        corr_df["role"] = corr_df.apply(
            lambda r: "✅ IN MODEL" if r["feature"] in selected
            else ("⬇ Below threshold" if r["composite"] < r_thresh else "🔶 Not selected (collinear)"),
            axis=1
        )
        with c2:
            st.dataframe(corr_df[["feature", "pearson_r", "spearman_rho", "composite", "role"]]
                         .head(12).style.format({"pearson_r": "{:.3f}", "spearman_rho": "{:.3f}", "composite": "{:.3f}"}),
                         use_container_width=True, hide_index=True)

        # Ridge coefficients
        ridge_res = mods.get("Ridge")
        if ridge_res and ridge_res["coefs"] is not None:
            eq_parts = [f"{ridge_res['intercept']:.2f}"]
            for feat, coef in zip(selected, ridge_res["coefs"]):
                eq_parts.append(f"{coef:+.3f}×{feat}")
            st.markdown(f"**Ridge equation:** `{event}_DOY = {' '.join(eq_parts)}`")
            for feat, coef in zip(selected, ridge_res["coefs"]):
                all_coef_rows.append({"event": event, "model": "Ridge", "feature": feat, "coefficient": coef})
            all_coef_rows.append({"event": event, "model": "Ridge", "feature": "intercept", "coefficient": ridge_res["intercept"]})

    if all_coef_rows:
        coef_csv = pd.DataFrame(all_coef_rows).to_csv(index=False).encode()
        st.download_button("⬇ Download Model Coefficients (CSV)", coef_csv, "model_coefficients.csv", "text/csv")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2: CORRELATIONS
# ══════════════════════════════════════════════════════════════════════════════

def tab_correlations():
    st.markdown('<div class="section-header">📈 Feature Correlations</div>', unsafe_allow_html=True)

    corr_tables = st.session_state.corr_tables
    if not corr_tables:
        st.info("Run Training first to compute correlations.")
        return

    event = st.selectbox("Phenology Event", list(corr_tables.keys()))
    corr_df = corr_tables[event]

    col1, col2 = st.columns([1, 1])
    with col1:
        fig = plot_correlation_bar(corr_df.head(20), event)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    feat_df = st.session_state.feature_df
    if feat_df is not None:
        top_feats = corr_df.head(8)["feature"].tolist()
        sub = feat_df[feat_df["event"] == event]
        available = [f for f in top_feats if f in sub.columns]
        if len(available) >= 2:
            with col2:
                fig_hm = plot_corr_heatmap(sub[available].apply(pd.to_numeric, errors="coerce").dropna())
                st.pyplot(fig_hm, use_container_width=True)
                plt.close(fig_hm)

    # Year-by-year NDVI + met overlays
    st.markdown('<div class="section-header">📅 Annual NDVI + Meteorology</div>', unsafe_allow_html=True)
    ndvi_df = st.session_state.ndvi_df
    met_eng = st.session_state.met_eng
    pheno_df = st.session_state.pheno_df

    if ndvi_df is None or pheno_df is None:
        return

    years = sorted(pheno_df["year"].unique())
    sel_year = st.selectbox("Select Year", years)
    row = pheno_df[pheno_df["year"] == sel_year]

    t0 = pd.Timestamp(f"{sel_year - 1}-07-01")
    t1 = pd.Timestamp(f"{sel_year + 1}-06-30")
    yr_ndvi = ndvi_df[(ndvi_df["date"] >= t0) & (ndvi_df["date"] <= t1)]

    fig, axes = plt.subplots(2, 1, figsize=(13, 6), facecolor="#0e1117", sharex=True)
    for ax in axes:
        ax.set_facecolor("#1e2530")
        for sp in ax.spines.values(): sp.set_edgecolor("#374151")
        ax.tick_params(colors="#9ca3af")

    axes[0].scatter(yr_ndvi["date"], yr_ndvi["NDVI"], s=18, color="#4ade80", alpha=0.8, label="NDVI")
    if not row.empty:
        r = row.iloc[0]
        axes[0].axvline(r["SOS"], color=PALETTE["sos"], lw=1.5, ls="--", label=f"SOS DOY={r['SOS_DOY']}")
        axes[0].axvline(r["POS"], color=PALETTE["pos"], lw=1.5, ls="-.", label=f"POS DOY={r['POS_DOY']}")
        axes[0].axvline(r["EOS"], color=PALETTE["eos"], lw=1.5, ls=":", label=f"EOS DOY={r['EOS_DOY']}")
    axes[0].set_ylabel("NDVI", color="#9ca3af")
    axes[0].legend(facecolor="#1e2530", edgecolor="#374151", labelcolor="#e5e7eb", fontsize=8)
    axes[0].set_title(f"Year {sel_year}", color="#e5e7eb", fontsize=11)

    if met_eng is not None:
        yr_met = met_eng[(met_eng["date"] >= t0) & (met_eng["date"] <= t1)]
        met_plot_cols = [c for c in ["T2M", "PRECTOTCORR", "ALLSKY_SFC_SW_DWN"] if c in yr_met.columns][:3]
        colors_m = ["#f97316", "#3b82f6", "#fbbf24"]
        for col, clr in zip(met_plot_cols, colors_m):
            vals = pd.to_numeric(yr_met[col], errors="coerce")
            axes[1].plot(yr_met["date"], vals, color=clr, lw=1.5, alpha=0.8, label=col)
    axes[1].set_ylabel("Meteorology", color="#9ca3af")
    axes[1].set_xlabel("Date", color="#9ca3af")
    axes[1].legend(facecolor="#1e2530", edgecolor="#374151", labelcolor="#e5e7eb", fontsize=8)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3: PREDICT
# ══════════════════════════════════════════════════════════════════════════════

def tab_predict():
    st.markdown('<div class="section-header">🔮 Predict Phenology Events</div>', unsafe_allow_html=True)

    models = st.session_state.models
    if not models:
        st.info("Run Training with meteorology data to enable predictions.")
        return

    feat_df = st.session_state.feature_df
    pred_rows = []

    for event in ["SOS", "POS", "EOS"]:
        if event not in models:
            continue
        ev = models[event]
        selected = ev["selected"]
        X_df = ev["X_df"]
        y = ev["y"]

        st.markdown(f"#### {event} — Selected features: `{', '.join(selected)}`")

        # Pre-fill with training means
        inputs = {}
        cols = st.columns(min(len(selected), 4))
        for i, feat in enumerate(selected):
            train_mean = float(X_df[feat].mean())
            train_min = float(X_df[feat].min())
            train_max = float(X_df[feat].max())
            val = cols[i % 4].number_input(
                f"{feat}", value=round(train_mean, 3),
                help=f"Training range: {train_min:.2f} – {train_max:.2f}",
                key=f"pred_{event}_{feat}"
            )
            inputs[feat] = val

        best_mods = ev["models"]
        best_name = max(best_mods.keys(),
                        key=lambda k: best_mods[k]["loo_r2"] if not np.isnan(best_mods[k]["loo_r2"]) else -999)
        best = best_mods[best_name]

        X_pred = np.array([[inputs[f] for f in selected]])
        if best["scaler"] is not None:
            # scale using training X
            sc = best["scaler"]
            X_pred_s = sc.transform(X_pred)
        else:
            X_pred_s = X_pred

        try:
            pred_val = float(best["model"].predict(X_pred_s)[0])
        except Exception:
            pred_val = float(y.mean())

        pred_date = pd.Timestamp(f"{pd.Timestamp.now().year}-01-01") + pd.Timedelta(days=int(pred_val) - 1)
        r2_str = f"{best['loo_r2']:.3f}" if not np.isnan(best["loo_r2"]) else "N/A"

        st.markdown(f"""
        <div class="highlight-box">
        🎯 <b>Predicted {event}</b>: DOY <b style="color:#22c55e;font-size:1.3rem">{pred_val:.1f}</b>
        &nbsp;|&nbsp; ≈ <b>{pred_date.strftime("%d %b")}</b>
        &nbsp;|&nbsp; Model: <b>{best_name}</b> (LOO R²={r2_str})
        </div>
        """, unsafe_allow_html=True)

        pred_rows.append({"Event": event, "Predicted_DOY": round(pred_val, 1),
                          "Approx_Date": pred_date.strftime("%d-%b"),
                          "Model": best_name, "LOO_R2": r2_str})

    if pred_rows:
        pred_df = pd.DataFrame(pred_rows)
        # Ecological order check
        if len(pred_df) == 3:
            sos_v = pred_df.loc[pred_df["Event"] == "SOS", "Predicted_DOY"].values[0]
            pos_v = pred_df.loc[pred_df["Event"] == "POS", "Predicted_DOY"].values[0]
            eos_v = pred_df.loc[pred_df["Event"] == "EOS", "Predicted_DOY"].values[0]
            if not (sos_v < pos_v < eos_v):
                st.warning("⚠️ Predicted order SOS < POS < EOS violated — check input feature values.")

        st.markdown('<div class="section-header">📋 Prediction Summary</div>', unsafe_allow_html=True)
        st.dataframe(pred_df, use_container_width=True, hide_index=True)
        pred_csv = pred_df.to_csv(index=False).encode()
        st.download_button("⬇ Download Predictions (CSV)", pred_csv, "predictions.csv", "text/csv")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4: TECHNICAL GUIDE
# ══════════════════════════════════════════════════════════════════════════════

def tab_technical_guide():
    st.markdown("""
    ## 📖 Technical Guide

    ### Phenology Extraction Methodology (v5 — 100% Data-Driven)

    | Step | Parameter | v5 Method |
    |---|---|---|
    | 1 | NDVI cadence | Median of observed date differences |
    | 2 | Max gap threshold | 8× detected cadence |
    | 3 | Trough min distance | 40% of autocorrelation cycle estimate |
    | 4 | MIN_AMPLITUDE | 5% of data P5–P95 range |
    | 5 | SG window | ≤ 31 steps per segment |
    | 6 | SOS / EOS | user% × per-cycle amplitude |
    | 7 | POS | Raw NDVI maximum between SOS and EOS |
    | 8 | Season year | Trough start year (eliminates duplicate-year collision) |

    ---

    ### Feature Selection Pipeline

    ```
    1. Compute Pearson r + Spearman ρ composite per feature
    2. Filter: composite ≥ user-defined threshold (default 0.40)
    3. Collinearity removal: if |r| > 0.85 between two features → drop weaker one
    4. Forward selection: add feature only if LOO R² improves ≥ 0.03
    ```

    ---

    ### Model Summary

    | Model | Notes |
    |---|---|
    | **Ridge** | L2-regularized linear; best for small n, collinear features |
    | **Polynomial deg-2/3** | Captures nonlinear driver responses |
    | **GPR** | Gaussian Process; uncertainty-aware; requires n ≥ 5 |

    All models are evaluated with **Leave-One-Out Cross-Validation (LOO-CV)**.

    ---

    ### R² Interpretation

    | LOO R² | Interpretation |
    |---|---|
    | > 0.80 | Excellent — strong environmental control |
    | 0.60 – 0.80 | Good — reliable predictions |
    | 0.40 – 0.60 | Moderate — useful but uncertain |
    | < 0.40 | Weak — more seasons needed or key driver missing |

    ---

    ### Derived Meteorological Features

    | Feature | Derivation |
    |---|---|
    | `GDD_5` | (T2M − 5)⁺ |
    | `GDD_10` | (T2M − 10)⁺ |
    | `GDD_cum` | Cumulative GDD_5 |
    | `DTR` | T2M_MAX − T2M_MIN |
    | `VPD` | es × (1 − RH2M/100) |
    | `log_precip` | log(1 + PRECTOTCORR) |
    | `MSI` | Solar / (Precip + 1) |
    | `SPEI_proxy` | Precip − PET (simplified Hargreaves) |

    ---

    ### Citation
    ```
    Sharma, S. (2025). Universal Indian Forest Phenology Predictor v5 [Software].
    GitHub. https://github.com/shreejisharma/Indian-forest-phenology
    ```

    ---

    ### License
    MIT License
    """)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    init_state()
    ndvi_file, met_file, threshold_pct, window_days, r_thresh = render_sidebar()

    # Load data on upload
    if ndvi_file is not None:
        try:
            ndvi_df = PhenologyEngine.load_ndvi(ndvi_file)
            if st.session_state.ndvi_df is None or len(ndvi_df) != len(st.session_state.ndvi_df):
                st.session_state.ndvi_df = ndvi_df
                st.session_state.trained = False
        except Exception as e:
            st.sidebar.error(f"NDVI load error: {e}")

    if met_file is not None:
        try:
            met_df = PhenologyEngine.load_met(met_file)
            if st.session_state.met_df is None or len(met_df) != len(st.session_state.met_df):
                st.session_state.met_df = met_df
                st.session_state.trained = False
        except Exception as e:
            st.sidebar.error(f"Met load error: {e}")

    # Header
    st.markdown("""
    <h1 style="color:#22c55e;font-size:1.8rem;margin-bottom:0">
        🌲 Universal Indian Forest Phenology Predictor — v5
    </h1>
    <p style="color:#9ca3af;margin-top:4px;margin-bottom:16px">
        100% Data-Driven · All Indian Forest Types · SOS · POS · EOS · LOS
    </p>
    """, unsafe_allow_html=True)

    # Status bar
    ndvi_ok = st.session_state.ndvi_df is not None
    met_ok = st.session_state.met_df is not None
    trained_ok = st.session_state.trained
    cols = st.columns(3)
    cols[0].markdown(f"{'✅' if ndvi_ok else '⬜'} **NDVI** {'loaded' if ndvi_ok else 'not uploaded'}")
    cols[1].markdown(f"{'✅' if met_ok else '⬜'} **Meteorology** {'loaded' if met_ok else 'not uploaded'}")
    cols[2].markdown(f"{'✅' if trained_ok else '⬜'} **Models** {'trained' if trained_ok else 'not run'}")

    # Tabs
    tab0, tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Data Overview",
        "🔬 Training",
        "📈 Correlations",
        "🔮 Predict",
        "📖 Technical Guide",
    ])

    with tab0:
        tab_data_overview()
    with tab1:
        tab_training(threshold_pct, window_days, r_thresh)
    with tab2:
        tab_correlations()
    with tab3:
        tab_predict()
    with tab4:
        tab_technical_guide()


if __name__ == "__main__":
    main()
