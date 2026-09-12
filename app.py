"""
QuantMetrics Pro: Market Intelligence & Quantitative Risk Engine
==================================================================
"""

from __future__ import annotations

import io
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from scipy import stats

warnings.filterwarnings("ignore")

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except Exception:  # pragma: no cover
    YFINANCE_AVAILABLE = False

EPS = 1e-8
TRADING_DAYS = 252

# ---------------------------------------------------------------------------
# PAGE CONFIG & STREAMLIT SHELL CLEANUP
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="QuantMetrics Pro | Institutional Risk Engine",
    page_icon=":material/monitoring:",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Keep the left sidebar collapse control while minimizing Streamlit's
# production/developer toolbar.
try:
    st.set_option("client.toolbarMode", "minimal")
except Exception:
    pass

hide_st_style = """
<style>

/* ---------------------------------------------------------
   KEEP STREAMLIT HEADER ALIVE
   The sidebar expand button is part of Streamlit's header UI.
   --------------------------------------------------------- */

[data-testid="stHeader"],
header.stAppHeader {
    display: flex !important;
    visibility: visible !important;
    background: transparent !important;
    box-shadow: none !important;
}


/* ---------------------------------------------------------
   HIDE ONLY UNWANTED STREAMLIT CONTROLS
   Do NOT hide the whole toolbar/header.
   --------------------------------------------------------- */

#MainMenu,
[data-testid="stAppDeployButton"],
[data-testid="stStatusWidget"],
.stDeployButton {
    display: none !important;
    visibility: hidden !important;
}


/* ---------------------------------------------------------
   HIDE GITHUB / FORK
   --------------------------------------------------------- */

header a[href*="github.com"],
header a[href*="/fork"],
header button[aria-label*="Fork"],
header [title*="Fork"],
header [data-testid*="Fork"],
header [data-testid*="GitHub"] {
    display: none !important;
    visibility: hidden !important;
}


/* ---------------------------------------------------------
   KEEP SIDEBAR COLLAPSE / EXPAND CONTROLS
   Expanded + collapsed states
   --------------------------------------------------------- */

[data-testid="stSidebarCollapseButton"],
[data-testid="stExpandSidebarButton"],
[data-testid="stBaseButton-headerNoPadding"] {
    display: flex !important;
    visibility: visible !important;
    opacity: 1 !important;
    pointer-events: auto !important;
}


/* Make sure the actual button is clickable */

[data-testid="stExpandSidebarButton"] button,
[data-testid="stBaseButton-headerNoPadding"] button {
    display: flex !important;
    visibility: visible !important;
    opacity: 1 !important;
    pointer-events: auto !important;
}


/* ---------------------------------------------------------
   STREAMLIT FOOTER
   --------------------------------------------------------- */

footer {
    display: none !important;
    visibility: hidden !important;
}

</style>
"""

st.markdown(hide_st_style, unsafe_allow_html=True)


BG = "#0e1117"
CARD = "#1e222d"
BORDER = "#2e3440"
GREEN = "#00FFA3"
RED = "#FF4B4B"
CYAN = "#00D4FF"
TEXT_MUTED = "#8b93a7"
TEXT = "#e6e9f0"

CUSTOM_CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500&display=swap');
@import url('https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined:opsz,wght,FILL,GRAD@20..48,100..700,0..1,-50..200');
html, body, [class*="css"] {{
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}}
.material-symbols-outlined {{
    font-family: 'Material Symbols Outlined';
    font-weight: normal;
    font-style: normal;
    line-height: 1;
    letter-spacing: normal;
    text-transform: none;
    display: inline-block;
    white-space: nowrap;
    word-wrap: normal;
    direction: ltr;
    -webkit-font-smoothing: antialiased;
    font-variation-settings: 'FILL' 0, 'wght' 400, 'GRAD' 0, 'opsz' 24;
    vertical-align: text-bottom;
}}
.stApp {{
    background-color: {BG};
    color: {TEXT};
}}
section[data-testid="stSidebar"] {{
    background-color: #12151c;
    border-right: 1px solid {BORDER};
}}
div[data-testid="stMetric"] {{
    background-color: {CARD};
    border: 1px solid {BORDER};
    border-radius: 10px;
    padding: 14px 16px 8px 16px;
}}
div[data-testid="stMetricLabel"] {{
    color: {TEXT_MUTED} !important;
    font-size: 0.78rem !important;
    text-transform: uppercase;
    letter-spacing: 0.04em;
}}
.block-container {{
    padding-top: 3.25rem;
}}
.qm-card {{
    background-color: {CARD};
    border: 1px solid {BORDER};
    border-radius: 12px;
    padding: 18px 20px;
    margin-bottom: 14px;
}}
.qm-title {{
    font-size: 1.7rem;
    font-weight: 700;
    letter-spacing: -0.02em;
    line-height: 1.4;
    margin-top: 0px;
    margin-bottom: 0px;
}}
.qm-subtitle {{
    color: {TEXT_MUTED};
    font-size: 0.92rem;
    margin-top: -6px;
    margin-bottom: 18px;
}}
.qm-kpi {{
    background-color: {CARD};
    border: 1px solid {BORDER};
    border-radius: 10px;
    padding: 14px 16px 12px 16px;
    height: 100%;
}}
.qm-kpi-label {{
    color: {TEXT_MUTED};
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    font-weight: 500;
}}
.qm-kpi-value {{
    font-size: 1.65rem;
    font-weight: 700;
    font-family: 'JetBrains Mono', monospace;
    margin-top: 4px;
    letter-spacing: -0.01em;
}}
.qm-badge {{
    display: inline-block;
    font-size: 0.76rem;
    padding: 3px 10px;
    border-radius: 20px;
    border: 1px solid {BORDER};
    margin-right: 8px;
    color: {TEXT_MUTED};
}}
.qm-badge b {{
    color: {TEXT};
}}
.qm-sidebar-footer {{
    color: {TEXT_MUTED};
    font-size: 0.72rem;
    border-top: 1px solid {BORDER};
    padding-top: 10px;
    margin-top: 18px;
    line-height: 1.4;
}}
hr {{
    border-color: {BORDER};
}}
.stTabs [data-baseweb="tab-list"] {{
    gap: 8px;
    border-bottom: 1px solid {BORDER};
    padding-bottom: 8px;
}}
.stTabs [data-baseweb="tab"] {{
    background-color: transparent !important;
    border-radius: 0 !important;
    padding: 10px 14px 11px !important;
    color: {TEXT_MUTED};
    border: 0 !important;
    border-bottom: 2px solid transparent !important;
    transition: color 0.18s ease, border-color 0.18s ease;
    display: flex;
    align-items: center;
    gap: 6px;
    box-sizing: border-box;
}}
.stTabs [data-baseweb="tab"] p {{
    margin: 0 !important;
    font-size: 0.92rem;
}}
.stTabs [data-baseweb="tab"]:hover {{
    background-color: transparent !important;
    color: {TEXT};
}}
.stTabs [aria-selected="true"] {{
    background-color: transparent !important;
    color: {RED} !important;
    border: 0 !important;
    border-bottom: 2px solid {RED} !important;
    border-radius: 0 !important;
    padding: 10px 14px 11px !important;
    font-weight: 600;
}}
.stTabs [data-baseweb="tab-highlight"],
.stTabs [data-baseweb="tab-border"] {{
    background: transparent !important;
    background-color: transparent !important;
    background-image: none !important;
    box-shadow: none !important;
    border: none !important;
    opacity: 0 !important;
    height: 0 !important;
    display: none !important;
}}
thead tr th {{
    background-color: {BORDER} !important;
    color: {TEXT} !important;
}}
.qm-chart-key {{
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    justify-content: center;
    gap: 12px;
    margin: 2px 0 2px 0;
    font-size: 0.72rem;
    line-height: 1.1;
    color: {TEXT_MUTED};
}}
.qm-chart-key span {{
    white-space: nowrap;
}}
.qm-chart-key-title {{
    color: {TEXT_MUTED};
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.04em;
}}

/* Responsive institutional KPI sizing: prevent Streamlit from truncating long values. */
div[data-testid="stMetric"] {{
    min-width: 0 !important;
    overflow: hidden !important;
}}
div[data-testid="stMetricLabel"] {{
    white-space: normal !important;
    overflow: visible !important;
    text-overflow: clip !important;
    line-height: 1.15 !important;
}}
div[data-testid="stMetricValue"],
div[data-testid="stMetricValue"] > div,
div[data-testid="stMetricValue"] p {{
    min-width: 0 !important;
    width: 100% !important;
    max-width: 100% !important;
    font-size: clamp(0.86rem, 1.72vw, 1.42rem) !important;
    line-height: 1.05 !important;
    letter-spacing: -0.025em !important;
    white-space: normal !important;
    overflow: visible !important;
    text-overflow: clip !important;
    word-break: normal !important;
    overflow-wrap: anywhere !important;
}}
div[data-testid="stMetricValue"] span {{
    max-width: 100% !important;
}}

/* Stable market snapshot layout: avoid seven cramped columns on desktop. */
.qm-snapshot-grid {{
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: 12px;
    margin-bottom: 12px;
}}
.qm-snapshot-grid-3 {{
    grid-template-columns: repeat(3, minmax(0, 1fr));
}}
.qm-price-hero {{
    background: {CARD};
    border: 1px solid {BORDER};
    border-radius: 12px;
    padding: 16px 20px;
    margin-bottom: 12px;
}}
.qm-price-hero-inner {{
    display: grid;
    grid-template-columns: minmax(0, 1fr) auto;
    align-items: center;
    gap: 24px;
}}
.qm-price-label {{
    color: {TEXT_MUTED}; font-size: .74rem; text-transform: uppercase;
    letter-spacing: .07em; font-weight: 600; margin-bottom: 5px;
}}
.qm-price-line {{ display:flex; align-items:baseline; gap:14px; flex-wrap:wrap; }}
.qm-price-value {{
    font-size: clamp(2rem, 4vw, 2.6rem); line-height: 1.05; font-weight: 800;
    font-family: 'JetBrains Mono', monospace; white-space: nowrap;
}}
.qm-price-change {{ font-size: 1rem; font-weight: 700; white-space: nowrap; }}
.qm-price-meta {{
    min-width: 245px; text-align: right; color: {TEXT_MUTED}; font-size: .78rem;
    line-height: 1.5; border-left: 1px solid {BORDER}; padding-left: 18px;
}}
.qm-price-meta b {{ color: {TEXT}; font-weight: 600; }}
.qm-kpi-grid {{ display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:12px; margin-bottom:12px; }}
.qm-kpi-grid-3 {{ grid-template-columns:repeat(3,minmax(0,1fr)); }}
.qm-kpi-grid .qm-kpi {{ min-width:0; min-height:92px; box-sizing:border-box; }}
.qm-kpi-value {{
    font-size: clamp(1.02rem, 1.8vw, 1.5rem); line-height:1.12;
    white-space:normal; overflow-wrap:anywhere; word-break:normal;
}}

/* Make every Plotly surface stretch to the Streamlit content column. */
div[data-testid="stPlotlyChart"], .stPlotlyChart, .js-plotly-plot, .plotly, .plot-container {{
    width: 100% !important; max-width: 100% !important;
}}
.qm-brand {{
    display:flex; align-items:center; gap:10px; padding:2px 2px 16px;
    margin-bottom:16px; border-bottom:1px solid {BORDER};
}}
.qm-brand-mark {{
    width:36px; height:36px; flex:0 0 36px; border-radius:9px;
    display:flex; align-items:center; justify-content:center;
    background:linear-gradient(145deg,#1c222d,#0e1117); border:1px solid #3a4251;
}}
.qm-brand-name {{ font-size:1.03rem; font-weight:800; letter-spacing:-.025em; line-height:1; }}
.qm-brand-sub {{ color:{TEXT_MUTED}; font-size:.67rem; letter-spacing:.07em; text-transform:uppercase; margin-top:4px; }}
@media (max-width:900px) {{
    .qm-snapshot-grid, .qm-snapshot-grid-3, .qm-kpi-grid {{ grid-template-columns:repeat(2,minmax(0,1fr)); }}
    .qm-price-hero-inner {{ grid-template-columns:1fr; }}
    .qm-price-meta {{ min-width:0; border-left:0; border-top:1px solid {BORDER}; padding-left:0; padding-top:10px; text-align:left; }}
}}
@media (max-width:560px) {{
    .qm-snapshot-grid, .qm-snapshot-grid-3, .qm-kpi-grid {{ grid-template-columns:1fr; }}
}}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

PLOTLY_TEMPLATE = dict(
    layout=dict(
        paper_bgcolor=BG,
        plot_bgcolor=CARD,
        font=dict(color=TEXT, family="Inter, sans-serif"),
        xaxis=dict(
            gridcolor=BORDER, zerolinecolor=BORDER,
            showspikes=True, spikemode="across", spikesnap="data",
            spikecolor=TEXT_MUTED, spikethickness=1,
        ),
        yaxis=dict(
            gridcolor=BORDER, zerolinecolor=BORDER,
            showspikes=True, spikemode="across", spikesnap="data",
            spikecolor=TEXT_MUTED, spikethickness=1,
        ),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            orientation="h",
            yanchor="top",
            y=-0.16,
            xanchor="left",
            x=0,
            font=dict(size=11),
        ),
        margin=dict(l=50, r=30, t=60, b=82),
        title=dict(font=dict(size=15, color=TEXT)),
    )
)


# ---------------------------------------------------------------------------
# RISK ENGINE
# ---------------------------------------------------------------------------

@dataclass
class RiskEngine:
    """
    Vectorized institutional risk analytics engine.

    Parameters
    ----------
    R : pd.Series
        Daily strategy returns (decimal, not %), indexed by date.
    Rb : Optional[pd.Series]
        Daily benchmark returns, aligned by date. Optional.
    risk_free_annual : float
        Annualized risk-free rate (decimal), e.g. 0.045 for 4.5%.
    """

    R: pd.Series
    Rb: Optional[pd.Series] = None
    risk_free_annual: float = 0.045
    metrics: dict = field(default_factory=dict, init=False)

    def __post_init__(self):
        # Normalize first, then recompute the observation count AFTER any
        # benchmark alignment. The old implementation used max(len(R), 1),
        # which allowed n=1 even when R was actually empty and later caused
        # cumulative_wealth.iloc[-1] to raise IndexError.
        self.R = pd.Series(self.R).dropna().astype(float)
        if self.Rb is not None:
            self.Rb = pd.Series(self.Rb).dropna().astype(float)
            common = self.R.index.intersection(self.Rb.index)
            self.R = self.R.loc[common]
            self.Rb = self.Rb.loc[common]
        self.n = len(self.R)
        self.rf_daily = self.risk_free_annual / TRADING_DAYS
        self._compute_all()

    # -- core wealth / drawdown series -------------------------------------
    @property
    def cumulative_wealth(self) -> pd.Series:
        return (1.0 + self.R).cumprod()

    @property
    def cumulative_returns(self) -> pd.Series:
        return self.cumulative_wealth - 1.0

    @property
    def high_water_mark(self) -> pd.Series:
        return self.cumulative_wealth.cummax()

    @property
    def drawdown_series(self) -> pd.Series:
        hwm = self.high_water_mark
        return (self.cumulative_wealth - hwm) / hwm.replace(0, EPS)

    # -- individual metric calculators --------------------------------------
    def _total_return(self) -> float:
        wealth = self.cumulative_wealth
        return float(wealth.iloc[-1] - 1.0) if not wealth.empty else 0.0

    def _cagr(self) -> float:
        total = self._total_return()
        years = self.n / TRADING_DAYS
        if years <= 0:
            return 0.0
        base = max(1.0 + total, EPS)
        return float(base ** (1.0 / years) - 1.0)

    def _ann_vol(self) -> float:
        return float(self.R.std(ddof=1) * np.sqrt(TRADING_DAYS)) if self.n > 1 else 0.0

    def _sharpe(self, cagr: float, ann_vol: float) -> float:
        return float((cagr - self.risk_free_annual) / (ann_vol + EPS))

    def _downside_deviation(self) -> float:
        excess = self.R - self.rf_daily
        downside_sq = np.minimum(0.0, excess) ** 2
        return float(np.sqrt(TRADING_DAYS) * np.sqrt(downside_sq.mean() + EPS))

    def _sortino(self, cagr: float, dd_dev: float) -> float:
        return float((cagr - self.risk_free_annual) / (dd_dev + EPS))

    def _max_drawdown(self) -> float:
        dd = self.drawdown_series
        return float(dd.min()) if len(dd) else 0.0

    def _max_dd_duration(self) -> int:
        dd = self.drawdown_series
        underwater = dd < -1e-9
        max_run, run = 0, 0
        for flag in underwater:
            run = run + 1 if flag else 0
            max_run = max(max_run, run)
        return int(max_run)

    def _calmar(self, cagr: float, mdd: float) -> float:
        return float(cagr / (abs(mdd) + EPS))

    def _recovery_factor(self, mdd: float) -> float:
        net_profit = self._total_return()
        return float(net_profit / (abs(mdd) + EPS))

    def _omega(self, threshold: float = 0.0) -> float:
        excess = self.R - threshold
        gains = excess[excess > 0].sum()
        losses = -excess[excess < 0].sum()
        return float(gains / (losses + EPS))

    def _var(self, level: float) -> float:
        if self.n == 0:
            return 0.0
        return float(np.percentile(self.R, (1 - level) * 100))

    def _cvar(self, level: float) -> float:
        var = self._var(level)
        tail = self.R[self.R <= var]
        return float(tail.mean()) if len(tail) else var

    def _skew_kurt(self):
        if self.n < 3:
            return 0.0, 0.0
        return float(stats.skew(self.R)), float(stats.kurtosis(self.R, fisher=True))

    def _win_rate(self) -> float:
        wins = (self.R > 0).sum()
        return float(wins / (self.n + EPS) * 100)

    def _profit_factor(self) -> float:
        gains = self.R[self.R > 0].sum()
        losses = -self.R[self.R < 0].sum()
        return float(gains / (losses + EPS))

    def _benchmark_metrics(self, cagr: float):
        out = dict(beta=np.nan, alpha=np.nan, information_ratio=np.nan,
                   up_capture=np.nan, down_capture=np.nan, bench_cagr=np.nan)
        if self.Rb is None or len(self.Rb) < 5:
            return out
        cov_matrix = np.cov(self.R, self.Rb)
        var_b = cov_matrix[1, 1]
        beta = cov_matrix[0, 1] / (var_b + EPS)
        bench_wealth = (1.0 + self.Rb).cumprod()
        bench_total = float(bench_wealth.iloc[-1] - 1.0)
        years = self.n / TRADING_DAYS
        bench_cagr = (max(1.0 + bench_total, EPS)) ** (1.0 / max(years, EPS)) - 1.0
        alpha = (cagr - self.risk_free_annual) - beta * (bench_cagr - self.risk_free_annual)

        active = self.R - self.Rb
        ir = float(active.mean() / (active.std(ddof=1) + EPS) * np.sqrt(TRADING_DAYS))

        up_mask = self.Rb > 0
        down_mask = self.Rb < 0
        up_capture = (self.R[up_mask].mean() / (self.Rb[up_mask].mean() + EPS)) if up_mask.sum() > 0 else np.nan
        down_capture = (self.R[down_mask].mean() / (self.Rb[down_mask].mean() + EPS)) if down_mask.sum() > 0 else np.nan

        out.update(dict(beta=float(beta), alpha=float(alpha), information_ratio=ir,
                         up_capture=float(up_capture) if up_capture == up_capture else np.nan,
                         down_capture=float(down_capture) if down_capture == down_capture else np.nan,
                         bench_cagr=float(bench_cagr)))
        return out

    def worst_drawdown_periods(self, top_n: int = 5) -> pd.DataFrame:
        dd = self.drawdown_series
        wealth = self.cumulative_wealth
        hwm = self.high_water_mark
        is_uw = dd < -1e-9

        periods = []
        start_idx = None
        for i, flag in enumerate(is_uw):
            if flag and start_idx is None:
                start_idx = i
            elif not flag and start_idx is not None:
                periods.append((start_idx, i - 1))
                start_idx = None
        if start_idx is not None:
            periods.append((start_idx, len(is_uw) - 1))

        rows = []
        idx = dd.index
        for s, e in periods:
            window = dd.iloc[s:e + 1]
            valley_pos = window.values.argmin()
            valley_idx = s + valley_pos
            peak_loss = dd.iloc[valley_idx]

            # recovery date: first point after valley where wealth >= hwm at valley's hwm level
            recovery_date = None
            target = hwm.iloc[valley_idx]
            for j in range(valley_idx + 1, len(wealth)):
                if wealth.iloc[j] >= target - EPS:
                    recovery_date = idx[j]
                    break

            duration = (idx[e] - idx[s]).days if recovery_date is None else (recovery_date - idx[s]).days
            rows.append({
                "Start Date": idx[s].date(),
                "Valley Date": idx[valley_idx].date(),
                "Recovery Date": recovery_date.date() if recovery_date is not None else "Ongoing",
                "Duration (Days)": duration,
                "Peak Loss (%)": round(peak_loss * 100, 2),
            })
        df = pd.DataFrame(rows)
        if df.empty:
            return df
        return df.sort_values("Peak Loss (%)").head(top_n).reset_index(drop=True)

    def rolling_sharpe(self, window: int = 126) -> pd.Series:
        if self.n < window:
            return pd.Series(dtype=float)
        roll_mean = self.R.rolling(window).mean() * TRADING_DAYS
        roll_std = self.R.rolling(window).std(ddof=1) * np.sqrt(TRADING_DAYS)
        return (roll_mean - self.risk_free_annual) / (roll_std + EPS)

    def monthly_returns_table(self) -> pd.DataFrame:
        wealth = self.cumulative_wealth
        if wealth.empty:
            return pd.DataFrame()
        monthly = wealth.resample("ME").last()
        monthly_ret = monthly.pct_change()
        if len(monthly_ret) > 0:
            monthly_ret.iloc[0] = monthly.iloc[0] - 1.0
        df = monthly_ret.to_frame("ret")
        df["Year"] = df.index.year
        df["Month"] = df.index.strftime("%b")
        pivot = df.pivot_table(index="Year", columns="Month", values="ret")
        month_order = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        pivot = pivot.reindex(columns=[m for m in month_order if m in pivot.columns])
        yearly = self.R.groupby(self.R.index.year).apply(lambda x: (1 + x).prod() - 1)
        pivot["Year Total"] = yearly
        return pivot * 100

    def _compute_all(self):
        cagr = self._cagr()
        ann_vol = self._ann_vol()
        sharpe = self._sharpe(cagr, ann_vol)
        dd_dev = self._downside_deviation()
        sortino = self._sortino(cagr, dd_dev)
        mdd = self._max_drawdown()
        calmar = self._calmar(cagr, mdd)
        recovery = self._recovery_factor(mdd)
        omega = self._omega()
        skew, kurt = self._skew_kurt()
        bench = self._benchmark_metrics(cagr)

        self.metrics = {
            "total_return": self._total_return(),
            "cagr": cagr,
            "ann_vol": ann_vol,
            "sharpe": sharpe,
            "downside_deviation": dd_dev,
            "sortino": sortino,
            "max_drawdown": mdd,
            "max_dd_duration": self._max_dd_duration(),
            "calmar": calmar,
            "recovery_factor": recovery,
            "omega": omega,
            "var_95": self._var(0.95),
            "var_99": self._var(0.99),
            "cvar_95": self._cvar(0.95),
            "cvar_99": self._cvar(0.99),
            "skew": skew,
            "excess_kurtosis": kurt,
            "win_rate": self._win_rate(),
            "profit_factor": self._profit_factor(),
            **bench,
        }


# ---------------------------------------------------------------------------
# DATA INGESTION HELPERS
# ---------------------------------------------------------------------------

def generate_sample_strategy(seed: int = 42, n_days: int = 1260) -> pd.DataFrame:
    """
    Offline synthetic fallback only — used solely when live Yahoo Finance data
    cannot be reached. Never used as the primary sample; see
    fetch_authentic_sample_strategy() for the real-market-data path.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=n_days)
    drift = 0.0006
    vol = 0.011
    shocks = rng.normal(drift, vol, n_days)
    ar = np.zeros(n_days)
    for i in range(1, n_days):
        ar[i] = 0.05 * ar[i - 1] + shocks[i]
    tail_events = rng.random(n_days) < 0.01
    ar[tail_events] -= rng.uniform(0.02, 0.06, tail_events.sum())
    df = pd.DataFrame({"date": dates, "strategy_returns": ar})
    return df


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_authentic_sample_strategy(ticker: str = "SPY", years: int = 6):
    """
    Builds the "sample strategy" from REAL Yahoo Finance market data — a Dual
    Moving Average Crossover (50/200) backtest on `ticker` — so the demo
    dashboard reflects an authentic, verifiable historical track record rather
    than random noise. Falls back to a clearly-labeled offline synthetic
    series only if live data genuinely cannot be reached (e.g. no internet).

    Returns
    -------
    (returns: pd.Series, is_authentic: bool)
    """
    if YFINANCE_AVAILABLE:
        try:
            end = pd.Timestamp.today().strftime("%Y-%m-%d")
            start = (pd.Timestamp.today() - pd.DateOffset(years=years)).strftime("%Y-%m-%d")
            data = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
            if data is not None and not data.empty:
                close = data["Close"]
                if isinstance(close, pd.DataFrame):
                    close = close.iloc[:, 0]
                close.index = pd.to_datetime(close.index).tz_localize(None)
                returns = simulate_sma_crossover(close, fast=50, slow=200)
                returns = returns.dropna()
                if len(returns) > 30:
                    return returns.astype(float), True
        except Exception:
            pass
    # Offline fallback — clearly labeled downstream, never silently passed off as live data.
    df = generate_sample_strategy()
    series = detect_and_parse_csv(df)
    return series, False


@st.cache_data(ttl=600, show_spinner=False)
def download_benchmark(ticker: str, start: str, end: str) -> pd.Series:
    if not YFINANCE_AVAILABLE:
        return pd.Series(dtype=float)
    try:
        data = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
        if data.empty:
            return pd.Series(dtype=float)
        close = data["Close"]
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        returns = close.pct_change().dropna()
        returns.index = pd.to_datetime(returns.index).tz_localize(None)
        return returns
    except Exception:
        return pd.Series(dtype=float)


def detect_and_parse_csv(raw_df: pd.DataFrame) -> pd.Series:
    """
    Auto-detects CSV format and returns a clean daily-returns Series.
    Supports:
      Format 1: date + return/strategy_returns column
      Format 2: date + equity/nav column (converted to pct returns)
      Format 3: entry_date/exit_date + pnl (aggregated to daily returns)
    """
    df = raw_df.copy()
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    date_col = next((c for c in df.columns if "date" in c and "exit" not in c and "entry" not in c), None)
    entry_col = next((c for c in df.columns if "entry" in c and "date" in c), None)
    exit_col = next((c for c in df.columns if "exit" in c and "date" in c), None)
    ret_col = next((c for c in df.columns if c in ("return", "returns", "strategy_returns", "daily_return", "pct_return")), None)
    nav_col = next((c for c in df.columns if c in ("nav", "equity", "portfolio_value", "balance", "close")), None)
    pnl_col = next((c for c in df.columns if "pnl" in c or "profit" in c), None)

    # Format 3: trade log with entry/exit + pnl
    if entry_col and exit_col and pnl_col:
        df[exit_col] = pd.to_datetime(df[exit_col])
        trades = df[[exit_col, pnl_col]].dropna()
        trades = trades.sort_values(exit_col)
        daily_pnl = trades.groupby(exit_col)[pnl_col].sum()
        # Treat pnl as % if magnitude suggests it, else normalize against rolling capital base
        if daily_pnl.abs().median() > 1.5:
            base_capital = max(daily_pnl.cumsum().abs().max() * 5, 1.0)
            returns = daily_pnl / base_capital
        else:
            returns = daily_pnl
        returns.index.name = "date"
        return returns.astype(float)

    if date_col is None:
        raise ValueError("Could not detect a date column in the uploaded CSV.")

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col]).sort_values(date_col).set_index(date_col)

    # Format 1: explicit returns column
    if ret_col is not None:
        series = pd.to_numeric(df[ret_col], errors="coerce").dropna()
        if series.abs().median() > 1.0:
            series = series / 100.0
        return series.astype(float)

    # Format 2: NAV / equity curve -> convert to returns
    if nav_col is not None:
        nav = pd.to_numeric(df[nav_col], errors="coerce").dropna()
        returns = nav.pct_change().dropna()
        return returns.astype(float)

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        fallback = pd.to_numeric(df[numeric_cols[0]], errors="coerce").dropna()
        if fallback.abs().median() > 1.0:
            return fallback.pct_change().dropna().astype(float)
        return fallback.astype(float)

    raise ValueError("Unrecognized CSV structure. Please provide date + return/NAV/PnL columns.")


def simulate_sma_crossover(prices: pd.Series, fast: int = 50, slow: int = 200) -> pd.Series:
    """Used internally by the authentic 'Sample Strategy' loader (long-only, kept simple)."""
    fast_ma = prices.rolling(fast).mean()
    slow_ma = prices.rolling(slow).mean()
    signal = (fast_ma > slow_ma).astype(int).shift(1).fillna(0)
    daily_returns = prices.pct_change().fillna(0)
    strategy_returns = signal * daily_returns
    return strategy_returns.dropna()


# ---------------------------------------------------------------------------
# QUANTITATIVE INDICATOR LIBRARY (vectorized, textbook definitions)
# ---------------------------------------------------------------------------

def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Wilder's Relative Strength Index."""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / (avg_loss + EPS)
    return 100 - (100 / (1 + rs))


def compute_macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    """MACD line and signal line via exponential moving averages."""
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line, signal_line


def compute_bollinger(close: pd.Series, window: int = 20, num_std: float = 2.0):
    """Bollinger Bands: SMA +/- num_std * rolling standard deviation."""
    sma = close.rolling(window).mean()
    std = close.rolling(window).std(ddof=0)
    return sma, sma + num_std * std, sma - num_std * std


def compute_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Average True Range (Wilder smoothing) — requires OHLC data."""
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()


def _flip_signal_returns(close: pd.Series, raw_signal: pd.Series, allow_short: bool) -> pd.Series:
    """Applies an instantaneous long/flat or long/short signal with a 1-day
    execution lag (decision made on t-1's close, executed on t's return)."""
    position = raw_signal.copy()
    if not allow_short:
        position = position.clip(lower=0)
    position = position.shift(1).fillna(0)
    daily_returns = close.pct_change().fillna(0)
    return (position * daily_returns).dropna()


def _stateful_strategy_returns(close: pd.Series, long_entry: pd.Series, long_exit: pd.Series,
                                short_entry: pd.Series, short_exit: pd.Series,
                                allow_short: bool) -> pd.Series:
    """
    Runs a finite-state long/flat/short position machine from already-lagged
    entry/exit boolean signals (each representing information known as of the
    previous close), so day t's position trades day t's return with zero
    look-ahead bias. Used for threshold-based mean-reversion and breakout
    strategies where positions are held across multiple days rather than
    flipping every bar.
    """
    n = len(close)
    le = long_entry.fillna(False).values
    lx = long_exit.fillna(False).values
    se = short_entry.fillna(False).values if allow_short else np.zeros(n, dtype=bool)
    sx = short_exit.fillna(False).values if allow_short else np.zeros(n, dtype=bool)

    position = np.zeros(n)
    state = 0
    for i in range(n):
        if state == 0:
            if le[i]:
                state = 1
            elif se[i]:
                state = -1
        elif state == 1:
            if lx[i]:
                state = 0
            elif se[i]:
                state = -1
        elif state == -1:
            if sx[i]:
                state = 0
            elif le[i]:
                state = 1
        position[i] = state

    position_s = pd.Series(position, index=close.index)
    daily_returns = close.pct_change().fillna(0)
    return (position_s * daily_returns).dropna()



# ---------------------------------------------------------------------------
# MARKET INTELLIGENCE DATA LAYER
# ---------------------------------------------------------------------------

CURRENCY_SYMBOLS = {
    "INR": "₹", "USD": "$", "GBP": "£", "EUR": "€", "JPY": "¥",
    "CAD": "C$", "AUD": "A$", "CHF": "CHF ", "HKD": "HK$",
    "SGD": "S$", "CNY": "¥", "KRW": "₩", "NZD": "NZ$",
    "SEK": "kr ", "NOK": "kr ", "DKK": "kr ", "ZAR": "R",
    "BRL": "R$", "MXN": "MX$", "TWD": "NT$", "THB": "฿",
}

MARKET_DEFAULTS = {
    "NS": {"country": "India", "currency": "INR", "benchmark": "^NSEI"},
    "BO": {"country": "India", "currency": "INR", "benchmark": "^BSESN"},
    "L": {"country": "United Kingdom", "currency": "GBP", "benchmark": "^FTSE"},
    "TO": {"country": "Canada", "currency": "CAD", "benchmark": "^GSPTSE"},
    "HK": {"country": "Hong Kong", "currency": "HKD", "benchmark": "^HSI"},
    "T": {"country": "Japan", "currency": "JPY", "benchmark": "^N225"},
    "AX": {"country": "Australia", "currency": "AUD", "benchmark": "^AXJO"},
    "PA": {"country": "France", "currency": "EUR", "benchmark": "^FCHI"},
    "DE": {"country": "Germany", "currency": "EUR", "benchmark": "^GDAXI"},
    "AS": {"country": "Netherlands", "currency": "EUR", "benchmark": "^AEX"},
    "MI": {"country": "Italy", "currency": "EUR", "benchmark": "FTSEMIB.MI"},
    "SI": {"country": "Singapore", "currency": "SGD", "benchmark": "^STI"},
}

# Curated benchmark universe. Yahoo Finance exposes both regional indices and
# liquid ETF proxies. The UI shows a human-readable label while calculations
# continue to use the raw Yahoo Finance symbol.
BENCHMARK_NAMES = {
    # India — broad indices
    "^NSEI": "India · NIFTY 50",
    "^NSEBANK": "India · NIFTY BANK",
    "^BSESN": "India · BSE SENSEX",
    "^NSEMDCP50": "India · NIFTY MIDCAP 50",
    "^NSEMDCP100": "India · NIFTY MIDCAP 100",
    "^NSESMCP100": "India · NIFTY SMALLCAP 100",
    "^NSEI50": "India · NIFTY 50 (alternate)",
    # India — sector / ETF reference benchmarks
    "^CNXIT": "India · NIFTY IT",
    "^CNXAUTO": "India · NIFTY AUTO",
    "^CNXPHARMA": "India · NIFTY PHARMA",
    "^CNXFMCG": "India · NIFTY FMCG",
    "^CNXMETAL": "India · NIFTY METAL",
    "^CNXPSUBANK": "India · NIFTY PSU BANK",
    "NIFTYBEES.NS": "India · NIFTY 50 ETF (NIFTYBEES)",
    "BANKBEES.NS": "India · BANK NIFTY ETF (BANKBEES)",
    "JUNIORBEES.NS": "India · NIFTY NEXT 50 ETF (JUNIORBEES)",
    "ITBEES.NS": "India · NIFTY IT ETF (ITBEES)",
    "MID150BEES.NS": "India · NIFTY MIDCAP 150 ETF (MID150BEES)",
    "SETFNIF50.NS": "India · NIFTY 50 ETF (SETFNIF50)",
    # United States — broad indices
    "^GSPC": "US · S&P 500",
    "^DJI": "US · Dow Jones Industrial Average",
    "^IXIC": "US · NASDAQ Composite",
    "^RUT": "US · Russell 2000",
    "^VIX": "US · CBOE Volatility Index",
    "^NYA": "US · NYSE Composite",
    # United States — liquid ETF proxies / sectors / fixed income
    "SPY": "US · S&P 500 ETF (SPY)",
    "QQQ": "US · NASDAQ-100 ETF (QQQ)",
    "IWM": "US · Russell 2000 ETF (IWM)",
    "DIA": "US · Dow Jones ETF (DIA)",
    "XLK": "US · Technology ETF (XLK)",
    "XLF": "US · Financials ETF (XLF)",
    "XLE": "US · Energy ETF (XLE)",
    "XLV": "US · Health Care ETF (XLV)",
    "XLI": "US · Industrials ETF (XLI)",
    "XLY": "US · Consumer Discretionary ETF (XLY)",
    "XLP": "US · Consumer Staples ETF (XLP)",
    "XLU": "US · Utilities ETF (XLU)",
    "XLC": "US · Communication Services ETF (XLC)",
    "XLRE": "US · Real Estate ETF (XLRE)",
    "AGG": "US · US Aggregate Bond ETF (AGG)",
    "BND": "US · Total Bond Market ETF (BND)",
    "IEF": "US · 7–10Y Treasury ETF (IEF)",
    "TLT": "US · 20+Y Treasury ETF (TLT)",
    # Global / developed / emerging
    "ACWI": "Global · MSCI All Country World ETF (ACWI)",
    "VT": "Global · Total World Stock ETF (VT)",
    "EFA": "International · MSCI EAFE ETF (EFA)",
    "EEM": "Emerging Markets ETF (EEM)",
    "VEA": "Developed Markets ETF (VEA)",
    "VWO": "Emerging Markets ETF (VWO)",
    # Europe
    "^FTSE": "UK · FTSE 100",
    "^FTMC": "UK · FTSE 250",
    "^FCHI": "France · CAC 40",
    "^GDAXI": "Germany · DAX",
    "^AEX": "Netherlands · AEX",
    "FTSEMIB.MI": "Italy · FTSE MIB",
    "^STOXX50E": "Eurozone · EURO STOXX 50",
    # Asia-Pacific / Canada
    "^N225": "Japan · Nikkei 225",
    "^HSI": "Hong Kong · Hang Seng",
    "^KS11": "South Korea · KOSPI",
    "^TWII": "Taiwan · TAIEX",
    "^STI": "Singapore · STI",
    "^AXJO": "Australia · S&P/ASX 200",
    "^GSPTSE": "Canada · S&P/TSX Composite",
    # Cross-asset context
    "GC=F": "Gold Futures (GC=F)",
    "CL=F": "WTI Crude Oil Futures (CL=F)",
    "SI=F": "Silver Futures (SI=F)",
    "^TNX": "US 10Y Treasury Yield (^TNX)",
}

BENCHMARK_CHOICES = list(BENCHMARK_NAMES.keys())

# Quick-select instrument universe. These are convenience labels only; all
# market data still comes from Yahoo Finance for the selected symbol.
INSTRUMENT_NAMES = {
    # India
    "RELIANCE.NS": "India · Reliance Industries",
    "TCS.NS": "India · Tata Consultancy Services",
    "HDFCBANK.NS": "India · HDFC Bank",
    "ICICIBANK.NS": "India · ICICI Bank",
    "INFY.NS": "India · Infosys",
    "SBIN.NS": "India · State Bank of India",
    "BHARTIARTL.NS": "India · Bharti Airtel",
    "ITC.NS": "India · ITC",
    "LT.NS": "India · Larsen & Toubro",
    "AXISBANK.NS": "India · Axis Bank",
    "KOTAKBANK.NS": "India · Kotak Mahindra Bank",
    "MARUTI.NS": "India · Maruti Suzuki",
    "M&M.NS": "India · Mahindra & Mahindra",
    "SUNPHARMA.NS": "India · Sun Pharmaceutical",
    "HINDUNILVR.NS": "India · Hindustan Unilever",
    "BAJFINANCE.NS": "India · Bajaj Finance",
    "ADANIENT.NS": "India · Adani Enterprises",
    "TATAMOTORS.NS": "India · Tata Motors",
    "^NSEI": "India · NIFTY 50",
    "^NSEBANK": "India · NIFTY BANK",
    "^BSESN": "India · BSE SENSEX",
    # Indian-origin international listings / ADRs — often quoted in USD
    "HDB": "Indian-origin · HDFC Bank ADR (NYSE)",
    "IBN": "Indian-origin · ICICI Bank ADR (NYSE)",
    "INFY": "Indian-origin · Infosys (NYSE)",
    "WIT": "Indian-origin · Wipro ADR (NYSE)",
    "RDY": "Indian-origin · Dr. Reddy's Laboratories ADR (NYSE)",
    # United States
    "AAPL": "US · Apple",
    "MSFT": "US · Microsoft",
    "NVDA": "US · NVIDIA",
    "AMZN": "US · Amazon",
    "GOOGL": "US · Alphabet",
    "META": "US · Meta Platforms",
    "AVGO": "US · Broadcom",
    "TSLA": "US · Tesla",
    "JPM": "US · JPMorgan Chase",
    "AMD": "US · AMD",
    "SPY": "US · S&P 500 ETF",
    "QQQ": "US · NASDAQ-100 ETF",
    # International
    "ASML": "Netherlands · ASML",
    "TSM": "Taiwan · TSMC ADR",
    "BABA": "China · Alibaba",
    "0700.HK": "Hong Kong · Tencent",
    "7203.T": "Japan · Toyota Motor",
    "9984.T": "Japan · SoftBank Group",
    "NESN.SW": "Switzerland · Nestlé",
    "HSBA.L": "UK · HSBC Holdings",
}
INSTRUMENT_CHOICES = list(INSTRUMENT_NAMES.keys())

# Indian-origin international listings can have Indian economic domicile but a
# non-INR listing currency. These values are fallbacks only; Yahoo metadata is
# authoritative when it is available.
INSTRUMENT_OVERRIDES = {
    "HDB": {"country": "India", "currency": "USD", "benchmark": "^NSEI"},
    "IBN": {"country": "India", "currency": "USD", "benchmark": "^NSEI"},
    "INFY": {"country": "India", "currency": "USD", "benchmark": "^NSEI"},
    "WIT": {"country": "India", "currency": "USD", "benchmark": "^NSEI"},
    "RDY": {"country": "India", "currency": "USD", "benchmark": "^NSEI"},
}

PERIOD_OPTIONS = ["1D", "5D", "1M", "3M", "6M", "1Y", "3Y", "5Y", "Max"]
INTERVAL_OPTIONS = ["1m", "5m", "15m", "30m", "1h", "1d", "1wk", "1mo"]
ROLLING_WINDOWS = [20, 60, 126, 252]


@dataclass
class InstrumentProfile:
    ticker: str
    name: str = "N/A"
    exchange: str = "N/A"
    country: str = "N/A"
    currency: str = "N/A"
    sector: str = "N/A"
    industry: str = "N/A"
    quote_type: str = "N/A"
    market_state: str = "N/A"
    timezone: str = "N/A"
    benchmark: str = "N/A"
    data_source: str = "Market data provider"
    raw_info: dict[str, Any] = field(default_factory=dict)
    fast_info: dict[str, Any] = field(default_factory=dict)


@dataclass
class MarketSnapshot:
    latest_price: float = np.nan
    previous_close: float = np.nan
    change: float = np.nan
    change_pct: float = np.nan
    open: float = np.nan
    day_high: float = np.nan
    day_low: float = np.nan
    week52_high: float = np.nan
    week52_low: float = np.nan
    volume: float = np.nan
    average_volume: float = np.nan
    market_cap: float = np.nan
    timestamp: Optional[pd.Timestamp] = None
    price_kind: str = "Latest available price"


def _safe_float(value: Any) -> float:
    try:
        if value is None:
            return np.nan
        x = float(value)
        return x if np.isfinite(x) else np.nan
    except Exception:
        return np.nan


def _safe_int(value: Any) -> Optional[int]:
    try:
        if value is None:
            return None
        return int(value)
    except Exception:
        return None


def _is_valid_number(value: Any) -> bool:
    x = _safe_float(value)
    return bool(np.isfinite(x))


def _first_valid(mapping: Mapping[str, Any], keys: Sequence[str], default=np.nan):
    for key in keys:
        if key in mapping:
            value = mapping.get(key)
            if _is_valid_number(value):
                return _safe_float(value)
    return default


def _sanitize_index(index: pd.Index) -> pd.DatetimeIndex:
    idx = pd.to_datetime(index, errors="coerce")
    if isinstance(idx, pd.DatetimeIndex):
        if idx.tz is not None:
            idx = idx.tz_localize(None)
        return idx
    return pd.DatetimeIndex(idx)


def format_market_timestamp(value: Optional[pd.Timestamp], timezone_name: str) -> str:
    """Format a market timestamp in the instrument's exchange timezone."""
    if value is None:
        return "N/A"
    try:
        ts = pd.Timestamp(value)
        if ts.tzinfo is None and timezone_name and timezone_name != "N/A":
            # Snapshot timestamps are already expressed in exchange-local time.
            zone = ZoneInfo(timezone_name)
            aware = ts.to_pydatetime().replace(tzinfo=zone)
        elif ts.tzinfo is not None:
            aware = ts.to_pydatetime()
        else:
            aware = ts.to_pydatetime()
        zone_label = aware.tzname() or timezone_name
        return f"{ts.strftime('%d %b %Y %H:%M')} {zone_label}"
    except Exception:
        try:
            return pd.Timestamp(value).strftime("%d %b %Y %H:%M")
        except Exception:
            return "N/A"


def _flatten_yf_columns(data: pd.DataFrame, ticker: str = "") -> pd.DataFrame:
    if data is None or data.empty:
        return pd.DataFrame()
    out = data.copy()
    if isinstance(out.columns, pd.MultiIndex):
        # yfinance can return either (field, ticker) or (ticker, field).
        fields = {"Open", "High", "Low", "Close", "Adj Close", "Volume", "Dividends", "Stock Splits"}
        if any(c in fields for c in out.columns.get_level_values(0)):
            out.columns = out.columns.get_level_values(0)
        elif any(c in fields for c in out.columns.get_level_values(-1)):
            out.columns = out.columns.get_level_values(-1)
        else:
            out.columns = [
                "_".join(str(v) for v in c if str(v) not in {"", "nan"})
                for c in out.columns.to_flat_index()
            ]
    # De-duplicate fields while retaining the first valid column.
    cleaned = {}
    for name in ("Open", "High", "Low", "Close", "Adj Close", "Volume", "Dividends", "Stock Splits"):
        matches = [c for c in out.columns if str(c).strip().lower() == name.lower()]
        if matches:
            cleaned[name] = pd.to_numeric(out[matches[0]], errors="coerce")
    if not cleaned:
        return pd.DataFrame()
    result = pd.DataFrame(cleaned, index=_sanitize_index(out.index))
    return result[~result.index.duplicated(keep="last")].sort_index()


def _period_to_yf(period_label: str) -> str:
    # Yahoo Finance's documented period set does not include 3y; request 5y
    # and trim the result locally for the 3Y presentation window.
    return {
        "1D": "1d", "5D": "5d", "1M": "1mo", "3M": "3mo",
        "6M": "6mo", "1Y": "1y", "3Y": "5y", "5Y": "5y", "Max": "max"
    }.get(period_label, "1y")


def _safe_history_args(period_label: str, interval: str) -> tuple[str, str, Optional[str]]:
    """Return a Yahoo-supported period/interval combination plus an explanatory fallback."""
    period = _period_to_yf(period_label)
    fallback_note = None
    intraday = interval not in {"1d", "1wk", "1mo"}
    if intraday:
        # Yahoo limits intraday history. Keep the request inside a conservative
        # supported window so the application does not repeatedly issue invalid calls.
        if interval == "1m":
            max_period = "5d"
        elif interval in {"5m", "15m", "30m"}:
            max_period = "1mo"
        else:
            max_period = "2y"
        allowed = {"5d": 5, "1mo": 30, "3mo": 90, "6mo": 180, "1y": 365, "3y": 1095, "5y": 1825, "max": 99999}
        requested_days = allowed.get(period, 365)
        max_days = allowed.get(max_period, 30)
        if requested_days > max_days:
            fallback_note = f"{interval} history is limited by Yahoo Finance; using {max_period}."
            period = max_period
    return period, interval, fallback_note


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_instrument_profile(ticker: str) -> InstrumentProfile:
    tkr = ticker.strip().upper()
    profile = InstrumentProfile(ticker=tkr)
    if not YFINANCE_AVAILABLE or not tkr:
        return profile
    try:
        obj = yf.Ticker(tkr)
        info = {}
        fast = {}
        try:
            raw = obj.info
            if isinstance(raw, dict):
                info = raw
        except Exception:
            pass
        try:
            raw_fast = obj.fast_info
            if raw_fast is not None:
                try:
                    fast = dict(raw_fast)
                except Exception:
                    fast = {k: raw_fast[k] for k in raw_fast.keys()} if hasattr(raw_fast, "keys") else {}
        except Exception:
            pass
        suffix = tkr.rsplit(".", 1)[-1] if "." in tkr else ""
        default = MARKET_DEFAULTS.get(suffix, {})
        override = INSTRUMENT_OVERRIDES.get(tkr, {})
        exchange = str(info.get("exchange") or info.get("fullExchangeName") or default.get("exchange") or "N/A")
        # Yahoo's actual metadata wins. The override/default is only a fallback
        # for sparse metadata responses.
        country = str(info.get("country") or override.get("country") or default.get("country") or "N/A")
        currency = str(info.get("currency") or fast.get("currency") or override.get("currency") or default.get("currency") or "N/A").upper()

        quote_type = str(info.get("quoteType") or "").upper()
        benchmark = override.get("benchmark") or default.get("benchmark")
        if country.strip().lower() == "india":
            benchmark = benchmark or "^NSEI"
        elif quote_type == "INDEX":
            benchmark = benchmark or tkr
        elif country == "United States" or exchange in {"NMS", "NYQ", "NGM", "PCX", "BTS"}:
            benchmark = benchmark or "^GSPC"
        else:
            benchmark = benchmark or "^GSPC"
        profile = InstrumentProfile(
            ticker=tkr,
            name=str(info.get("longName") or info.get("shortName") or tkr),
            exchange=exchange,
            country=country,
            currency=currency,
            sector=str(info.get("sector") or "N/A"),
            industry=str(info.get("industry") or "N/A"),
            quote_type=str(info.get("quoteType") or "N/A"),
            market_state=str(info.get("marketState") or "N/A"),
            timezone=str(info.get("exchangeTimezoneName") or fast.get("timezone") or "N/A"),
            benchmark=str(benchmark),
            raw_info=info,
            fast_info=fast,
        )
    except Exception:
        # Retain a useful suffix-based profile when the metadata endpoint is unavailable.
        suffix = tkr.rsplit(".", 1)[-1] if "." in tkr else ""
        default = MARKET_DEFAULTS.get(suffix, {})
        override = INSTRUMENT_OVERRIDES.get(tkr, {})
        profile.country = override.get("country", default.get("country", "N/A"))
        profile.currency = override.get("currency", default.get("currency", "N/A"))
        profile.benchmark = override.get("benchmark", default.get("benchmark", "^GSPC"))
    return profile


@st.cache_data(ttl=600, show_spinner=False)
def fetch_market_history(ticker: str, period_label: str = "5Y", interval: str = "1d",
                         auto_adjust: bool = True) -> tuple[pd.DataFrame, str]:
    """Fetch market history robustly across yfinance versions/endpoints.

    Primary path uses ``Ticker.history``. A ``yf.download`` fallback is used when
    history is empty, malformed, or contains no usable Close observations. This
    is important because Yahoo/yfinance can intermittently return an empty
    ``Ticker.history`` response while ``download`` succeeds for the same symbol.
    """
    if not YFINANCE_AVAILABLE:
        return pd.DataFrame(), "yfinance is unavailable."

    tkr = ticker.strip().upper()
    period, interval, fallback_note = _safe_history_args(period_label, interval)
    notes: list[str] = []
    if fallback_note:
        notes.append(fallback_note)

    def _usable(frame: pd.DataFrame) -> bool:
        return (
            frame is not None
            and not frame.empty
            and "Close" in frame.columns
            and pd.to_numeric(frame["Close"], errors="coerce").notna().sum() >= 2
        )

    # Primary endpoint: Ticker.history. Keep the call conservative for
    # compatibility with both older and newer yfinance releases.
    try:
        obj = yf.Ticker(tkr)
        try:
            data = obj.history(
                period=period,
                interval=interval,
                auto_adjust=auto_adjust,
                actions=True,
                prepost=False,
                repair=False,
            )
        except TypeError:
            # Older yfinance versions may not accept ``repair``.
            data = obj.history(
                period=period,
                interval=interval,
                auto_adjust=auto_adjust,
                actions=True,
                prepost=False,
            )
        clean = _flatten_yf_columns(data, tkr)
        if _usable(clean):
            close = pd.to_numeric(clean["Close"], errors="coerce")
            clean = clean.loc[close.notna()].copy()
            if period_label == "3Y" and not clean.empty:
                cutoff = clean.index.max() - pd.Timedelta(days=365 * 3)
                clean = clean.loc[clean.index >= cutoff]
            notes.append("Yahoo Finance Ticker.history")
            return clean, " · ".join(notes)
        notes.append("Ticker.history returned no usable Close data")
    except Exception as exc:
        notes.append(f"Ticker.history failed ({type(exc).__name__})")

    # Fallback endpoint: yf.download. This frequently succeeds when a direct
    # Ticker.history request is affected by a transient Yahoo/crumb/session issue.
    try:
        data = yf.download(
            tkr,
            period=period,
            interval=interval,
            progress=False,
            auto_adjust=auto_adjust,
            actions=False,
            threads=False,
        )
        clean = _flatten_yf_columns(data, tkr)
        if _usable(clean):
            close = pd.to_numeric(clean["Close"], errors="coerce")
            clean = clean.loc[close.notna()].copy()
            if period_label == "3Y" and not clean.empty:
                cutoff = clean.index.max() - pd.Timedelta(days=365 * 3)
                clean = clean.loc[clean.index >= cutoff]
            notes.append("Yahoo Finance yf.download fallback")
            return clean, " · ".join(notes)
        notes.append("yf.download returned no usable Close data")
    except Exception as exc:
        notes.append(f"yf.download failed ({type(exc).__name__})")

    return pd.DataFrame(), " · ".join(notes)


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_fundamentals(ticker: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if not YFINANCE_AVAILABLE:
        return out
    try:
        info = yf.Ticker(ticker.strip().upper()).info
        if isinstance(info, dict):
            out.update(info)
    except Exception:
        pass
    return out


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_actions(ticker: str) -> pd.DataFrame:
    if not YFINANCE_AVAILABLE:
        return pd.DataFrame()
    try:
        tkr = yf.Ticker(ticker.strip().upper())
        actions = tkr.actions
        if actions is None or actions.empty:
            return pd.DataFrame()
        out = actions.copy()
        out.index = _sanitize_index(out.index)
        return out
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_calendar_data(ticker: str) -> dict[str, Any]:
    if not YFINANCE_AVAILABLE:
        return {}
    try:
        cal = yf.Ticker(ticker.strip().upper()).calendar
        return dict(cal) if isinstance(cal, dict) else {}
    except Exception:
        return {}



@st.cache_data(ttl=1800, show_spinner=False)
def fetch_latest_quote(ticker: str) -> dict[str, Any]:
    """Fetch a fresh quote snapshot and latest trade timestamp.

    The quote snapshot is cached with the loaded instrument data so normal
    Streamlit widget reruns do not repeatedly hit Yahoo Finance. Yahoo's
    ``regularMarketTime`` is used first because it matches the quote timestamp
    displayed by Yahoo Finance; a 1-minute bar is only a fallback.
    """
    out: dict[str, Any] = {}
    if not YFINANCE_AVAILABLE or not ticker.strip():
        return out
    tkr = ticker.strip().upper()
    try:
        obj = yf.Ticker(tkr)
    except Exception:
        return out

    # Fast quote fields are preferable for the current price when available.
    for source_name, loader in (("fast", lambda: obj.fast_info), ("info", lambda: obj.info)):
        try:
            raw = loader()
            if raw is None:
                continue
            try:
                mapping = dict(raw)
            except Exception:
                mapping = raw if isinstance(raw, dict) else {}
            if source_name == "fast":
                out["fast"] = mapping
            else:
                out["info"] = mapping
        except Exception:
            continue

    info = out.get("info", {})
    ts = _safe_int(info.get("regularMarketTime")) if isinstance(info, dict) else None
    if ts:
        try:
            out["info_timestamp"] = pd.Timestamp(ts, unit="s", tz="UTC")
        except Exception:
            pass

    # If Yahoo already supplied regularMarketTime, do not make an additional
    # intraday request. The loaded quote metadata and its timestamp should remain
    # a single consistent snapshot rather than mixing two market-data responses.
    if out.get("info_timestamp") is not None:
        return out

    # Fallback: obtain an intraday bar only when Yahoo quote metadata has no
    # usable regularMarketTime.
    try:
        intraday = obj.history(
            period="5d",
            interval="1m",
            auto_adjust=False,
            actions=False,
            prepost=False,
            repair=False,
        )
    except TypeError:
        try:
            intraday = obj.history(
                period="5d",
                interval="1m",
                auto_adjust=False,
                actions=False,
                prepost=False,
            )
        except Exception:
            intraday = pd.DataFrame()
    except Exception:
        intraday = pd.DataFrame()

    if intraday is not None and not intraday.empty:
        clean = _flatten_yf_columns(intraday, tkr)
        if not clean.empty and "Close" in clean.columns:
            valid = pd.to_numeric(clean["Close"], errors="coerce").dropna()
            if not valid.empty:
                latest_label = valid.index[-1]
                out["latest_price"] = _safe_float(valid.iloc[-1])
                out["timestamp"] = latest_label
                if "Volume" in clean:
                    try:
                        out["latest_volume"] = _safe_float(clean.loc[latest_label, "Volume"])
                    except Exception:
                        pass

    return out


def get_market_snapshot(profile: InstrumentProfile, history: pd.DataFrame,
                        fresh_quote: Optional[dict[str, Any]] = None) -> MarketSnapshot:
    info = profile.raw_info or {}
    fast = profile.fast_info or {}
    fresh = fresh_quote or {}
    fresh_info = fresh.get("info") or {}
    fresh_fast = fresh.get("fast") or {}
    snap = MarketSnapshot()
    # Keep price and timestamp on the same Yahoo quote snapshot whenever possible.
    # The intraday-bar price is only a fallback when quote metadata is incomplete.
    snap.latest_price = _first_valid(fresh_info, ["currentPrice", "regularMarketPrice"])
    if not np.isfinite(snap.latest_price):
        snap.latest_price = _first_valid(fresh_fast, ["lastPrice", "regularMarketPrice"])
    if not np.isfinite(snap.latest_price):
        snap.latest_price = _safe_float(fresh.get("latest_price"))
    if not np.isfinite(snap.latest_price):
        snap.latest_price = _first_valid(info, ["currentPrice", "regularMarketPrice", "previousClose"])
    if not np.isfinite(snap.latest_price):
        snap.latest_price = _first_valid(fast, ["lastPrice", "regularMarketPrice", "previousClose"])
    if not np.isfinite(snap.latest_price) and not history.empty:
        snap.latest_price = _safe_float(history["Close"].dropna().iloc[-1])
    snap.previous_close = _first_valid(fresh_info, ["previousClose", "regularMarketPreviousClose"])
    if not np.isfinite(snap.previous_close):
        snap.previous_close = _first_valid(info, ["previousClose", "regularMarketPreviousClose"])
    if not np.isfinite(snap.previous_close) and len(history) >= 2:
        snap.previous_close = _safe_float(history["Close"].dropna().iloc[-2])
    snap.change = _first_valid(fresh_info, ["regularMarketChange", "priceChange"])
    if not np.isfinite(snap.change):
        snap.change = _first_valid(info, ["regularMarketChange", "priceChange"])
    if not np.isfinite(snap.change) and np.isfinite(snap.latest_price) and np.isfinite(snap.previous_close):
        snap.change = snap.latest_price - snap.previous_close
    snap.change_pct = _first_valid(fresh_info, ["regularMarketChangePercent", "priceChangePercent"])
    if not np.isfinite(snap.change_pct):
        snap.change_pct = _first_valid(info, ["regularMarketChangePercent", "priceChangePercent"])
    if np.isfinite(snap.change_pct) and abs(snap.change_pct) > 1:
        snap.change_pct /= 100.0
    if not np.isfinite(snap.change_pct) and np.isfinite(snap.latest_price) and np.isfinite(snap.previous_close):
        snap.change_pct = snap.latest_price / max(snap.previous_close, EPS) - 1
    for attr, keys in {
        "open": ["open", "regularMarketOpen"],
        "day_high": ["dayHigh", "regularMarketDayHigh"],
        "day_low": ["dayLow", "regularMarketDayLow"],
        "week52_high": ["fiftyTwoWeekHigh", "52WeekHigh"],
        "week52_low": ["fiftyTwoWeekLow", "52WeekLow"],
        "volume": ["volume", "regularMarketVolume"],
        "average_volume": ["averageVolume", "averageDailyVolume10Day", "averageVolume10days"],
        "market_cap": ["marketCap"],
    }.items():
        value = _first_valid(fresh_info, keys)
        if not np.isfinite(value):
            value = _first_valid(fresh_fast, keys)
        if not np.isfinite(value):
            value = _first_valid(info, keys)
        if not np.isfinite(value):
            value = _first_valid(fast, keys)
        setattr(snap, attr, value)
    if (not np.isfinite(snap.day_high)) and not history.empty:
        snap.day_high = _safe_float(history["High"].dropna().iloc[-1]) if "High" in history else np.nan
    if (not np.isfinite(snap.day_low)) and not history.empty:
        snap.day_low = _safe_float(history["Low"].dropna().iloc[-1]) if "Low" in history else np.nan
    if not np.isfinite(snap.volume) and "Volume" in history:
        snap.volume = _safe_float(history["Volume"].dropna().iloc[-1])
    if not np.isfinite(snap.average_volume) and "Volume" in history:
        snap.average_volume = _safe_float(history["Volume"].tail(20).mean())
    # Use Yahoo's regularMarketTime first. This is the quote timestamp shown by
    # Yahoo Finance itself. The unix timestamp is UTC and is converted once into
    # the instrument's exchange timezone. A fresh intraday bar is only a fallback.
    info_ts = fresh.get("info_timestamp")
    if info_ts is None:
        ts = _safe_int(fresh_info.get("regularMarketTime")) or _safe_int(info.get("regularMarketTime"))
        if ts:
            try:
                info_ts = pd.Timestamp(ts, unit="s", tz="UTC")
            except Exception:
                info_ts = None
    if info_ts is not None:
        try:
            ts_value = pd.Timestamp(info_ts)
            if ts_value.tzinfo is not None:
                tz_name = profile.timezone if profile.timezone != "N/A" else "UTC"
                ts_value = ts_value.tz_convert(tz_name).tz_localize(None)
            snap.timestamp = ts_value
        except Exception:
            snap.timestamp = None

    if snap.timestamp is None:
        fresh_ts = fresh.get("timestamp")
        if fresh_ts is not None:
            try:
                ts_value = pd.Timestamp(fresh_ts)
                if ts_value.tzinfo is not None:
                    tz_name = profile.timezone if profile.timezone != "N/A" else "UTC"
                    ts_value = ts_value.tz_convert(tz_name).tz_localize(None)
                snap.timestamp = ts_value
            except Exception:
                snap.timestamp = None

    if snap.timestamp is None and not history.empty:
        snap.timestamp = pd.Timestamp(history.index[-1])
    return snap


def currency_symbol(currency: str) -> str:
    return CURRENCY_SYMBOLS.get(str(currency).upper(), f"{str(currency).upper()} ")


def format_price(value: Any, currency: str, decimals: int = 2) -> str:
    x = _safe_float(value)
    if not np.isfinite(x):
        return "N/A"
    return f"{currency_symbol(currency)}{x:,.{decimals}f}"


def format_money(value: Any, currency: str, decimals: int = 2) -> str:
    x = _safe_float(value)
    if not np.isfinite(x):
        return "N/A"
    abs_x = abs(x)
    if abs_x >= 1e12:
        return f"{currency_symbol(currency)}{x / 1e12:,.{decimals}f}T"
    if abs_x >= 1e9:
        return f"{currency_symbol(currency)}{x / 1e9:,.{decimals}f}B"
    if abs_x >= 1e6:
        return f"{currency_symbol(currency)}{x / 1e6:,.{decimals}f}M"
    if abs_x >= 1e3:
        return f"{currency_symbol(currency)}{x / 1e3:,.{decimals}f}K"
    return f"{currency_symbol(currency)}{x:,.{decimals}f}"


def format_value(value: Any, kind: str = "number", currency: str = "USD") -> str:
    if kind == "price":
        return format_price(value, currency)
    if kind == "money":
        return format_money(value, currency)
    if kind == "pct":
        return pct(_safe_float(value))
    return num(_safe_float(value))


def relative_volume(snapshot: MarketSnapshot) -> float:
    if np.isfinite(snapshot.volume) and np.isfinite(snapshot.average_volume) and snapshot.average_volume > EPS:
        return snapshot.volume / snapshot.average_volume
    return np.nan


def period_return(close: pd.Series, days: int) -> float:
    s = close.dropna()
    if len(s) <= days:
        return np.nan
    return float(s.iloc[-1] / s.iloc[-(days + 1)] - 1)


def period_return_calendar(close: pd.Series, period: str) -> float:
    s = close.dropna()
    if s.empty:
        return np.nan
    if period == "1D":
        return period_return(s, 1)
    if period == "5D":
        return period_return(s, 5)
    if period == "1M":
        return period_return(s, 21)
    if period == "3M":
        return period_return(s, 63)
    if period == "6M":
        return period_return(s, 126)
    if period == "1Y":
        return period_return(s, 252)
    return np.nan


def ytd_return(close: pd.Series) -> float:
    s = close.dropna()
    if s.empty:
        return np.nan
    last = s.iloc[-1]
    year_start = s[s.index.year == s.index[-1].year]
    if year_start.empty:
        return np.nan
    return float(last / year_start.iloc[0] - 1)


def cagr_from_series(close: pd.Series, years: int) -> float:
    s = close.dropna()
    if len(s) <= years * 200:
        return np.nan
    start = s.iloc[-min(len(s), years * TRADING_DAYS + 1)]
    end = s.iloc[-1]
    if start <= 0:
        return np.nan
    return float((end / start) ** (TRADING_DAYS / max(len(s.iloc[-min(len(s), years * TRADING_DAYS + 1):]), 1)) - 1)


def drawdown_stats_from_close(close: pd.Series) -> dict[str, Any]:
    s = close.dropna().astype(float)
    if s.empty:
        return {"current": np.nan, "max": np.nan, "average": np.nan,
                "peak_date": None, "trough_date": None, "recovery_date": None,
                "recovery_days": np.nan, "duration_days": np.nan, "series": pd.Series(dtype=float)}
    hwm = s.cummax()
    dd = s / hwm - 1
    trough_pos = int(dd.values.argmin())
    trough_date = dd.index[trough_pos]
    peak_before = s.iloc[:trough_pos + 1]
    peak_date = peak_before.idxmax()
    recovery_date = None
    target = hwm.iloc[trough_pos]
    for j in range(trough_pos + 1, len(s)):
        if s.iloc[j] >= target - EPS:
            recovery_date = s.index[j]
            break
    duration_days = (trough_date - peak_date).days if peak_date is not None else np.nan
    recovery_days = (recovery_date - trough_date).days if recovery_date is not None else np.nan
    return {
        "current": float(dd.iloc[-1]),
        "max": float(dd.min()),
        "average": float(dd.mean()),
        "peak_date": peak_date,
        "trough_date": trough_date,
        "recovery_date": recovery_date,
        "recovery_days": recovery_days,
        "duration_days": duration_days,
        "series": dd,
    }


def realized_volatility(returns: pd.Series, window: int) -> float:
    r = returns.dropna()
    if len(r) < 2:
        return np.nan
    sample = r.tail(window)
    if len(sample) < max(5, min(window, 20)):
        return np.nan
    return float(sample.std(ddof=1) * np.sqrt(TRADING_DAYS))


def rolling_max_drawdown(close: pd.Series, window: int) -> pd.Series:
    roll_hwm = close.rolling(window, min_periods=max(5, window // 4)).max()
    return close / (roll_hwm + EPS) - 1


def compute_stochastic(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> tuple[pd.Series, pd.Series]:
    ll = low.rolling(period).min()
    hh = high.rolling(period).max()
    k = 100 * (close - ll) / (hh - ll + EPS)
    d = k.rolling(3).mean()
    return k, d


def compute_obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    direction = np.sign(close.diff()).fillna(0)
    return (direction * volume.fillna(0)).cumsum()


def compute_adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    up = high.diff()
    down = -low.diff()
    plus_dm = pd.Series(np.where((up > down) & (up > 0), up, 0.0), index=high.index)
    minus_dm = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=high.index)
    tr = compute_true_range(high, low, close)
    atr = tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    plus_di = 100 * plus_dm.ewm(alpha=1 / period, min_periods=period, adjust=False).mean() / (atr + EPS)
    minus_di = 100 * minus_dm.ewm(alpha=1 / period, min_periods=period, adjust=False).mean() / (atr + EPS)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + EPS)
    return dx.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()


def compute_true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev = close.shift(1)
    return pd.concat([
        high - low,
        (high - prev).abs(),
        (low - prev).abs(),
    ], axis=1).max(axis=1)


def technical_frame(history: pd.DataFrame) -> pd.DataFrame:
    df = history.copy()
    if df.empty or "Close" not in df:
        return df
    c = df["Close"].astype(float)
    h = df["High"].astype(float) if "High" in df else c
    l = df["Low"].astype(float) if "Low" in df else c
    v = df["Volume"].astype(float) if "Volume" in df else pd.Series(index=df.index, dtype=float)
    for w in (20, 50, 100, 200):
        df[f"SMA{w}"] = c.rolling(w).mean()
        df[f"EMA{w}"] = c.ewm(span=w, adjust=False).mean()
    df["RSI14"] = compute_rsi(c, 14)
    macd, macd_signal = compute_macd(c, 12, 26, 9)
    df["MACD"] = macd
    df["MACDSignal"] = macd_signal
    df["MACDHist"] = macd - macd_signal
    sma20, upper, lower = compute_bollinger(c, 20, 2.0)
    df["BBMid"], df["BBUpper"], df["BBLower"] = sma20, upper, lower
    df["BBWidth"] = (upper - lower) / (sma20.abs() + EPS)
    df["TR"] = compute_true_range(h, l, c)
    df["ATR14"] = df["TR"].ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    df["ADX14"] = compute_adx(h, l, c, 14)
    k, d = compute_stochastic(h, l, c, 14)
    df["StochK"], df["StochD"] = k, d
    df["ROC20"] = c.pct_change(20) * 100
    df["OBV"] = compute_obv(c, v)
    df["RV20"] = c.pct_change().rolling(20).std(ddof=1) * np.sqrt(TRADING_DAYS)
    df["RV60"] = c.pct_change().rolling(60).std(ddof=1) * np.sqrt(TRADING_DAYS)
    df["RV252"] = c.pct_change().rolling(252).std(ddof=1) * np.sqrt(TRADING_DAYS)
    return df


def distance_from(value: float, reference: float) -> float:
    if not np.isfinite(value) or not np.isfinite(reference) or abs(reference) < EPS:
        return np.nan
    return float(value / reference - 1)


def classify_proximity(distance_pct: float, near_threshold: float = 1.0) -> str:
    if not np.isfinite(distance_pct):
        return "N/A"
    if abs(distance_pct) <= near_threshold:
        return "Near"
    return "Above" if distance_pct > 0 else "Below"


def _crossover_events(fast: pd.Series, slow: pd.Series, close: pd.Series, name: str) -> pd.DataFrame:
    spread = fast - slow
    sign = np.sign(spread)
    # Treat exact zeros as the previous sign to avoid duplicate zero-cross events.
    sign = sign.replace(0, np.nan).ffill().fillna(0)
    up = (sign > 0) & (sign.shift(1) <= 0)
    down = (sign < 0) & (sign.shift(1) >= 0)
    positions = np.where((up | down).fillna(False))[0]
    rows = []
    for i in positions:
        state = "Bullish Cross" if bool(up.iloc[i]) else "Bearish Cross"
        price = _safe_float(close.iloc[i])
        row = {
            "Signal": name,
            "State": state,
            "Date": close.index[i].date(),
            "Days Since Cross": len(close) - 1 - i,
            "Price at Cross": price,
            "Current Price": _safe_float(close.iloc[-1]),
            "Return Since Cross": _safe_float(close.iloc[-1] / price - 1) if price > EPS else np.nan,
            "Previous Fast MA": _safe_float(fast.iloc[i - 1]) if i > 0 else np.nan,
            "Previous Slow MA": _safe_float(slow.iloc[i - 1]) if i > 0 else np.nan,
            "Fast MA": _safe_float(fast.iloc[i]),
            "Slow MA": _safe_float(slow.iloc[i]),
            "_pos": i,
        }
        for horizon in (5, 20, 60, 120):
            j = i + horizon
            row[f"{horizon}D Return"] = (
                _safe_float(close.iloc[j] / price - 1) if j < len(close) and price > EPS else np.nan
            )
        if price > EPS:
            fwd = close.iloc[i + 1:min(len(close), i + 121)]
            if not fwd.empty:
                path = fwd / price - 1
                row["Max Forward Gain"] = _safe_float(path.max())
                row["Max Forward Drawdown"] = _safe_float(path.min())
            else:
                row["Max Forward Gain"] = np.nan
                row["Max Forward Drawdown"] = np.nan
        else:
            row["Max Forward Gain"] = np.nan
            row["Max Forward Drawdown"] = np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def crossover_intelligence(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if df.empty or "Close" not in df:
        return pd.DataFrame(), pd.DataFrame()
    c = df["Close"]
    specs = [
        ("SMA 20/50", c.rolling(20).mean(), c.rolling(50).mean()),
        ("SMA 50/200", c.rolling(50).mean(), c.rolling(200).mean()),
        ("EMA 12/26", c.ewm(span=12, adjust=False).mean(), c.ewm(span=26, adjust=False).mean()),
        ("EMA 50/200", c.ewm(span=50, adjust=False).mean(), c.ewm(span=200, adjust=False).mean()),
    ]
    tables = []
    status_rows = []
    for label, fast, slow in specs:
        events = _crossover_events(fast, slow, c, label)
        if not events.empty:
            latest = events.iloc[-1].to_dict()
            status_rows.append(latest)
            tables.append(events)
        else:
            fv, sv = _safe_float(fast.iloc[-1]), _safe_float(slow.iloc[-1])
            current = "Bullish" if np.isfinite(fv) and np.isfinite(sv) and fv > sv else "Bearish" if np.isfinite(fv) and np.isfinite(sv) else "N/A"
            status_rows.append({
                "Signal": label, "State": current, "Date": "N/A",
                "Days Since Cross": np.nan, "Price at Cross": np.nan,
                "Current Price": _safe_float(c.iloc[-1]), "Return Since Cross": np.nan,
                "Previous Fast MA": np.nan, "Previous Slow MA": np.nan,
                "Fast MA": fv, "Slow MA": sv,
            })
    return pd.DataFrame(status_rows), pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()


def crossover_summary_stats(events: pd.DataFrame) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame()
    rows = []
    for signal, grp in events.groupby("Signal"):
        for state, sub in grp.groupby("State"):
            vals = pd.to_numeric(sub["60D Return"], errors="coerce").dropna()
            rows.append({
                "Signal": signal, "Cross": state,
                "Count": int(len(sub)),
                "Avg 60D Return": float(vals.mean()) if len(vals) else np.nan,
                "Median 60D Return": float(vals.median()) if len(vals) else np.nan,
                "Win Rate": float((vals > 0).mean()) if len(vals) else np.nan,
                "Worst 60D": float(vals.min()) if len(vals) else np.nan,
                "Best 60D": float(vals.max()) if len(vals) else np.nan,
            })
    return pd.DataFrame(rows)


def strategy_positions(strategy_name: str, close: pd.Series, high: Optional[pd.Series] = None,
                       low: Optional[pd.Series] = None, allow_short: bool = False,
                       params: Optional[dict[str, Any]] = None) -> pd.Series:
    """Replicate the strategy state used by the existing catalog for exposure/trade diagnostics."""
    params = params or {}
    c = close.astype(float)
    h = high if high is not None else c
    l = low if low is not None else c
    p = pd.Series(0.0, index=c.index)

    if strategy_name == "Buy & Hold":
        return pd.Series(1.0, index=c.index)

    if strategy_name == "SMA Crossover":
        fast = c.rolling(int(params.get("fast", 50))).mean()
        slow = c.rolling(int(params.get("slow", 200))).mean()
        raw = np.where(fast > slow, 1.0, -1.0)
        p = pd.Series(raw, index=c.index)
        p[fast.isna() | slow.isna()] = 0
    elif strategy_name == "EMA Crossover":
        fast = c.ewm(span=int(params.get("fast", 12)), adjust=False).mean()
        slow = c.ewm(span=int(params.get("slow", 26)), adjust=False).mean()
        p = pd.Series(np.where(fast > slow, 1.0, -1.0), index=c.index)
    elif strategy_name == "Triple SMA Trend Filter":
        f = c.rolling(int(params.get("fast", 20))).mean()
        m = c.rolling(int(params.get("mid", 50))).mean()
        s = c.rolling(int(params.get("slow", 200))).mean()
        p = pd.Series(np.select([(c > f) & (f > m) & (m > s), (c < f) & (f < m) & (m < s)],
                                [1.0, -1.0], default=0.0), index=c.index)
    elif strategy_name == "MACD Signal Crossover":
        macd, sig = compute_macd(c, int(params.get("fast", 12)), int(params.get("slow", 26)), int(params.get("signal", 9)))
        p = pd.Series(np.where(macd > sig, 1.0, -1.0), index=c.index)
    elif strategy_name == "Time-Series Momentum":
        mom = c.pct_change(int(params.get("lookback", 90)))
        p = pd.Series(np.where(mom > 0, 1.0, -1.0), index=c.index)
        p[mom.isna()] = 0
    elif strategy_name == "Rate of Change (ROC)":
        roc = c.pct_change(int(params.get("lookback", 20))) * 100
        p = pd.Series(np.select([roc > float(params.get("threshold_pct", 2.0)),
                                 roc < -float(params.get("threshold_pct", 2.0))],
                                [1.0, -1.0], default=0.0), index=c.index)
    elif strategy_name in {"RSI Mean Reversion", "Bollinger Band Reversion", "Z-Score Mean Reversion",
                           "Donchian Channel Breakout", "ATR Volatility Breakout"}:
        if strategy_name == "RSI Mean Reversion":
            rsi = compute_rsi(c, int(params.get("period", 14))).shift(1)
            le, lx = rsi < float(params.get("oversold", 30)), rsi > 50
            se, sx = rsi > float(params.get("overbought", 70)), rsi < 50
        elif strategy_name == "Bollinger Band Reversion":
            sma, upper, lower = compute_bollinger(c, int(params.get("window", 20)), float(params.get("num_std", 2.0)))
            cs, ssu, ssl = c.shift(1), upper.shift(1), lower.shift(1)
            le, lx = cs < ssl, cs > sma_s
            se, sx = cs > ssu, cs < sma_s
        elif strategy_name == "Z-Score Mean Reversion":
            window = int(params.get("window", 30))
            entry_z = float(params.get("entry_z", 2.0))
            exit_z = float(params.get("exit_z", 0.5))
            z = ((c - c.rolling(window).mean()) / (c.rolling(window).std(ddof=0) + EPS)).shift(1)
            le, lx = z < -entry_z, z > -exit_z
            se, sx = z > entry_z, z < exit_z
        elif strategy_name == "Donchian Channel Breakout":
            ew, xw = int(params.get("entry_window", 20)), int(params.get("exit_window", 10))
            href, lref, clag = h.shift(2), l.shift(2), c.shift(1)
            ue, lew = href.rolling(ew).max(), lref.rolling(ew).min()
            ux, lxw = href.rolling(xw).max(), lref.rolling(xw).min()
            le, lx = clag > ue, clag < lxw
            se, sx = clag < lew, clag > ux
        else:
            period, k = int(params.get("period", 14)), float(params.get("k", 1.5))
            atr = compute_atr(h, l, c, period).shift(1)
            clag, pclag = c.shift(1), c.shift(2)
            trail = clag.rolling(10).mean()
            le, se = clag > pclag + k * atr, clag < pclag - k * atr
            lx, sx = clag < trail, clag > trail
        le = le.fillna(False).values
        lx = lx.fillna(False).values
        se = se.fillna(False).values if allow_short else np.zeros(len(c), dtype=bool)
        sx = sx.fillna(False).values if allow_short else np.zeros(len(c), dtype=bool)
        state = 0
        arr = np.zeros(len(c))
        for i in range(len(c)):
            if state == 0:
                if le[i]:
                    state = 1
                elif se[i]:
                    state = -1
            elif state == 1:
                if lx[i]:
                    state = 0
                elif se[i]:
                    state = -1
            else:
                if sx[i]:
                    state = 0
                elif le[i]:
                    state = 1
            arr[i] = state
        p = pd.Series(arr, index=c.index)
    # Signal-based crossover/momentum strategies execute with a 1-bar lag in the
    # existing engine. Stateful threshold/breakout strategies already shift their
    # decision inputs by one bar before the state machine runs, so shifting again
    # would introduce an unintended two-bar lag.
    stateful_names = {
        "RSI Mean Reversion", "Bollinger Band Reversion", "Z-Score Mean Reversion",
        "Donchian Channel Breakout", "ATR Volatility Breakout",
    }
    if strategy_name not in {"Buy & Hold"} and strategy_name not in stateful_names:
        p = p.shift(1).fillna(0)
    if not allow_short:
        p = p.clip(lower=0)
    return p.astype(float)


def strategy_trade_stats(strategy_name: str, strategy_returns: pd.Series, close: pd.Series,
                         high: pd.Series, low: pd.Series, allow_short: bool,
                         params: dict[str, Any], initial_capital: float = 100000.0) -> dict[str, Any]:
    pos = strategy_positions(strategy_name, close, high, low, allow_short, params).reindex(strategy_returns.index).fillna(0)
    changed = pos.diff().fillna(pos).abs()
    nonzero = pos != 0
    entries = int(((nonzero) & (~nonzero.shift(1, fill_value=False))).sum())
    flips = int(((pos != 0) & (pos.shift(1) != 0) & (pos != pos.shift(1))).sum())
    trade_id = ((nonzero) & (~nonzero.shift(1, fill_value=False))).cumsum()
    trade_rets = []
    for tid in sorted(trade_id[trade_id > 0].unique()):
        block = strategy_returns[trade_id == tid]
        if len(block):
            trade_rets.append(float((1 + block).prod() - 1))
    td = pd.Series(trade_rets)
    return {
        "position": pos,
        "entries": entries + flips,
        "number_of_trades": len(td),
        "average_trade": float(td.mean()) if len(td) else np.nan,
        "median_trade": float(td.median()) if len(td) else np.nan,
        "best_trade": float(td.max()) if len(td) else np.nan,
        "worst_trade": float(td.min()) if len(td) else np.nan,
        "avg_holding_days": float(nonzero.astype(int).groupby((~nonzero).cumsum()).sum().replace(0, np.nan).mean()) if nonzero.any() else np.nan,
        "exposure_pct": float(nonzero.mean() * 100),
        "long_exposure_pct": float((pos > 0).mean() * 100),
        "short_exposure_pct": float((pos < 0).mean() * 100),
        "turnover": float(changed.sum() / max(len(pos), 1)),
    }


def apply_transaction_costs(returns: pd.Series, positions: pd.Series, cost_bps: float) -> pd.Series:
    turnover = positions.diff().abs().fillna(positions.abs())
    return returns - turnover * (max(cost_bps, 0.0) / 10000.0)


def detect_regime(tech: pd.DataFrame, benchmark_returns: Optional[pd.Series] = None) -> dict[str, Any]:
    if tech.empty or len(tech) < 30:
        return {"regime": "Data unavailable", "conditions": [], "trend": "N/A", "momentum": "N/A",
                "volatility": "N/A", "relative_strength": "N/A", "risk": "N/A"}
    last = tech.iloc[-1]
    price = _safe_float(last["Close"])
    s50, s200 = _safe_float(last.get("SMA50")), _safe_float(last.get("SMA200"))
    rsi = _safe_float(last.get("RSI14"))
    adx = _safe_float(last.get("ADX14"))
    rv20 = _safe_float(last.get("RV20"))
    mom60 = _safe_float(tech["Close"].pct_change(60).iloc[-1])
    d50 = distance_from(price, s50)
    d200 = distance_from(price, s200)
    ma_gap = distance_from(s50, s200)
    conditions = []
    score = 0
    if np.isfinite(d200):
        if d200 > 0.03: score += 2; conditions.append(f"Price {d200 * 100:+.1f}% vs SMA 200")
        elif d200 < -0.03: score -= 2; conditions.append(f"Price {d200 * 100:+.1f}% vs SMA 200")
        else: conditions.append(f"Price {d200 * 100:+.1f}% vs SMA 200")
    if np.isfinite(ma_gap):
        if ma_gap > 0.02: score += 1; conditions.append(f"SMA 50 {ma_gap * 100:+.1f}% vs SMA 200")
        elif ma_gap < -0.02: score -= 1; conditions.append(f"SMA 50 {ma_gap * 100:+.1f}% vs SMA 200")
        else: conditions.append(f"SMA 50 {ma_gap * 100:+.1f}% vs SMA 200")
    if np.isfinite(rsi):
        conditions.append(f"RSI {rsi:.1f}")
    if np.isfinite(adx):
        conditions.append(f"ADX {adx:.1f}")
    if np.isfinite(mom60):
        conditions.append(f"60D momentum {mom60 * 100:+.1f}%")
        if mom60 > 0.08: score += 1
        elif mom60 < -0.08: score -= 1

    if score >= 3:
        regime = "Strong Bull Trend"
        trend = "Bullish"
    elif score >= 1:
        regime = "Weak Bull Trend"
        trend = "Bullish"
    elif score <= -3:
        regime = "Strong Bear Trend"
        trend = "Bearish"
    elif score <= -1:
        regime = "Weak Bear Trend"
        trend = "Bearish"
    else:
        regime = "Sideways / Range"
        trend = "Neutral"

    if np.isfinite(rv20):
        vol_bucket = "Low" if rv20 < 0.15 else "Normal" if rv20 < 0.30 else "Elevated" if rv20 < 0.50 else "Extreme"
        if vol_bucket in {"Elevated", "Extreme"}:
            regime = f"{regime} + {vol_bucket} Volatility"
    else:
        vol_bucket = "N/A"
    if np.isfinite(rsi):
        momentum = "Strong" if 55 <= rsi <= 70 or rsi >= 70 else "Moderate" if 45 <= rsi < 55 else "Weak"
    else:
        momentum = "N/A"
    if np.isfinite(adx):
        if adx < 18 and "Range" not in regime:
            conditions.append(f"ADX {adx:.1f} suggests limited trend strength")
    risk = "Low" if vol_bucket == "Low" else "Moderate" if vol_bucket == "Normal" else "High" if vol_bucket == "Elevated" else "Very High"
    return {
        "regime": regime, "conditions": conditions, "trend": trend,
        "momentum": momentum, "volatility": vol_bucket,
        "relative_strength": "N/A", "risk": risk,
        "price_vs_sma50": d50, "price_vs_sma200": d200,
        "sma50_vs_sma200": ma_gap, "rsi": rsi, "adx": adx, "rv20": rv20, "momentum60": mom60,
    }


def technical_scorecard(tech: pd.DataFrame, benchmark_asset: Optional[pd.Series] = None) -> dict[str, Any]:
    if tech.empty:
        return {"Trend": (0, 5, []), "Momentum": (0, 5, []), "Volatility": (0, 5, []),
                "Relative Strength": (0, 5, []), "Overall": 0}
    last = tech.iloc[-1]
    price = _safe_float(last["Close"])
    details: dict[str, list[str]] = {k: [] for k in ["Trend", "Momentum", "Volatility", "Relative Strength"]}
    trend = 0
    for key, label, points in [
        ("SMA50", "Price vs SMA 50", lambda d: 1 if d > 0 else 0),
        ("SMA200", "Price vs SMA 200", lambda d: 1 if d > 0 else 0),
    ]:
        d = distance_from(price, _safe_float(last.get(key)))
        if np.isfinite(d):
            trend += points(d)
            details["Trend"].append(f"{label}: {d * 100:+.2f}%")
    gap = distance_from(_safe_float(last.get("SMA50")), _safe_float(last.get("SMA200")))
    if np.isfinite(gap):
        trend += 1 if gap > 0 else 0
        details["Trend"].append(f"SMA 50 vs SMA 200: {gap * 100:+.2f}%")
    trend += 1 if _safe_float(last.get("ADX14")) >= 25 else 0
    trend = min(trend, 5)

    rsi = _safe_float(last.get("RSI14"))
    roc = _safe_float(last.get("ROC20"))
    hist = _safe_float(last.get("MACDHist"))
    momentum = int(np.isfinite(rsi) and 45 <= rsi <= 70) + int(np.isfinite(roc) and roc > 0) + int(np.isfinite(hist) and hist > 0)
    if np.isfinite(rsi) and 40 < rsi < 80:
        details["Momentum"].append(f"RSI: {rsi:.1f}")
    if np.isfinite(roc):
        details["Momentum"].append(f"ROC 20D: {roc:+.2f}%")
    if np.isfinite(hist):
        details["Momentum"].append(f"MACD histogram: {hist:+.4f}")
    momentum = min(momentum, 5)

    rv20 = _safe_float(last.get("RV20"))
    bbw = _safe_float(last.get("BBWidth"))
    atr_pct = _safe_float(last.get("ATR14")) / price * 100 if price > EPS and np.isfinite(_safe_float(last.get("ATR14"))) else np.nan
    vol_score = 5
    if np.isfinite(rv20):
        vol_score -= 1 if rv20 > 0.50 else 0
        vol_score -= 1 if rv20 > 0.30 else 0
    if np.isfinite(bbw):
        vol_score -= 1 if bbw > 0.15 else 0
    if np.isfinite(atr_pct):
        vol_score -= 1 if atr_pct > 4 else 0
        details["Volatility"].append(f"ATR as % price: {atr_pct:.2f}%")
    if np.isfinite(rv20):
        details["Volatility"].append(f"20D realized vol: {rv20 * 100:.2f}%")
    if np.isfinite(bbw):
        details["Volatility"].append(f"BB width: {bbw * 100:.2f}%")
    vol_score = int(np.clip(vol_score, 0, 5))

    rs_score = 0
    if benchmark_asset is not None and not benchmark_asset.empty:
        aligned = pd.concat([tech["Close"].rename("asset"), benchmark_asset.rename("bench")], axis=1).dropna()
        if len(aligned) > 21:
            asset_r = aligned["asset"].iloc[-1] / aligned["asset"].iloc[-22] - 1
            bench_r = aligned["bench"].iloc[-1] / aligned["bench"].iloc[-22] - 1
            excess = asset_r - bench_r
            rs_score = 5 if excess > 0.05 else 4 if excess > 0.02 else 3 if excess > -0.02 else 2 if excess > -0.05 else 1
            details["Relative Strength"].append(f"1M excess return: {excess * 100:+.2f}%")
    overall = trend + momentum + vol_score + rs_score
    return {
        "Trend": (trend, 5, details["Trend"]),
        "Momentum": (momentum, 5, details["Momentum"]),
        "Volatility": (vol_score, 5, details["Volatility"]),
        "Relative Strength": (rs_score, 5, details["Relative Strength"]),
        "Overall": overall,
    }


def benchmark_analytics(asset_returns: pd.Series, benchmark_returns: Optional[pd.Series],
                        asset_close: Optional[pd.Series] = None, benchmark_close: Optional[pd.Series] = None,
                        risk_free_annual: float = 0.045) -> dict[str, Any]:
    out = {
        "asset_return": np.nan, "benchmark_return": np.nan, "excess_return": np.nan,
        "beta": np.nan, "alpha": np.nan, "correlation": np.nan, "information_ratio": np.nan,
        "up_capture": np.nan, "down_capture": np.nan, "relative_strength": np.nan,
        "rs_20d": np.nan, "rs_60d": np.nan, "rs_1y": np.nan,
    }
    if asset_returns is None or asset_returns.empty or benchmark_returns is None or benchmark_returns.empty:
        return out
    aligned = pd.concat([asset_returns.rename("asset"), benchmark_returns.rename("bench")], axis=1).dropna()
    if len(aligned) < 5:
        return out
    ar, br = aligned["asset"], aligned["bench"]
    out["asset_return"] = float((1 + ar).prod() - 1)
    out["benchmark_return"] = float((1 + br).prod() - 1)
    out["excess_return"] = out["asset_return"] - out["benchmark_return"]
    cov = np.cov(ar, br)
    out["beta"] = float(cov[0, 1] / (cov[1, 1] + EPS))
    out["correlation"] = float(ar.corr(br))
    active = ar - br
    out["information_ratio"] = float(active.mean() / (active.std(ddof=1) + EPS) * np.sqrt(TRADING_DAYS))
    bench_cagr = (1 + out["benchmark_return"]) ** (TRADING_DAYS / max(len(br), 1)) - 1
    asset_cagr = (1 + out["asset_return"]) ** (TRADING_DAYS / max(len(ar), 1)) - 1
    out["alpha"] = float((asset_cagr - risk_free_annual) - out["beta"] * (bench_cagr - risk_free_annual))
    up = br > 0
    down = br < 0
    if up.any():
        out["up_capture"] = float(ar[up].mean() / (br[up].mean() + EPS))
    if down.any():
        out["down_capture"] = float(ar[down].mean() / (br[down].mean() + EPS))
    if asset_close is not None and benchmark_close is not None:
        rs = pd.concat([asset_close.rename("asset"), benchmark_close.rename("bench")], axis=1).dropna()
        if not rs.empty:
            ratio = rs["asset"] / (rs["bench"].abs() + EPS)
            out["relative_strength"] = float(ratio.iloc[-1])
            for d, key in [(20, "rs_20d"), (60, "rs_60d"), (252, "rs_1y")]:
                if len(ratio) > d:
                    out[key] = float(ratio.iloc[-1] / ratio.iloc[-(d + 1)] - 1)
    return out


def relative_strength_series(asset_close: pd.Series, bench_close: pd.Series) -> pd.Series:
    aligned = pd.concat([asset_close.rename("asset"), bench_close.rename("bench")], axis=1).dropna()
    if aligned.empty:
        return pd.Series(dtype=float)
    return aligned["asset"] / (aligned["bench"].abs() + EPS)


def support_resistance_levels(close: pd.Series, atr: float = np.nan) -> dict[str, list[float]]:
    s = close.dropna()
    if len(s) < 20:
        return {"support": [], "resistance": []}
    recent = s.tail(min(len(s), 126))
    # Local extrema are analytical reference levels, not guaranteed support/resistance.
    lows = recent[(recent.shift(1) > recent) & (recent.shift(-1) >= recent)]
    highs = recent[(recent.shift(1) < recent) & (recent.shift(-1) <= recent)]
    supports = sorted(set(round(float(x), 4) for x in lows.tail(10).values), reverse=True)[:5]
    resistances = sorted(set(round(float(x), 4) for x in highs.tail(10).values))[:5]
    if np.isfinite(atr) and atr > 0 and not supports:
        supports = [float(s.iloc[-1] - atr), float(s.iloc[-1] - 2 * atr)]
    if np.isfinite(atr) and atr > 0 and not resistances:
        resistances = [float(s.iloc[-1] + atr), float(s.iloc[-1] + 2 * atr)]
    return {"support": supports, "resistance": resistances}


def backtest_performance_summary(strategy_returns: pd.Series, benchmark_returns: Optional[pd.Series] = None) -> pd.DataFrame:
    rows = []
    r = strategy_returns.dropna()
    if r.empty:
        return pd.DataFrame()
    wealth = (1 + r).cumprod()
    dd = wealth / wealth.cummax() - 1
    years = len(r) / TRADING_DAYS
    annual = float(wealth.iloc[-1] ** (1 / max(years, EPS)) - 1)
    ann_vol = r.std(ddof=1) * np.sqrt(TRADING_DAYS) if len(r) > 1 else np.nan
    sharpe = (annual - 0.045) / (ann_vol + EPS) if np.isfinite(ann_vol) else np.nan
    downside = np.minimum(r - 0.045 / TRADING_DAYS, 0)
    sortino = (annual - 0.045) / (np.sqrt((downside ** 2).mean() * TRADING_DAYS) + EPS)
    calmar = annual / (abs(dd.min()) + EPS)
    rows.extend([
        ("Strategy Return", wealth.iloc[-1] - 1),
        ("CAGR", annual),
        ("Annualized Volatility", ann_vol),
        ("Sharpe", sharpe),
        ("Sortino", sortino),
        ("Calmar", calmar),
        ("Max Drawdown", dd.min()),
        ("Win Rate", (r > 0).mean()),
        ("Profit Factor", r[r > 0].sum() / (abs(r[r < 0].sum()) + EPS)),
        ("Best Day", r.max()),
        ("Worst Day", r.min()),
    ])
    if benchmark_returns is not None and not benchmark_returns.empty:
        b = pd.concat([r.rename("strategy"), benchmark_returns.rename("benchmark")], axis=1).dropna()["benchmark"]
        if len(b):
            bh = (1 + b).prod() - 1
            rows.append(("Buy & Hold Benchmark", bh))
    return pd.DataFrame(rows, columns=["Metric", "Value"])


def cross_asset_context(ticker: str, is_india: bool = False) -> list[str]:
    if is_india:
        return ["^NSEI", "^NSEBANK", "GC=F", "CL=F", "^TNX"]
    return ["^GSPC", "^IXIC", "GC=F", "CL=F", "^TNX"]


@st.cache_data(ttl=600, show_spinner=False)
def download_cross_asset_prices(tickers: tuple[str, ...], period: str = "1y") -> pd.DataFrame:
    if not YFINANCE_AVAILABLE or not tickers:
        return pd.DataFrame()
    try:
        data = yf.download(
            list(tickers), period=period, interval="1d", auto_adjust=True,
            progress=False, group_by="column", multi_level_index=True,
        )
        if data is None or data.empty:
            return pd.DataFrame()
        idx = _sanitize_index(data.index)
        result = pd.DataFrame(index=idx)
        if isinstance(data.columns, pd.MultiIndex):
            # Current yfinance default is commonly (PriceField, Ticker).
            for symbol in tickers:
                candidates = [
                    ("Close", symbol), (symbol, "Close"),
                ]
                series = None
                for candidate in candidates:
                    if candidate in data.columns:
                        series = data[candidate]
                        break
                if series is not None:
                    result[symbol] = pd.to_numeric(series, errors="coerce").values
        else:
            # Single-ticker fall-back.
            if "Close" in data.columns:
                result[tickers[0]] = pd.to_numeric(data["Close"], errors="coerce").values
        return result.dropna(how="all")
    except Exception:
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# STRATEGY LIBRARY — each returns daily strategy returns from OHLC data
# ---------------------------------------------------------------------------

def strat_buy_and_hold(close, high=None, low=None, allow_short=False, **kwargs) -> pd.Series:
    return close.pct_change().fillna(0)


def strat_sma_crossover(close, high=None, low=None, allow_short=False, fast=50, slow=200, **kwargs) -> pd.Series:
    fast_ma, slow_ma = close.rolling(int(fast)).mean(), close.rolling(int(slow)).mean()
    raw = pd.Series(np.where(fast_ma > slow_ma, 1.0, -1.0), index=close.index)
    raw[fast_ma.isna() | slow_ma.isna()] = 0.0
    return _flip_signal_returns(close, raw, allow_short)


def strat_ema_crossover(close, high=None, low=None, allow_short=False, fast=12, slow=26, **kwargs) -> pd.Series:
    fast_ma = close.ewm(span=int(fast), adjust=False).mean()
    slow_ma = close.ewm(span=int(slow), adjust=False).mean()
    raw = pd.Series(np.where(fast_ma > slow_ma, 1.0, -1.0), index=close.index)
    return _flip_signal_returns(close, raw, allow_short)


def strat_triple_sma_trend(close, high=None, low=None, allow_short=False, fast=20, mid=50, slow=200, **kwargs) -> pd.Series:
    f, m, s = close.rolling(int(fast)).mean(), close.rolling(int(mid)).mean(), close.rolling(int(slow)).mean()
    uptrend = (close > f) & (f > m) & (m > s)
    downtrend = (close < f) & (f < m) & (m < s)
    raw = pd.Series(np.select([uptrend, downtrend], [1.0, -1.0], default=0.0), index=close.index)
    return _flip_signal_returns(close, raw, allow_short)


def strat_macd_crossover(close, high=None, low=None, allow_short=False, fast=12, slow=26, signal=9, **kwargs) -> pd.Series:
    macd_line, signal_line = compute_macd(close, int(fast), int(slow), int(signal))
    raw = pd.Series(np.where(macd_line > signal_line, 1.0, -1.0), index=close.index)
    return _flip_signal_returns(close, raw, allow_short)


def strat_time_series_momentum(close, high=None, low=None, allow_short=False, lookback=90, **kwargs) -> pd.Series:
    momentum = close.pct_change(int(lookback))
    raw = pd.Series(np.where(momentum > 0, 1.0, -1.0), index=close.index)
    raw[momentum.isna()] = 0.0
    return _flip_signal_returns(close, raw, allow_short)


def strat_roc_momentum(close, high=None, low=None, allow_short=False, lookback=20, threshold_pct=2.0, **kwargs) -> pd.Series:
    roc = close.pct_change(int(lookback)) * 100
    raw = pd.Series(np.select([roc > threshold_pct, roc < -threshold_pct], [1.0, -1.0], default=0.0), index=close.index)
    return _flip_signal_returns(close, raw, allow_short)


def strat_rsi_reversion(close, high=None, low=None, allow_short=False, period=14, oversold=30, overbought=70, **kwargs) -> pd.Series:
    rsi = compute_rsi(close, int(period)).shift(1)
    exit_level = 50.0
    long_entry, long_exit = rsi < oversold, rsi > exit_level
    short_entry, short_exit = rsi > overbought, rsi < exit_level
    return _stateful_strategy_returns(close, long_entry, long_exit, short_entry, short_exit, allow_short)


def strat_bollinger_reversion(close, high=None, low=None, allow_short=False, window=20, num_std=2.0, **kwargs) -> pd.Series:
    sma, upper, lower = compute_bollinger(close, int(window), float(num_std))
    sma_s, upper_s, lower_s, close_s = sma.shift(1), upper.shift(1), lower.shift(1), close.shift(1)
    long_entry, long_exit = close_s < lower_s, close_s > sma_s
    short_entry, short_exit = close_s > upper_s, close_s < sma_s
    return _stateful_strategy_returns(close, long_entry, long_exit, short_entry, short_exit, allow_short)


def strat_zscore_reversion(close, high=None, low=None, allow_short=False, window=30, entry_z=2.0, exit_z=0.5, **kwargs) -> pd.Series:
    roll_mean = close.rolling(int(window)).mean()
    roll_std = close.rolling(int(window)).std(ddof=0)
    z = ((close - roll_mean) / (roll_std + EPS)).shift(1)
    long_entry, long_exit = z < -entry_z, z > -exit_z
    short_entry, short_exit = z > entry_z, z < exit_z
    return _stateful_strategy_returns(close, long_entry, long_exit, short_entry, short_exit, allow_short)


def strat_donchian_breakout(close, high=None, low=None, allow_short=False, entry_window=20, exit_window=10, **kwargs) -> pd.Series:
    # Reference channels use data through t-2 so that close[t-1] (the most
    # recent fully-known close) is compared against a channel that does NOT
    # already contain that same bar's own high/low — otherwise a breakout
    # above "the max high including yesterday" can mathematically never fire.
    high_ref, low_ref, close_lag = high.shift(2), low.shift(2), close.shift(1)
    upper_entry = high_ref.rolling(int(entry_window)).max()
    lower_entry = low_ref.rolling(int(entry_window)).min()
    upper_exit = high_ref.rolling(int(exit_window)).max()
    lower_exit = low_ref.rolling(int(exit_window)).min()
    long_entry, long_exit = close_lag > upper_entry, close_lag < lower_exit
    short_entry, short_exit = close_lag < lower_entry, close_lag > upper_exit
    return _stateful_strategy_returns(close, long_entry, long_exit, short_entry, short_exit, allow_short)


def strat_atr_breakout(close, high=None, low=None, allow_short=False, period=14, k=1.5, **kwargs) -> pd.Series:
    atr_lag = compute_atr(high, low, close, int(period)).shift(1)
    close_lag, prev_close_lag = close.shift(1), close.shift(2)
    trail_sma = close_lag.rolling(10).mean()
    long_entry = close_lag > prev_close_lag + k * atr_lag
    short_entry = close_lag < prev_close_lag - k * atr_lag
    long_exit = close_lag < trail_sma
    short_exit = close_lag > trail_sma
    return _stateful_strategy_returns(close, long_entry, long_exit, short_entry, short_exit, allow_short)


STRATEGY_CATALOG = {
    "Buy & Hold":                 {"category": "Baseline",             "func": strat_buy_and_hold,
                                    "params": {},
                                    "desc": "Fully invested benchmark exposure with no timing signal — the reference every active strategy must beat."},
    "SMA Crossover":              {"category": "Trend-Following",      "func": strat_sma_crossover,
                                    "params": {"fast": (10, 100, 50, 5), "slow": (100, 300, 200, 10)},
                                    "desc": "Classic Dual Moving Average Crossover — long when the fast SMA trades above the slow SMA, short (or flat) otherwise."},
    "EMA Crossover":               {"category": "Trend-Following",      "func": strat_ema_crossover,
                                    "params": {"fast": (5, 50, 12, 1), "slow": (20, 200, 26, 2)},
                                    "desc": "Exponential moving average crossover — reacts faster to new information than a simple moving average."},
    "Triple SMA Trend Filter":    {"category": "Trend-Following",      "func": strat_triple_sma_trend,
                                    "params": {"fast": (5, 50, 20, 1), "mid": (20, 100, 50, 5), "slow": (100, 300, 200, 10)},
                                    "desc": "Requires full trend alignment (price > fast > mid > slow SMA) before entering — filters out weak or choppy trends."},
    "MACD Signal Crossover":      {"category": "Trend-Following",      "func": strat_macd_crossover,
                                    "params": {"fast": (5, 30, 12, 1), "slow": (15, 60, 26, 1), "signal": (3, 20, 9, 1)},
                                    "desc": "Moving Average Convergence Divergence — trades crossovers of the MACD line and its signal line."},
    "Time-Series Momentum":       {"category": "Momentum",             "func": strat_time_series_momentum,
                                    "params": {"lookback": (10, 250, 90, 5)},
                                    "desc": "Absolute (time-series) momentum — goes long when trailing N-day return is positive, short when negative."},
    "Rate of Change (ROC)":       {"category": "Momentum",             "func": strat_roc_momentum,
                                    "params": {"lookback": (5, 120, 20, 5), "threshold_pct": (0.0, 10.0, 2.0, 0.5)},
                                    "desc": "Trades only once momentum exceeds a minimum threshold, reducing whipsaws from marginal moves."},
    "RSI Mean Reversion":         {"category": "Mean-Reversion",       "func": strat_rsi_reversion,
                                    "params": {"period": (5, 30, 14, 1), "oversold": (10, 40, 30, 1), "overbought": (60, 90, 70, 1)},
                                    "desc": "Buys oversold / shorts overbought RSI extremes, exiting once RSI reverts to the neutral 50 midline."},
    "Bollinger Band Reversion":   {"category": "Mean-Reversion",       "func": strat_bollinger_reversion,
                                    "params": {"window": (10, 60, 20, 1), "num_std": (1.0, 3.5, 2.0, 0.1)},
                                    "desc": "Fades price extremes outside N standard-deviation Bollinger Bands, exiting at the mean (middle band)."},
    "Z-Score Mean Reversion":     {"category": "Mean-Reversion",       "func": strat_zscore_reversion,
                                    "params": {"window": (10, 90, 30, 5), "entry_z": (1.0, 3.5, 2.0, 0.1), "exit_z": (0.0, 1.5, 0.5, 0.1)},
                                    "desc": "Statistical arbitrage-style reversion on the price's rolling z-score, with independent entry/exit thresholds."},
    "Donchian Channel Breakout":  {"category": "Breakout / Volatility", "func": strat_donchian_breakout,
                                    "params": {"entry_window": (10, 60, 20, 5), "exit_window": (5, 30, 10, 1)},
                                    "desc": "The original Turtle Trading rules — enters on an N1-day high/low breakout, exits on a shorter N2-day reversal."},
    "ATR Volatility Breakout":    {"category": "Breakout / Volatility", "func": strat_atr_breakout,
                                    "params": {"period": (5, 30, 14, 1), "k": (0.5, 4.0, 1.5, 0.1)},
                                    "desc": "Enters when a single day's move exceeds k × Average True Range, riding volatility expansions until price reverts to its 10-day trend."},
}

CATEGORY_ORDER = ["Baseline", "Trend-Following", "Momentum", "Mean-Reversion", "Breakout / Volatility"]


@st.cache_data(ttl=600, show_spinner=False)
def download_ohlc(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Download strategy OHLC data with robust yfinance MultiIndex handling."""
    if not YFINANCE_AVAILABLE:
        return pd.DataFrame()
    tkr = ticker.strip().upper()
    try:
        data = yf.download(
            tkr, start=start, end=end, progress=False, auto_adjust=True,
            actions=False, threads=False,
        )
        out = _flatten_yf_columns(data, tkr)
        required = {"Close", "High", "Low"}
        if not required.issubset(out.columns):
            return pd.DataFrame()
        out = out[["Close", "High", "Low"]].apply(pd.to_numeric, errors="coerce").dropna()
        return out
    except Exception:
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# FORMATTING HELPERS
# ---------------------------------------------------------------------------

def pct(x, decimals: int = 2) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    return f"{x * 100:.{decimals}f}%"


def num(x, decimals: int = 2) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    return f"{x:.{decimals}f}"


def render_kpi(col, label: str, value: str, positive: bool) -> None:
    """Renders a Bloomberg-style KPI card with a green/red value color."""
    color = GREEN if positive else RED
    col.markdown(
        f"""<div class="qm-kpi">
            <div class="qm-kpi-label">{label}</div>
            <div class="qm-kpi-value" style="color:{color};">{value}</div>
        </div>""",
        unsafe_allow_html=True,
    )


def mi(icon_name: str, size: int = 18, color: Optional[str] = None) -> str:
    """Returns an inline Material Symbols icon <span> for use inside raw HTML/markdown."""
    style = f"font-size:{size}px;"
    if color:
        style += f" color:{color};"
    return f'<span class="material-symbols-outlined" style="{style}">{icon_name}</span>'


# ---------------------------------------------------------------------------
# CHART BUILDERS
# ---------------------------------------------------------------------------

def monte_carlo_projection(
    R: pd.Series,
    last_date: pd.Timestamp,
    last_value: float,
    horizon_days: int,
    n_sims: int = 2000,
    seed: int = 7,
    lookback: int = 252,
) -> dict:
    """
    Bootstrap future daily returns from the asset's recent realized distribution.

    This preserves empirical tail behavior better than a single Gaussian path.
    It is a scenario generator, not a forecast. The output contains complete
    day-by-day simulated price paths plus percentile trajectories.
    """
    r = pd.Series(R).dropna().astype(float)
    if len(r) < 20:
        raise ValueError("At least 20 clean returns are required.")
    sample = r.tail(min(int(lookback), len(r))).values
    rng = np.random.default_rng(seed)
    sampled = rng.choice(sample, size=(int(n_sims), int(horizon_days)), replace=True)
    sampled = np.clip(sampled, -0.995, None)
    value_paths = float(last_value) * np.cumprod(1.0 + sampled, axis=1)

    future_dates = pd.bdate_range(
        start=pd.Timestamp(last_date) + pd.Timedelta(days=1),
        periods=int(horizon_days),
    )
    display_count = min(16, int(n_sims))
    display_idx = np.linspace(0, int(n_sims) - 1, display_count, dtype=int)

    return {
        "dates": future_dates,
        "paths": value_paths,
        "display_paths": value_paths[display_idx],
        "p05": np.percentile(value_paths, 5, axis=0),
        "p10": np.percentile(value_paths, 10, axis=0),
        "p25": np.percentile(value_paths, 25, axis=0),
        "p50": np.percentile(value_paths, 50, axis=0),
        "p75": np.percentile(value_paths, 75, axis=0),
        "p90": np.percentile(value_paths, 90, axis=0),
        "p95": np.percentile(value_paths, 95, axis=0),
    }


def chart_future_projection(
    projection: dict,
    currency: str,
    last_price: float,
    show_sample_paths: bool = False,
) -> go.Figure:
    """Show observed anchor + day-by-day scenario paths + low/base/high bands."""
    dates = pd.DatetimeIndex(projection["dates"])
    x0 = dates[0] - pd.Timedelta(days=1)
    x = pd.DatetimeIndex([x0]).append(dates)
    last = float(last_price)
    fig = go.Figure()

    if show_sample_paths:
        display_paths = np.asarray(projection.get("display_paths", []), dtype=float)
        # Show only a few representative paths when explicitly requested.
        display_paths = display_paths[:4]
        for i, path in enumerate(display_paths):
            fig.add_trace(go.Scatter(
                x=x, y=np.concatenate([[last], path]), mode="lines",
                line=dict(width=0.8, color="rgba(139,147,167,0.24)"),
                name="Representative path", legendgroup="sample_paths", showlegend=(i == 0),
                hovertemplate=f"Representative path<br>Date=%{{x|%d %b %Y}}<br>Price=%{{y:,.2f}} {currency}<extra></extra>",
            ))

    fig.add_trace(go.Scatter(
        x=x, y=np.concatenate([[last], projection["p90"]]),
        mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=x, y=np.concatenate([[last], projection["p10"]]),
        mode="lines", line=dict(width=0),
        fill="tonexty", fillcolor="rgba(0,212,255,0.10)",
        name="10–90% scenario band",
        hovertemplate=f"Date=%{{x|%d %b %Y}}<br>Low band P10=%{{y:,.2f}} {currency}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=x, y=np.concatenate([[last], projection["p75"]]),
        mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=x, y=np.concatenate([[last], projection["p25"]]),
        mode="lines", line=dict(width=0),
        fill="tonexty", fillcolor="rgba(0,212,255,0.07)",
        name="25–75% scenario band", hoverinfo="skip",
    ))

    fig.add_trace(go.Scatter(
        x=dates, y=projection["p05"], mode="lines",
        line=dict(color=RED, width=1.2, dash="dot"), name="Stress low (P05)",
        hovertemplate=f"Date=%{{x|%d %b %Y}}<br>Stress low P05=%{{y:,.2f}} {currency}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=dates, y=projection["p10"], mode="lines",
        line=dict(color=RED, width=2.2), name="Low-case (P10)",
        hovertemplate=f"Date=%{{x|%d %b %Y}}<br>Low-case P10=%{{y:,.2f}} {currency}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=dates, y=projection["p50"], mode="lines",
        line=dict(color=CYAN, width=2.4), name="Base-case median (P50)",
        hovertemplate=f"Date=%{{x|%d %b %Y}}<br>Median P50=%{{y:,.2f}} {currency}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=dates, y=projection["p90"], mode="lines",
        line=dict(color=GREEN, width=2.0), name="High-case (P90)",
        hovertemplate=f"Date=%{{x|%d %b %Y}}<br>High-case P90=%{{y:,.2f}} {currency}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=[x0], y=[last], mode="markers", name="Latest observed",
        marker=dict(size=10),
        hovertemplate=f"Latest observed<br>Date=%{{x|%d %b %Y}}<br>Price=%{{y:,.2f}} {currency}<extra></extra>",
    ))
    fig.add_hline(
        y=last, line_dash="dot", line_width=1,
        annotation_text="Latest observed", annotation_position="top left",
    )

    fig.update_layout(**PLOTLY_TEMPLATE["layout"])
    fig.update_layout(
        title="Future Price Scenarios — Daily Path Dispersion",
        yaxis_title=f"Projected Price ({currency})",
        height=560,
        width=None,
        autosize=True,
        hovermode="x",
        legend=dict(
            orientation="h", yanchor="top", y=-0.15,
            xanchor="left", x=0, bgcolor="rgba(0,0,0,0)", font=dict(size=10),
        ),
        margin=dict(l=60, r=25, t=70, b=115),
        transition=dict(duration=420, easing="cubic-in-out"),
        uirevision="quantmetrics-projection",
    )
    fig.update_xaxes(
        title_text="Trading date", showspikes=True, spikemode="across",
        spikesnap="data", spikecolor=TEXT_MUTED, spikethickness=1,
    )
    fig.update_yaxes(
        showspikes=True, spikemode="across",
        spikesnap="data", spikecolor=TEXT_MUTED, spikethickness=1,
    )
    return fig


def chart_equity_curve(engine: RiskEngine, log_scale: bool, benchmark_name: str,
                        projection: Optional[dict] = None, currency: str = "") -> go.Figure:
    fig = go.Figure()
    strat_curve = (1 + engine.cumulative_returns) * 100
    fig.add_trace(go.Scatter(
        x=strat_curve.index, y=strat_curve.values, mode="lines",
        name="Strategy", line=dict(color=GREEN, width=2),
        fill="tozeroy", fillcolor="rgba(0,255,163,0.08)",
    ))
    if engine.Rb is not None and len(engine.Rb) > 0:
        bench_curve = (1 + engine.Rb).cumprod() * 100
        fig.add_trace(go.Scatter(
            x=bench_curve.index, y=bench_curve.values, mode="lines",
            name=f"Benchmark ({benchmark_name})", line=dict(color=CYAN, width=1.6, dash="dot"),
        ))

    if projection is not None:
        last_date, last_value = strat_curve.index[-1], strat_curve.values[-1]
        px = np.concatenate([[last_date], projection["dates"]])
        p10 = np.concatenate([[last_value], projection["p10"]])
        p25 = np.concatenate([[last_value], projection["p25"]])
        p50 = np.concatenate([[last_value], projection["p50"]])
        p75 = np.concatenate([[last_value], projection["p75"]])
        p90 = np.concatenate([[last_value], projection["p90"]])

        fig.add_trace(go.Scatter(x=px, y=p90, mode="lines", line=dict(width=0),
                                  showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=px, y=p10, mode="lines", line=dict(width=0),
                                  fill="tonexty", fillcolor="rgba(0,212,255,0.10)",
                                  name="10th–90th Percentile", hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=px, y=p75, mode="lines", line=dict(width=0),
                                  showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=px, y=p25, mode="lines", line=dict(width=0),
                                  fill="tonexty", fillcolor="rgba(0,212,255,0.22)",
                                  name="25th–75th Percentile", hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=px, y=p50, mode="lines", name="Projected (Median)",
                                  line=dict(color=CYAN, width=2, dash="dash")))
        # Manual shape + annotation instead of fig.add_vline(): add_vline's internal
        # annotation-placement helper computes sum(x)/len(x) over the shape's x
        # endpoints, which raises on pandas Timestamp / datetime objects on some
        # plotly+pandas version combinations. A plain ISO date string sidesteps
        # that code path entirely while rendering identically on a datetime axis.
        last_date_str = pd.Timestamp(last_date).strftime("%Y-%m-%d")
        fig.add_shape(
            type="line", xref="x", yref="paper",
            x0=last_date_str, x1=last_date_str, y0=0, y1=1,
            line=dict(color=TEXT_MUTED, width=1, dash="dot"),
        )
        fig.add_annotation(
            x=last_date_str, y=1, yref="paper", yanchor="bottom",
            text="Today", showarrow=False, font=dict(color=TEXT_MUTED, size=11),
        )

    fig.update_layout(**PLOTLY_TEMPLATE["layout"])
    fig.update_layout(
        title="Cumulative Growth of $100",
        yaxis_type="log" if log_scale else "linear",
        yaxis_title="Indexed Wealth (100 = Start)",
        hovermode="x unified",
        height=440,
        width=None,
        autosize=True,
    )
    xrange_values = list(strat_curve.index)
    if projection is not None:
        xrange_values = xrange_values + list(pd.DatetimeIndex(projection["dates"]))
    _set_plot_xrange(fig, xrange_values)
    return _apply_hover_crosshair(fig)


def chart_rolling_sharpe(engine: RiskEngine) -> go.Figure:
    rs = engine.rolling_sharpe(window=126)
    fig = go.Figure()
    if len(rs) > 0:
        fig.add_trace(go.Scatter(
            x=rs.index, y=rs.values, mode="lines", name="Rolling 6M Sharpe",
            line=dict(color=CYAN, width=1.8), fill="tozeroy", fillcolor="rgba(0,212,255,0.08)",
        ))
        fig.add_hline(y=0, line_color=BORDER, line_width=1)
    fig.update_layout(**PLOTLY_TEMPLATE["layout"])
    fig.update_layout(title="Rolling 6-Month Sharpe Ratio", height=280, width=None, autosize=True, showlegend=False)
    if not rs.empty:
        _set_plot_xrange(fig, rs.index)
    return _apply_hover_crosshair(fig)


def chart_underwater(engine: RiskEngine) -> go.Figure:
    dd = engine.drawdown_series * 100
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dd.index, y=dd.values, mode="lines", name="Drawdown",
        line=dict(color=RED, width=1.6), fill="tozeroy", fillcolor="rgba(255,75,75,0.18)",
    ))
    fig.update_layout(**PLOTLY_TEMPLATE["layout"])
    fig.update_layout(title="Underwater Drawdown Plot", yaxis_title="Drawdown (%)", height=420,
                       width=None, autosize=True, hovermode="x unified")
    _set_plot_xrange(fig, dd.index)
    return _apply_hover_crosshair(fig)


def chart_monthly_heatmap(pivot: pd.DataFrame) -> go.Figure:
    if pivot.empty:
        return go.Figure()
    display_cols = [c for c in pivot.columns if c != "Year Total"] + (["Year Total"] if "Year Total" in pivot.columns else [])
    z = pivot[display_cols].values.astype(float)
    text = np.where(np.isnan(z), "", np.vectorize(lambda v: f"{v:.2f}")(np.nan_to_num(z)))
    fig = go.Figure(data=go.Heatmap(
        z=z,
        x=display_cols,
        y=[str(y) for y in pivot.index],
        colorscale=[[0, RED], [0.5, "#1e222d"], [1, GREEN]],
        zmid=0,
        text=text,
        texttemplate="%{text}",
        textfont=dict(size=11),
        colorbar=dict(title="%"),
        hoverongaps=False,
        xgap=2,
        ygap=2,
    ))
    fig.update_layout(**PLOTLY_TEMPLATE["layout"])
    fig.update_layout(title="Monthly Returns Heatmap (%)", height=max(320, 60 * len(pivot) + 100))
    return fig


def chart_return_distribution(engine: RiskEngine) -> go.Figure:
    r = engine.R * 100
    mu, sigma = r.mean(), r.std(ddof=1) if len(r) > 1 else 1.0
    x_range = np.linspace(r.min(), r.max(), 200)
    normal_curve = stats.norm.pdf(x_range, mu, sigma)
    normal_curve_scaled = normal_curve * len(r) * (r.max() - r.min()) / 40

    fig = go.Figure()
    fig.add_trace(go.Histogram(x=r, nbinsx=60, name="Daily Returns",
                                marker_color=CYAN, opacity=0.6))
    fig.add_trace(go.Scatter(x=x_range, y=normal_curve_scaled, mode="lines",
                              name="Normal Fit", line=dict(color=TEXT, width=1.5, dash="dash")))
    fig.add_vline(x=engine.metrics["var_95"] * 100, line_color="orange", line_width=1.5,
                  annotation_text="VaR 95%", annotation_font_color="orange")
    fig.add_vline(x=engine.metrics["var_99"] * 100, line_color=RED, line_width=1.5,
                  annotation_text="VaR 99%", annotation_font_color=RED)
    fig.update_layout(**PLOTLY_TEMPLATE["layout"])
    fig.update_layout(title="Daily Returns Distribution & Tail Risk", xaxis_title="Daily Return (%)",
                       height=420, bargap=0.02)
    return fig


def chart_beta_scatter(engine: RiskEngine, benchmark_name: str) -> go.Figure:
    fig = go.Figure()
    if engine.Rb is None or len(engine.Rb) < 5:
        fig.update_layout(**PLOTLY_TEMPLATE["layout"])
        fig.update_layout(title="Strategy vs Benchmark (benchmark unavailable)", height=420)
        return fig
    x = engine.Rb.values * 100
    y = engine.R.values * 100
    slope, intercept = np.polyfit(x, y, 1)
    x_line = np.linspace(x.min(), x.max(), 50)
    y_line = slope * x_line + intercept

    fig.add_trace(go.Scatter(x=x, y=y, mode="markers", name="Daily Pairs",
                              marker=dict(color=CYAN, size=5, opacity=0.5)))
    fig.add_trace(go.Scatter(x=x_line, y=y_line, mode="lines", name="Regression",
                              line=dict(color=GREEN, width=2)))
    beta = engine.metrics.get("beta", np.nan)
    alpha = engine.metrics.get("alpha", np.nan)
    fig.update_layout(**PLOTLY_TEMPLATE["layout"])
    fig.update_layout(
        title=f"Strategy vs {benchmark_name} — Beta: {num(beta)}  |  Alpha (ann.): {pct(alpha)}",
        xaxis_title=f"{benchmark_name} Daily Return (%)",
        yaxis_title="Strategy Daily Return (%)",
        height=420,
    )
    return fig



# ---------------------------------------------------------------------------
# ENHANCED MARKET CHART BUILDERS
# ---------------------------------------------------------------------------

def _apply_hover_crosshair(fig: go.Figure) -> go.Figure:
    """Add a professional crosshair/spike-line interaction to every Plotly axis."""
    fig.update_xaxes(
        showspikes=True,
        spikemode="across",
        spikesnap="data",
        showline=True,
        spikecolor=TEXT_MUTED,
        spikethickness=1,
    )
    fig.update_yaxes(
        showspikes=True,
        spikemode="across",
        spikesnap="data",
        showline=True,
        spikecolor=TEXT_MUTED,
        spikethickness=1,
    )
    return fig


def _set_plot_xrange(fig: go.Figure, x_values: Any) -> go.Figure:
    """Set the actual data extent and remove non-trading calendar gaps.

    Daily market series contain observations only on open-market sessions, but
    Plotly's date axis normally reserves visual space for weekends and holidays.
    We keep the real timestamps for hover/labels while using date-axis range
    breaks to collapse every missing calendar date between observations.
    """
    try:
        x = pd.DatetimeIndex(pd.to_datetime(pd.Series(x_values).dropna(), errors="coerce")).dropna().unique().sort_values()
        if len(x) == 0:
            return fig

        lo, hi = x.min(), x.max()
        if lo == hi:
            pad = pd.Timedelta(days=1)
            lo, hi = lo - pad, hi + pad
        else:
            pad = max((hi - lo) * 0.015, pd.Timedelta(days=1))
            lo, hi = lo - pad, hi + pad
        fig.update_xaxes(range=[lo, hi], autorange=False)

        # Only apply the precise missing-day compression to daily/longer data.
        # For intraday series the market's overnight session boundaries are
        # exchange-specific, so we avoid guessing them here.
        if len(x) >= 2:
            deltas = pd.Series(x[1:] - x[:-1])
            median_delta = deltas.median()
            if pd.notna(median_delta) and median_delta >= pd.Timedelta(hours=20):
                all_days = pd.date_range(
                    start=x.min().normalize(), end=x.max().normalize(), freq="D"
                )
                present_days = pd.DatetimeIndex(x.normalize().unique())
                missing_days = all_days.difference(present_days)
                if len(missing_days):
                    # Hide weekends through a compact bounds rule; explicit
                    # values handle exchange holidays and other closed dates.
                    non_weekend_missing = [
                        d.strftime("%Y-%m-%d") for d in missing_days if d.dayofweek < 5
                    ]
                    breaks = [dict(bounds=["sat", "mon"]) ]
                    if non_weekend_missing:
                        breaks.append(dict(values=non_weekend_missing))
                    fig.update_xaxes(rangebreaks=breaks)
    except Exception:
        pass
    return fig


def _plotly_base(fig: go.Figure, title: str, height: int = 440, x_values: Any = None) -> go.Figure:
    fig.update_layout(**PLOTLY_TEMPLATE["layout"])
    fig.update_layout(
        title=title,
        height=height,
        width=None,
        autosize=True,
        hovermode="x",
        uirevision="quantmetrics",
    )
    if x_values is not None:
        _set_plot_xrange(fig, x_values)
    return _apply_hover_crosshair(fig)


def _add_subplot_titles(fig: go.Figure, titles: Sequence[str], gap: float = 0.014) -> go.Figure:
    """Place compact subplot titles in the whitespace above each panel.

    Plotly's default subplot title annotations sit directly on panel boundaries,
    which becomes cramped when several indicator rows are enabled. Using the
    calculated y-axis domains gives every panel its own breathing room.
    """
    for idx, title in enumerate(titles, start=1):
        axis_name = "yaxis" if idx == 1 else f"yaxis{idx}"
        axis = getattr(fig.layout, axis_name, None)
        domain = getattr(axis, "domain", None) if axis is not None else None
        if domain and len(domain) == 2:
            fig.add_annotation(
                x=0.5, y=min(0.995, float(domain[1]) + gap),
                xref="paper", yref="paper",
                text=title, showarrow=False,
                xanchor="center", yanchor="bottom",
                font=dict(size=11, color=TEXT, family="Inter, sans-serif"),
                align="center",
            )
    return fig


def chart_price_technical(
    tech: pd.DataFrame,
    currency: str,
    show_candles: bool,
    ma_selection: Sequence[str],
    indicator_selection: Sequence[str],
) -> go.Figure:
    if tech.empty:
        return go.Figure()

    lower_indicators = [
        x for x in indicator_selection
        if x in {"RSI", "MACD", "ATR", "ADX", "Stochastic", "ROC", "OBV"}
    ]
    n_indicator_rows = len(lower_indicators)
    nrows = 2 + n_indicator_rows

    if nrows == 2:
        row_heights = [0.68, 0.32]
    else:
        row_heights = [0.56, 0.16] + [0.28 / n_indicator_rows] * n_indicator_rows

    panel_titles = ["Price", "Volume"]
    panel_titles.extend({
        "RSI": "RSI (14)",
        "MACD": "MACD (12, 26, 9)",
        "ATR": "ATR (14)",
        "ADX": "ADX (14)",
        "Stochastic": "Stochastic",
        "ROC": "ROC (20D)",
        "OBV": "OBV",
    }[x] for x in lower_indicators)

    fig = make_subplots(
        rows=nrows,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.065,
        row_heights=row_heights,
    )

    c = tech["Close"]
    if show_candles and {"Open", "High", "Low"}.issubset(tech.columns):
        fig.add_trace(go.Candlestick(
            x=tech.index, open=tech["Open"], high=tech["High"], low=tech["Low"], close=c,
            name="OHLC", increasing_line_color=GREEN, decreasing_line_color=RED,
            showlegend=False,
            hoverinfo="all",
            xhoverformat="%d %b %Y",
            yhoverformat=",.2f",
        ), row=1, col=1)
    else:
        fig.add_trace(go.Scatter(
            x=tech.index, y=c, mode="lines", name="Close",
            line=dict(color=CYAN, width=1.8),
            hovertemplate=f"Date=%{{x|%d %b %Y}}<br>Close=%{{y:,.2f}} {currency}<extra></extra>",
        ), row=1, col=1)

    ma_map = {
        "SMA 20": "SMA20", "SMA 50": "SMA50", "SMA 100": "SMA100", "SMA 200": "SMA200",
        "EMA 20": "EMA20", "EMA 50": "EMA50", "EMA 200": "EMA200",
    }
    for label in ma_selection:
        col = ma_map.get(label)
        if col in tech:
            fig.add_trace(go.Scatter(
                x=tech.index, y=tech[col], mode="lines", name=label,
                line=dict(width=1.4),
                hovertemplate=f"Date=%{{x|%d %b %Y}}<br>{label}=%{{y:,.2f}} {currency}<extra></extra>",
            ), row=1, col=1)

    if "Bollinger Bands" in indicator_selection:
        fig.add_trace(go.Scatter(
            x=tech.index, y=tech["BBUpper"], line=dict(width=1, dash="dot"),
            name="BB Upper", showlegend=True,
            hovertemplate="Date=%{x|%d %b %Y}<br>BB Upper=%{y:,.2f}<extra></extra>",
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=tech.index, y=tech["BBLower"], line=dict(width=1, dash="dot"),
            name="BB Lower", fill="tonexty", fillcolor="rgba(0,212,255,0.06)",
            showlegend=True,
            hovertemplate="Date=%{x|%d %b %Y}<br>BB Lower=%{y:,.2f}<extra></extra>",
        ), row=1, col=1)

    if "Volume" in tech.columns:
        vol = tech["Volume"]
        avg20 = vol.rolling(20).mean()
        fig.add_trace(go.Bar(
            x=tech.index, y=vol, name="Volume", opacity=0.35,
            hovertemplate="Date=%{x|%d %b %Y}<br>Volume=%{y:,.0f}<extra></extra>",
        ), row=2, col=1)
        fig.add_trace(go.Scatter(
            x=tech.index, y=avg20, name="20D Avg Volume", line=dict(width=1.2),
            hovertemplate="Date=%{x|%d %b %Y}<br>20D Avg Volume=%{y:,.0f}<extra></extra>",
        ), row=2, col=1)

    indicator_row = 3
    indicator_map = {
        "RSI": ("RSI14", "RSI 14"),
        "ATR": ("ATR14", "ATR 14"),
        "ADX": ("ADX14", "ADX 14"),
        "Stochastic": ("StochK", "Stoch %K"),
        "ROC": ("ROC20", "ROC 20D"),
        "OBV": ("OBV", "OBV"),
    }

    for indicator in lower_indicators:
        if indicator == "MACD":
            fig.add_trace(go.Scatter(
                x=tech.index, y=tech["MACD"], name="MACD", line=dict(width=1.4),
                hovertemplate="Date=%{x|%d %b %Y}<br>MACD=%{y:,.4f}<extra></extra>",
            ), row=indicator_row, col=1)
            fig.add_trace(go.Scatter(
                x=tech.index, y=tech["MACDSignal"], name="MACD Signal", line=dict(width=1.1),
                hovertemplate="Date=%{x|%d %b %Y}<br>Signal=%{y:,.4f}<extra></extra>",
            ), row=indicator_row, col=1)
            fig.add_trace(go.Bar(
                x=tech.index, y=tech["MACDHist"], name="MACD Histogram", opacity=0.35,
                hovertemplate="Date=%{x|%d %b %Y}<br>Histogram=%{y:,.4f}<extra></extra>",
            ), row=indicator_row, col=1)
        elif indicator in indicator_map:
            col, label = indicator_map[indicator]
            if col in tech:
                fig.add_trace(go.Scatter(
                    x=tech.index, y=tech[col], name=label, line=dict(width=1.4),
                    hovertemplate=f"Date=%{{x|%d %b %Y}}<br>{label}=%{{y:,.3f}}<extra></extra>",
                ), row=indicator_row, col=1)
                if indicator == "RSI":
                    fig.add_hline(y=70, line_dash="dot", line_width=1, row=indicator_row, col=1)
                    fig.add_hline(y=30, line_dash="dot", line_width=1, row=indicator_row, col=1)
                elif indicator == "ADX":
                    fig.add_hline(y=25, line_dash="dot", line_width=1, row=indicator_row, col=1)
                elif indicator == "Stochastic":
                    fig.add_hline(y=80, line_dash="dot", line_width=1, row=indicator_row, col=1)
                    fig.add_hline(y=20, line_dash="dot", line_width=1, row=indicator_row, col=1)
        indicator_row += 1

    # Compact left-side axis labels keep indicator identity visible without
    # consuming vertical chart space. Titles remain inside each subplot axis.
    fig.update_yaxes(
        title_text=f"Price ({currency})", row=1, col=1, title_standoff=8,
        title_font=dict(size=10, color=TEXT_MUTED),
    )
    fig.update_yaxes(
        title_text="Volume", row=2, col=1, title_standoff=8,
        title_font=dict(size=10, color=TEXT_MUTED),
    )
    indicator_axis_labels = {
        "RSI": "RSI (14)",
        "MACD": "MACD",
        "ATR": "ATR (14)",
        "ADX": "ADX (14)",
        "Stochastic": "Stoch",
        "ROC": "ROC (20D)",
        "OBV": "OBV",
    }
    for r, indicator in enumerate(lower_indicators, start=3):
        label = indicator_axis_labels.get(indicator)
        if label:
            fig.update_yaxes(
                title_text=label, row=r, col=1, title_standoff=8,
                title_font=dict(size=10, color=TEXT_MUTED),
            )
    for r in range(1, nrows + 1):
        fig.update_xaxes(
            rangeslider_visible=False,
            row=r, col=1,
            showspikes=True, spikemode="across", spikesnap="data",
            spikecolor=TEXT_MUTED, spikethickness=1,
        )
        fig.update_yaxes(
            showspikes=True, spikemode="across", spikesnap="data",
            spikecolor=TEXT_MUTED, spikethickness=1,
        )

    fig.update_layout(**PLOTLY_TEMPLATE["layout"])
    _add_subplot_titles(fig, panel_titles, gap=0.012)
    fig.update_layout(
        title="Price & Volume — Technical Dashboard",
        hovermode="x unified",
        legend=dict(
            orientation="h", yanchor="top", y=-0.10,
            xanchor="left", x=0, bgcolor="rgba(0,0,0,0)", font=dict(size=10),
        ),
        margin=dict(l=60, r=20, t=70, b=115),
        height=560 + 105 * n_indicator_rows,
        width=None,
        autosize=True,
        transition=dict(duration=280, easing="cubic-in-out"),
    )
    _set_plot_xrange(fig, tech.index)
    return _apply_hover_crosshair(fig)


def chart_crossover_intelligence(tech: pd.DataFrame, events: pd.DataFrame, currency: str) -> go.Figure:
    if tech.empty:
        return go.Figure()
    c = tech["Close"].astype(float)
    fig = go.Figure()
    if {"Open", "High", "Low"}.issubset(tech.columns):
        fig.add_trace(go.Candlestick(
            x=tech.index, open=tech["Open"], high=tech["High"], low=tech["Low"], close=c,
            name="Price", increasing_line_color=GREEN, decreasing_line_color=RED,
            showlegend=False,
            hoverinfo="all",
            xhoverformat="%d %b %Y",
            yhoverformat=",.2f",
        ))
    else:
        fig.add_trace(go.Scatter(
            x=tech.index, y=c, mode="lines", name="Price", line=dict(width=1.7),
            hovertemplate=f"Date=%{{x|%d %b %Y}}<br>Price=%{{y:,.2f}} {currency}<extra></extra>",
        ))

    for label, col in [
        ("SMA 50", "SMA50"), ("SMA 200", "SMA200"),
        ("EMA 50", "EMA50"), ("EMA 200", "EMA200"),
    ]:
        if col in tech:
            fig.add_trace(go.Scatter(
                x=tech.index, y=tech[col], mode="lines", name=label, line=dict(width=1.2),
                hovertemplate=f"Date=%{{x|%d %b %Y}}<br>{label}=%{{y:,.2f}} {currency}<extra></extra>",
            ))

    if not events.empty:
        work = events.copy()
        work["Date"] = pd.to_datetime(work["Date"], errors="coerce")
        work["Price at Cross"] = pd.to_numeric(work["Price at Cross"], errors="coerce")
        work = work.dropna(subset=["Date", "Price at Cross"])
        work = work[(work["Date"] >= tech.index.min()) & (work["Date"] <= tech.index.max())]

        signal_groups = [
            ("Bullish Cross", "Bullish crossover", GREEN, "triangle-up"),
            ("Bearish Cross", "Bearish crossover", RED, "triangle-down"),
        ]
        for signal_state, trace_name, color, symbol in signal_groups:
            sub = work[work["State"] == signal_state].copy()
            if sub.empty:
                continue
            symbols = []
            for _, row in sub.iterrows():
                sig = str(row.get("Signal", ""))
                if sig == "SMA 50/200" and signal_state == "Bullish Cross":
                    symbols.append("star")
                elif sig == "SMA 50/200" and signal_state == "Bearish Cross":
                    symbols.append("x")
                else:
                    symbols.append(symbol)

            custom_cols = [
                "Signal", "Date", "Price at Cross", "Previous Fast MA", "Previous Slow MA",
                "Fast MA", "Slow MA", "5D Return", "20D Return", "60D Return", "120D Return",
            ]
            available_cols = [c for c in custom_cols if c in sub.columns]
            custom = sub[available_cols].astype(object).values
            col_idx = {col: i for i, col in enumerate(available_cols)}

            def cd(name: str) -> int:
                return col_idx.get(name, -1)

            # Hover contains all analytical detail; chart itself stays visually clean.
            hover_parts = [
                f"Signal=%{{customdata[{cd('Signal')}]}}",
                f"Date=%{{customdata[{cd('Date')} ]|%d %b %Y}}" if cd("Date") >= 0 else "",
                f"Price=%{{customdata[{cd('Price at Cross')}]:,.2f}} {currency}" if cd("Price at Cross") >= 0 else "",
            ]
            for label, key in [
                ("Prev fast", "Previous Fast MA"), ("Prev slow", "Previous Slow MA"),
                ("New fast", "Fast MA"), ("New slow", "Slow MA"),
            ]:
                if cd(key) >= 0:
                    hover_parts.append(f"{label}=%{{customdata[{cd(key)}]:,.2f}}")
            for horizon, key in [("5D", "5D Return"), ("20D", "20D Return"), ("60D", "60D Return"), ("120D", "120D Return")]:
                if cd(key) >= 0:
                    hover_parts.append(f"{horizon}=%{{customdata[{cd(key)}]:.2%}}")

            fig.add_trace(go.Scatter(
                x=sub["Date"], y=sub["Price at Cross"],
                mode="markers",
                name=trace_name,
                marker=dict(size=10, color=color, symbol=symbols, line=dict(width=1)),
                customdata=custom,
                hovertemplate="<br>".join([p for p in hover_parts if p]) + "<extra></extra>",
            ))

    last_idx = tech.index[-1]
    last_price = _safe_float(c.iloc[-1])
    if np.isfinite(last_price):
        fig.add_trace(go.Scatter(
            x=[last_idx], y=[last_price], mode="markers",
            name="Current price", marker=dict(size=9, color=CYAN, symbol="circle"),
            hovertemplate=f"CURRENT<br>Date=%{{x|%d %b %Y}}<br>Price=%{{y:,.2f}} {currency}<extra></extra>",
        ))

    fig.update_layout(**PLOTLY_TEMPLATE["layout"])
    fig.update_layout(
        title="Crossover Intelligence — Price, Major MAs & Signal Events",
        height=560,
        width=None,
        autosize=True,
        hovermode="x unified",
        showlegend=False,
        margin=dict(l=60, r=20, t=65, b=40),
    )
    fig.update_xaxes(showspikes=True, spikemode="across", spikesnap="data", spikecolor=TEXT_MUTED, spikethickness=1)
    fig.update_yaxes(showspikes=True, spikemode="across", spikesnap="data", spikecolor=TEXT_MUTED, spikethickness=1)
    _set_plot_xrange(fig, tech.index)
    return _apply_hover_crosshair(fig)


def chart_relative_strength(asset_close: pd.Series, benchmark_close: pd.Series,
                            asset_name: str, benchmark_name: str) -> go.Figure:
    rs = relative_strength_series(asset_close, benchmark_close)
    if rs.empty:
        return go.Figure()
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=rs.index, y=rs.values, mode="lines",
        name=f"{asset_name} / {benchmark_name}", line=dict(width=1.8),
        hovertemplate=f"Date=%{{x|%d %b %Y}}<br>Relative ratio=%{{y:.4f}}<extra></extra>",
    ))
    fig.add_hline(
        y=rs.iloc[0], line_dash="dot", line_width=1,
        annotation_text="Starting ratio", annotation_position="top left",
    )
    fig.update_layout(**PLOTLY_TEMPLATE["layout"])
    fig.update_layout(
        title=f"Relative Strength Ratio — {asset_name} / {benchmark_name}",
        yaxis_title="Asset / Benchmark", height=390,
    )
    _set_plot_xrange(fig, rs.index)
    return _apply_hover_crosshair(fig)


def chart_rolling_analytics(asset_returns: pd.Series, benchmark_returns: Optional[pd.Series],
                            metric: str, window: int) -> go.Figure:
    r = asset_returns.dropna()
    fig = go.Figure()
    if len(r) < max(20, window):
        fig.update_layout(**PLOTLY_TEMPLATE["layout"])
        return fig
    if metric == "Rolling Return":
        series = r.rolling(window).apply(lambda x: np.prod(1 + x) - 1, raw=True)
    elif metric == "Rolling Volatility":
        series = r.rolling(window).std(ddof=1) * np.sqrt(TRADING_DAYS)
    elif metric == "Rolling Sharpe":
        ann = r.rolling(window).mean() * TRADING_DAYS
        vol = r.rolling(window).std(ddof=1) * np.sqrt(TRADING_DAYS)
        series = (ann - 0.045) / (vol + EPS)
    elif metric == "Rolling Maximum Drawdown":
        wealth = (1 + r).cumprod()
        series = wealth.rolling(window).apply(
            lambda x: np.min(x / np.maximum.accumulate(x) - 1), raw=True
        )
    elif metric == "Rolling RSI":
        close_proxy = (1 + r).cumprod()
        series = compute_rsi(close_proxy, 14)
    elif metric == "Rolling Momentum":
        series = (1 + r).rolling(window).apply(np.prod, raw=True) - 1
    elif metric == "Rolling Beta":
        if benchmark_returns is None:
            series = pd.Series(index=r.index, dtype=float)
        else:
            aligned = pd.concat([r.rename("asset"), benchmark_returns.rename("b")], axis=1).dropna()
            series = aligned["asset"].rolling(window).cov(aligned["b"]) / (aligned["b"].rolling(window).var() + EPS)
    elif metric == "Rolling Correlation":
        if benchmark_returns is None:
            series = pd.Series(index=r.index, dtype=float)
        else:
            aligned = pd.concat([r.rename("asset"), benchmark_returns.rename("b")], axis=1).dropna()
            series = aligned["asset"].rolling(window).corr(aligned["b"])
    else:
        series = pd.Series(index=r.index, dtype=float)
    fig.add_trace(go.Scatter(x=series.index, y=series.values, mode="lines", name=metric, line=dict(width=1.8)))
    fig.add_hline(y=0, line_dash="dot", line_width=1)
    return _plotly_base(fig, f"{metric} — {window} Trading Days", 360, x_values=series.index)


def chart_detailed_drawdown(close: pd.Series) -> go.Figure:
    stats = drawdown_stats_from_close(close)
    dd = stats["series"] * 100
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dd.index, y=dd.values, mode="lines", name="Drawdown",
                             line=dict(width=1.6), fill="tozeroy"))
    if stats["trough_date"] is not None:
        fig.add_trace(go.Scatter(
            x=[stats["peak_date"], stats["trough_date"]],
            y=[dd.loc[stats["peak_date"]], dd.loc[stats["trough_date"]]],
            mode="markers+text", text=["Peak", "Trough"], textposition="top center",
            name="Max DD markers", marker=dict(size=9),
        ))
    if stats["recovery_date"] is not None:
        fig.add_trace(go.Scatter(
            x=[stats["recovery_date"]], y=[dd.loc[stats["recovery_date"]]],
            mode="markers+text", text=["Recovery"], textposition="top center",
            name="Recovery", marker=dict(size=9),
        ))
    return _plotly_base(fig, "Detailed Drawdown — Current, Maximum, Trough & Recovery", 430, x_values=dd.index)


def chart_return_distribution_detailed(r: pd.Series, engine: Optional[RiskEngine] = None) -> go.Figure:
    pct_r = pd.Series(r).dropna() * 100
    if pct_r.empty:
        return go.Figure()

    mu = float(pct_r.mean())
    med = float(pct_r.median())
    sigma = float(pct_r.std(ddof=1)) if len(pct_r) > 1 else 1.0
    q001, q999 = float(pct_r.quantile(0.001)), float(pct_r.quantile(0.999))
    x_range = np.linspace(q001, q999, 250)
    normal = stats.norm.pdf(x_range, mu, sigma)
    scale = max(len(pct_r) * (pct_r.quantile(0.95) - pct_r.quantile(0.05)) / 20, 1e-6)

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=pct_r, nbinsx=60, name="Daily Returns", opacity=0.66,
        hovertemplate="Return bin=%{x:.2f}%<br>Observations=%{y}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=x_range, y=normal * scale, mode="lines",
        name="Normal Reference", line=dict(width=1.4, dash="dash"),
        hovertemplate="Normal reference<br>Return=%{x:.2f}%<extra></extra>",
    ))

    var95 = float(pct_r.quantile(0.05))
    var99 = float(pct_r.quantile(0.01))
    marker_specs = [
        (var99, "VaR 99%", RED, 0.94),
        (var95, "VaR 95%", RED, 0.84),
        (med, f"Median {med:.2f}%", CYAN, 0.74),
        (mu, f"Mean {mu:.2f}%", TEXT, 0.64),
    ]
    for xval, label, color, ypaper in marker_specs:
        fig.add_shape(
            type="line", x0=xval, x1=xval, y0=0, y1=1, yref="paper",
            line=dict(color=color, width=1.3, dash="dot"),
        )
        fig.add_annotation(
            x=xval, y=ypaper, xref="x", yref="paper",
            text=label, showarrow=False, yanchor="middle", xanchor="center",
            font=dict(size=10, color=color),
            bgcolor="rgba(14,17,23,0.88)",
            bordercolor="rgba(139,147,167,0.18)", borderwidth=1, borderpad=3,
        )

    left_cut = float(pct_r.quantile(0.01))
    right_cut = float(pct_r.quantile(0.99))
    left = pct_r[pct_r <= left_cut]
    right = pct_r[pct_r >= right_cut]
    fig.add_trace(go.Scatter(
        x=left, y=np.zeros(len(left)), mode="markers",
        name="Left-tail observations", marker=dict(size=4),
        hovertemplate="Left tail<br>Return=%{x:.2f}%<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=right, y=np.zeros(len(right)), mode="markers",
        name="Right-tail observations", marker=dict(size=4),
        hovertemplate="Right tail<br>Return=%{x:.2f}%<extra></extra>",
    ))

    fig.update_layout(**PLOTLY_TEMPLATE["layout"])
    fig.update_layout(
        title="Return Distribution — Empirical Tails, VaR & Central Tendency",
        xaxis_title="Daily return (%)", yaxis_title="Observations",
        height=450, hovermode="x",
        legend=dict(
            orientation="h", yanchor="top", y=-0.14,
            xanchor="left", x=0, bgcolor="rgba(0,0,0,0)", font=dict(size=10),
        ),
        margin=dict(l=55, r=25, t=90, b=100),
    )
    return _apply_hover_crosshair(fig)


def chart_cross_asset_matrix(context_df: pd.DataFrame) -> go.Figure:
    if context_df.empty:
        return go.Figure()
    cols = [c for c in context_df.columns if c != "Ticker"]
    fig = go.Figure(data=go.Heatmap(
        z=context_df[cols].apply(pd.to_numeric, errors="coerce").values,
        x=cols, y=context_df["Ticker"].astype(str).tolist(),
        colorscale=[[0, RED], [0.5, CARD], [1, GREEN]], zmid=0,
        colorbar=dict(title="%"),
        hovertemplate="%{y} — %{x}: %{z:.2f}%<extra></extra>",
    ))
    return _plotly_base(fig, "Cross-Asset Returns Context", 360)


# ---------------------------------------------------------------------------
# STREAMLIT APPLICATION / DECISION TERMINAL
# ---------------------------------------------------------------------------

st.sidebar.markdown(
    f"""
    <div class="qm-brand">
        <div class="qm-brand-mark">
            <svg width="24" height="24" viewBox="0 0 24 24" aria-hidden="true">
                <path d="M4 16.5 8.5 12l3.1 3.1L20 6.8" fill="none" stroke="{RED}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                <path d="M4 20h16" fill="none" stroke="#697386" stroke-width="1.3" stroke-linecap="round"/>
            </svg>
        </div>
        <div>
            <div class="qm-brand-name">QuantMetrics Pro</div>
            <div class="qm-brand-sub">Quantitative Backtesting & Risk Analytics Platform</div>
        </div>
    </div>""",
    unsafe_allow_html=True,
)
st.sidebar.markdown(
    f'<div style="font-size:1.15rem;font-weight:700;margin-bottom:2px;">{mi("tune", 22)}&nbsp; Control Panel</div>',
    unsafe_allow_html=True,
)

current_ticker = st.session_state.get("_qm_ticker", "NVDA").strip().upper()

st.sidebar.markdown("### Instrument")
# One control serves both as a searchable ticker picker and a free-form Yahoo Finance
# symbol entry. Streamlit matches known options while accept_new_options=True keeps
# arbitrary valid Yahoo symbols fully supported without adding them to the dropdown.
instrument_option_labels = [
    f"{symbol} — {INSTRUMENT_NAMES.get(symbol, symbol)}"
    for symbol in INSTRUMENT_CHOICES
]
instrument_default_idx = None
for i, option in enumerate(instrument_option_labels):
    if option.split(" — ", 1)[0].upper() == current_ticker:
        instrument_default_idx = i
        break

selected_instrument = st.sidebar.selectbox(
    "Ticker",
    instrument_option_labels,
    index=instrument_default_idx,
    key="qm_ticker_selector",
    placeholder="Search ticker or type any Yahoo Finance symbol",
    accept_new_options=True,
    filter_mode="contains",
    format_func=lambda x: x,
    help="Search the curated universe by ticker/company name, or type any Yahoo Finance ticker. A typed symbol outside the list is used unchanged.",
)

if selected_instrument is None:
    ticker_input = current_ticker
else:
    raw_selected = str(selected_instrument).strip()
    ticker_input = raw_selected.split(" — ", 1)[0].strip().upper()

st.session_state["_qm_ticker"] = ticker_input

# Market profile is cheap once cached and is also used to choose a benchmark default.
profile = fetch_instrument_profile(ticker_input) if ticker_input else InstrumentProfile(ticker="")
default_benchmark = profile.benchmark if profile.benchmark and profile.benchmark != "N/A" else "^GSPC"

auto_benchmark = st.sidebar.checkbox("Auto-select benchmark", value=True)
if default_benchmark not in BENCHMARK_CHOICES:
    benchmark_choices = BENCHMARK_CHOICES + [default_benchmark]
else:
    benchmark_choices = BENCHMARK_CHOICES
default_index = benchmark_choices.index(default_benchmark) if default_benchmark in benchmark_choices else 0
benchmark_ticker = st.sidebar.selectbox(
    "Benchmark",
    benchmark_choices,
    index=default_index,
    disabled=auto_benchmark,
    format_func=lambda x: f"{BENCHMARK_NAMES.get(x, x)} ({x})",
)
st.sidebar.caption(
    f"Auto benchmark: {BENCHMARK_NAMES.get(default_benchmark, default_benchmark)} ({default_benchmark})"
    if auto_benchmark else
    "Broad + sector benchmarks: India, US, Europe, Asia-Pacific and global ETFs/indices."
)
custom_benchmark = st.sidebar.text_input("Custom benchmark override", value="")
if auto_benchmark:
    benchmark_ticker = default_benchmark
if custom_benchmark.strip():
    benchmark_ticker = custom_benchmark.strip().upper()

risk_free_default = 4.5
risk_free_pct = st.sidebar.number_input(
    "Risk-Free Rate (annual %)", min_value=0.0, max_value=20.0,
    value=risk_free_default, step=0.1,
    help="Explicit model assumption. Yahoo Finance does not provide a single universal risk-free rate for every country."
)
risk_free_annual = risk_free_pct / 100.0

st.sidebar.markdown("---")
adjusted_prices = st.sidebar.checkbox(
    "Use adjusted prices", value=True,
    help="Adjusted prices incorporate Yahoo Finance's adjustment series. Keep this consistent for returns/backtests."
)
chart_period = st.sidebar.selectbox("Primary chart timeframe", PERIOD_OPTIONS, index=5)
chart_interval = st.sidebar.selectbox("Chart interval", INTERVAL_OPTIONS, index=5)
show_candles = st.sidebar.checkbox("Candlestick OHLC", value=True)

st.sidebar.markdown("### Technical overlays")
ma_selection = st.sidebar.multiselect(
    "Moving averages",
    ["SMA 20", "SMA 50", "SMA 100", "SMA 200", "EMA 20", "EMA 50", "EMA 200"],
    default=["SMA 50", "SMA 200"],
)
indicator_selection = st.sidebar.multiselect(
    "Indicator panels / overlays",
    ["RSI", "MACD", "Bollinger Bands", "ATR", "ADX", "Stochastic", "ROC", "OBV"],
    default=["RSI", "MACD"],
)

st.sidebar.markdown("---")
st.sidebar.markdown("### Strategy research")
mode = st.sidebar.radio(
    "Strategy Data Source",
    ["Upload Custom Strategy CSV", "Benchmark Strategy Simulator"],
    index=0,
)
returns_raw: Optional[pd.Series] = None
strategy_label = "Custom Strategy"
strategy_description: Optional[str] = None
strategy_type = "Buy & Hold"
allow_short = False
user_params: dict[str, Any] = {}
strategy_ohlc: Optional[pd.DataFrame] = None

if mode == "Upload Custom Strategy CSV":
    uploaded_file = st.sidebar.file_uploader("Upload Trade Log / Returns CSV", type=["csv"])
    load_sample = st.sidebar.button(
        "Load Sample Institutional Strategy",
        icon=":material/bar_chart:",
        use_container_width=True,
    )
    st.sidebar.caption("Sample path: SPY 50/200 SMA crossover built from market history.")
    if uploaded_file is not None:
        try:
            raw_df = pd.read_csv(uploaded_file)
            returns_raw = detect_and_parse_csv(raw_df)
            strategy_label = uploaded_file.name.rsplit(".", 1)[0]
            strategy_description = (
                "Custom return series loaded from the uploaded CSV. Metrics are computed from "
                "the provided return/NAV/trade-log data without assuming a strategy definition."
            )
            st.session_state["_sample_loaded"] = False
        except Exception as exc:
            st.sidebar.error(f"Parse error: {exc}", icon=":material/error:")
    elif load_sample:
        st.session_state["_sample_loaded"] = True
    if uploaded_file is None and st.session_state.get("_sample_loaded"):
        with st.spinner("Fetching SPY history for the sample crossover..."):
            sample_returns, is_authentic = fetch_authentic_sample_strategy(ticker="SPY", years=6)
        returns_raw = sample_returns
        if is_authentic:
            strategy_label = "SPY — SMA 50/200 Crossover"
            strategy_description = (
                "Dual moving-average crossover: long when SMA 50 exceeds SMA 200 and flat otherwise. "
                "The sample is built from the downloaded daily price series."
            )
        else:
            strategy_label = "Sample Strategy — Offline Synthetic Fallback"
            strategy_description = (
                "Live market data could not be reached, so the explicit offline synthetic fallback in the "
                "original application was used. Treat these metrics as illustrative only."
            )
            st.sidebar.warning("Live market data unavailable; sample metrics are synthetic.", icon=":material/warning:")
else:
    ticker_for_strategy = ticker_input
    strategy_category = st.sidebar.selectbox("Strategy Category", CATEGORY_ORDER, index=1)
    strategies_in_cat = [name for name in STRATEGY_CATALOG if STRATEGY_CATALOG[name]["category"] == strategy_category]
    strategy_type = st.sidebar.selectbox("Strategy", strategies_in_cat)
    st.sidebar.caption(STRATEGY_CATALOG[strategy_type]["desc"])
    allow_short = st.sidebar.checkbox("Allow Short Positions", value=False)
    param_spec = STRATEGY_CATALOG[strategy_type]["params"]
    if param_spec:
        with st.sidebar.expander("Strategy Parameters", icon=":material/tune:"):
            for pname, (pmin, pmax, pdefault, pstep) in param_spec.items():
                plabel = pname.replace("_", " ").title()
                if isinstance(pdefault, int):
                    user_params[pname] = st.slider(plabel, int(pmin), int(pmax), int(pdefault), int(pstep))
                else:
                    user_params[pname] = st.slider(plabel, float(pmin), float(pmax), float(pdefault), float(pstep))
    sim_years = st.sidebar.slider("Strategy lookback (years)", 1, 15, 6)
    run_sim = st.sidebar.button("Run / Refresh Strategy", icon=":material/rocket_launch:", use_container_width=True)
    sim_key = (ticker_for_strategy, strategy_type, tuple(sorted(user_params.items())), allow_short, sim_years, adjusted_prices)
    if run_sim or st.session_state.get("_last_sim_key") == sim_key:
        st.session_state["_last_sim_key"] = sim_key
        end = pd.Timestamp.today().strftime("%Y-%m-%d")
        start = (pd.Timestamp.today() - pd.DateOffset(years=sim_years + 1)).strftime("%Y-%m-%d")
        if YFINANCE_AVAILABLE:
            with st.spinner(f"Fetching {ticker_for_strategy} strategy history..."):
                strategy_ohlc = download_ohlc(ticker_for_strategy, start, end)
            if not strategy_ohlc.empty and len(strategy_ohlc) > 50:
                strategy_func = STRATEGY_CATALOG[strategy_type]["func"]
                returns_raw = strategy_func(
                    close=strategy_ohlc["Close"], high=strategy_ohlc["High"], low=strategy_ohlc["Low"],
                    allow_short=allow_short, **user_params,
                )
                side_label = "Long/Short" if allow_short else "Long-Only"
                strategy_label = f"{ticker_for_strategy} — {strategy_type} ({side_label})"
                param_str = ", ".join(f"{k.replace('_',' ').title()} = {v}" for k, v in user_params.items())
                strategy_description = STRATEGY_CATALOG[strategy_type]["desc"]
                if param_str:
                    strategy_description += f" Configured with {param_str}."
            elif run_sim:
                st.sidebar.error("No usable price history returned for the selected ticker.", icon=":material/error:")
        else:
            st.sidebar.error("yfinance is not available in this environment.", icon=":material/error:")

st.sidebar.markdown("---")
# Transaction-cost assumptions are explicit and user-overridable. Defaults are zero
# rather than a hard-coded current tax schedule.
with st.sidebar.expander("Transaction Costs", icon=":material/payments:"):
    brokerage_pct = st.number_input("Brokerage / commission (%)", 0.0, 5.0, 0.0, 0.005)
    slippage_pct = st.number_input("Slippage (%)", 0.0, 5.0, 0.02, 0.005)
    transaction_pct = st.number_input("Other transaction cost (%)", 0.0, 5.0, 0.0, 0.005)
    st.caption("All rates are assumptions. No current tax schedule is hard-coded.")
    if profile.country == "India":
        st.markdown("**India-specific optional inputs**")
        stt_pct = st.number_input("STT (%)", 0.0, 5.0, 0.0, 0.005)
        exchange_pct = st.number_input("Exchange charges (%)", 0.0, 1.0, 0.0, 0.005)
        gst_pct = st.number_input("GST (%)", 0.0, 5.0, 0.0, 0.005)
        sebi_pct = st.number_input("SEBI charges (%)", 0.0, 0.1, 0.0, 0.0001)
        stamp_pct = st.number_input("Stamp duty (%)", 0.0, 5.0, 0.0, 0.005)
    else:
        stt_pct = exchange_pct = gst_pct = sebi_pct = stamp_pct = 0.0
total_transaction_cost_pct = brokerage_pct + slippage_pct + transaction_pct + stt_pct + exchange_pct + gst_pct + sebi_pct + stamp_pct

with st.sidebar.expander("Position Sizing", icon=":material/calculate:"):
    capital = st.number_input("Capital", min_value=0.0, value=100000.0, step=5000.0)
    max_risk_pct = st.number_input("Max risk per trade (%)", min_value=0.0, max_value=100.0, value=1.0, step=0.1)
    entry_price = st.number_input("Entry price", min_value=0.0, value=0.0, step=0.01)
    stop_price = st.number_input("Stop price", min_value=0.0, value=0.0, step=0.01)
    allocation_cap_pct = st.number_input("Max portfolio allocation (%)", min_value=0.0, max_value=100.0, value=20.0, step=1.0)

st.sidebar.markdown(
    f"""
    <style>
        .qm-sidebar-footer a {{
            color: {TEXT_MUTED};
            text-decoration: none;
            transition: color 0.2s ease, transform 0.2s ease;
        }}
        .qm-sidebar-footer a:hover {{
            color: #ffffff !important;
            transform: translateY(-1px);
        }}
        .qm-sidebar-footer a svg {{
            transition: color 0.2s ease;
        }}
    </style>
    <div class="qm-sidebar-footer">
        <div style="
            color:{TEXT_MUTED};
            font-size:11px;
            text-transform:uppercase;
            letter-spacing:0.8px;
        ">
            Built by
        </div>
        <div style="
            margin-top:4px;
            color:{CYAN};
            font-size:12px;
            font-weight:400;
        ">
            Sudhir
            <span style="color:{TEXT_MUTED};"><strong> | </strong></span>
            <span style="color:{TEXT};">QuantMetrics Pro</span>
        </div>
        <div style="
            margin-top:5px;
            color:{TEXT_MUTED};
            font-size:11px;
        ">
            Quantitative Analytics &amp; Risk Engine Portfolio
        </div>
        <div style="
            margin-top:12px;
            padding-top:10px;
            border-top:1px solid {BORDER};
            display:flex;
            align-items:center;
            gap:18px;
        ">
            <a href="https://github.com/spelsudhir"
               target="_blank"
               style="
                   font-size:12px;
                   font-weight:600;
                   display:flex;
                   align-items:center;
                   gap:6px;
               ">
                <svg width="15" height="15" viewBox="0 0 24 24"
                     fill="currentColor">
                    <path d="M12 .5C5.65.5.5 5.65.5 12c0 5.08 3.29 9.39 7.86 10.91.57.1.78-.25.78-.55
                    0-.27-.01-1.16-.01-2.1-3.2.7-3.88-1.36-3.88-1.36-.52-1.33-1.28-1.68-1.28-1.68
                    -1.04-.71.08-.7.08-.7 1.15.08 1.75 1.18 1.75 1.18 1.02 1.75 2.67 1.24 3.32.95
                    .1-.74.4-1.24.72-1.52-2.55-.29-5.23-1.28-5.23-5.69 0-1.26.45-2.29 1.18-3.1
                    -.12-.29-.51-1.47.11-3.06 0 0 .96-.31 3.15 1.18a10.9 10.9 0 0 1 5.74 0
                    c2.19-1.49 3.15-1.18 3.15-1.18.62 1.59.23 2.77.11 3.06.73.81 1.18 1.84 1.18 3.1
                    0 4.42-2.69 5.4-5.25 5.69.41.35.77 1.04.77 2.1 0 1.52-.01 2.75-.01 3.12
                    0 .3.2.66.79.55C20.21 21.39 23.5 17.08 23.5 12 23.5 5.65 18.35.5 12 .5z"/>
                </svg>
                GitHub
            </a>
            <a href="mailto:spelsudhir@gmail.com"
               style="
                   font-size:12px;
                   font-weight:600;
                   display:flex;
                   align-items:center;
                   gap:6px;
               ">
                <svg width="15" height="15" viewBox="0 0 24 24"
                     fill="none"
                     stroke="currentColor"
                     stroke-width="2"
                     stroke-linecap="round"
                     stroke-linejoin="round">
                    <rect x="3" y="5" width="18" height="14" rx="2"/>
                    <path d="m3 7 9 6 9-6"/>
                </svg>
                Contact
            </a>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# MARKET DATA LOAD
# ---------------------------------------------------------------------------

if not ticker_input:
    st.info("Enter a market ticker to begin.", icon=":material/search:")
    st.stop()

with st.spinner(f"Loading {ticker_input} market intelligence..."):
    daily_history, daily_note = fetch_market_history(ticker_input, "5Y", "1d", adjusted_prices)
    chart_history, chart_note = fetch_market_history(ticker_input, chart_period, chart_interval, adjusted_prices)
    benchmark_history, benchmark_note = fetch_market_history(benchmark_ticker, "5Y", "1d", True)
    benchmark_profile = fetch_instrument_profile(benchmark_ticker) if benchmark_ticker else InstrumentProfile(ticker="")
    fundamentals = fetch_fundamentals(ticker_input)
    actions = fetch_actions(ticker_input)
    calendar_data = fetch_calendar_data(ticker_input)

if daily_history.empty:
    st.error(
        f"Data unavailable for **{ticker_input}**. Check the symbol or try again later.",
        icon=":material/error:",
    )
    if returns_raw is None or len(returns_raw) < 5:
        st.stop()

# Use daily data for signal/risk calculations. Intraday charting is visual-only.
tech = technical_frame(daily_history)
fresh_quote = fetch_latest_quote(ticker_input)
snapshot = get_market_snapshot(profile, daily_history, fresh_quote)
close = daily_history["Close"].dropna()
asset_returns = close.pct_change().dropna()
if len(close) < 2 or asset_returns.empty:
    st.error(
        f"Insufficient historical observations for **{ticker_input}** to calculate returns and risk analytics. "
        "The data provider returned fewer than 2 usable price observations.",
        icon=":material/error:",
    )
    st.stop()

# Single source of truth for the current selected asset. Export, risk, and analytics
# all use this engine so market analytics remain anchored to the current selected asset.
asset_engine = RiskEngine(
    R=asset_returns,
    Rb=None,
    risk_free_annual=risk_free_annual,
)
bench_close = benchmark_history["Close"].dropna() if not benchmark_history.empty and "Close" in benchmark_history else pd.Series(dtype=float)
bench_returns = bench_close.pct_change().dropna()
bench_stats = benchmark_analytics(
    asset_returns, bench_returns, close, bench_close, risk_free_annual=risk_free_annual
)
regime = detect_regime(tech, bench_returns)
if not bench_close.empty:
    rs = relative_strength_series(close, bench_close)
else:
    rs = pd.Series(dtype=float)
if not rs.empty:
    rs1m = float(rs.iloc[-1] / rs.iloc[-min(22, len(rs))] - 1) if len(rs) > 2 else np.nan
    rs3m = float(rs.iloc[-1] / rs.iloc[-min(63, len(rs))] - 1) if len(rs) > 20 else np.nan
    rs1y = float(rs.iloc[-1] / rs.iloc[-min(253, len(rs))] - 1) if len(rs) > 50 else np.nan
    rs_level = bench_stats.get("relative_strength", np.nan)
    rs_bucket = "Outperforming" if np.nanmean([rs1m, rs3m]) > 0.02 else "Underperforming" if np.nanmean([rs1m, rs3m]) < -0.02 else "Neutral"
else:
    rs1m = rs3m = rs1y = rs_level = np.nan
    rs_bucket = "N/A"
regime["relative_strength"] = rs_bucket
score = technical_scorecard(tech, bench_close if not bench_close.empty else None)

# ---------------------------------------------------------------------------
# HEADER + SNAPSHOT
# ---------------------------------------------------------------------------

name = profile.name if profile.name != "N/A" else ticker_input
st.markdown(
    f'<div class="qm-title">{mi("monitoring", 30)}&nbsp; {name}</div>',
    unsafe_allow_html=True,
)
st.markdown(
    f'<div class="qm-subtitle"><b>{ticker_input}</b> · {profile.exchange} · {profile.country} · '
    f'{profile.currency} · {profile.quote_type} · {profile.market_state}</div>',
    unsafe_allow_html=True,
)

# The listing currency comes from Yahoo's actual instrument metadata. This is
# intentionally independent of domicile, which matters for ADRs such as Indian
# companies trading in USD.
if profile.currency != "N/A":
    st.caption(f"Listing currency: **{profile.currency}** · Economic domicile: **{profile.country}**")

last_dt = format_market_timestamp(snapshot.timestamp, profile.timezone)
rv_ratio = relative_volume(snapshot)
price_color = GREEN if snapshot.change_pct >= 0 else RED
price_sign = "+" if snapshot.change_pct > 0 else ""

st.markdown(
    f"""<div class="qm-price-hero">
        <div class="qm-price-hero-inner">
            <div>
                <div class="qm-price-label">Latest available price</div>
                <div class="qm-price-line">
                    <span class="qm-price-value">{format_price(snapshot.latest_price, profile.currency, 2)}</span>
                    <span class="qm-price-change" style="color:{price_color};">{format_price(snapshot.change, profile.currency, 2)} &nbsp; {price_sign}{pct(snapshot.change_pct)}</span>
                </div>
            </div>
            <div class="qm-price-meta">
                <div><b>Market timestamp</b><br>{last_dt}</div>
                <div><b>Price basis</b><br>{"Adjusted" if adjusted_prices else "Unadjusted"}</div>
            </div>
        </div>
    </div>""",
    unsafe_allow_html=True,
)

st.markdown(
    f"""<div class="qm-kpi-grid">
        <div class="qm-kpi"><div class="qm-kpi-label">Previous Close</div><div class="qm-kpi-value" style="color:{GREEN};">{format_price(snapshot.previous_close, profile.currency)}</div></div>
        <div class="qm-kpi"><div class="qm-kpi-label">Day High</div><div class="qm-kpi-value" style="color:{GREEN};">{format_price(snapshot.day_high, profile.currency)}</div></div>
        <div class="qm-kpi"><div class="qm-kpi-label">Day Low</div><div class="qm-kpi-value" style="color:{RED};">{format_price(snapshot.day_low, profile.currency)}</div></div>
        <div class="qm-kpi"><div class="qm-kpi-label">52W High</div><div class="qm-kpi-value" style="color:{GREEN};">{format_price(snapshot.week52_high, profile.currency)}</div></div>
    </div>
    <div class="qm-kpi-grid qm-kpi-grid-3">
        <div class="qm-kpi"><div class="qm-kpi-label">52W Low</div><div class="qm-kpi-value" style="color:{RED};">{format_price(snapshot.week52_low, profile.currency)}</div></div>
        <div class="qm-kpi"><div class="qm-kpi-label">Relative Volume</div><div class="qm-kpi-value" style="color:{RED if np.isfinite(rv_ratio) and rv_ratio > 1.5 else TEXT};">{num(rv_ratio)}</div></div>
        <div class="qm-kpi"><div class="qm-kpi-label">Market Cap</div><div class="qm-kpi-value" style="color:{GREEN};">{format_money(snapshot.market_cap, profile.currency)}</div></div>
    </div>""",
    unsafe_allow_html=True,
)

dist_high = snapshot.latest_price / snapshot.week52_high - 1 if np.isfinite(snapshot.latest_price) and np.isfinite(snapshot.week52_high) and snapshot.week52_high > 0 else np.nan
dist_low = snapshot.latest_price / snapshot.week52_low - 1 if np.isfinite(snapshot.latest_price) and np.isfinite(snapshot.week52_low) and snapshot.week52_low > 0 else np.nan
wcols = st.columns(2)
wcols[0].metric("Distance from 52W High", pct(dist_high))
wcols[1].metric("Distance from 52W Low", pct(dist_low))

# ---------------------------------------------------------------------------
# DECISION DASHBOARD
# ---------------------------------------------------------------------------

st.markdown(
    f"<div class='qm-subtitle'>Research strategy: <b>{strategy_label}</b></div>",
    unsafe_allow_html=True,
)

decision_cols = st.columns(5)
decision_cols[0].metric("Trend", regime["trend"])
decision_cols[1].metric("Momentum", regime["momentum"])
decision_cols[2].metric("Volatility", regime["volatility"])
decision_cols[3].metric("Relative Strength", rs_bucket)
decision_cols[4].metric("Risk", regime["risk"])

positives = []
risks = []
if np.isfinite(regime.get("price_vs_sma200", np.nan)) and regime["price_vs_sma200"] > 0:
    positives.append(f"Price is {regime['price_vs_sma200'] * 100:.1f}% above SMA 200.")
if np.isfinite(regime.get("sma50_vs_sma200", np.nan)) and regime["sma50_vs_sma200"] > 0:
    positives.append(f"SMA 50 is {regime['sma50_vs_sma200'] * 100:.1f}% above SMA 200.")
if np.isfinite(regime.get("momentum60", np.nan)) and regime["momentum60"] > 0:
    positives.append(f"60D momentum is {regime['momentum60'] * 100:.1f}%.")
if np.isfinite(rs1m):
    positives.append(f"1M relative-strength change is {rs1m * 100:+.1f}% vs {benchmark_ticker}.")
if np.isfinite(regime.get("rsi", np.nan)) and regime["rsi"] >= 70:
    risks.append(f"RSI is {regime['rsi']:.1f}, an elevated momentum reading.")
if np.isfinite(snapshot.latest_price) and np.isfinite(snapshot.week52_high) and snapshot.week52_high > 0:
    dist_high = snapshot.latest_price / snapshot.week52_high - 1
    if dist_high > -0.02:
        risks.append(f"Price is within {abs(dist_high) * 100:.1f}% of its 52-week high.")
rv20 = _safe_float(tech["RV20"].iloc[-1]) if "RV20" in tech and len(tech) else np.nan
rv60 = _safe_float(tech["RV60"].iloc[-1]) if "RV60" in tech and len(tech) else np.nan
if np.isfinite(rv20) and np.isfinite(rv60) and rv60 > 0 and rv20 > rv60 * 1.35:
    risks.append(f"20D realized volatility ({rv20 * 100:.1f}%) is >35% above 60D volatility ({rv60 * 100:.1f}%).")
dd_asset = drawdown_stats_from_close(close)
if np.isfinite(dd_asset["current"]) and dd_asset["current"] < -0.10:
    risks.append(f"Current drawdown is {dd_asset['current'] * 100:.1f}% from the high-water mark.")

with st.expander("Decision Dashboard — Evidence, Risk Flags & Decision Context", expanded=True):
    dc1, dc2, dc3 = st.columns(3)
    with dc1:
        st.markdown("**Evidence**")
        for item in positives[:6] or ["No strong positive evidence detected from the current rule set."]:
            st.write("• " + item)
    with dc2:
        st.markdown("**Risk Flags**")
        for item in risks[:6] or ["No high-priority quantitative risk flag triggered by the current thresholds."]:
            st.write("• " + item)
    with dc3:
        st.markdown("**Decision Context**")
        st.write(
            f"The quantitative evidence is currently more consistent with a **{regime['trend'].lower()} trend regime** "
            f"with **{regime['momentum'].lower()} momentum** and **{regime['volatility'].lower()} volatility**. "
            "This is an evidence summary, not a prediction or investment recommendation."
        )
        st.caption("Primary regime inputs: price-vs-MA alignment, RSI, ADX, 60D momentum and realized volatility.")

# ---------------------------------------------------------------------------
# MARKET INTELLIGENCE TABS
# ---------------------------------------------------------------------------

tabs = st.tabs([
    "Price & Technical", "Crossover Intelligence", "Performance", "Risk",
    "Relative Strength", "Strategy Research", "Future Projections", "Fundamentals", "Data Quality"
])

# PRICE & TECHNICAL
with tabs[0]:
    if chart_history.empty:
        st.warning(f"Chart data unavailable. {chart_note}", icon=":material/warning:")
    else:
        chart_tech = technical_frame(chart_history)
        fig = chart_price_technical(
            chart_tech, profile.currency, show_candles, ma_selection, indicator_selection
        )
        st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False, "scrollZoom": True})

    st.markdown("### Benchmark Context")
    bench_close_clean = benchmark_history["Close"].dropna() if not benchmark_history.empty and "Close" in benchmark_history else pd.Series(dtype=float)
    bench_last = _safe_float(bench_close_clean.iloc[-1]) if not bench_close_clean.empty else np.nan
    bench_1m = period_return(bench_close_clean, 21) if len(bench_close_clean) >= 2 else np.nan
    bench_1y = period_return(bench_close_clean, 252) if len(bench_close_clean) >= 2 else np.nan
    bench_name_display = benchmark_profile.name if benchmark_profile.name != "N/A" else benchmark_ticker
    bench_currency = benchmark_profile.currency if benchmark_profile.currency != "N/A" else profile.currency
    bench_context_cols = st.columns(4)
    bench_context_cols[0].markdown(
        f"<div class='qm-kpi'><div class='qm-kpi-label'>Benchmark</div>"
        f"<div class='qm-kpi-value' style='color:{CYAN};font-size:1.15rem;'>{benchmark_ticker}</div>"
        f"<div style='color:{TEXT_MUTED};font-size:.78rem;margin-top:5px;'>{bench_name_display}</div></div>",
        unsafe_allow_html=True,
    )
    bench_context_cols[1].markdown(
        f"<div class='qm-kpi'><div class='qm-kpi-label'>Benchmark Price</div>"
        f"<div class='qm-kpi-value'>{format_price(bench_last, bench_currency)}</div>"
        f"<div style='color:{TEXT_MUTED};font-size:.78rem;margin-top:5px;'>{benchmark_profile.exchange} · {bench_currency}</div></div>",
        unsafe_allow_html=True,
    )
    bench_context_cols[2].metric("Benchmark 1M", pct(bench_1m))
    bench_context_cols[3].metric("Benchmark 1Y", pct(bench_1y))
    st.caption(
        f"Relative-performance reference: {ticker_input} vs {bench_name_display} ({benchmark_ticker}). "
        f"Benchmark listing currency: {bench_currency}. Returns are compared on a percentage basis; price levels are not mixed across currencies."
    )

    st.markdown("### Current technical state")
    if not tech.empty:
        last = tech.iloc[-1]
        ma_rows = []
        for label, col in [
            ("SMA 20", "SMA20"), ("SMA 50", "SMA50"), ("SMA 100", "SMA100"), ("SMA 200", "SMA200"),
            ("EMA 20", "EMA20"), ("EMA 50", "EMA50"), ("EMA 200", "EMA200"),
        ]:
            value = _safe_float(last.get(col))
            dist = distance_from(_safe_float(last["Close"]), value)
            ma_rows.append({
                "Moving Average": label,
                "Value": value,
                "Distance from Price (%)": dist * 100 if np.isfinite(dist) else np.nan,
                "Relationship": classify_proximity(dist * 100 if np.isfinite(dist) else np.nan),
            })
        ma_df = pd.DataFrame(ma_rows)
        ma_df["Value"] = ma_df["Value"].map(lambda x: format_price(x, profile.currency))
        ma_df["Distance from Price (%)"] = ma_df["Distance from Price (%)"].map(lambda x: f"{x:+.2f}%" if pd.notna(x) else "N/A")
        st.dataframe(ma_df, use_container_width=True, hide_index=True)

        tech_cols = st.columns(3)
        tech_cols[0].metric("RSI 14", num(_safe_float(last.get("RSI14"))))
        tech_cols[1].metric("MACD Hist", num(_safe_float(last.get("MACDHist")), 4))
        tech_cols[2].metric("ATR 14", format_price(_safe_float(last.get("ATR14")), profile.currency))
        tech_cols2 = st.columns(3)
        tech_cols2[0].metric("20D Realized Vol", pct(_safe_float(last.get("RV20"))))
        tech_cols2[1].metric("ADX 14", num(_safe_float(last.get("ADX14"))))
        tech_cols2[2].metric("ROC 20D", pct(_safe_float(last.get("ROC20")) / 100 if np.isfinite(_safe_float(last.get("ROC20"))) else np.nan))
        st.caption("MA distance = (Price / MA − 1). 'Near' uses a ±1% threshold. Indicator calculations use daily adjusted/unadjusted data selected in the sidebar.")

    levels = support_resistance_levels(close, _safe_float(tech["ATR14"].iloc[-1]) if "ATR14" in tech and not tech.empty else np.nan)
    st.markdown("### Analytical reference levels")
    lvl1, lvl2 = st.columns(2)
    with lvl1:
        st.write("**Potential support zones**")
        for x in levels["support"][:5]:
            st.write(f"• {format_price(x, profile.currency)}")
    with lvl2:
        st.write("**Potential resistance zones**")
        for x in levels["resistance"][:5]:
            st.write(f"• {format_price(x, profile.currency)}")
    st.caption("Levels are local historical reference points, not guaranteed support or resistance.")

# CROSSOVER
with tabs[1]:
    status_df, events_df = crossover_intelligence(tech)
    stats_df = crossover_summary_stats(events_df)
    if status_df.empty:
        st.info("Not enough price history for crossover analysis.", icon=":material/info:")
    else:
        st.markdown("### Crossover status")
        display_status = status_df.copy()
        for col in ["Fast MA", "Slow MA"]:
            if col in display_status:
                display_status[col] = display_status[col].map(lambda x: format_price(x, profile.currency))
        if "Return Since Cross" in display_status:
            display_status["Return Since Cross"] = display_status["Return Since Cross"].map(lambda x: pct(x))
        st.dataframe(display_status.drop(columns=["_pos"], errors="ignore"), use_container_width=True, hide_index=True)

        st.plotly_chart(
            chart_crossover_intelligence(tech, events_df, profile.currency),
            use_container_width=True, config={"displaylogo": False, "scrollZoom": True}
        )

        if not events_df.empty:
            bullish_count = int((events_df["State"] == "Bullish Cross").sum()) if "State" in events_df else 0
            bearish_count = int((events_df["State"] == "Bearish Cross").sum()) if "State" in events_df else 0
            golden_count = int(((events_df.get("Signal", pd.Series(dtype=str)) == "SMA 50/200") & (events_df.get("State", pd.Series(dtype=str)) == "Bullish Cross")).sum()) if not events_df.empty else 0
            death_count = int(((events_df.get("Signal", pd.Series(dtype=str)) == "SMA 50/200") & (events_df.get("State", pd.Series(dtype=str)) == "Bearish Cross")).sum()) if not events_df.empty else 0
            st.markdown(
            f"""
            <div class="qm-chart-key">
                <span class="qm-chart-key-title">Signal Key&nbsp;:</span>
                <span style="color:{GREEN};">▲&nbsp;&nbsp; Bull <small>({bullish_count})</small></span>
                <span style="color:{GREEN};">★&nbsp;&nbsp; Golden <small>({golden_count})</small></span>
                <span style="color:{RED};">▼&nbsp;&nbsp; Bear <small>({bearish_count})</small></span>
                <span style="color:{RED};">✕&nbsp;&nbsp; Death <small>({death_count})</small></span>
                <span style="color:{CYAN};">●&nbsp;&nbsp; Current</span>
            </div>
            """,
            unsafe_allow_html=True,
        )


            show_cols = ["Signal", "State", "Date", "Days Since Cross", "Price at Cross", "Current Price",
                         "Return Since Cross", "Previous Fast MA", "Previous Slow MA", "Fast MA", "Slow MA",
                         "5D Return", "20D Return", "60D Return", "120D Return",
                         "Max Forward Gain", "Max Forward Drawdown"]
            event_display = events_df[show_cols].copy()
            for col in ["Price at Cross", "Current Price", "Previous Fast MA", "Previous Slow MA", "Fast MA", "Slow MA"]:
                event_display[col] = event_display[col].map(lambda x: format_price(x, profile.currency))
            for col in [c for c in show_cols if "Return" in c or "Gain" in c or "Drawdown" in c]:
                if col in event_display:
                    event_display[col] = event_display[col].map(pct)
            st.markdown("<div style='padding-top: 20px;'></div>", unsafe_allow_html=True)
            st.markdown("### Historical crossover events")
            st.dataframe(event_display.sort_values("Date", ascending=False), use_container_width=True, hide_index=True)

            st.markdown("### Crossover outcome statistics")
            stats_display = stats_df.copy()
            for col in ["Avg 60D Return", "Median 60D Return", "Win Rate", "Worst 60D", "Best 60D"]:
                if col in stats_display:
                    if col == "Win Rate":
                        stats_display[col] = stats_display[col].map(pct)
                    else:
                        stats_display[col] = stats_display[col].map(pct)
            st.dataframe(stats_display, use_container_width=True, hide_index=True)
            st.caption("Golden Cross = SMA 50 crossing above SMA 200; Death Cross = SMA 50 crossing below SMA 200. Forward returns are only shown where enough future data exists.")
        else:
            st.info("No historical crossover events were detected in the loaded history.", icon=":material/info:")

# PERFORMANCE
with tabs[2]:
    st.markdown("### Monthly returns heatmap")
    monthly_asset_engine = asset_engine.monthly_returns_table()
    if monthly_asset_engine.empty:
        st.info("Monthly return heatmap is unavailable because the loaded history does not yet contain enough observations.", icon=":material/info:")
    else:
        st.plotly_chart(
            chart_monthly_heatmap(monthly_asset_engine),
            use_container_width=True,
            config={"displaylogo": False, "scrollZoom": True},
        )
        st.caption("Monthly compounded returns from the currently selected asset price series. Year Total is shown in the final column.")

    perf_rows = []
    for label, days in [("1D", 1), ("5D", 5), ("1M", 21), ("3M", 63), ("6M", 126), ("1Y", 252)]:
        value = period_return(close, days)
        bench = period_return(bench_close, days) if not bench_close.empty else np.nan
        perf_rows.append({"Period": label, "Asset": value, "Benchmark": bench, "Excess": value - bench if np.isfinite(value) and np.isfinite(bench) else np.nan})
    ytd = ytd_return(close)
    bench_ytd = ytd_return(bench_close) if not bench_close.empty else np.nan
    perf_rows.append({"Period": "YTD", "Asset": ytd, "Benchmark": bench_ytd, "Excess": ytd - bench_ytd if np.isfinite(ytd) and np.isfinite(bench_ytd) else np.nan})
    perf_rows.extend([
        {"Period": "3Y CAGR", "Asset": cagr_from_series(close, 3), "Benchmark": cagr_from_series(bench_close, 3) if not bench_close.empty else np.nan, "Excess": np.nan},
        {"Period": "5Y CAGR", "Asset": cagr_from_series(close, 5), "Benchmark": cagr_from_series(bench_close, 5) if not bench_close.empty else np.nan, "Excess": np.nan},
    ])
    perf_df = pd.DataFrame(perf_rows)
    perf_df["Excess"] = perf_df["Asset"] - perf_df["Benchmark"]
    for col in ["Asset", "Benchmark", "Excess"]:
        perf_df[col] = perf_df[col].map(pct)
    st.markdown("### Return comparison")
    st.dataframe(perf_df, use_container_width=True, hide_index=True)

    annual_return = (1 + asset_returns.mean()) ** TRADING_DAYS - 1 if len(asset_returns) else np.nan
    ann_vol = asset_returns.std(ddof=1) * np.sqrt(TRADING_DAYS) if len(asset_returns) > 1 else np.nan
    sharpe = (annual_return - risk_free_annual) / (ann_vol + EPS) if np.isfinite(ann_vol) else np.nan
    downside = np.minimum(asset_returns - risk_free_annual / TRADING_DAYS, 0.0)
    sortino = (annual_return - risk_free_annual) / (np.sqrt((downside ** 2).mean() * TRADING_DAYS) + EPS) if len(asset_returns) else np.nan
    calmar = annual_return / (abs(dd_asset["max"]) + EPS) if np.isfinite(dd_asset["max"]) else np.nan
    best_month = close.resample("ME").last().pct_change().max() if len(close) > 30 else np.nan
    worst_month = close.resample("ME").last().pct_change().min() if len(close) > 30 else np.nan
    perf_kpi = st.columns(4)

    perf_kpi[0].metric("Annualized Return", pct(annual_return))
    perf_kpi[1].metric("Ann. Volatility", pct(ann_vol))
    perf_kpi[2].metric("Sharpe", num(sharpe))
    perf_kpi[3].metric("Sortino", num(sortino))
    perf_kpi2 = st.columns(4)
    perf_kpi2[0].metric("Calmar", num(calmar))
    perf_kpi2[1].metric("Current DD", pct(dd_asset["current"]))
    perf_kpi2[2].metric("Best Day", pct(asset_returns.max()))
    perf_kpi2[3].metric("Worst Day", pct(asset_returns.min()))

    if not bench_returns.empty:
        bench_vol = bench_returns.std(ddof=1) * np.sqrt(TRADING_DAYS) if len(bench_returns) > 1 else np.nan
        bench_wealth = (1 + bench_returns).cumprod()
        bench_dd = (bench_wealth / bench_wealth.cummax() - 1).min() if len(bench_wealth) else np.nan
        bench_ann = (1 + bench_returns.mean()) ** TRADING_DAYS - 1 if len(bench_returns) else np.nan
        bench_sharpe = (bench_ann - risk_free_annual) / (bench_vol + EPS) if np.isfinite(bench_vol) else np.nan
        st.markdown("### Risk-adjusted comparison")
        bench_compare = pd.DataFrame([
            ("Annualized return", pct(annual_return), pct(bench_ann)),
            ("Annualized volatility", pct(ann_vol), pct(bench_vol)),
            ("Sharpe ratio", num(sharpe), num(bench_sharpe)),
            ("Maximum drawdown", pct(dd_asset["max"]), pct(bench_dd)),
        ], columns=["Metric", ticker_input, benchmark_ticker])
        st.dataframe(bench_compare, use_container_width=True, hide_index=True)
    mcols = st.columns(4)
    mcols[0].metric("Best Month", pct(best_month))
    mcols[1].metric("Worst Month", pct(worst_month))
    mcols[2].metric("Positive Days", pct((asset_returns > 0).mean()))
    mcols[3].metric("Profit Factor", num(asset_returns[asset_returns > 0].sum() / (abs(asset_returns[asset_returns < 0].sum()) + EPS)))

    st.markdown("### Monthly / yearly performance summary")
    monthly_asset = close.resample("ME").last().pct_change()
    positive_month_pct = float((monthly_asset.dropna() > 0).mean()) if len(monthly_asset.dropna()) else np.nan
    yearly_asset = close.resample("YE").last().pct_change().dropna()
    positive_year_pct = float((yearly_asset > 0).mean()) if len(yearly_asset) else np.nan
    st.caption(f"Positive months: {pct(positive_month_pct)} · Positive years: {pct(positive_year_pct)}")

# RISK
with tabs[3]:
    r = asset_returns
    risk_metrics = {
        "Daily Volatility": r.std(ddof=1),
        "Weekly Volatility": r.std(ddof=1) * np.sqrt(5),
        "Monthly Volatility": r.std(ddof=1) * np.sqrt(21),
        "Annualized Volatility": ann_vol,
        "20D Realized Volatility": realized_volatility(r, 20),
        "60D Realized Volatility": realized_volatility(r, 60),
        "252D Realized Volatility": realized_volatility(r, 252),
        "Historical VaR 95%": r.quantile(0.05),
        "Historical VaR 99%": r.quantile(0.01),
        "CVaR 95%": r[r <= r.quantile(0.05)].mean(),
        "CVaR 99%": r[r <= r.quantile(0.01)].mean(),
        "Current Drawdown": dd_asset["current"],
        "Maximum Drawdown": dd_asset["max"],
        "Average Drawdown": dd_asset["average"],
        "Skewness": stats.skew(r) if len(r) > 2 else np.nan,
        "Excess Kurtosis": stats.kurtosis(r, fisher=True) if len(r) > 2 else np.nan,
    }
    risk_k1, risk_k2, risk_k3, risk_k4 = st.columns(4)
    risk_k1.metric("20D Realized Vol", pct(risk_metrics["20D Realized Volatility"]))
    risk_k2.metric("60D Realized Vol", pct(risk_metrics["60D Realized Volatility"]))
    risk_k3.metric("VaR 95%", pct(risk_metrics["Historical VaR 95%"]))
    risk_k4.metric("CVaR 95%", pct(risk_metrics["CVaR 95%"]))

    risk_df = pd.DataFrame(list(risk_metrics.items()), columns=["Metric", "Value"])
    risk_df["Value"] = risk_df.apply(
        lambda row: pct(row["Value"]) if "Volatility" in row["Metric"] or "VaR" in row["Metric"] or "CVaR" in row["Metric"] or "Drawdown" in row["Metric"]
        else num(row["Value"]), axis=1
    )
    st.dataframe(risk_df, use_container_width=True, hide_index=True)

    st.plotly_chart(chart_detailed_drawdown(close), use_container_width=True)
    st.markdown("### Maximum drawdown lifecycle")
    dd_lifecycle = pd.DataFrame([{
        "Peak Date": dd_asset["peak_date"].date() if dd_asset["peak_date"] is not None else "N/A",
        "Trough Date": dd_asset["trough_date"].date() if dd_asset["trough_date"] is not None else "N/A",
        "Recovery Date": dd_asset["recovery_date"].date() if dd_asset["recovery_date"] is not None else "Ongoing / N/A",
        "Max Drawdown": pct(dd_asset["max"]),
        "Current Drawdown": pct(dd_asset["current"]),
        "Recovery Duration (days)": num(dd_asset["recovery_days"], 0),
    }])
    st.dataframe(dd_lifecycle, use_container_width=True, hide_index=True)

    st.markdown("### Top 5 drawdown periods")
    # Reuse the single current-asset engine created before the tabs.
    st.dataframe(asset_engine.worst_drawdown_periods(5), use_container_width=True, hide_index=True)

    st.plotly_chart(chart_return_distribution_detailed(r, asset_engine), use_container_width=True)
    fat_tail = "Elevated" if abs(risk_metrics["Excess Kurtosis"]) > 3 else "Normal-ish"
    asymmetry = "Negative" if abs(r[r < 0].mean()) > abs(r[r > 0].mean()) else "Positive"
    st.info(
        f"Distribution diagnostics: skew {num(risk_metrics['Skewness'])}, excess kurtosis {num(risk_metrics['Excess Kurtosis'])}, "
        f"fat-tail indicator = **{fat_tail}**, return asymmetry = **{asymmetry}**.",
        icon=":material/analytics:",
    )

    st.markdown("### Rolling analytics")
    roll_c1, roll_c2 = st.columns(2)
    with roll_c1:
        rolling_metric = st.selectbox(
            "Rolling measure",
            ["Rolling Return", "Rolling Volatility", "Rolling Sharpe", "Rolling Beta",
             "Rolling Correlation", "Rolling Maximum Drawdown", "Rolling RSI", "Rolling Momentum"],
            key="rolling_metric",
        )
    with roll_c2:
        rolling_window = st.selectbox("Window (trading days)", ROLLING_WINDOWS, index=1, key="rolling_window")
    st.plotly_chart(
        chart_rolling_analytics(asset_returns, bench_returns if not bench_returns.empty else None,
                                rolling_metric, rolling_window),
        use_container_width=True, config={"responsive": True},
    )

    st.markdown("### Tail distribution statistics")
    tail_tbl = pd.DataFrame([
        ("Mean", pct(r.mean())),
        ("Median", pct(r.median())),
        ("Standard deviation", pct(r.std(ddof=1))),
        ("VaR 95%", pct(r.quantile(0.05))),
        ("VaR 99%", pct(r.quantile(0.01))),
        ("CVaR 95%", pct(r[r <= r.quantile(0.05)].mean())),
        ("CVaR 99%", pct(r[r <= r.quantile(0.01)].mean())),
        ("Left-tail observations", num((r <= r.quantile(0.01)).sum(), 0)),
        ("Right-tail observations", num((r >= r.quantile(0.99)).sum(), 0)),
    ], columns=["Measure", "Value"])
    st.dataframe(tail_tbl, use_container_width=True, hide_index=True)

# RELATIVE STRENGTH
with tabs[4]:
    if bench_close.empty:
        st.warning(f"Benchmark {benchmark_ticker} unavailable. Relative analytics are disabled for this view.", icon=":material/warning:")
    else:

        rs_cols = st.columns(4)
        rs_cols[0].metric("Asset Return", pct(bench_stats["asset_return"]))
        rs_cols[1].metric("Benchmark Return", pct(bench_stats["benchmark_return"]))
        rs_cols[2].metric("Excess Return", pct(bench_stats["excess_return"]))
        rs_cols[3].metric("Beta", num(bench_stats["beta"]))
        rs_cols2 = st.columns(4)
        rs_cols2[0].metric("Alpha", pct(bench_stats["alpha"]))
        rs_cols2[1].metric("Correlation", num(bench_stats["correlation"]))
        rs_cols2[2].metric("Information Ratio", num(bench_stats["information_ratio"]))
        rs_cols2[3].metric("Up / Down Capture", f"{num(bench_stats['up_capture'])} / {num(bench_stats['down_capture'])}")

        st.plotly_chart(
            chart_relative_strength(close, bench_close, ticker_input, benchmark_ticker),
            use_container_width=True, config={"responsive": True},
        )
        rs_tbl = pd.DataFrame([{
            "Measure": "Current relative strength ratio",
            "Value": num(bench_stats["relative_strength"]),
        }, {
            "Measure": "20D change",
            "Value": pct(bench_stats["rs_20d"]),
        }, {
            "Measure": "60D change",
            "Value": pct(bench_stats["rs_60d"]),
        }, {
            "Measure": "1Y change",
            "Value": pct(bench_stats["rs_1y"]),
        }])
        st.dataframe(rs_tbl, use_container_width=True, hide_index=True)
        st.caption(f"Relative-strength classification uses 1M/3M change: >+2% = outperforming, <−2% = underperforming, otherwise neutral.")

        st.markdown("### Cross-asset context")
        is_india = profile.country == "India" or profile.currency == "INR"
        context_tickers = tuple(x for x in cross_asset_context(ticker_input, is_india) if x != ticker_input)
        context_prices = download_cross_asset_prices(context_tickers, "1y")
        if not context_prices.empty:
            context_rows = []
            for symbol in context_tickers:
                if symbol not in context_prices:
                    continue
                s = context_prices[symbol].dropna()
                if len(s) < 2:
                    continue
                row = {"Ticker": symbol}
                for label, d in [("1D", 1), ("1M", 21), ("3M", 63), ("1Y", 252)]:
                    row[label] = period_return(s, d)
                row["Corr vs Asset"] = float(s.pct_change().corr(asset_returns.reindex(s.index))) if len(s) > 20 else np.nan
                context_rows.append(row)
            context_df = pd.DataFrame(context_rows)
            if not context_df.empty:
                context_display = context_df.copy()
                for col in context_display.columns:
                    if col == "Ticker":
                        continue
                    context_display[col] = context_display[col].map(
                        lambda x: num(x) if col == "Corr vs Asset" else pct(x)
                    )
                st.dataframe(context_display, use_container_width=True, hide_index=True)
                st.plotly_chart(chart_cross_asset_matrix(context_df.drop(columns=["Corr vs Asset"], errors="ignore").assign(
                    **{c: pd.to_numeric(context_df[c], errors="coerce") * 100 for c in context_df.columns if c not in {"Ticker","Corr vs Asset"}}
                )), use_container_width=True, config={"responsive": True})

# FUTURE PROJECTIONS
with tabs[6]:
    st.markdown("### Future Price Scenarios")
    st.info(
        "A statistical scenario engine, not a forecast. Future paths are generated by resampling recent realized daily returns. "
        "Use the low/base/high bands to understand dispersion and downside sensitivity rather than treating them as target prices.",
        icon=":material/science:",
    )
    proj_c1, proj_c2, proj_c3 = st.columns(3)
    with proj_c1:
        projection_horizon = st.selectbox(
            "Projection horizon (trading days)", [20, 60, 120, 252], index=1, key="projection_horizon"
        )
    with proj_c2:
        projection_sims = st.selectbox(
            "Monte Carlo simulations", [500, 1000, 2000, 5000], index=2, key="projection_sims"
        )
    with proj_c3:
        projection_seed = st.number_input("Random seed", min_value=1, max_value=999999, value=7, step=1, key="projection_seed")
    show_sample_paths = st.checkbox(
        "Show 4 representative simulated paths", value=False, key="show_sample_paths",
        help="Off by default to keep the projection readable. Percentile scenarios remain visible."
    )

    if len(asset_returns) < 20 or not np.isfinite(snapshot.latest_price):
        st.info("At least 20 clean daily returns and a valid latest price are required for the projection module.", icon=":material/info:")
    else:
        projection = monte_carlo_projection(
            asset_returns, close.index[-1], snapshot.latest_price,
            horizon_days=int(projection_horizon),
            n_sims=int(projection_sims),
            seed=int(projection_seed),
            lookback=252,
        )
        st.plotly_chart(
            chart_future_projection(
                projection, profile.currency, snapshot.latest_price, show_sample_paths=show_sample_paths
            ),
            use_container_width=True,
            config={"displaylogo": False, "scrollZoom": True},
        )

        final_p05 = float(projection["p05"][-1])
        final_p10 = float(projection["p10"][-1])
        final_p50 = float(projection["p50"][-1])
        final_p90 = float(projection["p90"][-1])
        path_low = float(np.min(projection["paths"]))
        path_high = float(np.max(projection["paths"]))

        proj_metrics = st.columns(4)
        proj_metrics[0].metric("Latest Price", format_price(snapshot.latest_price, profile.currency))
        proj_metrics[1].metric("Base / P50 End", format_price(final_p50, profile.currency))
        proj_metrics[2].metric("Low / P10 End", format_price(final_p10, profile.currency))
        proj_metrics[3].metric("High / P90 End", format_price(final_p90, profile.currency))

        proj_metrics2 = st.columns(4)
        proj_metrics2[0].metric("Stress / P05 End", format_price(final_p05, profile.currency))
        proj_metrics2[1].metric("Scenario Min", format_price(path_low, profile.currency))
        proj_metrics2[2].metric("Scenario Max", format_price(path_high, profile.currency))
        proj_metrics2[3].metric("P10–P90 Width", format_price(final_p90 - final_p10, profile.currency))

        st.markdown("### Day-by-day scenario levels")
        projection_table = pd.DataFrame({
            "Day": np.arange(1, len(projection["dates"]) + 1),
            "Date": projection["dates"].date,
            "Stress P05": projection["p05"],
            "Low P10": projection["p10"],
            "Base P50": projection["p50"],
            "High P90": projection["p90"],
            "Stress P95": projection["p95"],
        })
        for col in ["Stress P05", "Low P10", "Base P50", "High P90", "Stress P95"]:
            projection_table[col] = projection_table[col].map(
                lambda x: format_price(x, profile.currency)
            )
        st.dataframe(
            projection_table, use_container_width=True, hide_index=True, height=320
        )

        st.markdown("### Scenario methodology")
        st.write(
            f"{projection_sims:,} paths × {projection_horizon} trading days. "
            "Each future day samples one realized return from the most recent 252 available observations, "
            "with replacement, and compounds from the latest observed price. "
            "This retains empirical skew and tail events better than a single Gaussian path, "
            "but it still assumes the recent return distribution is informative for scenario construction. "
            "It is not a prediction of the actual future path."
        )

# STRATEGY RESEARCH
with tabs[5]:
    st.markdown("### Strategy Definition & Execution")
    if mode == "Benchmark Strategy Simulator" and strategy_type in STRATEGY_CATALOG:
        strategy_meta = STRATEGY_CATALOG[strategy_type]
        cat = strategy_meta["category"]
        param_summary = ", ".join(
            f"{k.replace('_', ' ').title()} = {v}" for k, v in user_params.items()
        ) if user_params else "Default parameters"
        exec_model = "Long/Short" if allow_short else "Long-only / flat"
        strategy_info_cols = st.columns(3)
        strategy_info_cols[0].markdown(
            f"<div class='qm-kpi'><div class='qm-kpi-label'>Selected Strategy</div>"
            f"<div class='qm-kpi-value' style='font-size:1.15rem;color:{CYAN};'>{strategy_type}</div>"
            f"<div style='color:{TEXT_MUTED};font-size:.78rem;margin-top:5px;'>{cat}</div></div>",
            unsafe_allow_html=True,
        )
        strategy_info_cols[1].markdown(
            f"<div class='qm-kpi'><div class='qm-kpi-label'>Signal Logic</div>"
            f"<div style='font-size:.90rem;line-height:1.35;margin-top:6px;'>{strategy_meta['desc']}</div></div>",
            unsafe_allow_html=True,
        )
        strategy_info_cols[2].markdown(
            f"<div class='qm-kpi'><div class='qm-kpi-label'>Configuration</div>"
            f"<div style='font-family:JetBrains Mono,monospace;font-size:.86rem;line-height:1.5;margin-top:6px;'>{param_summary}</div>"
            f"<div style='color:{TEXT_MUTED};font-size:.75rem;margin-top:5px;'>Execution: {exec_model} · Signals use prior-known information</div></div>",
            unsafe_allow_html=True,
        )
        st.info(
            f"How to read this strategy: {strategy_meta['desc']} "
            "Performance below is backtested on historical observations; it is not a forward forecast.",
            icon=":material/strategy:",
        )
    else:
        custom_text = strategy_description or "Upload a return/NAV/trade-log CSV to define the strategy without assuming a signal model."
        st.info(f"Strategy definition: {custom_text}", icon=":material/description:")

    st.markdown("### Existing strategy engine")
    if returns_raw is None or len(returns_raw) < 5:
        st.info("Run the strategy simulator or upload a strategy CSV from the sidebar to populate strategy research.", icon=":material/info:")
    else:
        returns_raw = returns_raw.sort_index()
        min_date, max_date = returns_raw.index.min().date(), returns_raw.index.max().date()
        if min_date == max_date:
            date_range = (min_date, max_date)
        else:
            date_range = st.slider(
                "Strategy analysis date range",
                min_value=min_date, max_value=max_date, value=(min_date, max_date),
                format="YYYY-MM-DD", key="strategy_date_range",
            )
        mask = (returns_raw.index.date >= date_range[0]) & (returns_raw.index.date <= date_range[1])
        R = returns_raw.loc[mask]
        if len(R) < 5:
            st.warning("Not enough strategy observations in the selected range.", icon=":material/warning:")
        else:
            strategy_bench = bench_returns.reindex(R.index).dropna() if not bench_returns.empty else pd.Series(dtype=float)
            strategy_engine = RiskEngine(R=R, Rb=strategy_bench if len(strategy_bench) >= 5 else None, risk_free_annual=risk_free_annual)
            sm = strategy_engine.metrics
            # If we have the OHLC associated with a simulated strategy, compute true exposure/trade stats.
            trade_stats = None
            if strategy_ohlc is not None and not strategy_ohlc.empty:
                trade_stats = strategy_trade_stats(
                    strategy_type, R, strategy_ohlc["Close"], strategy_ohlc["High"], strategy_ohlc["Low"],
                    allow_short, user_params, capital
                )
            if trade_stats is None:
                # Uploaded return series: expose statistics that can be inferred from the return stream
                # and explicitly mark position-dependent fields as unavailable.
                trade_stats = {
                    "number_of_trades": np.nan, "average_trade": np.nan, "median_trade": np.nan,
                    "best_trade": np.nan, "worst_trade": np.nan, "avg_holding_days": np.nan,
                    "exposure_pct": np.nan, "long_exposure_pct": np.nan, "short_exposure_pct": np.nan,
                    "turnover": np.nan,
                }

            sk1, sk2, sk3 = st.columns(3)
            sk1.metric("CAGR", pct(sm["cagr"]))
            sk2.metric("Sharpe", num(sm["sharpe"]))
            sk3.metric("Sortino", num(sm["sortino"]))
            sk4, sk5, sk6 = st.columns(3)
            sk4.metric("Max DD", pct(sm["max_drawdown"]))
            sk5.metric("Win Rate", pct(sm["win_rate"] / 100))
            sk6.metric("Profit Factor", num(sm["profit_factor"]))

            strategy_bh_r = close.pct_change().reindex(R.index).dropna()
            strategy_bh_return = float((1 + strategy_bh_r).prod() - 1) if not strategy_bh_r.empty else np.nan
            strategy_metrics_df = pd.DataFrame([
                ("Strategy Return", sm["total_return"]),
                ("Buy & Hold Return", strategy_bh_return),
                ("CAGR", sm["cagr"]),
                ("Volatility", sm["ann_vol"]),
                ("Sharpe", sm["sharpe"]),
                ("Sortino", sm["sortino"]),
                ("Calmar", sm["calmar"]),
                ("Max Drawdown", sm["max_drawdown"]),
                ("Win Rate", sm["win_rate"] / 100),
                ("Profit Factor", sm["profit_factor"]),
                ("Number of Trades", trade_stats["number_of_trades"]),
                ("Average Trade", trade_stats["average_trade"]),
                ("Median Trade", trade_stats["median_trade"]),
                ("Best Trade", trade_stats["best_trade"]),
                ("Worst Trade", trade_stats["worst_trade"]),
                ("Average Holding (days)", trade_stats["avg_holding_days"]),
                ("Exposure %", trade_stats["exposure_pct"] / 100 if np.isfinite(trade_stats["exposure_pct"]) else np.nan),
                ("Long Exposure %", trade_stats["long_exposure_pct"] / 100 if np.isfinite(trade_stats["long_exposure_pct"]) else np.nan),
                ("Short Exposure %", trade_stats["short_exposure_pct"] / 100 if np.isfinite(trade_stats["short_exposure_pct"]) else np.nan),
                ("Turnover / day", trade_stats["turnover"]),
            ], columns=["Metric", "Value"])
            def _strategy_fmt(row):
                v = row["Value"]
                if "Return" in row["Metric"] or "%" in row["Metric"] or "Rate" in row["Metric"] or row["Metric"] in {"Volatility","Max Drawdown","Average Trade","Median Trade","Best Trade","Worst Trade"}:
                    return pct(v)
                return num(v)
            strategy_metrics_df["Value"] = strategy_metrics_df.apply(_strategy_fmt, axis=1)
            st.dataframe(strategy_metrics_df, use_container_width=True, hide_index=True)

            # Cost-aware performance when a position series is available.
            if strategy_ohlc is not None and strategy_type in STRATEGY_CATALOG:
                pos = strategy_trade_stats(strategy_type, R, strategy_ohlc["Close"], strategy_ohlc["High"], strategy_ohlc["Low"],
                                           allow_short, user_params)["position"]
                cost_returns = apply_transaction_costs(R, pos, total_transaction_cost_pct * 100)
                gross = (1 + R).prod() - 1
                net = (1 + cost_returns).prod() - 1
                st.info(
                    f"Gross strategy return = {pct(gross)}; estimated net return after configurable transaction-cost assumptions = "
                    f"{pct(net)}. Turnover is position-change based.",
                    icon=":material/payments:",
                )

            # Robustness analysis
            st.markdown("### Crossover robustness")
            if strategy_type in {"SMA Crossover", "EMA Crossover"} and strategy_ohlc is not None and not strategy_ohlc.empty:
                run_robustness = st.button("Run Parameter Robustness Grid", use_container_width=True, key="robustness_btn")
                if run_robustness:
                    base_close, base_high, base_low = strategy_ohlc["Close"], strategy_ohlc["High"], strategy_ohlc["Low"]
                    fast_grid = [20, 30, 40, 50, 60]
                    slow_grid = [100, 150, 200, 250]
                    cagr_grid = pd.DataFrame(index=fast_grid, columns=slow_grid, dtype=float)
                    sharpe_grid = cagr_grid.copy()
                    dd_grid = cagr_grid.copy()
                    for f in fast_grid:
                        for s in slow_grid:
                            if f >= s:
                                continue
                            tmp = (strat_sma_crossover if strategy_type == "SMA Crossover" else strat_ema_crossover)(
                                base_close, base_high, base_low, allow_short=allow_short, fast=f, slow=s
                            )
                            eng = RiskEngine(tmp.dropna())
                            cagr_grid.loc[f, s] = eng.metrics["cagr"]
                            sharpe_grid.loc[f, s] = eng.metrics["sharpe"]
                            dd_grid.loc[f, s] = eng.metrics["max_drawdown"]
                    rc1, rc2, rc3 = st.columns(3)
                    for col, grid, title, fmt in [
                        (rc1, cagr_grid, "CAGR", ".1%"),
                        (rc2, sharpe_grid, "Sharpe", ".2f"),
                        (rc3, dd_grid, "Max Drawdown", ".1%"),
                    ]:
                        with col:
                            fig = go.Figure(data=go.Heatmap(
                                z=grid.values, x=[str(x) for x in grid.columns], y=[str(x) for x in grid.index],
                                colorscale=[[0, RED], [0.5, CARD], [1, GREEN]], zmid=0,
                                colorbar=dict(title=title),
                            ))
                            _plotly_base(fig, f"Robustness: {title}", 360)
                            st.plotly_chart(fig, use_container_width=True)
                    st.warning(
                        "Parameter stability is a diagnostic, not proof of robustness. A narrow performance peak is a classic overfitting warning.",
                        icon=":material/warning:",
                    )
            else:
                st.caption("Robustness grid is enabled for SMA/EMA crossover strategies when simulated OHLC data is available.")

            st.markdown("### Position sizing calculator")
            current_price = snapshot.latest_price
            current_atr = _safe_float(tech["ATR14"].iloc[-1]) if "ATR14" in tech and not tech.empty else np.nan
            entry = entry_price if entry_price > 0 else current_price
            stop = stop_price
            risk_budget = capital * max_risk_pct / 100
            risk_per_share = abs(entry - stop) if entry > 0 and stop > 0 else np.nan
            risk_shares = risk_budget / risk_per_share if np.isfinite(risk_per_share) and risk_per_share > 0 else np.nan
            alloc_shares = (capital * allocation_cap_pct / 100) / entry if entry > 0 else np.nan
            max_shares = min(risk_shares, alloc_shares) if np.isfinite(risk_shares) and np.isfinite(alloc_shares) else np.nan
            position_value = max_shares * entry if np.isfinite(max_shares) else np.nan
            ps_df = pd.DataFrame([
                ("Risk budget", format_money(risk_budget, profile.currency)),
                ("ATR 14 reference", format_price(current_atr, profile.currency)),
                ("Risk per share", format_price(risk_per_share, profile.currency)),
                ("Risk-based shares", num(risk_shares, 0)),
                ("Allocation-based max shares", num(alloc_shares, 0)),
                ("Maximum shares", num(max_shares, 0)),
                ("Position value", format_money(position_value, profile.currency)),
                ("Portfolio allocation", pct(position_value / capital if capital > 0 and np.isfinite(position_value) else np.nan)),
            ], columns=["Measure", "Value"])
            st.dataframe(ps_df, use_container_width=True, hide_index=True)

# FUNDAMENTALS
with tabs[7]:
    quote_type = profile.quote_type.upper()
    st.markdown("### Instrument overview")
    overview = pd.DataFrame([
        ("Name", profile.name), ("Ticker", profile.ticker), ("Exchange", profile.exchange),
        ("Country", profile.country), ("Sector", profile.sector), ("Industry", profile.industry),
        ("Currency", profile.currency), ("Instrument Type", profile.quote_type),
        ("Market State", profile.market_state), ("Timezone", profile.timezone),
    ], columns=["Field", "Value"])
    st.dataframe(overview, use_container_width=True, hide_index=True)

    if quote_type in {"EQUITY", "COMMONSTOCK", "ADR"} or any(k in fundamentals for k in ["trailingPE", "forwardPE", "returnOnEquity"]):
        valuation_keys = [
            ("Market Capitalization", "marketCap", "money"),
            ("Enterprise Value", "enterpriseValue", "money"),
            ("Trailing P/E", "trailingPE", "number"),
            ("Forward P/E", "forwardPE", "number"),
            ("PEG", "pegRatio", "number"),
            ("Price / Book", "priceToBook", "number"),
            ("Price / Sales", "priceToSalesTrailing12Months", "number"),
            ("EV / EBITDA", "enterpriseToEbitda", "number"),
            ("Dividend Yield", "dividendYield", "pct_raw"),
            ("EPS", "trailingEps", "price"),
            ("Revenue", "totalRevenue", "money"),
            ("Gross Margin", "grossMargins", "pct_raw"),
            ("Operating Margin", "operatingMargins", "pct_raw"),
            ("Net Margin", "profitMargins", "pct_raw"),
            ("ROE", "returnOnEquity", "pct_raw"),
            ("ROA", "returnOnAssets", "pct_raw"),
            ("Revenue Growth", "revenueGrowth", "pct_raw"),
            ("Earnings Growth", "earningsGrowth", "pct_raw"),
            ("Quarterly Earnings Growth", "earningsQuarterlyGrowth", "pct_raw"),
            ("Debt / Equity", "debtToEquity", "number"),
            ("Current Ratio", "currentRatio", "number"),
            ("Payout Ratio", "payoutRatio", "pct_raw"),
            ("Dividend Rate", "dividendRate", "money"),
        ]
        val_rows = []
        for label, key, kind in valuation_keys:
            value = fundamentals.get(key, np.nan)
            if not _is_valid_number(value):
                display = "N/A"
            elif kind == "money":
                display = format_money(value, profile.currency)
            elif kind == "price":
                display = format_price(value, profile.currency)
            elif kind == "pct_raw":
                display = pct(_safe_float(value))
            else:
                display = num(value)
            val_rows.append((label, display))
        st.markdown("### Valuation, profitability & balance sheet")
        st.dataframe(pd.DataFrame(val_rows, columns=["Metric", "Value"]), use_container_width=True, hide_index=True)
    else:
        st.info("A conventional equity-fundamentals field set is not available for this instrument. Relevant instrument metadata is shown above.", icon=":material/info:")

    st.markdown("### Corporate actions")
    if actions.empty:
        st.info("No dividend/split action series was returned for this ticker.", icon=":material/info:")
    else:
        action_display = actions.tail(30).reset_index()
        action_display["Date"] = pd.to_datetime(action_display[action_display.columns[0]]).dt.date
        st.dataframe(action_display, use_container_width=True, hide_index=True)
    st.caption(
        f"Price mode: {'adjusted' if adjusted_prices else 'unadjusted'}. "
        "Do not mix adjusted and unadjusted series inside a single return calculation."
    )

    if calendar_data:
        with st.expander("Calendar fields"):
            st.json(calendar_data)

# DATA QUALITY
with tabs[8]:
    latest_hist = daily_history.index[-1] if not daily_history.empty else None
    missing = int(daily_history["Close"].isna().sum()) if "Close" in daily_history else 0
    quality = pd.DataFrame([
        ("Data Provider", "Yahoo Finance"),
        ("Data Status", "Live / latest response" if not daily_history.empty else "Data unavailable"),
        ("Latest Quote Timestamp", last_dt),
        ("Latest Daily Data", str(latest_hist.date()) if latest_hist is not None else "N/A"),
        ("Historical Start", str(daily_history.index.min().date()) if not daily_history.empty else "N/A"),
        ("Historical End", str(daily_history.index.max().date()) if not daily_history.empty else "N/A"),
        ("Observations", f"{len(daily_history):,}"),
        ("Missing Close Observations", f"{missing:,}"),
        ("Frequency", "Daily for analytics"),
        ("Currency", profile.currency),
        ("Exchange", profile.exchange),
        ("Adjusted Prices", "Yes" if adjusted_prices else "No"),
        ("Yahoo cache layer", "Streamlit cache, TTL 10–30 min depending dataset"),
        ("Chart request", f"{chart_period} / {chart_interval}"),
        ("Chart data note", chart_note),
        ("Benchmark", benchmark_ticker),
        ("Benchmark status", "Available" if not benchmark_history.empty else "Unavailable"),
    ], columns=["Data Quality Field", "Value"])
    st.dataframe(quality, use_container_width=True, hide_index=True)
    st.markdown("### Data source & integrity")
    st.write(
        "Market data is sourced from Yahoo Finance. "
        "The dashboard does not label delayed or last-session observations as live. "
        "Any offline synthetic series is restricted to the explicit sample-strategy fallback and is labeled when used."
    )

# ---------------------------------------------------------------------------
# APP FOOTER
st.caption(
    f"QuantMetrics Pro • {ticker_input} • Latest data: {last_dt} • "
    f"Benchmark: {benchmark_ticker} • Research / decision-support use only — not investment advice. "
    "Scenario projections are statistical simulations, not forecasts."
)
