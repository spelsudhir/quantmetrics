# QuantMetrics Pro

**Market Intelligence & Quantitative Risk Engine** — an institutional-style Streamlit terminal for backtest analytics, portfolio risk decomposition, and technical analysis, built on live Yahoo Finance data.

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![Streamlit](https://img.shields.io/badge/streamlit-app-FF4B4B)
![License](https://img.shields.io/badge/license-MIT-lightgrey)

---

## Overview

QuantMetrics Pro is a single-file Streamlit application that turns raw price history into a full quantitative research workflow. Point it at any Yahoo Finance ticker (equities, indices, ETFs, or international listings) and it will:

- Pull live OHLCV data, fundamentals, corporate actions, and quote metadata
- Compute an institutional-grade risk/performance metric library (Sharpe, Sortino, Calmar, VaR/CVaR, drawdown lifecycle, etc.)
- Run and backtest a catalog of rules-based trading strategies against your own uploaded return series or a live-data simulator
- Generate technical indicator dashboards, crossover signal intelligence, relative-strength analysis, and Monte Carlo price scenario projections
- Present everything in a dark, Bloomberg-terminal-inspired UI with responsive KPI cards and interactive Plotly charts

The app is designed for **research and decision support only** — it is explicitly not investment advice, and it says so throughout the UI.

---

## Features

### 📊 Market Intelligence
- Searchable instrument picker (curated universe of Indian, US, and international tickers) that also accepts any raw Yahoo Finance symbol
- Live price snapshot: last price, change %, day range, 52-week range, volume, relative volume, market cap
- Auto-detected benchmark selection based on the instrument's country/exchange, with 50+ curated indices/ETFs and a free-text override
- Multi-currency formatting (₹, $, £, €, ¥, and more) driven by the instrument's actual listing currency

### 📈 Price & Technical Dashboard
- Candlestick or line chart with configurable timeframe (1D–Max) and interval (1m–1mo)
- Overlay library: SMA/EMA (20/50/100/200), Bollinger Bands
- Indicator sub-panels: RSI(14), MACD(12,26,9), ATR(14), ADX(14), Stochastic, ROC(20), OBV
- Auto-detected support/resistance reference levels

### 🔀 Crossover Intelligence
- Tracks SMA 20/50, SMA 50/200 (Golden/Death Cross), EMA 12/26, and EMA 50/200
- Full historical crossover event log with forward returns (5D/20D/60D/120D) and max forward gain/drawdown
- Aggregate win-rate and return statistics per signal type

### 🧮 Risk Engine
A vectorized `RiskEngine` dataclass computes 20+ metrics from a daily return series:
- Total return, CAGR, annualized volatility, Sharpe, Sortino, Calmar, Omega
- Max drawdown, max drawdown duration, recovery factor
- Historical VaR/CVaR at 95%/99%
- Skewness, excess kurtosis, win rate, profit factor
- Beta, alpha, information ratio, up/down capture (vs. benchmark)
- Rolling Sharpe, monthly returns heatmap, top-5 worst drawdown periods (with recovery dates)

### 📉 Performance & Risk Tabs
- Period return comparison (1D → 5Y CAGR) vs. benchmark, with excess return
- Risk-adjusted comparison table (return, vol, Sharpe, max DD) vs. benchmark
- Detailed drawdown lifecycle chart with peak/trough/recovery markers
- Return distribution histogram with VaR markers and tail-observation scatter
- Configurable rolling analytics: return, volatility, Sharpe, beta, correlation, max drawdown, RSI, momentum

### 🧪 Strategy Research
An 11-strategy catalog across 5 categories, all built from vectorized OHLC logic with proper signal lag (no look-ahead bias):

| Category | Strategies |
|---|---|
| Baseline | Buy & Hold |
| Trend-Following | SMA Crossover, EMA Crossover, Triple SMA Trend Filter, MACD Signal Crossover |
| Momentum | Time-Series Momentum, Rate of Change (ROC) |
| Mean-Reversion | RSI Mean Reversion, Bollinger Band Reversion, Z-Score Mean Reversion |
| Breakout / Volatility | Donchian Channel Breakout, ATR Volatility Breakout |

Two data-source modes:
1. **Upload Custom Strategy CSV** — auto-detects returns, NAV/equity curves, or entry/exit trade logs
2. **Benchmark Strategy Simulator** — runs any catalog strategy live against real Yahoo Finance history with adjustable parameters, long/short toggle, and lookback window

Additional strategy tooling:
- Full trade statistics: number of trades, average/median/best/worst trade, average holding period, exposure %, turnover
- Configurable transaction-cost model (brokerage, slippage, and India-specific STT/GST/stamp duty/exchange/SEBI charges)
- Parameter robustness heatmaps (CAGR / Sharpe / Max Drawdown) for SMA/EMA crossover strategies
- Position-sizing calculator (risk-per-trade, ATR-based sizing, allocation caps)

### 🔮 Future Price Scenarios
- Bootstrap Monte Carlo engine that resamples the asset's own recent realized daily returns (not a Gaussian assumption) to preserve empirical skew and fat tails
- Configurable horizon (20/60/120/252 trading days), simulation count (500–5,000), and random seed
- P05/P10/P25/P50/P75/P90/P95 scenario bands plus optional representative sample paths
- Day-by-day scenario table
- Explicitly labeled as a statistical scenario tool, **not a forecast**

### 🌍 Relative Strength & Cross-Asset Context
- Asset-vs-benchmark relative strength ratio and its 20D/60D/1Y change
- Beta, alpha, correlation, information ratio, up/down capture
- Cross-asset return/correlation matrix (regional index, gold, oil, 10Y yield, etc.)

### 🏦 Fundamentals
- Company overview (sector, industry, exchange, currency, market state)
- Valuation & profitability block (P/E, PEG, P/B, EV/EBITDA, margins, ROE/ROA, growth rates, leverage ratios) when available
- Corporate actions (dividends/splits) and Yahoo calendar data

### ✅ Data Quality Tab
- Transparent reporting of data provider, cache TTLs, observation counts, missing data, and whether the currently loaded strategy series is authentic (live market data) or a labeled offline synthetic fallback

---

## Screenshots

<img width="1440" height="900" alt="Screenshot 2026-09-13 at 1 21 12 AM" src="https://github.com/user-attachments/assets/1a781aca-6870-41f9-b3fa-c301e9dc4ca0" />

<img width="1440" height="900" alt="Screenshot 2026-09-13 at 1 21 32 AM" src="https://github.com/user-attachments/assets/f225179d-99fa-4d83-8c36-15660cd2eab9" />

<img width="1440" height="900" alt="Screenshot 2026-09-13 at 1 22 06 AM" src="https://github.com/user-attachments/assets/39743096-3118-4f5d-966c-1a5e73ea9a01" />

<img width="1440" height="900" alt="Screenshot 2026-09-13 at 1 22 15 AM" src="https://github.com/user-attachments/assets/746faf38-2f90-41ac-b129-9c21958f627c" />

<img width="1440" height="900" alt="Screenshot 2026-09-13 at 1 22 40 AM" src="https://github.com/user-attachments/assets/ba8d4611-91db-441e-9fc5-f50ca117a647" />


---

## Tech Stack

| Layer | Library |
|---|---|
| App framework | [Streamlit](https://streamlit.io/) |
| Market data | [yfinance](https://pypi.org/project/yfinance/) (Yahoo Finance) |
| Data processing | pandas, NumPy |
| Statistics | SciPy (`scipy.stats`) |
| Charting | Plotly (`graph_objects`, `subplots`) |

---

## Getting Started

### Prerequisites
- Python 3.10+
- Internet access (for live Yahoo Finance data — the app degrades gracefully to a labeled synthetic fallback for the sample strategy only, if unreachable)

### Installation

```bash
git clone https://github.com/spelsudhir/quantmetrics-pro.git
cd quantmetrics-pro
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### `requirements.txt`

```
streamlit
pandas
numpy
scipy
plotly
yfinance
```

### Run the app

```bash
streamlit run app.py
```

Then open the local URL Streamlit prints (typically `http://localhost:8501`).

---

## Usage Guide

1. **Pick an instrument** — search the curated dropdown or type any Yahoo Finance ticker (e.g. `RELIANCE.NS`, `AAPL`, `^NSEI`, `BTC-USD`).
2. **Set a benchmark** — leave "Auto-select benchmark" on to have the app pick a sensible regional index, or choose/override manually.
3. **Configure the chart** — timeframe, interval, candlesticks, moving averages, and indicator panels are all in the sidebar.
4. **Load a strategy**:
   - Upload a CSV of returns, NAV/equity values, or an entry/exit trade log, **or**
   - Switch to "Benchmark Strategy Simulator", pick a strategy + parameters, and click **Run / Refresh Strategy**.
5. **Explore the tabs** — Price & Technical, Crossover Intelligence, Performance, Risk, Relative Strength, Strategy Research, Future Projections, Fundamentals, and Data Quality.
6. **Tune transaction costs and position sizing** in the sidebar expanders to see cost-adjusted and sizing-adjusted views in the Strategy Research tab.

### Custom Strategy CSV formats

The uploader (`detect_and_parse_csv`) auto-detects three formats:

| Format | Required columns | Behavior |
|---|---|---|
| Returns | `date` + `return` / `returns` / `strategy_returns` / `daily_return` / `pct_return` | Used directly (auto-scaled if values look like percentages) |
| NAV / Equity | `date` + `nav` / `equity` / `portfolio_value` / `balance` / `close` | Converted to daily % returns |
| Trade log | `entry_date`, `exit_date` + `pnl`/`profit` | Aggregated to a daily return series |

---

## Architecture Notes

- **`RiskEngine`** (dataclass) is the single source of truth for all risk/performance metrics on a given return series; it aligns strategy and benchmark series by date before computing anything, so `n` reflects the actual overlapping sample.
- **Signal lag & look-ahead bias**: crossover/momentum strategies apply a 1-bar execution lag on top of already-lagged decision inputs; threshold/breakout strategies (RSI, Bollinger, Z-Score, Donchian, ATR) run as an explicit finite-state long/flat/short machine (`_stateful_strategy_returns`) so a position on day *t* only ever uses information known as of the previous close.
- **Caching**: `st.cache_data` is used throughout (`fetch_market_history`, `fetch_instrument_profile`, `fetch_fundamentals`, `download_ohlc`, etc.) with TTLs from 10–30 minutes to keep the app responsive without hammering Yahoo Finance.
- **Robust data fetching**: `fetch_market_history` tries `yf.Ticker.history` first and transparently falls back to `yf.download` if the primary endpoint returns no usable `Close` data — a known intermittent Yahoo/yfinance issue.
- **No silent synthetic data**: the only synthetic fallback in the entire app is the "Sample Strategy" path, used solely when live data cannot be reached, and it is always visibly labeled as such in the UI.

---

## Disclaimer

QuantMetrics Pro is provided for **research, education, and decision-support purposes only**. Nothing in this application constitutes investment, financial, tax, or legal advice. Backtested performance is not indicative of future results. Market data is sourced from Yahoo Finance via `yfinance` and may be delayed, incomplete, or subject to provider outages. Always do your own due diligence and consult a licensed financial advisor before making investment decisions.

---

## Roadmap Ideas

- [ ] Multi-asset portfolio construction and correlation-aware risk budgeting
- [ ] Walk-forward optimization for strategy parameters
- [ ] Options analytics (IV surface, Greeks)
- [ ] Persistent watchlists and saved strategy configurations

---

## Contributing

Issues and pull requests are welcome. Please open an issue first to discuss significant changes.

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

## Author

**Sudhir**
- GitHub: [@spelsudhir](https://github.com/spelsudhir)
- Contact: [spelsudhir@gmail.com](mailto:spelsudhir@gmail.com)
