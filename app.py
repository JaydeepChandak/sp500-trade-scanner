from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

from sp500_universe import SP500_TICKERS, normalize_ticker, to_yfinance_ticker


TRADING_DAYS_PER_YEAR = 252
MIN_HOLDING_DAYS = 7
DEFAULT_HISTORY_YEARS = 3
MOMENTUM_WINDOWS = {
    "Momentum 1M": 21,
    "Momentum 3M": 63,
    "Momentum 6M": 126,
    "Momentum 12M": 252,
}
SCENARIO_CATEGORIES = [
    "Geopolitical tension",
    "War escalation",
    "War de-escalation",
    "Sanctions",
    "Oil supply shock",
    "Inflation surprise",
    "Interest rate increase",
    "Interest rate cut",
    "Recession risk",
    "Strong economic growth",
    "US dollar strengthening",
    "US dollar weakening",
    "Commodity supply disruption",
    "China demand shock",
    "Election / policy uncertainty",
    "Trade tariffs",
    "Banking / credit stress",
    "AI / technology boom",
    "Defensive market rotation",
    "Risk-on market",
    "Risk-off market",
]

SCENARIO_DEFINITIONS = {
    "Geopolitical tension": {"return_impact": 0.018, "risk_impact": 0.045, "growth_tilt": -0.15},
    "War escalation": {"return_impact": 0.030, "risk_impact": 0.080, "growth_tilt": -0.25},
    "War de-escalation": {"return_impact": 0.020, "risk_impact": 0.050, "growth_tilt": 0.20},
    "Sanctions": {"return_impact": 0.018, "risk_impact": 0.045, "growth_tilt": -0.10},
    "Oil supply shock": {"return_impact": 0.022, "risk_impact": 0.060, "growth_tilt": -0.20},
    "Inflation surprise": {"return_impact": 0.024, "risk_impact": 0.060, "growth_tilt": -0.25},
    "Interest rate increase": {"return_impact": 0.026, "risk_impact": 0.055, "growth_tilt": -0.35},
    "Interest rate cut": {"return_impact": 0.022, "risk_impact": 0.050, "growth_tilt": 0.30},
    "Recession risk": {"return_impact": 0.032, "risk_impact": 0.080, "growth_tilt": -0.30},
    "Strong economic growth": {"return_impact": 0.026, "risk_impact": 0.035, "growth_tilt": 0.25},
    "US dollar strengthening": {"return_impact": 0.016, "risk_impact": 0.035, "growth_tilt": -0.10},
    "US dollar weakening": {"return_impact": 0.014, "risk_impact": 0.030, "growth_tilt": 0.10},
    "Commodity supply disruption": {"return_impact": 0.020, "risk_impact": 0.055, "growth_tilt": -0.15},
    "China demand shock": {"return_impact": 0.022, "risk_impact": 0.055, "growth_tilt": -0.15},
    "Election / policy uncertainty": {"return_impact": 0.018, "risk_impact": 0.050, "growth_tilt": -0.10},
    "Trade tariffs": {"return_impact": 0.020, "risk_impact": 0.050, "growth_tilt": -0.15},
    "Banking / credit stress": {"return_impact": 0.032, "risk_impact": 0.085, "growth_tilt": -0.25},
    "AI / technology boom": {"return_impact": 0.026, "risk_impact": 0.045, "growth_tilt": 0.40},
    "Defensive market rotation": {"return_impact": 0.018, "risk_impact": 0.040, "growth_tilt": -0.35},
    "Risk-on market": {"return_impact": 0.028, "risk_impact": 0.045, "growth_tilt": 0.35},
    "Risk-off market": {"return_impact": 0.030, "risk_impact": 0.085, "growth_tilt": -0.35},
}


st.set_page_config(
    page_title="S&P 500 Trade Scanner",
    layout="wide",
    initial_sidebar_state="expanded",
)


def apply_ui_theme() -> None:
    """Apply a compact research-terminal visual system."""
    st.markdown(
        """
        <style>
            :root {
                --surface: #111827;
                --surface-2: #1f2937;
                --ink: #f8fafc;
                --muted: #cbd5e1;
                --line: #475569;
                --line-soft: #334155;
                --accent: #a855f7;
                --accent-2: #020617;
                --good: #22c55e;
                --warn: #f59e0b;
                --bad: #ef4444;
            }
            html, body, [data-testid="stAppViewContainer"] {
                color: var(--ink);
                background:
                    radial-gradient(circle at 20% 0%, rgba(126, 34, 206, 0.36), transparent 32rem),
                    radial-gradient(circle at 85% 10%, rgba(37, 99, 235, 0.22), transparent 28rem),
                    #020617;
            }
            .block-container {
                padding-top: 1.15rem;
                padding-bottom: 3rem;
                max-width: 1500px;
            }
            section[data-testid="stSidebar"] {
                background: linear-gradient(180deg, #090b1a 0%, #111827 55%, #1e1b4b 100%);
                border-right: 1px solid #4c1d95;
            }
            section[data-testid="stSidebar"] label,
            section[data-testid="stSidebar"] p {
                color: #f8fafc;
            }
            section[data-testid="stSidebar"] h1,
            section[data-testid="stSidebar"] h2,
            section[data-testid="stSidebar"] h3 {
                color: #ffffff;
            }
            div[data-testid="stMetric"] {
                background: linear-gradient(180deg, rgba(17, 24, 39, 0.96), rgba(30, 27, 75, 0.86));
                border: 1px solid #4c1d95;
                border-left: 5px solid var(--accent);
                border-radius: 8px;
                padding: 0.85rem 0.95rem;
                box-shadow: 0 14px 34px rgba(0, 0, 0, 0.35);
            }
            div[data-testid="stMetricLabel"] p {
                color: var(--muted);
                font-size: 0.78rem;
                letter-spacing: 0;
                font-weight: 750;
            }
            div[data-testid="stMetricValue"] {
                color: var(--ink);
                font-size: 1.45rem;
                font-weight: 850;
            }
            .app-hero {
                border: 1px solid #7e22ce;
                background: linear-gradient(135deg, #020617 0%, #312e81 48%, #7e22ce 100%);
                border-radius: 10px;
                padding: 1.25rem 1.35rem;
                margin-bottom: 1rem;
                box-shadow: 0 18px 42px rgba(0, 0, 0, 0.42);
            }
            .app-hero h1 {
                margin: 0 0 0.25rem 0;
                font-size: 2rem;
                line-height: 1.12;
                letter-spacing: 0;
                color: #ffffff;
            }
            .app-hero p {
                margin: 0;
                color: #e9d5ff;
                font-size: 0.98rem;
                font-weight: 600;
            }
            .section-label {
                margin-top: 1.25rem;
                margin-bottom: 0.45rem;
                padding: 0.5rem 0.7rem;
                border-left: 5px solid var(--accent);
                background: linear-gradient(90deg, rgba(88, 28, 135, 0.92), rgba(17, 24, 39, 0.92));
                border: 1px solid #4c1d95;
                border-radius: 6px;
                font-weight: 850;
                color: var(--ink);
                font-size: 1.02rem;
            }
            .fine-print {
                color: var(--muted);
                font-size: 0.82rem;
                line-height: 1.45;
            }
            .badge-row {
                display: flex;
                flex-wrap: wrap;
                gap: 0.5rem;
                margin: 0.35rem 0 0.85rem;
            }
            .badge {
                border: 1px solid #a855f7;
                border-radius: 999px;
                padding: 0.3rem 0.72rem;
                background: #111827;
                color: var(--ink);
                font-size: 0.78rem;
                font-weight: 800;
            }
            .badge-good { border-color: #22c55e; color: #052e16; background: #86efac; }
            .badge-warn { border-color: #f59e0b; color: #451a03; background: #fcd34d; }
            .badge-bad { border-color: #ef4444; color: #450a0a; background: #fca5a5; }
            div[data-testid="stDataFrame"] {
                border: 1px solid #4c1d95;
                border-radius: 8px;
                overflow: hidden;
                box-shadow: 0 14px 30px rgba(0, 0, 0, 0.30);
            }
            .stTabs [data-baseweb="tab-list"] {
                gap: 0.35rem;
                border-bottom: 1px solid #4c1d95;
            }
            .stTabs [data-baseweb="tab"] {
                border-radius: 8px 8px 0 0;
                padding: 0.65rem 0.9rem;
            }
            .stButton > button {
                width: 100%;
                border-radius: 8px;
                font-weight: 850;
                border: 1px solid #c084fc;
                background: linear-gradient(135deg, #7e22ce, #4f46e5);
                color: #ffffff;
            }
            div[data-testid="stAlert"] {
                border: 1px solid var(--line);
                color: #f8fafc;
                background: rgba(17, 24, 39, 0.88);
            }
            .stMarkdown, .stText, p, li, label, span {
                color: #f8fafc;
            }
            input, textarea, [data-baseweb="select"] {
                color: #f8fafc;
            }
            div[data-baseweb="input"],
            div[data-baseweb="textarea"],
            div[data-baseweb="select"] > div {
                background-color: #111827;
                border-color: #64748b;
                color: #f8fafc;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_badges(items: list[tuple[str, str]]) -> None:
    badges = "".join(f'<span class="badge badge-{tone}">{label}</span>' for label, tone in items)
    st.markdown(f'<div class="badge-row">{badges}</div>', unsafe_allow_html=True)


def render_section_label(label: str) -> None:
    st.markdown(f'<div class="section-label">{label}</div>', unsafe_allow_html=True)


def render_fine_print(text: str) -> None:
    st.markdown(f'<div class="fine-print">{text}</div>', unsafe_allow_html=True)


def format_percent(value: float, decimals: int = 2) -> str:
    if pd.isna(value):
        return "N/A"
    return f"{value:.{decimals}%}"


def format_number(value: float, decimals: int = 2) -> str:
    if pd.isna(value):
        return "N/A"
    return f"{value:.{decimals}f}"


def format_money(value: float, decimals: int = 0) -> str:
    if pd.isna(value):
        return "N/A"
    return f"${value:,.{decimals}f}"


def style_numeric_table(styler: pd.io.formats.style.Styler, gradient_columns: list[str] | None = None):
    return styler.set_properties(**{"font-size": "12px", "color": "#f8fafc", "background-color": "#111827"}).set_table_styles(
        [
            {
                "selector": "th",
                "props": [
                    ("font-size", "12px"),
                    ("font-weight", "850"),
                    ("background-color", "#312e81"),
                    ("color", "#ffffff"),
                    ("border-color", "#7e22ce"),
                ],
            },
            {"selector": "td", "props": [("border-color", "#334155")]},
        ]
    )


def parse_tickers(raw_tickers: str) -> list[str]:
    """Parse comma/space-separated tickers and keep order without duplicates."""
    cleaned = raw_tickers.replace("\n", ",").replace(" ", ",")
    parsed = [normalize_ticker(token) for token in cleaned.split(",") if token.strip()]
    return list(dict.fromkeys(parsed))


def validate_sp500_tickers(tickers: list[str]) -> tuple[list[str], list[str]]:
    valid = [ticker for ticker in tickers if ticker in SP500_TICKERS]
    invalid = [ticker for ticker in tickers if ticker not in SP500_TICKERS]
    return valid, invalid


def get_sp500_universe() -> list[str]:
    """Return the built-in version 1 S&P 500 stock universe."""
    return sorted(SP500_TICKERS)


@st.cache_data(ttl=60 * 60 * 24, show_spinner=False)
def get_company_metadata(tickers: tuple[str, ...]) -> pd.DataFrame:
    """Fetch lightweight company labels for display.

    Metadata is best-effort only; if yfinance does not return a field, the app
    falls back to the ticker so the scanner remains usable.
    """
    rows = []
    for ticker in tickers:
        name = ticker
        sector = "Unavailable"
        try:
            info = yf.Ticker(to_yfinance_ticker(ticker)).get_info()
            name = info.get("shortName") or info.get("longName") or ticker
            sector = info.get("sector") or "Unavailable"
        except Exception:
            pass

        rows.append(
            {
                "Ticker": ticker,
                "Company Name": name,
                "Sector": sector,
                "Company": f"{name} ({ticker})" if name != ticker else ticker,
            }
        )

    return pd.DataFrame(rows).set_index("Ticker") if rows else pd.DataFrame()


@st.cache_data(ttl=60 * 60, show_spinner=False)
def load_data(tickers: tuple[str, ...], start_date: date, end_date: date) -> pd.DataFrame:
    """Load adjusted close prices from yfinance.

    Adjusted prices are used so splits and dividends are reflected in return
    calculations. This is still historical market data, not a forecast.
    """
    if not tickers:
        return pd.DataFrame()

    yahoo_tickers = [to_yfinance_ticker(ticker) for ticker in tickers]
    yahoo_to_display = dict(zip(yahoo_tickers, tickers))

    raw = yf.download(
        yahoo_tickers,
        start=start_date,
        end=end_date + timedelta(days=1),
        auto_adjust=False,
        progress=False,
        group_by="column",
        threads=True,
    )

    if raw.empty:
        return pd.DataFrame()

    if isinstance(raw.columns, pd.MultiIndex):
        if "Adj Close" in raw.columns.get_level_values(0):
            prices = raw["Adj Close"]
        else:
            prices = raw["Close"]
        prices = prices.rename(columns=yahoo_to_display)
    else:
        price_col = "Adj Close" if "Adj Close" in raw else "Close"
        prices = raw[[price_col]].rename(columns={price_col: tickers[0]})

    prices = prices.sort_index().dropna(axis=1, how="all")
    return prices.ffill().dropna(how="all")


def calculate_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return prices.pct_change(fill_method=None).dropna(how="all")


def calculate_holding_period_stats(returns: pd.DataFrame, holding_period_days: int) -> pd.DataFrame:
    """Estimate 1-period forward outcomes from historical rolling returns."""
    trading_horizon = max(2, int(round(holding_period_days / 365 * TRADING_DAYS_PER_YEAR)))
    rows = []

    for ticker in returns.columns:
        series = returns[ticker].dropna()
        rolling_forward = (1 + series).rolling(trading_horizon).apply(np.prod, raw=True) - 1
        rolling_forward = rolling_forward.dropna()
        if rolling_forward.empty:
            rows.append(
                {
                    "Ticker": ticker,
                    "Expected Return Estimate": np.nan,
                    "Downside Risk Estimate": np.nan,
                    "Probability Positive Historical Return": np.nan,
                    "Best Historical Period": np.nan,
                    "Worst Historical Period": np.nan,
                    "Holding Period Observations": 0,
                }
            )
            continue

        rows.append(
            {
                "Ticker": ticker,
                "Expected Return Estimate": rolling_forward.mean(),
                "Downside Risk Estimate": rolling_forward.quantile(0.10),
                "Probability Positive Historical Return": (rolling_forward > 0).mean(),
                "Best Historical Period": rolling_forward.max(),
                "Worst Historical Period": rolling_forward.min(),
                "Holding Period Observations": int(rolling_forward.count()),
            }
        )

    return pd.DataFrame(rows).set_index("Ticker") if rows else pd.DataFrame()


def calculate_rsi(price: pd.Series, window: int = 14) -> pd.Series:
    """Calculate RSI using rolling average gains/losses."""
    delta = price.diff()
    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)
    avg_gain = gains.rolling(window).mean()
    avg_loss = losses.rolling(window).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def classify_bullish_score(score: float) -> str:
    if pd.isna(score):
        return "Insufficient data"
    if score <= 30:
        return "Weak / avoid"
    if score <= 50:
        return "Neutral"
    if score <= 70:
        return "Bullish watchlist"
    if score <= 85:
        return "Strong bullish setup"
    return "Very strong bullish setup"


def score_confidence(observations: int, annualized_volatility: float, max_drawdown: float) -> str:
    """Rate confidence in the score from data depth and risk stability."""
    points = 0
    if observations >= 504:
        points += 2
    elif observations >= 252:
        points += 1

    if annualized_volatility <= 0.30:
        points += 2
    elif annualized_volatility <= 0.45:
        points += 1

    if max_drawdown >= -0.25:
        points += 2
    elif max_drawdown >= -0.40:
        points += 1

    if points >= 5:
        return "High"
    if points >= 3:
        return "Medium"
    return "Low"


def score_ratio(value: float, low: float, high: float) -> float:
    """Map a metric into a 0-1 score where higher is better."""
    if pd.isna(value):
        return 0.0
    return float(np.clip((value - low) / (high - low), 0, 1))


def score_ratio_series(values: pd.Series, low: float, high: float) -> pd.Series:
    """Vectorized 0-1 score mapper."""
    return ((values - low) / (high - low)).clip(0, 1).fillna(0)


def calculate_indicators(
    prices: pd.DataFrame,
    returns: pd.DataFrame,
    benchmark_prices: pd.DataFrame | None = None,
) -> dict[str, pd.DataFrame]:
    """Calculate technical indicators used by the Step 2 bullish framework."""
    indicators: dict[str, pd.DataFrame] = {}
    benchmark_series = None

    if benchmark_prices is not None and not benchmark_prices.empty:
        benchmark_series = benchmark_prices.iloc[:, 0].dropna()

    for ticker in prices.columns:
        price = prices[ticker].dropna()
        ticker_returns = returns[ticker].dropna() if ticker in returns else pd.Series(dtype=float)
        if price.empty:
            continue

        frame = pd.DataFrame(index=price.index)
        frame["Price"] = price
        frame["MA20"] = price.rolling(20).mean()
        frame["MA50"] = price.rolling(50).mean()
        frame["MA200"] = price.rolling(200).mean()
        frame["RSI14"] = calculate_rsi(price)

        ema12 = price.ewm(span=12, adjust=False).mean()
        ema26 = price.ewm(span=26, adjust=False).mean()
        frame["MACD"] = ema12 - ema26
        frame["MACD Signal"] = frame["MACD"].ewm(span=9, adjust=False).mean()
        frame["MACD Histogram"] = frame["MACD"] - frame["MACD Signal"]

        rolling_mean = price.rolling(20).mean()
        rolling_std = price.rolling(20).std()
        frame["Bollinger Upper"] = rolling_mean + 2 * rolling_std
        frame["Bollinger Lower"] = rolling_mean - 2 * rolling_std
        frame["Bollinger %B"] = (price - frame["Bollinger Lower"]) / (
            frame["Bollinger Upper"] - frame["Bollinger Lower"]
        )

        high_52w = price.rolling(252, min_periods=60).max()
        low_52w = price.rolling(252, min_periods=60).min()
        frame["52W High"] = high_52w
        frame["52W Low"] = low_52w
        frame["52W Position"] = (price - low_52w) / (high_52w - low_52w)
        frame["Discount From 52W High"] = price / high_52w - 1

        for label, window in MOMENTUM_WINDOWS.items():
            frame[label] = price.pct_change(window)

        frame["Rolling Vol 20D"] = ticker_returns.rolling(20).std() * np.sqrt(TRADING_DAYS_PER_YEAR)
        frame["Rolling Vol 252D"] = ticker_returns.rolling(252, min_periods=60).std() * np.sqrt(TRADING_DAYS_PER_YEAR)
        frame["Volatility Regime"] = frame["Rolling Vol 20D"] / frame["Rolling Vol 252D"]
        frame["Trend Strength"] = (frame["MA50"] / frame["MA200"]) - 1
        frame["Recent Drawdown"] = price / price.rolling(63, min_periods=20).max() - 1

        if benchmark_series is not None:
            aligned = pd.concat([price, benchmark_series], axis=1, join="inner").dropna()
            if len(aligned) > 63:
                relative = aligned.iloc[:, 0] / aligned.iloc[:, 1]
                frame["Relative Strength 3M"] = relative.pct_change(63).reindex(frame.index)
                frame["Relative Strength 6M"] = relative.pct_change(126).reindex(frame.index)
            else:
                frame["Relative Strength 3M"] = np.nan
                frame["Relative Strength 6M"] = np.nan
        else:
            frame["Relative Strength 3M"] = np.nan
            frame["Relative Strength 6M"] = np.nan

        indicators[ticker] = frame

    return indicators


def calculate_bullish_score(
    indicators: dict[str, pd.DataFrame],
    metrics: pd.DataFrame,
) -> pd.DataFrame:
    """Create a transparent 0-100 bullish score from technical and risk inputs.

    The score is a historical-data heuristic for long-only research. It is not
    a prediction engine and it does not guarantee future returns.
    """
    rows = []

    for ticker, frame in indicators.items():
        latest = frame.dropna(subset=["Price"]).iloc[-1]
        price = latest["Price"]
        ma20 = latest.get("MA20")
        ma50 = latest.get("MA50")
        ma200 = latest.get("MA200")
        rsi = latest.get("RSI14")
        macd_hist = latest.get("MACD Histogram")
        position_52w = latest.get("52W Position")
        discount_high = latest.get("Discount From 52W High")
        vol_regime = latest.get("Volatility Regime")
        recent_drawdown = latest.get("Recent Drawdown")
        rel_3m = latest.get("Relative Strength 3M")
        rel_6m = latest.get("Relative Strength 6M")

        metric_row = metrics.loc[ticker] if ticker in metrics.index else pd.Series(dtype=float)
        annualized_vol = metric_row.get("Annualized Volatility", np.nan)
        max_drawdown = metric_row.get("Max Drawdown", np.nan)
        observations = int(metric_row.get("Observations", 0))

        trend_score = 0
        trend_score += 0.35 if price > ma200 else 0
        trend_score += 0.25 if ma50 > ma200 else 0
        trend_score += 0.20 if ma20 > ma50 else 0
        trend_score += 0.20 if price > ma50 else 0

        support_score = 0
        if pd.notna(discount_high):
            support_score += 0.30 if -0.25 <= discount_high <= -0.04 else 0.15 if -0.35 <= discount_high < -0.25 else 0
        if pd.notna(rsi):
            support_score += 0.30 if 40 <= rsi <= 62 else 0.15 if 30 <= rsi < 70 else 0
        if pd.notna(price) and pd.notna(ma50):
            support_score += 0.25 if 0.95 <= price / ma50 <= 1.06 else 0.10 if 0.90 <= price / ma50 <= 1.12 else 0
        if pd.notna(position_52w):
            support_score += 0.15 if 0.35 <= position_52w <= 0.85 else 0.05 if 0.20 <= position_52w < 0.95 else 0

        momentum_score = 0
        momentum_score += 0.25 * score_ratio(latest.get("Momentum 1M"), -0.03, 0.08)
        momentum_score += 0.25 * score_ratio(latest.get("Momentum 3M"), -0.05, 0.15)
        momentum_score += 0.20 * score_ratio(latest.get("Momentum 6M"), -0.08, 0.25)
        momentum_score += 0.15 * score_ratio(latest.get("Momentum 12M"), -0.10, 0.35)
        momentum_score += 0.15 if pd.notna(macd_hist) and macd_hist > 0 else 0

        risk_score = 0
        risk_score += 0.30 * (1 - score_ratio(annualized_vol, 0.18, 0.60))
        risk_score += 0.25 * (1 - score_ratio(vol_regime, 0.80, 1.60))
        risk_score += 0.25 * score_ratio(recent_drawdown, -0.25, -0.03)
        risk_score += 0.20 * score_ratio(max_drawdown, -0.60, -0.15)

        relative_strength_score = 0.50 * score_ratio(rel_3m, -0.08, 0.08)
        relative_strength_score += 0.50 * score_ratio(rel_6m, -0.12, 0.12)

        if pd.notna(price) and pd.notna(ma200) and pd.notna(latest.get("52W High")):
            upside_to_high = latest["52W High"] / price - 1
            downside_to_ma200 = abs(price / ma200 - 1)
            risk_reward = upside_to_high / downside_to_ma200 if downside_to_ma200 > 0 else np.nan
        else:
            risk_reward = np.nan
        risk_reward_score = score_ratio(risk_reward, 0.75, 2.50)

        score = 100 * (
            0.25 * trend_score
            + 0.20 * support_score
            + 0.20 * momentum_score
            + 0.15 * risk_score
            + 0.10 * relative_strength_score
            + 0.10 * risk_reward_score
        )

        if price < ma200 and ma50 < ma200:
            score = min(score, 45)
        if pd.notna(rsi) and rsi > 75:
            score -= 8
        if pd.notna(recent_drawdown) and recent_drawdown < -0.30:
            score -= 10

        score = float(np.clip(score, 0, 100))
        buy_low_flags = [
            pd.notna(discount_high) and -0.25 <= discount_high <= -0.04,
            pd.notna(rsi) and 35 <= rsi <= 65,
            pd.notna(price) and pd.notna(ma50) and 0.94 <= price / ma50 <= 1.08,
            pd.notna(price) and pd.notna(ma200) and price > ma200,
            pd.notna(recent_drawdown) and recent_drawdown > -0.25,
        ]
        sell_high_flags = [
            pd.notna(position_52w) and position_52w >= 0.90,
            pd.notna(rsi) and rsi >= 70,
            pd.notna(macd_hist) and macd_hist < 0,
            pd.notna(price) and pd.notna(ma50) and price / ma50 >= 1.12,
            pd.notna(discount_high) and discount_high >= -0.03,
        ]

        suggested_buy_zone = np.nan
        suggested_sell_zone = np.nan
        stop_loss = np.nan
        if pd.notna(ma50) and pd.notna(latest.get("Bollinger Lower")):
            suggested_buy_zone = min(ma50, latest["Bollinger Lower"] * 1.03)
        if pd.notna(latest.get("52W High")) and pd.notna(price):
            suggested_sell_zone = max(latest["52W High"] * 0.98, price * 1.08)
        if pd.notna(ma200) and pd.notna(price):
            stop_loss = min(price * 0.92, ma200 * 0.97)

        explanation = []
        explanation.append("long-term trend positive" if price > ma200 else "price below 200-day trend")
        explanation.append("momentum improving" if momentum_score >= 0.55 else "momentum mixed or weak")
        explanation.append("reasonable pullback" if sum(buy_low_flags) >= 3 else "buy-low conditions incomplete")
        explanation.append("relative strength positive" if relative_strength_score >= 0.55 else "relative strength not yet convincing")
        explanation.append("risk profile acceptable" if risk_score >= 0.50 else "risk profile elevated")

        rows.append(
            {
                "Ticker": ticker,
                "Bullish Score": score,
                "Interpretation": classify_bullish_score(score),
                "Confidence": score_confidence(observations, annualized_vol, max_drawdown),
                "Trend Score": trend_score * 100,
                "Buy-Low Score": support_score * 100,
                "Momentum Score": momentum_score * 100,
                "Risk Score": risk_score * 100,
                "Relative Strength Score": relative_strength_score * 100,
                "Risk/Reward Score": risk_reward_score * 100,
                "Annualized Volatility": annualized_vol,
                "Max Drawdown": max_drawdown,
                "Observations": observations,
                "Price": price,
                "Risk/Reward Ratio": risk_reward,
                "RSI14": rsi,
                "MA20": ma20,
                "MA50": ma50,
                "MA200": ma200,
                "52W Position": position_52w,
                "Discount From 52W High": discount_high,
                "Recent Drawdown": recent_drawdown,
                "Volatility Regime": vol_regime,
                "Momentum 1M": latest.get("Momentum 1M"),
                "Momentum 3M": latest.get("Momentum 3M"),
                "Momentum 6M": latest.get("Momentum 6M"),
                "Momentum 12M": latest.get("Momentum 12M"),
                "Buy-Low Signals Met": sum(bool(flag) for flag in buy_low_flags),
                "Sell-High Signals Met": sum(bool(flag) for flag in sell_high_flags),
                "Suggested Buy Zone": suggested_buy_zone,
                "Suggested Sell / Take-Profit Zone": suggested_sell_zone,
                "Stop-Loss Suggestion": stop_loss,
                "Explanation": "; ".join(explanation) + ".",
            }
        )

    return pd.DataFrame(rows).set_index("Ticker") if rows else pd.DataFrame()


def rank_trade_candidates(
    bullish_summary: pd.DataFrame,
    holding_stats: pd.DataFrame,
    risk_preference: str,
) -> pd.DataFrame:
    """Rank S&P 500 stocks for the long-only bullish scanner."""
    if bullish_summary.empty:
        return pd.DataFrame()

    candidates = bullish_summary.copy()
    if holding_stats is not None and not holding_stats.empty:
        new_columns = [column for column in holding_stats.columns if column not in candidates.columns]
        candidates = candidates.join(holding_stats[new_columns], how="left")

    risk_weights = {
        "Conservative": {"score": 0.50, "risk": 0.30, "expected": 0.20, "min_score": 55},
        "Balanced": {"score": 0.60, "risk": 0.20, "expected": 0.20, "min_score": 50},
        "Aggressive": {"score": 0.70, "risk": 0.10, "expected": 0.20, "min_score": 45},
    }
    config = risk_weights[risk_preference]

    expected_rank = candidates["Expected Return Estimate"].rank(pct=True).fillna(0)
    downside_rank = candidates["Downside Risk Estimate"].rank(pct=True).fillna(0)
    candidates["Opportunity Rank Score"] = (
        config["score"] * candidates["Bullish Score"]
        + config["risk"] * candidates["Risk Score"]
        + config["expected"] * expected_rank * 100
        + 0.10 * downside_rank * 100
    )

    candidates["Liquidity/Data Quality Check"] = np.where(
        candidates["Holding Period Observations"].fillna(0) >= 60,
        "OK",
        "Limited history",
    )
    candidates["Risk Warning"] = np.select(
        [
            candidates["Risk Score"] < 35,
            candidates["Recent Drawdown"] < -0.20,
            candidates["Volatility Regime"] > 1.4,
            candidates["RSI14"] > 72,
            candidates["Liquidity/Data Quality Check"] != "OK",
        ],
        [
            "Elevated volatility/drawdown risk",
            "Recent drawdown is large",
            "Short-term volatility is elevated",
            "Potentially overbought",
            "Historical sample is limited",
        ],
        default="No major scanner warning",
    )
    candidates["Main Reason"] = candidates["Explanation"]

    ranked = candidates[
        (candidates["Bullish Score"] >= config["min_score"])
        & (candidates["Liquidity/Data Quality Check"] == "OK")
    ].sort_values("Opportunity Rank Score", ascending=False)
    return ranked


def run_scenario_analysis(candidates: pd.DataFrame, scenario_assumptions: list[dict[str, float | str]]) -> dict[str, pd.DataFrame]:
    """Apply manual macro/geopolitical assumptions to return and risk estimates.

    Scenario inputs are subjective assumptions. They are not facts, forecasts,
    live news, or guaranteed predictors of future market behavior.
    """
    adjusted = candidates.copy()
    if adjusted.empty:
        return {"adjusted_candidates": adjusted, "scenario_summary": pd.DataFrame()}

    adjusted["Baseline Expected Return Estimate"] = adjusted["Expected Return Estimate"]
    adjusted["Baseline Downside Risk Estimate"] = adjusted["Downside Risk Estimate"]
    adjusted["Baseline Probability Positive"] = adjusted["Probability Positive Historical Return"]
    adjusted["Baseline Bullish Score"] = adjusted["Bullish Score"]
    adjusted["Scenario Return Adjustment"] = 0.0
    adjusted["Scenario Risk Adjustment"] = 0.0

    active_assumptions = [
        assumption
        for assumption in scenario_assumptions
        if assumption["direction"] != "Neutral" and assumption["probability"] > 0 and assumption["severity"] > 0
    ]
    if not active_assumptions:
        adjusted["Scenario Note"] = "No active scenario adjustment"
        return {"adjusted_candidates": adjusted, "scenario_summary": pd.DataFrame()}

    growth_exposure = (
        0.35 * score_ratio_series(adjusted["Annualized Volatility"], 0.15, 0.55)
        + 0.30 * score_ratio_series(adjusted["Momentum 6M"], -0.10, 0.35)
        + 0.25 * score_ratio_series(adjusted["Relative Strength Score"], 0, 100)
        + 0.10 * (1 - score_ratio_series(adjusted["Risk Score"], 0, 100))
    ).clip(0, 1)
    risk_exposure = (
        0.50 * score_ratio_series(adjusted["Annualized Volatility"], 0.15, 0.60)
        + 0.30 * score_ratio_series(adjusted["Volatility Regime"], 0.80, 1.70)
        + 0.20 * (1 - score_ratio_series(adjusted["Risk Score"], 0, 100))
    ).clip(0.50, 1.50)

    summary_rows = []
    for assumption in active_assumptions:
        category = str(assumption["category"])
        definition = SCENARIO_DEFINITIONS[category]
        direction = 1 if assumption["direction"] == "Bullish" else -1
        severity = float(assumption["severity"]) / 5
        probability = float(assumption["probability"]) / 100
        scale = severity * probability
        style_multiplier = (1 + definition["growth_tilt"] * (growth_exposure - 0.5)).clip(0.60, 1.40)

        return_adjustment = direction * definition["return_impact"] * scale * style_multiplier
        if direction < 0:
            risk_adjustment = definition["risk_impact"] * scale * risk_exposure
        else:
            risk_adjustment = -0.40 * definition["risk_impact"] * scale * risk_exposure

        adjusted["Scenario Return Adjustment"] += return_adjustment
        adjusted["Scenario Risk Adjustment"] += risk_adjustment
        summary_rows.append(
            {
                "Scenario": category,
                "Direction": assumption["direction"],
                "Severity": assumption["severity"],
                "Probability": assumption["probability"] / 100,
                "Avg Return Adjustment": return_adjustment.mean(),
                "Avg Risk Adjustment": risk_adjustment.mean(),
                "Interpretation": (
                    "Manual bullish assumption lowers risk modestly and raises return estimates."
                    if direction > 0
                    else "Manual bearish assumption lowers return estimates and raises downside risk."
                ),
            }
        )

    adjusted["Expected Return Estimate"] = (
        adjusted["Baseline Expected Return Estimate"] + adjusted["Scenario Return Adjustment"]
    )
    adjusted["Downside Risk Estimate"] = (
        adjusted["Baseline Downside Risk Estimate"] - adjusted["Scenario Risk Adjustment"].clip(lower=0)
        - 0.25 * adjusted["Scenario Risk Adjustment"].clip(upper=0)
    )
    adjusted["Probability Positive Historical Return"] = (
        adjusted["Baseline Probability Positive"]
        + 1.50 * adjusted["Scenario Return Adjustment"]
        - 0.80 * adjusted["Scenario Risk Adjustment"].clip(lower=0)
    ).clip(0.05, 0.95)
    adjusted["Bullish Score"] = (
        adjusted["Baseline Bullish Score"]
        + 250 * adjusted["Scenario Return Adjustment"]
        - 120 * adjusted["Scenario Risk Adjustment"].clip(lower=0)
    ).clip(0, 100)
    adjusted["Scenario Note"] = np.where(
        adjusted["Scenario Risk Adjustment"] > 0.025,
        "Scenario assumptions materially increase risk",
        np.where(
            adjusted["Scenario Return Adjustment"] > 0.015,
            "Scenario assumptions materially improve expected return",
            "Scenario adjustment is modest",
        ),
    )

    return {
        "adjusted_candidates": adjusted,
        "scenario_summary": pd.DataFrame(summary_rows),
    }


def classify_current_holdings(current_holdings: list[str], ranked: pd.DataFrame, all_candidates: pd.DataFrame) -> pd.DataFrame:
    """Classify optional current holdings into buy/hold/sell style actions."""
    if not current_holdings:
        return pd.DataFrame()

    rows = []
    for ticker in current_holdings:
        if ticker not in all_candidates.index:
            rows.append(
                {
                    "Ticker": ticker,
                    "Suggested Action": "Unavailable",
                    "Current Score": np.nan,
                    "Condition": "No usable scanner data",
                    "1M Risk/Reward": np.nan,
                    "Suggested Sell Zone": np.nan,
                    "Suggested Stop Loss": np.nan,
                    "Explanation": "Ticker had insufficient data or was excluded.",
                }
            )
            continue

        row = all_candidates.loc[ticker]
        score = row["Bullish Score"]
        rsi = row["RSI14"]
        rank_in_buys = ticker in ranked.index

        if rank_in_buys and score >= 70:
            action = "Buy more"
        elif score >= 55 and rsi <= 72:
            action = "Hold"
        elif score >= 50 and rsi > 72:
            action = "Trim / partial sell"
        elif score >= 40:
            action = "Avoid adding"
        else:
            action = "Sell"

        condition = []
        condition.append("overbought" if rsi > 70 else "oversold" if rsi < 35 else "neutral RSI")
        condition.append("trend improving" if row["Trend Score"] >= 60 else "trend weakening")
        condition.append("risk/reward favorable" if row["Risk/Reward Score"] >= 55 else "risk/reward mixed")

        rows.append(
            {
                "Ticker": ticker,
                "Suggested Action": action,
                "Current Score": score,
                "Condition": "; ".join(condition),
                "1M Risk/Reward": row.get("Risk/Reward Ratio"),
                "Expected Return Estimate": row.get("Expected Return Estimate"),
                "Downside Risk Estimate": row.get("Downside Risk Estimate"),
                "Scenario Return Adjustment": row.get("Scenario Return Adjustment"),
                "Scenario Risk Adjustment": row.get("Scenario Risk Adjustment"),
                "Scenario Note": row.get("Scenario Note"),
                "Suggested Sell Zone": row.get("Suggested Sell / Take-Profit Zone"),
                "Suggested Stop Loss": row.get("Stop-Loss Suggestion"),
                "Explanation": row.get("Explanation"),
            }
        )

    return pd.DataFrame(rows).set_index("Ticker")


def cap_and_redistribute_weights(weights: pd.Series, max_position_size: float) -> pd.Series:
    """Cap position weights and redistribute excess to uncapped names."""
    weights = weights.dropna()
    if weights.empty:
        return weights

    weights = weights / weights.sum()
    effective_cap = max(max_position_size, 1 / len(weights))

    for _ in range(20):
        over_cap = weights > effective_cap
        if not over_cap.any():
            break

        excess = (weights[over_cap] - effective_cap).sum()
        weights[over_cap] = effective_cap
        under_cap = ~over_cap
        if not under_cap.any() or excess <= 0:
            break

        redistribution_base = weights[under_cap] / weights[under_cap].sum()
        weights[under_cap] += redistribution_base * excess

    return weights / weights.sum()


def calculate_suggested_allocation(
    buy_recommendations: pd.DataFrame,
    returns: pd.DataFrame,
    risk_preference: str,
    allocation_method: str = "Risk-aware",
    max_position_size: float | None = None,
) -> pd.Series:
    """Create risk-aware long-only allocation weights for suggested buys."""
    if buy_recommendations.empty:
        return pd.Series(dtype=float)

    tickers = [ticker for ticker in buy_recommendations.index if ticker in returns.columns]
    if not tickers:
        return pd.Series(dtype=float)

    candidates = buy_recommendations.loc[tickers]
    if allocation_method == "Equal weight":
        raw_weights = pd.Series(1 / len(tickers), index=tickers)
        return cap_and_redistribute_weights(raw_weights, max_position_size or 1.0)

    annualized_vol = returns[tickers].std() * np.sqrt(TRADING_DAYS_PER_YEAR)
    inverse_vol = 1 / annualized_vol.replace(0, np.nan)
    inverse_vol = inverse_vol.fillna(inverse_vol.median()).fillna(1)
    inverse_vol_weights = inverse_vol / inverse_vol.sum()

    if allocation_method == "Inverse volatility":
        return cap_and_redistribute_weights(inverse_vol_weights, max_position_size or 1.0)

    rank_score = candidates["Opportunity Rank Score"].clip(lower=0)
    if rank_score.sum() <= 0:
        rank_weights = pd.Series(1 / len(tickers), index=tickers)
    else:
        rank_weights = rank_score / rank_score.sum()

    allocation_rules = {
        "Conservative": {"inverse_vol": 0.75, "rank": 0.25, "cap": 0.15},
        "Balanced": {"inverse_vol": 0.50, "rank": 0.50, "cap": 0.20},
        "Aggressive": {"inverse_vol": 0.25, "rank": 0.75, "cap": 0.30},
    }
    rule = allocation_rules[risk_preference]
    raw_weights = rule["inverse_vol"] * inverse_vol_weights + rule["rank"] * rank_weights
    return cap_and_redistribute_weights(raw_weights, max_position_size or rule["cap"])


def calculate_portfolio_metrics(
    returns: pd.DataFrame,
    weights: pd.Series,
    candidate_estimates: pd.DataFrame,
    risk_free_rate: float,
    holding_period_days: int,
    transaction_cost: float,
    slippage: float,
) -> dict[str, float]:
    """Calculate proposed-portfolio risk and return estimates."""
    if weights.empty:
        return {
            "expected_return": np.nan,
            "net_expected_return": np.nan,
            "downside_risk": np.nan,
            "annualized_volatility": np.nan,
            "holding_period_volatility": np.nan,
            "sharpe": np.nan,
            "sortino": np.nan,
            "max_drawdown": np.nan,
            "probability_positive": np.nan,
            "average_correlation": np.nan,
            "highest_correlation": np.nan,
        }

    tickers = list(weights.index)
    aligned_returns = returns[tickers].dropna(how="all")
    aligned_weights = weights.reindex(tickers).fillna(0)
    portfolio_returns = calculate_portfolio_returns(aligned_returns, aligned_weights)

    expected_return = (candidate_estimates.loc[tickers, "Expected Return Estimate"] * aligned_weights).sum()
    round_trip_cost = 2 * (transaction_cost + slippage)
    net_expected_return = expected_return - round_trip_cost
    downside_risk = (candidate_estimates.loc[tickers, "Downside Risk Estimate"] * aligned_weights).sum()
    probability_positive = (
        candidate_estimates.loc[tickers, "Probability Positive Historical Return"] * aligned_weights
    ).sum()

    annualized_volatility = portfolio_returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR)
    holding_period_volatility = annualized_volatility * np.sqrt(holding_period_days / 365)
    annualized_expected = (1 + net_expected_return) ** (365 / holding_period_days) - 1
    sharpe = (
        (annualized_expected - risk_free_rate) / annualized_volatility
        if annualized_volatility and annualized_volatility > 0
        else np.nan
    )

    downside_returns = portfolio_returns[portfolio_returns < 0]
    downside_deviation = downside_returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR)
    sortino = (
        (annualized_expected - risk_free_rate) / downside_deviation
        if downside_deviation and downside_deviation > 0
        else np.nan
    )

    corr = aligned_returns[tickers].corr()
    upper_corr = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool)).stack()
    average_correlation = upper_corr.mean() if not upper_corr.empty else np.nan
    highest_correlation = upper_corr.max() if not upper_corr.empty else np.nan

    return {
        "expected_return": expected_return,
        "net_expected_return": net_expected_return,
        "downside_risk": downside_risk,
        "annualized_volatility": annualized_volatility,
        "holding_period_volatility": holding_period_volatility,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": calculate_max_drawdown(portfolio_returns),
        "probability_positive": probability_positive,
        "average_correlation": average_correlation,
        "highest_correlation": highest_correlation,
    }


def generate_portfolio_warnings(
    allocation: pd.DataFrame,
    portfolio_metrics: dict[str, float],
    risk_settings: dict[str, float | str | bool],
) -> list[str]:
    """Generate concentration, volatility, correlation, and risk/reward warnings."""
    warnings = []
    if allocation.empty:
        return ["No allocation generated because there were no qualifying buy recommendations."]

    max_weight = allocation["Suggested Allocation"].max()
    max_position_size = float(risk_settings["max_position_size"])
    if max_weight > max_position_size + 1e-6:
        warnings.append("Position concentration exceeds the configured maximum position size.")
    elif max_weight > 0.30:
        warnings.append("Position concentration is high; one suggested weight exceeds 30%.")
    elif max_weight > 0.20:
        warnings.append("Position concentration is moderate; one suggested weight exceeds 20%.")

    if portfolio_metrics["annualized_volatility"] > 0.35:
        warnings.append("Estimated annualized portfolio volatility is high.")
    if portfolio_metrics["max_drawdown"] < -float(risk_settings["max_portfolio_drawdown"]):
        warnings.append("Historical drawdown estimate breaches the configured portfolio drawdown limit.")
    if portfolio_metrics["highest_correlation"] > 0.85:
        warnings.append("Some suggested holdings are highly correlated, reducing diversification benefit.")
    if portfolio_metrics["net_expected_return"] <= abs(portfolio_metrics["downside_risk"]) * 0.50:
        warnings.append("Expected return estimate may not justify the downside risk estimate.")
    if (allocation["Annualized Volatility"] > float(risk_settings["high_volatility_threshold"])).any():
        warnings.append("At least one suggested buy exceeds the configured high-volatility threshold.")

    return warnings or ["No major portfolio-level warnings from the allocation checks."]


def apply_risk_management_rules(
    buys: pd.DataFrame,
    risk_settings: dict[str, float | str | bool],
) -> pd.DataFrame:
    """Add stop, trailing-stop, take-profit, and per-position risk diagnostics."""
    if buys.empty:
        return buys

    managed = buys.copy()
    stop_loss_pct = float(risk_settings["stop_loss_pct"])
    trailing_stop_pct = float(risk_settings["trailing_stop_pct"])
    take_profit_pct = float(risk_settings["take_profit_pct"])
    max_position_size = float(risk_settings["max_position_size"])
    high_volatility_threshold = float(risk_settings["high_volatility_threshold"])

    managed["Configured Stop Loss"] = managed["Price"] * (1 - stop_loss_pct)
    managed["Configured Trailing Stop"] = managed["Price"] * (1 - trailing_stop_pct)
    managed["Configured Take Profit"] = managed["Price"] * (1 + take_profit_pct)
    managed["Effective Stop Loss"] = managed[
        ["Stop-Loss Suggestion", "Configured Stop Loss", "Configured Trailing Stop"]
    ].max(axis=1)
    managed["Effective Take Profit"] = managed[
        ["Suggested Sell / Take-Profit Zone", "Configured Take Profit"]
    ].min(axis=1)
    managed["Capital At Risk"] = (
        managed["Suggested Dollars"] * ((managed["Price"] - managed["Effective Stop Loss"]) / managed["Price"]).clip(lower=0)
    )
    managed["Position Risk % Capital"] = managed["Capital At Risk"] / managed["Suggested Dollars"].sum()
    managed["Risk Management Warning"] = np.select(
        [
            managed["Suggested Allocation"] > max_position_size + 1e-6,
            managed["Annualized Volatility"] > high_volatility_threshold,
            managed["Recent Drawdown"] < -0.20,
            managed["Expected Return Estimate"] < abs(managed["Downside Risk Estimate"]) * 0.50,
        ],
        [
            "Position size exceeds limit",
            "High volatility",
            "Recent drawdown is high",
            "Expected return may not justify downside",
        ],
        default="Within configured risk checks",
    )
    return managed


def generate_trade_plan(
    buy_recommendations: pd.DataFrame,
    holdings_review: pd.DataFrame,
    returns: pd.DataFrame,
    initial_capital: float,
    max_recommendations: int,
    risk_preference: str,
    risk_free_rate: float,
    holding_period_days: int,
    transaction_cost: float,
    slippage: float,
    risk_settings: dict[str, float | str | bool],
) -> dict[str, pd.DataFrame | float | str]:
    """Generate a one-period trade plan with risk-aware allocation."""
    buys = buy_recommendations.head(max_recommendations).copy()
    allocation_weights = calculate_suggested_allocation(
        buys,
        returns,
        risk_preference,
        allocation_method=str(risk_settings["allocation_method"]),
        max_position_size=float(risk_settings["max_position_size"]),
    )

    if not allocation_weights.empty:
        buys = buys.loc[allocation_weights.index].copy()
        buys["Suggested Allocation"] = allocation_weights
        buys["Suggested Dollars"] = initial_capital * buys["Suggested Allocation"]
        buys["Estimated Dollar Gain"] = buys["Suggested Dollars"] * buys["Expected Return Estimate"]
        buys["Estimated Downside Dollars"] = buys["Suggested Dollars"] * buys["Downside Risk Estimate"]
    else:
        buys["Suggested Allocation"] = pd.Series(dtype=float)
        buys["Suggested Dollars"] = pd.Series(dtype=float)
        buys["Estimated Dollar Gain"] = pd.Series(dtype=float)
        buys["Estimated Downside Dollars"] = pd.Series(dtype=float)

    buys = apply_risk_management_rules(buys, risk_settings)

    portfolio_metrics = calculate_portfolio_metrics(
        returns=returns,
        weights=allocation_weights,
        candidate_estimates=buys,
        risk_free_rate=risk_free_rate,
        holding_period_days=holding_period_days,
        transaction_cost=transaction_cost,
        slippage=slippage,
    )
    portfolio_warnings = generate_portfolio_warnings(buys, portfolio_metrics, risk_settings)

    hold_actions = pd.DataFrame()
    sell_actions = pd.DataFrame()
    if not holdings_review.empty:
        hold_actions = holdings_review[
            holdings_review["Suggested Action"].isin(["Buy more", "Hold", "Avoid adding"])
        ]
        sell_actions = holdings_review[
            holdings_review["Suggested Action"].isin(["Sell", "Trim / partial sell"])
        ]

    return {
        "buys": buys,
        "holds": hold_actions,
        "sells": sell_actions,
        "portfolio_metrics": portfolio_metrics,
        "portfolio_warnings": portfolio_warnings,
        "summary": (
            "Risk-aware allocation blends inverse-volatility sizing with opportunity ranking. "
            "Transaction cost and slippage are included as a simple round-trip drag on expected return."
        ),
    }


def scan_sp500_opportunities(
    planned_buy_date: date,
    holding_period_days: int,
    benchmark: str,
    risk_preference: str,
    max_recommendations: int,
) -> dict[str, pd.DataFrame]:
    """Scan the S&P 500 and return ranked trade candidates."""
    analysis_end = min(planned_buy_date, date.today())
    analysis_start = analysis_end - timedelta(days=365 * DEFAULT_HISTORY_YEARS)
    universe = get_sp500_universe()

    prices = load_data(tuple(universe), analysis_start, analysis_end)
    benchmark_prices = load_data((benchmark,), analysis_start, analysis_end)
    returns = calculate_returns(prices)
    weights = calculate_equal_weights(list(prices.columns))
    metrics = calculate_risk_metrics(returns, risk_free_rate=0.04, weights=weights)
    indicators = calculate_indicators(prices, returns, benchmark_prices)
    bullish_summary = calculate_bullish_score(indicators, metrics)
    holding_stats = calculate_holding_period_stats(returns, holding_period_days)
    ranked = rank_trade_candidates(bullish_summary, holding_stats, risk_preference)

    return {
        "prices": prices,
        "returns": returns,
        "benchmark_prices": benchmark_prices,
        "metrics": metrics,
        "indicators": indicators,
        "all_candidates": bullish_summary.join(holding_stats, how="left"),
        "ranked": ranked,
        "buy_recommendations": ranked.head(max_recommendations),
    }


def get_month_end_trading_dates(index: pd.Index) -> pd.DatetimeIndex:
    """Return the last available trading date for each calendar month."""
    date_index = pd.DatetimeIndex(index).sort_values()
    if date_index.empty:
        return pd.DatetimeIndex([])
    month_ends = date_index.to_series().groupby(date_index.to_period("M")).max()
    return pd.DatetimeIndex(month_ends.values)


def summarize_return_series(returns: pd.Series, risk_free_rate: float) -> dict[str, float]:
    """Summarize monthly backtest returns."""
    clean = returns.dropna()
    if clean.empty:
        return {
            "Cumulative Return": np.nan,
            "Annualized Return": np.nan,
            "Annualized Volatility": np.nan,
            "Sharpe Ratio": np.nan,
            "Max Drawdown": np.nan,
            "Win Rate": np.nan,
            "Average Monthly Return": np.nan,
            "Best Month": np.nan,
            "Worst Month": np.nan,
            "Months": 0,
        }

    cumulative_return = (1 + clean).prod() - 1
    annualized_return = (1 + cumulative_return) ** (12 / len(clean)) - 1 if cumulative_return > -1 else np.nan
    annualized_volatility = clean.std() * np.sqrt(12)
    monthly_rf = (1 + risk_free_rate) ** (1 / 12) - 1
    sharpe = (
        (clean.mean() - monthly_rf) / clean.std() * np.sqrt(12)
        if clean.std() and clean.std() > 0
        else np.nan
    )

    return {
        "Cumulative Return": cumulative_return,
        "Annualized Return": annualized_return,
        "Annualized Volatility": annualized_volatility,
        "Sharpe Ratio": sharpe,
        "Max Drawdown": calculate_max_drawdown(clean),
        "Win Rate": (clean > 0).mean(),
        "Average Monthly Return": clean.mean(),
        "Best Month": clean.max(),
        "Worst Month": clean.min(),
        "Months": int(clean.count()),
    }


def run_backtest(
    prices: pd.DataFrame,
    benchmark_prices: pd.DataFrame,
    holding_period_days: int,
    risk_preference: str,
    top_n: int,
    risk_free_rate: float,
    transaction_cost: float,
    slippage: float,
    risk_settings: dict[str, float | str | bool] | None = None,
    min_history_days: int = 252,
) -> dict[str, pd.DataFrame]:
    """Backtest the monthly S&P 500 ranking system without look-ahead bias.

    At each month-end signal date, the ranker only receives data through that
    date. Trades are entered at the next trading day's close, held to the next
    month-end close, and charged a full round-trip cost assumption.
    """
    if prices.empty or len(prices) < min_history_days + 22:
        return {
            "periods": pd.DataFrame(),
            "summary": pd.DataFrame(),
            "yearly": pd.DataFrame(),
            "rolling": pd.DataFrame(),
            "losing_periods": pd.DataFrame(),
            "drawdowns": pd.DataFrame(),
            "reliability": pd.DataFrame(),
        }

    all_trading_dates = pd.DatetimeIndex(prices.index).sort_values()
    month_ends = get_month_end_trading_dates(all_trading_dates)
    returns = calculate_returns(prices)
    benchmark_returns = calculate_returns(benchmark_prices) if not benchmark_prices.empty else pd.DataFrame()
    effective_risk_settings = risk_settings or {
        "allocation_method": "Risk-aware",
        "max_position_size": 0.20,
    }
    rows = []

    for signal_date, exit_date in zip(month_ends[:-1], month_ends[1:]):
        history = prices.loc[:signal_date]
        if len(history) < min_history_days:
            continue

        entry_position = all_trading_dates.searchsorted(signal_date, side="right")
        if entry_position >= len(all_trading_dates):
            continue
        entry_date = all_trading_dates[entry_position]
        if entry_date >= exit_date:
            continue

        history_returns = returns.loc[:signal_date]
        history_benchmark = benchmark_prices.loc[:signal_date] if not benchmark_prices.empty else pd.DataFrame()

        weights = calculate_equal_weights(list(history.columns))
        history_metrics = calculate_risk_metrics(history_returns, risk_free_rate=risk_free_rate, weights=weights)
        history_indicators = calculate_indicators(history, history_returns, history_benchmark)
        bullish_summary = calculate_bullish_score(history_indicators, history_metrics)
        holding_stats = calculate_holding_period_stats(history_returns, holding_period_days)
        ranked = rank_trade_candidates(bullish_summary, holding_stats, risk_preference)
        selected = ranked.head(top_n)

        allocation = calculate_suggested_allocation(
            selected,
            history_returns,
            risk_preference,
            allocation_method=str(effective_risk_settings["allocation_method"]),
            max_position_size=float(effective_risk_settings["max_position_size"]),
        )
        if allocation.empty:
            continue

        tradable_tickers = [
            ticker
            for ticker in allocation.index
            if ticker in prices.columns
            and pd.notna(prices.at[entry_date, ticker])
            and pd.notna(prices.at[exit_date, ticker])
        ]
        if not tradable_tickers:
            continue

        allocation = allocation.reindex(tradable_tickers).dropna()
        allocation = allocation / allocation.sum()
        period_asset_returns = prices.loc[exit_date, tradable_tickers] / prices.loc[entry_date, tradable_tickers] - 1
        gross_return = (period_asset_returns * allocation).sum()
        cost_drag = 2 * (transaction_cost + slippage)
        strategy_return = gross_return - cost_drag

        benchmark_return = np.nan
        if not benchmark_prices.empty:
            benchmark_col = benchmark_prices.columns[0]
            if entry_date in benchmark_prices.index and exit_date in benchmark_prices.index:
                entry_benchmark = benchmark_prices.at[entry_date, benchmark_col]
                exit_benchmark = benchmark_prices.at[exit_date, benchmark_col]
                if pd.notna(entry_benchmark) and pd.notna(exit_benchmark) and entry_benchmark > 0:
                    benchmark_return = exit_benchmark / entry_benchmark - 1

        rows.append(
            {
                "Signal Date": signal_date,
                "Entry Date": entry_date,
                "Exit Date": exit_date,
                "Strategy Return": strategy_return,
                "Gross Strategy Return": gross_return,
                "Cost Drag": cost_drag,
                "Benchmark Return": benchmark_return,
                "Excess Return vs Benchmark": strategy_return - benchmark_return if pd.notna(benchmark_return) else np.nan,
                "Selected Count": len(tradable_tickers),
                "Qualified Candidates": len(ranked),
                "Average Selected Score": selected.loc[tradable_tickers, "Bullish Score"].mean(),
                "Selected Tickers": ", ".join(tradable_tickers),
            }
        )

    periods = pd.DataFrame(rows)
    if periods.empty:
        return {
            "periods": periods,
            "summary": pd.DataFrame(),
            "yearly": pd.DataFrame(),
            "rolling": pd.DataFrame(),
            "losing_periods": pd.DataFrame(),
            "drawdowns": pd.DataFrame(),
            "reliability": pd.DataFrame(),
        }

    periods = periods.set_index("Exit Date").sort_index()
    strategy_summary = summarize_return_series(periods["Strategy Return"], risk_free_rate)
    benchmark_summary = summarize_return_series(periods["Benchmark Return"], risk_free_rate)
    summary = pd.DataFrame([strategy_summary, benchmark_summary], index=["Strategy", "Benchmark"])
    summary.loc["Strategy - Benchmark"] = summary.loc["Strategy"] - summary.loc["Benchmark"]
    summary.loc["Strategy - Benchmark", "Months"] = summary.loc["Strategy", "Months"]

    yearly = pd.DataFrame(
        {
            "Strategy Return": periods["Strategy Return"].groupby(periods.index.year).apply(lambda x: (1 + x).prod() - 1),
            "Benchmark Return": periods["Benchmark Return"].groupby(periods.index.year).apply(lambda x: (1 + x).prod() - 1),
        }
    )
    yearly["Excess Return vs Benchmark"] = yearly["Strategy Return"] - yearly["Benchmark Return"]

    rolling = pd.DataFrame(index=periods.index)
    rolling["Rolling 6M Strategy Return"] = (1 + periods["Strategy Return"]).rolling(6).apply(np.prod, raw=True) - 1
    rolling["Rolling 6M Benchmark Return"] = (1 + periods["Benchmark Return"]).rolling(6).apply(np.prod, raw=True) - 1
    rolling["Rolling 6M Excess Return"] = rolling["Rolling 6M Strategy Return"] - rolling["Rolling 6M Benchmark Return"]

    strategy_equity = (1 + periods["Strategy Return"]).cumprod()
    benchmark_equity = (1 + periods["Benchmark Return"]).cumprod()
    drawdowns = pd.DataFrame(index=periods.index)
    drawdowns["Strategy Drawdown"] = strategy_equity / strategy_equity.cummax() - 1
    drawdowns["Benchmark Drawdown"] = benchmark_equity / benchmark_equity.cummax() - 1

    losing_periods = periods[periods["Strategy Return"] < 0].sort_values("Strategy Return").head(12)
    sample_months = len(periods)
    beats_benchmark = periods["Strategy Return"].sum() > periods["Benchmark Return"].sum()
    reliability_points = 0
    reliability_points += 30 if sample_months >= 36 else 20 if sample_months >= 24 else 10 if sample_months >= 12 else 0
    reliability_points += 25 if strategy_summary["Win Rate"] >= 0.55 else 15 if strategy_summary["Win Rate"] >= 0.50 else 5
    reliability_points += 20 if strategy_summary["Max Drawdown"] >= -0.20 else 10 if strategy_summary["Max Drawdown"] >= -0.35 else 0
    reliability_points += 15 if periods["Selected Count"].mean() >= max(3, top_n * 0.70) else 5
    reliability_points += 10 if beats_benchmark else 0
    reliability = pd.DataFrame(
        [
            {
                "Reliability Score": min(reliability_points, 100),
                "Backtest Months": sample_months,
                "Average Selected Count": periods["Selected Count"].mean(),
                "Average Qualified Candidates": periods["Qualified Candidates"].mean(),
                "Strategy Beats Benchmark After Costs": bool(
                    summary.loc["Strategy", "Cumulative Return"] > summary.loc["Benchmark", "Cumulative Return"]
                ),
                "Statistical Meaningfulness": "Limited" if sample_months < 24 else "Moderate" if sample_months < 60 else "Stronger",
                "Key Limitation": "Uses the current S&P 500 universe, so historical results have survivorship bias.",
            }
        ]
    )

    return {
        "periods": periods,
        "summary": summary,
        "yearly": yearly,
        "rolling": rolling,
        "losing_periods": losing_periods,
        "drawdowns": drawdowns,
        "reliability": reliability,
    }


def calculate_max_drawdown(return_series: pd.Series) -> float:
    equity_curve = (1 + return_series.dropna()).cumprod()
    if equity_curve.empty:
        return np.nan
    drawdown = equity_curve / equity_curve.cummax() - 1
    return float(drawdown.min())


def calculate_risk_metrics(
    returns: pd.DataFrame,
    risk_free_rate: float,
    weights: pd.Series | None = None,
) -> pd.DataFrame:
    """Calculate basic historical risk/return estimates."""
    daily_rf = risk_free_rate / TRADING_DAYS_PER_YEAR
    rows = []

    for ticker in returns.columns:
        series = returns[ticker].dropna()
        if series.empty:
            continue
        total_return = (1 + series).prod() - 1
        annualized_return = (1 + series.mean()) ** TRADING_DAYS_PER_YEAR - 1
        volatility = series.std()
        annualized_volatility = volatility * np.sqrt(TRADING_DAYS_PER_YEAR)
        excess = series - daily_rf
        sharpe = (
            excess.mean() / series.std() * np.sqrt(TRADING_DAYS_PER_YEAR)
            if series.std() > 0
            else np.nan
        )

        rows.append(
            {
                "Ticker": ticker,
                "Weight": weights.get(ticker, np.nan) if weights is not None else np.nan,
                "Total Return": total_return,
                "Annualized Return Estimate": annualized_return,
                "Daily Volatility": volatility,
                "Annualized Volatility": annualized_volatility,
                "Sharpe Ratio Estimate": sharpe,
                "Max Drawdown": calculate_max_drawdown(series),
                "Observations": int(series.count()),
            }
        )

    return pd.DataFrame(rows).set_index("Ticker") if rows else pd.DataFrame()


def calculate_portfolio_returns(returns: pd.DataFrame, weights: pd.Series) -> pd.Series:
    aligned_weights = weights.reindex(returns.columns).fillna(0)
    return returns.mul(aligned_weights, axis=1).sum(axis=1)


def calculate_equal_weights(tickers: list[str]) -> pd.Series:
    if not tickers:
        return pd.Series(dtype=float)
    return pd.Series(1 / len(tickers), index=tickers)


def render_metric_cards(metrics: pd.DataFrame, portfolio_returns: pd.Series) -> None:
    if metrics.empty or portfolio_returns.empty:
        return

    portfolio_metrics = calculate_risk_metrics(
        portfolio_returns.to_frame("Portfolio"),
        st.session_state["risk_free_rate"],
    )

    row = portfolio_metrics.loc["Portfolio"]
    cols = st.columns(4)
    cols[0].metric("Portfolio Total Return", f"{row['Total Return']:.2%}")
    cols[1].metric("Ann. Return Estimate", f"{row['Annualized Return Estimate']:.2%}")
    cols[2].metric("Ann. Volatility", f"{row['Annualized Volatility']:.2%}")
    cols[3].metric("Sharpe Estimate", f"{row['Sharpe Ratio Estimate']:.2f}")

    st.metric("Max Drawdown", f"{row['Max Drawdown']:.2%}")


def render_technical_chart(ticker: str, indicator_frame: pd.DataFrame) -> None:
    chart_data = indicator_frame.dropna(subset=["Price"])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=chart_data.index, y=chart_data["Price"], name="Price", line={"width": 2}))

    for column, color in [("MA20", "#4C78A8"), ("MA50", "#F58518"), ("MA200", "#54A24B")]:
        fig.add_trace(
            go.Scatter(
                x=chart_data.index,
                y=chart_data[column],
                name=column,
                line={"width": 1.4, "color": color},
            )
        )

    fig.add_trace(
        go.Scatter(
            x=chart_data.index,
            y=chart_data["Bollinger Upper"],
            name="Bollinger Upper",
            line={"width": 1, "dash": "dot", "color": "#B279A2"},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=chart_data.index,
            y=chart_data["Bollinger Lower"],
            name="Bollinger Lower",
            line={"width": 1, "dash": "dot", "color": "#B279A2"},
        )
    )

    fig.update_layout(
        title=f"{ticker} technical levels",
        xaxis_title="Date",
        yaxis_title="Adjusted Close",
        legend_title_text="Indicator",
        template="plotly_dark",
        paper_bgcolor="#0b1020",
        plot_bgcolor="#111827",
        font={"color": "#f8fafc"},
        hovermode="x unified",
        margin={"l": 20, "r": 20, "t": 55, "b": 25},
    )
    fig.update_xaxes(gridcolor="#334155", zerolinecolor="#94a3b8")
    fig.update_yaxes(gridcolor="#334155", zerolinecolor="#94a3b8")
    st.plotly_chart(fig, use_container_width=True)


def render_dashboard() -> None:
    apply_ui_theme()
    st.markdown(
        """
        <div class="app-hero">
            <h1>S&P 500 Long-Term Bullish Opportunity Scanner</h1>
            <p>One-month research workflow for bullish, long-only trade ideas with technical ranking, allocation, backtesting, manual scenarios, and risk controls.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    render_badges(
        [
            ("S&P 500 stocks only", "good"),
            ("Long-only", "good"),
            ("Holding period > 1 week", "good"),
            ("Historical estimates", "warn"),
            ("Not financial advice", "bad"),
        ]
    )

    with st.sidebar:
        st.header("Scanner Inputs")
        today = date.today()
        planned_buy_date = st.date_input("Planned buy date", value=today)
        holding_period_days = st.number_input(
            "Expected holding period, calendar days",
            min_value=MIN_HOLDING_DAYS,
            max_value=365,
            value=30,
            step=1,
        )
        raw_holdings = st.text_area("Current holdings, optional", value="")
        initial_capital = st.number_input("Initial capital", min_value=100.0, value=10000.0, step=500.0)
        max_recommendations = st.number_input(
            "Maximum number of suggested buys",
            min_value=1,
            max_value=50,
            value=10,
            step=1,
        )
        risk_preference = st.selectbox("Risk preference", ["Conservative", "Balanced", "Aggressive"], index=1)
        benchmark = normalize_ticker(st.text_input("Benchmark ticker", value="SPY"))
        risk_free_rate = st.number_input("Risk-free rate", min_value=0.0, max_value=0.25, value=0.04, step=0.005)
        st.session_state["risk_free_rate"] = risk_free_rate

        transaction_cost = st.number_input("Transaction cost per trade", min_value=0.0, value=0.001, step=0.0005, format="%.4f")
        slippage = st.number_input("Slippage per trade", min_value=0.0, value=0.0005, step=0.0005, format="%.4f")
        run_monthly_backtest = st.checkbox("Run monthly ranking backtest", value=False)
        show_advanced_details = st.checkbox("Show advanced diagnostics", value=False)
        with st.expander("Risk management"):
            allocation_method = st.selectbox(
                "Allocation method",
                ["Risk-aware", "Equal weight", "Inverse volatility"],
                index=0,
            )
            max_position_size = st.slider(
                "Maximum position size",
                min_value=0.05,
                max_value=0.50,
                value=0.20,
                step=0.01,
                format="%.2f",
            )
            stop_loss_pct = st.slider(
                "Stop-loss level",
                min_value=0.02,
                max_value=0.30,
                value=0.08,
                step=0.01,
                format="%.2f",
            )
            trailing_stop_pct = st.slider(
                "Trailing stop",
                min_value=0.02,
                max_value=0.35,
                value=0.10,
                step=0.01,
                format="%.2f",
            )
            take_profit_pct = st.slider(
                "Take-profit target",
                min_value=0.03,
                max_value=0.60,
                value=0.12,
                step=0.01,
                format="%.2f",
            )
            max_portfolio_drawdown = st.slider(
                "Maximum portfolio drawdown limit",
                min_value=0.05,
                max_value=0.60,
                value=0.25,
                step=0.01,
                format="%.2f",
            )
            high_volatility_threshold = st.slider(
                "High-volatility warning threshold",
                min_value=0.15,
                max_value=0.80,
                value=0.40,
                step=0.01,
                format="%.2f",
            )
            use_volatility_sizing = allocation_method in ["Risk-aware", "Inverse volatility"]
            st.caption(
                "Volatility-based sizing is active for Risk-aware and Inverse volatility allocation methods."
            )
        risk_settings = {
            "allocation_method": allocation_method,
            "max_position_size": max_position_size,
            "stop_loss_pct": stop_loss_pct,
            "trailing_stop_pct": trailing_stop_pct,
            "take_profit_pct": take_profit_pct,
            "max_portfolio_drawdown": max_portfolio_drawdown,
            "high_volatility_threshold": high_volatility_threshold,
            "use_volatility_sizing": use_volatility_sizing,
        }
        scenario_assumptions = []
        with st.expander("Scenario settings"):
            st.caption("Manual assumptions only. The app does not fetch or interpret live news in version 1.")
            enable_scenarios = st.checkbox("Apply manual scenario adjustments", value=False)
            selected_scenarios = st.multiselect(
                "Scenario categories",
                SCENARIO_CATEGORIES,
                default=[],
                disabled=not enable_scenarios,
            )
            for scenario_name in selected_scenarios:
                st.write(scenario_name)
                direction = st.selectbox(
                    "Direction",
                    ["Neutral", "Bullish", "Bearish"],
                    index=0,
                    key=f"scenario_direction_{scenario_name}",
                )
                severity = st.slider(
                    "Severity",
                    min_value=1,
                    max_value=5,
                    value=3,
                    key=f"scenario_severity_{scenario_name}",
                )
                probability = st.slider(
                    "Probability",
                    min_value=0,
                    max_value=100,
                    value=25,
                    step=5,
                    key=f"scenario_probability_{scenario_name}",
                )
                scenario_assumptions.append(
                    {
                        "category": scenario_name,
                        "direction": direction,
                        "severity": severity,
                        "probability": probability,
                    }
                )
        render_fine_print("Version 1 excludes ETFs, commodities, futures, crypto, forex, and non-S&P 500 assets.")
        render_fine_print(f"Cost assumptions: transaction cost {transaction_cost:.2%}, slippage {slippage:.2%}.")
        run_scan = st.button("Run S&P 500 scan", type="primary")

    if holding_period_days < MIN_HOLDING_DAYS:
        st.error("Holding period must be greater than 1 week. Choose dates at least 7 calendar days apart.")
        return

    current_holdings = parse_tickers(raw_holdings)
    valid_holdings, invalid_holdings = validate_sp500_tickers(current_holdings)

    if invalid_holdings:
        st.warning(
            "Excluded current holdings that are not in the built-in S&P 500 universe: "
            + ", ".join(invalid_holdings)
        )

    if planned_buy_date > today:
        st.warning(
            "The planned buy date is in the future, so the scanner uses the latest available historical data "
            f"through {today.isoformat()}."
        )

    render_fine_print(
        "Research tool only. Recommendations are historical-data estimates and assumptions, not financial advice or guaranteed predictions."
    )

    if not run_scan:
        st.info("Set the inputs, then run the S&P 500 scan.")
        return

    with st.spinner("Scanning the S&P 500 universe with yfinance data..."):
        scan_results = scan_sp500_opportunities(
            planned_buy_date=planned_buy_date,
            holding_period_days=int(holding_period_days),
            benchmark=benchmark,
            risk_preference=risk_preference,
            max_recommendations=int(max_recommendations),
        )

    prices = scan_results["prices"]
    returns = scan_results["returns"]
    metrics = scan_results["metrics"]
    indicators = scan_results["indicators"]
    scenario_results = run_scenario_analysis(scan_results["all_candidates"], scenario_assumptions)
    all_candidates = scenario_results["adjusted_candidates"]
    ranked = rank_trade_candidates(all_candidates, pd.DataFrame(), risk_preference)
    if "Scenario Note" in ranked.columns:
        ranked["Risk Warning"] = np.where(
            ranked["Scenario Note"].str.contains("increase risk", na=False),
            ranked["Risk Warning"] + "; scenario risk elevated",
            ranked["Risk Warning"],
        )
    buy_recommendations = ranked.head(int(max_recommendations))

    if scan_results["benchmark_prices"].empty:
        st.warning(
            f"No usable benchmark data returned for {benchmark}. Relative strength score components will be unavailable."
        )

    if prices.empty:
        st.error("No S&P 500 price data was returned from yfinance. Try again later.")
        return

    holdings_review = classify_current_holdings(valid_holdings, ranked, all_candidates)
    trade_plan = generate_trade_plan(
        buy_recommendations,
        holdings_review,
        returns,
        initial_capital,
        int(max_recommendations),
        risk_preference,
        risk_free_rate,
        int(holding_period_days),
        transaction_cost,
        slippage,
        risk_settings,
    )
    portfolio_metrics = trade_plan["portfolio_metrics"]

    if not buy_recommendations.empty:
        metadata = get_company_metadata(tuple(buy_recommendations.index))
        buy_recommendations = buy_recommendations.join(metadata, how="left")
        trade_plan["buys"] = trade_plan["buys"].join(metadata, how="left")
        trade_plan["buys"]["Company"] = trade_plan["buys"]["Company"].fillna(trade_plan["buys"].index.to_series())
        buy_recommendations["Company"] = buy_recommendations["Company"].fillna(buy_recommendations.index.to_series())
        buy_recommendations["Company Name"] = buy_recommendations["Company Name"].fillna(buy_recommendations.index.to_series())
        buy_recommendations["Sector"] = buy_recommendations["Sector"].fillna("Unavailable")

    render_section_label("Executive Overview")
    cols = st.columns(4)
    cols[0].metric("S&P 500 Stocks Screened", f"{len(prices.columns)}")
    cols[1].metric("Stocks Passing Buy Filters", f"{len(ranked)}")
    cols[2].metric("Risk Preference", risk_preference)
    cols[3].metric("Max Suggested Buys", f"{int(max_recommendations)}")
    risk_rule_cols = st.columns(4)
    risk_rule_cols[0].metric("Allocation Method", allocation_method)
    risk_rule_cols[1].metric("Max Position Size", f"{max_position_size:.0%}")
    risk_rule_cols[2].metric("Stop / Trail", f"{stop_loss_pct:.0%} / {trailing_stop_pct:.0%}")
    risk_rule_cols[3].metric("Take Profit", f"{take_profit_pct:.0%}")

    top_candidate = buy_recommendations["Company"].iloc[0] if not buy_recommendations.empty else "None"
    warning_count = len([warning for warning in trade_plan["portfolio_warnings"] if "No major" not in warning])
    overview_badges = [
        (f"Top candidate: {top_candidate}", "good" if top_candidate != "None" else "warn"),
        (f"Horizon: {int(holding_period_days)} days", "good"),
        (f"Warnings: {warning_count}", "warn" if warning_count else "good"),
        (f"Allocation: {allocation_method}", "good"),
    ]
    render_badges(overview_badges)

    if not buy_recommendations.empty:
        top_chart = buy_recommendations.head(10).sort_values("Bullish Score")
        score_fig = px.bar(
            top_chart,
            x="Bullish Score",
            y="Company",
            orientation="h",
            color="Risk Score",
            color_continuous_scale=[
                [0.00, "#a11212"],
                [0.45, "#d97706"],
                [0.70, "#facc15"],
                [1.00, "#006d3f"],
            ],
            labels={"Company": "Company", "Bullish Score": "Bullish score"},
            title="Top candidate score profile",
        )
        score_fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="#0b1020",
            plot_bgcolor="#111827",
            font={"color": "#f8fafc"},
            margin={"l": 20, "r": 20, "t": 55, "b": 25},
            height=360,
            coloraxis_colorbar_title="Risk score",
        )
        score_fig.update_xaxes(gridcolor="#334155", zerolinecolor="#94a3b8")
        score_fig.update_yaxes(gridcolor="#334155")
        st.plotly_chart(score_fig, use_container_width=True)

    render_section_label("Scenario Analysis")
    if scenario_results["scenario_summary"].empty:
        st.info("No manual scenario adjustments are active for this month.")
    else:
        st.caption(
            "Scenario adjustments are manual assumptions layered onto historical estimates. "
            "They are not facts, forecasts, or live news analysis."
        )
        st.dataframe(
            scenario_results["scenario_summary"].style.format(
                {
                    "Severity": "{:.0f}",
                    "Probability": "{:.0%}",
                    "Avg Return Adjustment": "{:.2%}",
                    "Avg Risk Adjustment": "{:.2%}",
                }
            ),
            use_container_width=True,
        )

    render_section_label("Buy Recommendations")
    if buy_recommendations.empty:
        st.warning("No stocks met the current scanner thresholds. Consider a different risk preference or review later.")
    else:
        display_buys = buy_recommendations.copy()
        buy_cols = [
            "Company",
            "Sector",
            "Bullish Score",
            "Suggested Buy Zone",
            "Suggested Sell / Take-Profit Zone",
            "Stop-Loss Suggestion",
            "Expected Return Estimate",
            "Downside Risk Estimate",
            "Probability Positive Historical Return",
            "Main Reason",
            "Risk Warning",
        ]
        st.dataframe(
            style_numeric_table(
                display_buys[buy_cols].style.format(
                    {
                        "Bullish Score": "{:.1f}",
                        "Suggested Buy Zone": "${:.2f}",
                        "Suggested Sell / Take-Profit Zone": "${:.2f}",
                        "Stop-Loss Suggestion": "${:.2f}",
                        "Expected Return Estimate": "{:.2%}",
                        "Downside Risk Estimate": "{:.2%}",
                        "Probability Positive Historical Return": "{:.1%}",
                    }
                ),
                ["Bullish Score", "Probability Positive Historical Return"],
            ),
            use_container_width=True,
        )

        st.caption(
            "Expected return and probability estimates use historical rolling holding-period returns. "
            "They are not predictions or guarantees."
        )

    render_section_label("Current Holdings Review")
    if holdings_review.empty:
        st.info("Add current holdings in the sidebar if you want a buy/hold/sell review.")
    else:
        st.dataframe(
            holdings_review.style.format(
                {
                    "Current Score": "{:.1f}",
                    "1M Risk/Reward": "{:.2f}",
                    "Expected Return Estimate": "{:.2%}",
                    "Downside Risk Estimate": "{:.2%}",
                    "Scenario Return Adjustment": "{:.2%}",
                    "Scenario Risk Adjustment": "{:.2%}",
                    "Suggested Sell Zone": "${:.2f}",
                    "Suggested Stop Loss": "${:.2f}",
                }
            ),
            use_container_width=True,
        )

    render_section_label("Trade Plan")
    st.caption(trade_plan["summary"])
    plan_cols = st.columns(4)
    plan_cols[0].metric("Gross Expected Return Estimate", format_percent(portfolio_metrics["expected_return"]))
    plan_cols[1].metric("Net Expected Return Estimate", format_percent(portfolio_metrics["net_expected_return"]))
    plan_cols[2].metric("Estimated Downside Risk", format_percent(portfolio_metrics["downside_risk"]))
    plan_cols[3].metric("Probability Positive", format_percent(portfolio_metrics["probability_positive"], 1))

    risk_cols = st.columns(4)
    risk_cols[0].metric("Ann. Volatility Estimate", format_percent(portfolio_metrics["annualized_volatility"]))
    risk_cols[1].metric("Sharpe Estimate", format_number(portfolio_metrics["sharpe"]))
    risk_cols[2].metric("Sortino Estimate", format_number(portfolio_metrics["sortino"]))
    risk_cols[3].metric("Max Drawdown Estimate", format_percent(portfolio_metrics["max_drawdown"]))

    corr_cols = st.columns(3)
    corr_cols[0].metric("Holding-Period Volatility", format_percent(portfolio_metrics["holding_period_volatility"]))
    corr_cols[1].metric("Avg. Pairwise Correlation", format_number(portfolio_metrics["average_correlation"]))
    corr_cols[2].metric("Highest Pairwise Correlation", format_number(portfolio_metrics["highest_correlation"]))

    if not trade_plan["buys"].empty and "Scenario Return Adjustment" in trade_plan["buys"]:
        weighted_scenario_return = (
            trade_plan["buys"]["Suggested Allocation"] * trade_plan["buys"]["Scenario Return Adjustment"]
        ).sum()
        weighted_scenario_risk = (
            trade_plan["buys"]["Suggested Allocation"] * trade_plan["buys"]["Scenario Risk Adjustment"]
        ).sum()
        scenario_cols = st.columns(2)
        scenario_cols[0].metric("Weighted Scenario Return Tilt", format_percent(weighted_scenario_return))
        scenario_cols[1].metric("Weighted Scenario Risk Tilt", format_percent(weighted_scenario_risk))

    if not trade_plan["buys"].empty:
        allocation_cols = [
            "Company",
            "Suggested Allocation",
            "Suggested Dollars",
            "Expected Return Estimate",
            "Estimated Dollar Gain",
            "Downside Risk Estimate",
            "Effective Stop Loss",
            "Effective Take Profit",
            "Capital At Risk",
            "Risk Management Warning",
            "Risk Warning",
        ]
        render_section_label("What To Buy")
        allocation_chart = trade_plan["buys"].copy()
        allocation_chart["Company"] = allocation_chart["Company"].fillna(allocation_chart.index.to_series())
        allocation_fig = px.pie(
            allocation_chart,
            names="Company",
            values="Suggested Allocation",
            hole=0.48,
            color_discrete_sequence=[
                "#0f4c81",
                "#006d3f",
                "#d97706",
                "#7c3aed",
                "#be123c",
                "#0891b2",
                "#4d7c0f",
                "#9333ea",
                "#b45309",
                "#1d4ed8",
            ],
            title="Suggested portfolio allocation for the selected holding period",
        )
        allocation_fig.update_traces(
            textinfo="percent+label",
            textposition="inside",
            hovertemplate="%{label}<br>Allocation: %{percent}<extra></extra>",
        )
        allocation_fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="#0b1020",
            plot_bgcolor="#111827",
            font={"color": "#f8fafc"},
            margin={"l": 20, "r": 20, "t": 55, "b": 20},
            height=430,
            showlegend=True,
        )
        st.plotly_chart(allocation_fig, use_container_width=True)
        st.caption(
            "This is the proposed one-period portfolio shape based on the scanner score, risk preference, scenario assumptions, and risk controls."
        )

        st.dataframe(
            style_numeric_table(
                trade_plan["buys"][allocation_cols].style.format(
                    {
                        "Suggested Allocation": "{:.1%}",
                        "Suggested Dollars": "${:,.0f}",
                        "Expected Return Estimate": "{:.2%}",
                        "Estimated Dollar Gain": "${:,.0f}",
                        "Downside Risk Estimate": "{:.2%}",
                        "Effective Stop Loss": "${:.2f}",
                        "Effective Take Profit": "${:.2f}",
                        "Capital At Risk": "${:,.0f}",
                    }
                ),
                ["Suggested Allocation", "Expected Return Estimate"],
            ),
            use_container_width=True,
        )

        total_capital_at_risk = trade_plan["buys"]["Capital At Risk"].sum()
        total_position_risk = total_capital_at_risk / initial_capital if initial_capital else np.nan
        rm_cols = st.columns(3)
        rm_cols[0].metric("Capital At Stop Risk", format_money(total_capital_at_risk))
        rm_cols[1].metric("Stop Risk % Capital", format_percent(total_position_risk))
        rm_cols[2].metric("Max Drawdown Limit", f"{max_portfolio_drawdown:.0%}")

        render_section_label("Portfolio Allocation Warnings")
        for warning in trade_plan["portfolio_warnings"]:
            st.warning(warning) if "No major" not in warning else st.info(warning)

        selected_returns = returns[list(trade_plan["buys"].index)].dropna(how="all")
        if show_advanced_details and len(trade_plan["buys"]) > 1 and not selected_returns.empty:
            render_section_label("Suggested-Buy Correlation Matrix")
            corr_matrix = selected_returns.corr()
            corr_fig = px.imshow(
                corr_matrix,
                text_auto=".2f",
                color_continuous_scale="RdBu_r",
                zmin=-1,
                zmax=1,
                aspect="auto",
            )
            corr_fig.update_layout(
                template="plotly_dark",
                paper_bgcolor="#0b1020",
                plot_bgcolor="#111827",
                font={"color": "#f8fafc"},
                margin={"l": 20, "r": 20, "t": 25, "b": 25},
            )
            st.plotly_chart(corr_fig, use_container_width=True)

    if run_monthly_backtest:
        render_section_label("Monthly Ranking Backtest")
        st.warning(
            "Backtest caveats: this is not a prediction, uses current S&P 500 membership, and therefore has "
            "survivorship bias. Signals use only data available through each month-end signal date. "
            "Manual scenario assumptions are not applied to historical backtest periods."
        )
        with st.spinner("Running no-look-ahead monthly ranking backtest..."):
            backtest = run_backtest(
                prices=prices,
                benchmark_prices=scan_results["benchmark_prices"],
                holding_period_days=int(holding_period_days),
                risk_preference=risk_preference,
                top_n=int(max_recommendations),
                risk_free_rate=risk_free_rate,
                transaction_cost=transaction_cost,
                slippage=slippage,
                risk_settings=risk_settings,
            )

        if backtest["summary"].empty:
            st.warning("Not enough historical data to run the monthly backtest. Try a longer data window later.")
        else:
            summary_cols = [
                "Cumulative Return",
                "Annualized Return",
                "Annualized Volatility",
                "Sharpe Ratio",
                "Max Drawdown",
                "Win Rate",
                "Average Monthly Return",
                "Best Month",
                "Worst Month",
                "Months",
            ]
            st.dataframe(
                style_numeric_table(
                    backtest["summary"][summary_cols].style.format(
                        {
                            "Cumulative Return": "{:.2%}",
                            "Annualized Return": "{:.2%}",
                            "Annualized Volatility": "{:.2%}",
                            "Sharpe Ratio": "{:.2f}",
                            "Max Drawdown": "{:.2%}",
                            "Win Rate": "{:.1%}",
                            "Average Monthly Return": "{:.2%}",
                            "Best Month": "{:.2%}",
                            "Worst Month": "{:.2%}",
                            "Months": "{:.0f}",
                        }
                    ),
                    ["Annualized Return", "Sharpe Ratio", "Win Rate"],
                ),
                use_container_width=True,
            )

            bt_periods = backtest["periods"]
            equity = pd.DataFrame(index=bt_periods.index)
            equity["Strategy"] = initial_capital * (1 + bt_periods["Strategy Return"]).cumprod()
            equity["Benchmark"] = initial_capital * (1 + bt_periods["Benchmark Return"]).cumprod()
            equity_fig = px.line(
                equity,
                x=equity.index,
                y=equity.columns,
                labels={"value": "Equity Value", "Exit Date": "Date"},
                title="Backtest Equity Curve",
            )
            equity_fig.update_layout(
                template="plotly_dark",
                paper_bgcolor="#0b1020",
                plot_bgcolor="#111827",
                font={"color": "#f8fafc"},
                hovermode="x unified",
                margin={"l": 20, "r": 20, "t": 55, "b": 25},
            )
            equity_fig.update_xaxes(gridcolor="#334155", zerolinecolor="#94a3b8")
            equity_fig.update_yaxes(gridcolor="#334155", zerolinecolor="#94a3b8")
            st.plotly_chart(equity_fig, use_container_width=True)

            drawdown_fig = px.line(
                backtest["drawdowns"],
                x=backtest["drawdowns"].index,
                y=backtest["drawdowns"].columns,
                labels={"value": "Drawdown", "Exit Date": "Date"},
                title="Backtest Drawdowns",
            )
            drawdown_fig.update_layout(yaxis_tickformat=".0%")
            drawdown_fig.update_layout(
                template="plotly_dark",
                paper_bgcolor="#0b1020",
                plot_bgcolor="#111827",
                font={"color": "#f8fafc"},
                hovermode="x unified",
                margin={"l": 20, "r": 20, "t": 55, "b": 25},
            )
            drawdown_fig.update_xaxes(gridcolor="#334155", zerolinecolor="#94a3b8")
            drawdown_fig.update_yaxes(gridcolor="#334155", zerolinecolor="#94a3b8")
            st.plotly_chart(drawdown_fig, use_container_width=True)

            yearly_cols = ["Strategy Return", "Benchmark Return", "Excess Return vs Benchmark"]
            render_section_label("Performance By Year")
            st.dataframe(
                backtest["yearly"][yearly_cols].style.format(
                    {
                        "Strategy Return": "{:.2%}",
                        "Benchmark Return": "{:.2%}",
                        "Excess Return vs Benchmark": "{:.2%}",
                    }
                ),
                use_container_width=True,
            )

            rolling_fig = px.line(
                backtest["rolling"],
                x=backtest["rolling"].index,
                y=backtest["rolling"].columns,
                labels={"value": "Rolling Return", "Exit Date": "Date"},
                title="Rolling 6-Month Performance",
            )
            rolling_fig.update_layout(yaxis_tickformat=".0%")
            rolling_fig.update_layout(
                template="plotly_dark",
                paper_bgcolor="#0b1020",
                plot_bgcolor="#111827",
                font={"color": "#f8fafc"},
                hovermode="x unified",
                margin={"l": 20, "r": 20, "t": 55, "b": 25},
            )
            rolling_fig.update_xaxes(gridcolor="#334155", zerolinecolor="#94a3b8")
            rolling_fig.update_yaxes(gridcolor="#334155", zerolinecolor="#94a3b8")
            st.plotly_chart(rolling_fig, use_container_width=True)

            render_section_label("Worst Losing Periods")
            if backtest["losing_periods"].empty:
                st.info("No losing monthly periods in this limited backtest sample.")
            else:
                losing_cols = [
                    "Signal Date",
                    "Entry Date",
                    "Strategy Return",
                    "Benchmark Return",
                    "Excess Return vs Benchmark",
                    "Selected Count",
                    "Selected Tickers",
                ]
                st.dataframe(
                    backtest["losing_periods"][losing_cols].style.format(
                        {
                            "Strategy Return": "{:.2%}",
                            "Benchmark Return": "{:.2%}",
                            "Excess Return vs Benchmark": "{:.2%}",
                            "Selected Count": "{:.0f}",
                        }
                    ),
                    use_container_width=True,
                )

            if show_advanced_details:
                render_section_label("Backtest Trade Log")
                trade_log_cols = [
                    "Signal Date",
                    "Entry Date",
                    "Strategy Return",
                    "Gross Strategy Return",
                    "Cost Drag",
                    "Benchmark Return",
                    "Excess Return vs Benchmark",
                    "Selected Count",
                    "Qualified Candidates",
                    "Average Selected Score",
                    "Selected Tickers",
                ]
                st.dataframe(
                    bt_periods[trade_log_cols].style.format(
                        {
                            "Strategy Return": "{:.2%}",
                            "Gross Strategy Return": "{:.2%}",
                            "Cost Drag": "{:.2%}",
                            "Benchmark Return": "{:.2%}",
                            "Excess Return vs Benchmark": "{:.2%}",
                            "Selected Count": "{:.0f}",
                            "Qualified Candidates": "{:.0f}",
                            "Average Selected Score": "{:.1f}",
                        }
                    ),
                    use_container_width=True,
                )

                render_section_label("Model Reliability")
                st.dataframe(
                    backtest["reliability"].style.format(
                        {
                            "Reliability Score": "{:.0f}",
                            "Backtest Months": "{:.0f}",
                            "Average Selected Count": "{:.1f}",
                            "Average Qualified Candidates": "{:.1f}",
                        }
                    ),
                    use_container_width=True,
                )

    if not trade_plan["holds"].empty:
        render_section_label("What To Hold / Avoid Adding")
        st.dataframe(trade_plan["holds"], use_container_width=True)

    if not trade_plan["sells"].empty:
        render_section_label("What To Sell Or Trim")
        st.dataframe(trade_plan["sells"], use_container_width=True)

    avoid = all_candidates.sort_values("Bullish Score", ascending=True).head(10)
    if not avoid.empty:
        avoid_metadata = get_company_metadata(tuple(avoid.index))
        avoid = avoid.join(avoid_metadata, how="left")
        avoid["Company"] = avoid["Company"].fillna(avoid.index.to_series())
        avoid["Sector"] = avoid["Sector"].fillna("Unavailable")
    render_section_label("Stocks To Avoid")
    st.dataframe(
        avoid[
            [
                "Company",
                "Sector",
                "Bullish Score",
                "Interpretation",
                "Recent Drawdown",
                "Explanation",
            ]
        ].style.format(
            {
                "Bullish Score": "{:.1f}",
                "Recent Drawdown": "{:.1%}",
            }
        ),
        use_container_width=True,
    )

    warnings_table = ranked[ranked["Risk Warning"] != "No major scanner warning"].head(20)
    render_section_label("Risk Warnings")
    if warnings_table.empty:
        st.info("No major scanner warnings among the top ranked candidates.")
    else:
        st.dataframe(
            warnings_table[
                [
                    "Bullish Score",
                    "Risk Warning",
                    "Annualized Volatility",
                    "Recent Drawdown",
                    "Volatility Regime",
                    "Downside Risk Estimate",
                ]
            ].style.format(
                {
                    "Bullish Score": "{:.1f}",
                    "Annualized Volatility": "{:.1%}",
                    "Recent Drawdown": "{:.1%}",
                    "Volatility Regime": "{:.2f}x",
                    "Downside Risk Estimate": "{:.2%}",
                }
            ),
            use_container_width=True,
        )

    if show_advanced_details and indicators:
        render_section_label("Manual Review Chart")
        inspectable = list(dict.fromkeys(list(buy_recommendations.index[:10]) + valid_holdings))
        if not inspectable:
            inspectable = list(indicators.keys())[:10]
        selected_ticker = st.selectbox("Ticker to inspect", options=inspectable)
        render_technical_chart(selected_ticker, indicators[selected_ticker])

    if show_advanced_details:
        with st.expander("Full Scanner Detail"):
            if not all_candidates.empty:
                st.dataframe(
                    all_candidates.sort_values("Bullish Score", ascending=False).style.format(
                        {
                            "Price": "${:.2f}",
                            "RSI14": "{:.1f}",
                            "MA20": "${:.2f}",
                            "MA50": "${:.2f}",
                            "MA200": "${:.2f}",
                            "52W Position": "{:.1%}",
                            "Discount From 52W High": "{:.1%}",
                            "Recent Drawdown": "{:.1%}",
                            "Volatility Regime": "{:.2f}x",
                            "Momentum 1M": "{:.1%}",
                            "Momentum 3M": "{:.1%}",
                            "Momentum 6M": "{:.1%}",
                            "Momentum 12M": "{:.1%}",
                            "Baseline Expected Return Estimate": "{:.2%}",
                            "Scenario Return Adjustment": "{:.2%}",
                            "Scenario Risk Adjustment": "{:.2%}",
                            "Expected Return Estimate": "{:.2%}",
                            "Downside Risk Estimate": "{:.2%}",
                            "Probability Positive Historical Return": "{:.1%}",
                            "Suggested Buy Zone": "${:.2f}",
                            "Suggested Sell / Take-Profit Zone": "${:.2f}",
                            "Stop-Loss Suggestion": "${:.2f}",
                        }
                    ),
                    use_container_width=True,
                )

    if show_advanced_details:
        with st.expander("Method Notes"):
            st.write(
                """
                - The scanner ranks current S&P 500 stocks only.
                - Scores use historical adjusted close prices from yfinance.
                - The bullish score uses moving averages, RSI, MACD, Bollinger Bands, 52-week position,
                  volatility regime, recent drawdown, momentum, relative strength, and a simple risk/reward estimate.
                - Expected return estimates use historical rolling returns over the selected holding period.
                - Step 4 backtesting ranks stocks at each month-end using only data available then, enters at the next
                  trading day's close, holds to the next month-end close, and subtracts round-trip cost/slippage.
                - The current backtest uses today's built-in S&P 500 universe, so it has survivorship bias.
                - Scenario analysis uses manual user assumptions only; it does not fetch live news.
                - This app is for research and education, not financial advice.
                """
            )


if __name__ == "__main__":
    render_dashboard()
