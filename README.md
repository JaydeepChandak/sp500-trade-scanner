# S&P 500 Bullish Trade Scanner

A Streamlit dashboard for researching bullish, long-only S&P 500 trade ideas over a holding period longer than one week. The app scans the S&P 500 universe, ranks potential buy candidates, reviews optional current holdings, suggests a one-period allocation plan, and highlights risk warnings.

This is designed for educational research and portfolio analysis. It does not predict the future, does not guarantee returns, and is not financial advice.

## What The App Does

- Scans a built-in S&P 500 stock universe
- Excludes non-S&P 500 tickers from version 1
- Ranks bullish, long-only trade ideas using technical and historical risk/return signals
- Generates suggested buy candidates for the selected holding period
- Reviews optional current holdings as buy more, hold, trim, sell, or avoid adding
- Builds a suggested one-period portfolio allocation
- Shows expected return estimates, downside risk estimates, Sharpe estimate, Sortino estimate, drawdown estimate, and probability of positive historical holding-period return
- Includes risk controls for maximum position size, stop-loss, trailing stop, take-profit, volatility warnings, and drawdown limits
- Supports manual macro/geopolitical scenario assumptions
- Includes a monthly ranking backtest module

## Important Limitations

This app is a research tool only.

- Outputs are estimates based on historical data and assumptions.
- The app does not guarantee accuracy or future returns.
- The app is not financial advice.
- Version 1 supports S&P 500 stocks only.
- ETFs, commodities, crypto, forex, futures, international stocks, and non-S&P 500 assets are intentionally excluded.
- Market data comes from `yfinance`, which may occasionally be delayed, incomplete, or unavailable.
- Backtests use the current S&P 500 universe, which can introduce survivorship bias.

Always verify data and assumptions before making any trading or investment decision.

## Strategy Style

The scanner is built for:

- Bullish trade ideas
- Long-only positions
- Buy-low / sell-high setups
- Holding periods greater than one week
- Portfolio research rather than day trading

The app uses a composite bullish score from 0 to 100 based on historical and technical factors such as:

- 20-day, 50-day, and 200-day moving averages
- RSI
- MACD
- Bollinger Bands
- 52-week high/low position
- Recent drawdown
- Volatility regime
- Trend strength
- 1-month, 3-month, 6-month, and 12-month momentum
- Relative strength versus benchmark
- Risk/reward estimate

## Main Workflow

1. Choose a planned buy date.
2. Select the expected holding period.
3. Optionally enter current holdings.
4. Set initial capital and risk preference.
5. Choose the maximum number of buy recommendations.
6. Run the S&P 500 scan.
7. Review:
   - Top buy recommendations
   - Suggested one-period portfolio allocation
   - Current holdings review
   - Stocks to avoid
   - Risk warnings
   - Backtest results, if enabled

## Files

```text
app.py              Main Streamlit application
sp500_universe.py   Built-in S&P 500 ticker universe and ticker helpers
requirements.txt    Python dependencies
README.md           Project documentation
```

## Local Setup

### 1. Clone Or Download The Project

If using Git:

```bash
git clone <your-repository-url>
cd <your-repository-folder>
```

If you downloaded the files manually, open a terminal in the folder containing `app.py`.

### 2. Install Dependencies

```bash
python3 -m pip install -r requirements.txt
```

### 3. Run The App

```bash
python3 -m streamlit run app.py
```

Streamlit will show a local URL similar to:

```text
http://localhost:8501
```

Open that URL in your browser.

## Deploying To Streamlit Community Cloud

1. Create a GitHub repository.
2. Upload these files:
   - `app.py`
   - `sp500_universe.py`
   - `requirements.txt`
   - `README.md`
3. Go to Streamlit Community Cloud:

```text
https://share.streamlit.io
```

4. Sign in with GitHub.
5. Click `Create app`.
6. Select your repository.
7. Set the main file path to:

```text
app.py
```

8. Click `Deploy`.

After deployment, Streamlit will give you a public link ending in:

```text
.streamlit.app
```

You can share that link with others.

## Updating The Deployed App

Streamlit Cloud deploys from GitHub. If you edit the app locally, the deployed app will not update until you upload or push the changed files to GitHub.

For UI or logic updates, usually update:

```text
app.py
```

Then commit the change on GitHub. Streamlit should redeploy automatically within a few minutes.

## Configuration

The sidebar lets users configure:

- Planned buy date
- Expected holding period
- Current holdings
- Initial capital
- Maximum number of suggested buys
- Risk preference
- Benchmark ticker
- Risk-free rate
- Transaction cost
- Slippage
- Allocation method
- Maximum position size
- Stop-loss
- Trailing stop
- Take-profit target
- Maximum portfolio drawdown limit
- High-volatility warning threshold
- Manual scenario assumptions

## Risk Metrics

The dashboard estimates:

- Expected return
- Net expected return after estimated cost and slippage
- Downside risk
- Annualized volatility
- Holding-period volatility
- Sharpe ratio
- Sortino ratio
- Max drawdown
- Average and highest pairwise correlation
- Probability of positive historical holding-period return

## Backtesting

The app includes a monthly ranking backtest that:

- Ranks S&P 500 stocks historically
- Selects the top candidates
- Holds for approximately one month
- Rebalances monthly
- Includes transaction cost and slippage assumptions
- Compares performance against the selected benchmark

Backtest results are historical simulations, not future predictions.

## Data Source

Historical price data is loaded through:

```text
yfinance
```

Because market data availability depends on Yahoo Finance and `yfinance`, occasional missing data or request failures can happen.

## Version 1 Scope

Version 1 is intentionally focused on reliability for S&P 500 stocks only.

Not included yet:

- ETFs
- Commodities
- Commodity ETFs
- Futures
- Crypto
- Forex
- International equities
- NSE/Nifty stocks

The code is structured so other universes can be added later as separate modules or separate apps.

## Disclaimer

This project is for education and research only. It does not provide financial, investment, tax, or legal advice. All outputs are estimates based on historical data and user-defined assumptions. Markets are uncertain, and losses are possible. Always do your own research and consult a qualified professional before making investment decisions.
