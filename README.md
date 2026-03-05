# 📊 Multi-Asset Sector-Constrained Portfolio Optimization

> Mean-Variance Model (Markowitz 1952) with Real-World Constraints

## Overview

Portfolio optimization using the **Mean-Variance (M-V) Model** across **26 assets** in **9 groups**, with sector weight constraints, transaction costs, lot-size adjustments, short-sell analysis, and out-of-sample validation.

---

## Assets (26 Tickers, 9 Groups)

| Group          | Tickers                       | Region |
| -------------- | ----------------------------- | ------ |
| US Technology  | AAPL, MSFT, NVDA, META        | 🇺🇸     |
| US Healthcare  | JNJ, ABBV, LLY                | 🇺🇸     |
| US Financial   | JPM, GS                       | 🇺🇸     |
| US Energy      | XOM, MPC                      | 🇺🇸     |
| US Consumer    | CAT, WMT                      | 🇺🇸     |
| Thai Stocks    | DELTA.BK, ADVANC.BK, KBANK.BK | 🇹🇭     |
| Chinese Stocks | PDD, FUTU, ZTO                | 🇨🇳     |
| Crypto         | BTC-USD, ETH-USD              | ₿      |
| Bonds          | HYG, VCSH, EMLC               | 🌏     |
| Commodities    | GLD, SLV                      | 🪙     |

> All assets screened for **Sharpe Ratio > 0.5** (4-year lookback)

---

## Workflow

```
Phase 1: DATA PREPARATION
  ├── Install & Import (yfinance, pandas, numpy, scipy)
  ├── Define 26 Assets (9 Groups)
  ├── Download 4-Year Price Data (yfinance API)
  ├── Currency Conversion → THB Base (USD/THB exchange rate)
  └── Calculate Returns & Correlation Matrix

Phase 2: M-V MODEL THEORY
  ├── Mean (μ), Variance (σ²), Covariance (Σ)
  ├── Dual Objectives: Max Return & Min Risk
  └── Heart of M-V: Correlation → Diversification Benefit

Phase 3: CONSTRAINTS
  └── Sector Weight Limits
        Sector: 2% ≤ w_sector ≤ 25%
        Stock:  1% ≤ w_i ≤ 15%
        Sum:    Σw = 1

Phase 4: OPTIMIZATION
  ├── Risk-Free Rate (US 10Y Treasury, ~4.5%)
  ├── Sharpe Ratio = (E(Rp) - Rf) / σp
  ├── SLSQP Solver (30 multi-start trials)
  └── Output: Unconstrained vs Sector-Constrained weights

Phase 5: ANALYSIS
  ├── Sector/Asset Weight Comparison
  ├── Efficient Frontier (Constrained vs Unconstrained)
  └── Backtest (4 years)

Phase 6: REAL-WORLD ADJUSTMENTS
  ├── Transaction Costs (0.20% per trade)
  ├── Transaction Lots (Thai: 100/lot, US: 1 share)
  ├── Short Sell with Leverage Constraint (1.0x–2.0x)
  └── Out-of-Sample Testing (Train 3yr / Test 1yr)
```

---

## Key Results (4-Year Backtest)

### Portfolio Performance

| Portfolio              | Total Return | Annual Return | Annual Vol | Sharpe Ratio | Max Drawdown |
| ---------------------- | ------------ | ------------- | ---------- | ------------ | ------------ |
| Unconstrained          | +117.89%     | 21.38%        | 7.31%      | 2.92         | -7.91%       |
| **Sector-Constrained** | **+168.32%** | **27.23%**    | **9.76%**  | **2.79**     | **-11.63%**  |
| Equal Weight           | +221.23%     | 32.83%        | 15.42%     | 2.13         | -17.31%      |

### Transaction Costs Impact

| Portfolio     | Without TC | With TC  | TC Drag |
| ------------- | ---------- | -------- | ------- |
| Unconstrained | +117.61%   | +116.12% | -0.69%  |
| Constrained   | +167.87%   | +165.17% | -1.01%  |

> TC = Broker 0.10% + Tax 0.05% + Bid-Ask 0.05% = **0.20%/trade**, rebalance monthly

### Out-of-Sample Validation

| Period                   | Ann Return | Sharpe (Rf-adj) | Max DD  |
| ------------------------ | ---------- | --------------- | ------- |
| In-Sample (Train 3yr)    | 33.06%     | 2.226           | -12.54% |
| Out-of-Sample (Test 1yr) | 41.42%     | 2.387           | -12.71% |

```
Sharpe degradation: -7.2% → GOOD (not overfitting)
```

---

## Sector Allocation (Constrained Portfolio)

| Sector         | Weight | Limit     |
| -------------- | ------ | --------- |
| Thai Stocks    | 25.00% | 25% (max) |
| Bonds          | 17.00% | 25%       |
| US Healthcare  | 16.44% | 25%       |
| Commodities    | 16.00% | 25%       |
| US Consumer    | 11.46% | 25%       |
| US Technology  | 4.59%  | 25%       |
| US Energy      | 3.51%  | 25%       |
| US Financial   | 2.00%  | 25%       |
| Chinese Stocks | 2.00%  | 25%       |
| Crypto         | 2.00%  | 25%       |

---

## Parameters

| Parameter           | Value         | Description                      |
| ------------------- | ------------- | -------------------------------- |
| `RISK_FREE_RATE`    | ~4.50%        | US 10Y Treasury (auto from ^TNX) |
| `SECTOR_MAX_WEIGHT` | 25%           | Max weight per sector            |
| `SECTOR_MIN_WEIGHT` | 2%            | Min weight per sector            |
| `STOCK_MAX_WEIGHT`  | 15%           | Max weight per asset             |
| `STOCK_MIN_WEIGHT`  | 1%            | Min weight per asset             |
| `TOTAL_TC`          | 0.20%         | Transaction cost per trade       |
| `REBALANCE_FREQ`    | 21 days       | Monthly rebalancing              |
| `PORTFOLIO_VALUE`   | 1,000,000 THB | Investment amount                |
| Data period         | 4 years       | 2022–2026                        |
| Currency base       | THB           | All prices converted to THB      |

---

## How to Run

```bash
# Install uv (if not installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Run
cd /Users/oattao/Desktop/ci
uv run jupyter notebook sector_constrained_portfolio.ipynb

# Or execute all cells from CLI
uv run jupyter nbconvert --execute sector_constrained_portfolio.ipynb
```

---

## File Structure

```
ci/
├── sector_constrained_portfolio.ipynb          # Source notebook
├── sector_constrained_portfolio_executed.ipynb  # Executed (with outputs)
├── portfolio_workflow.png                       # Workflow diagram
├── README.md                                    # This file
├── pyproject.toml                               # uv dependencies
└── uv.lock                                      # Lock file
```

---

## Dependencies

- `yfinance` — Market data download
- `pandas` / `numpy` — Data manipulation
- `matplotlib` / `seaborn` — Visualization
- `scipy` — SLSQP optimization solver
- `jupyter` / `ipykernel` — Notebook runtime
# sector_constrained_portfolio_executed
