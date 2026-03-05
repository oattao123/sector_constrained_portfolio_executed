# 📊 Multi-Asset Sector-Constrained Portfolio Optimization

> Mean-Variance Model (Markowitz 1952) with Real-World Constraints & Advanced Optimizers

## Overview

Portfolio optimization across **30 assets** representing **10 groups (Asset Classes/Sectors)**, incorporating sector weight constraints, transaction costs, lot-size adjustments, short-sell analysis, and out-of-sample validation.

This project goes beyond the standard SciPy SLSQP solver by introducing **Particle Swarm Optimization (PSO)** and **Differential Evolution (DE)** to overcome local optima and achieve a higher Sharpe ratio.

---

## Assets (30 Tickers, 10 Groups)

| Group          | Tickers                       | Region |
| -------------- | ----------------------------- | ------ |
| US Technology  | AAPL, NVDA, META, AVGO, GOOGL | 🇺🇸     |
| US Healthcare  | JNJ, ABBV, LLY                | 🇺🇸     |
| US Financial   | JPM, GS, V                    | 🇺🇸     |
| US Energy      | XOM, MPC                      | 🇺🇸     |
| US Consumer    | CAT, WMT, COST, MCD           | 🇺🇸     |
| Thai Stocks    | DELTA.BK, ADVANC.BK, KBANK.BK | 🇹🇭     |
| Chinese Stocks | PDD, FUTU                     | 🇨🇳     |
| Crypto         | BTC-USD, ETH-USD              | ₿      |
| Bonds          | HYG, VCSH, EMLC               | 🌏     |
| Commodities    | GLD, SLV, COPX                | 🪙     |

> All assets screened for **Sharpe Ratio > 0.5** (4-year lookback)

---

## Workflow

```text
Phase 1: DATA PREPARATION
  ├── Install & Import (yfinance, pandas, numpy, scipy, pyswarm)
  ├── Define 30 Assets (10 Groups)
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
  ├── 3 Optimizers Compared:
  │    ├── 1. SLSQP Solver (Standard)
  │    ├── 2. PSO Hybrid (Particle Swarm + SLSQP)
  │    └── 3. DE Hybrid (Differential Evolution + SLSQP) 🏆
  └── Output: Multi-optimizer comparison

Phase 5: ANALYSIS
  ├── Sector/Asset Weight Comparison across Optimizers
  ├── Efficient Frontier (Optimizers mapped)
  └── Backtest (4 years)

Phase 6: REAL-WORLD ADJUSTMENTS
  ├── Transaction Costs (0.20% per trade)
  ├── Transaction Lots (Thai: 100/lot, US: 1 share, Crypto: Fractions)
  ├── Short Sell with Leverage Constraint (1.0x–2.0x)
  └── Out-of-Sample Testing (Train 3yr / Test 1yr)
```

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

# Run notebook via uv
cd /Users/oattao/Desktop/ci
uv run jupyter notebook sector_constrained_portfolio_from_py.ipynb

# Or execute python script directly
uv run python sector_constrained_portfolio_executed.py
```

---

## File Structure

```text
ci/
├── pyproject.toml                               # uv dependencies
├── uv.lock                                      # Lock file
├── README.md                                    # This file
├── sector_constrained_portfolio_executed.py     # Main executable python script
└── sector_constrained_portfolio_from_py.ipynb   # Generated Notebook
```

---

## Dependencies

- `yfinance` — Market data download
- `pandas` / `numpy` — Data manipulation
- `matplotlib` / `seaborn` — Visualization
- `scipy` — SLSQP and DE optimization solvers
- `pyswarm` — Particle Swarm Optimization (PSO) solver
