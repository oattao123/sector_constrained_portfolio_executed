# 📋 Analysis Summary: Global Multi-Asset Simple Portfolio Optimization

Summary of the notebook `heath_sector_constrained_portfolio_executed_Simple.ipynb`, focusing on global portfolio diversification using baseline Mean-Variance Optimization.

## 🎯 Main Objectives
- Construct a portfolio of **50 assets** across **10 global industrial sectors**.
- Use **THB** as the base currency for precise calculations for Thai-based investors.
- Compare performance between Sector-Constrained strategies and unconstrained investment approaches.

## 📂 Asset Universe
Assets are categorized into 10 key sectors:
1.  **US Technology**: Leading tech stocks (AAPL, NVDA, MSFT, etc.)
2.  **US Healthcare / Financial / Energy / Consumer**: Major US industrial groups.
3.  **Thai Stocks**: Popular SET50 stocks (DELTA, ADVANC, etc.)
4.  **Chinese Stocks**: Large-cap Chinese firms traded as ADRs.
5.  **Crypto**: Blue-chip digital assets (BTC, ETH, SOL, etc.)
6.  **Bonds & Commodities**: Fixed-income and commodities for risk hedging.

## 📈 Performance Summary

### 1. Optimizer Efficiency
| Optimizer | Annual Return (%) | Volatility (%) | Sharpe Ratio |
| :--- | :--- | :--- | :--- |
| **SLSQP (Standard)** | 35.67% | 18.19% | **1.738** |
| **PSO (Swarm)** | 36.40% | 18.94% | **1.708** |
| **DE (Evolution)** | 34.48% | 18.57% | **1.638** |

### 2. Generalization (Train vs Test)
- **Sharpe Degradation**: The drop in Sharpe Ratio is only **11-15%** when applied to unseen data (Test Set). This confirms the model's robustness and lack of overfitting.

## 🛠 Execution Analysis

### 1. Lot Size Adjustment
- Calculates the actual number of shares to buy based on regional market rules (e.g., Thai stocks must be purchased in multiples of 100).
- **Tracking Error (RMSE)**: 5.749% (minimal deviation between the theoretical optimal and the practical execution).
- **Cash Management**: Approximately 23% cash reserve for a 1M THB portfolio to handle volatility and rounding.

### 2. Leverage and Short Sell Strategy
- **1.3x Leverage**: Identified as the **Sweet Spot**, providing a higher Sharpe Ratio through strategic short selling to manage overall risk.

## 💡 Conclusions and Recommendations
1.  **Leading Drivers**: `NVDA` and `DELTA.BK` remain the primary return engines for the portfolio.
2.  **Safety First**: Sector constraints successfully prevent over-concentration in technology during market downturns.
3.  **Execution-Ready**: This model is configured for live trading, accounting for both transaction fees and minimum lot sizes.
