# 📋 Analysis Summary: CVaR-Based Portfolio Optimization (Tail Risk Management)

Summary of the notebook `heath_sector_constrained_portfolio_executed_CVaR.ipynb`, which focuses on portfolio optimization utilizing **CVaR (Conditional Value at Risk)** to manage worst-case scenarios (Tail Risk).

## 🎯 Main Objectives
- Construct a portfolio focused on **Minimizing Tail Risk** (95% Confidence Level) instead of standard volatility.
- Analyze **50 assets** across **10 sectors** using **THB** as the base currency.
- Implement **Multi-Objective Optimization** to find a balance between Annual Return and CVaR risk.

## 📂 Advanced Optimization Algorithms
This project employs highly sophisticated simulation and calculation techniques:
1.  **PSO (Particle Swarm Optimization)**: Simulates swarm behavior to find the global minimum for CVaR.
2.  **ACO (Ant Colony Optimization)**: Uses ant foraging logic to identify paths (portfolios) with the highest STARR Ratio.
3.  **EBGWO (Enhanced Grey Wolf Optimizer)**: Implements pack hunting strategies for higher precision in reaching the optimal point.
4.  **NSGA-II**: A multi-objective genetic algorithm used to map the **Pareto Front** between "Return" and "CVaR Risk".

## 📊 Core Metric: STARR Ratio
In this project, we prioritize the **STARR Ratio (Return-to-CVaR)** over the standard Sharpe Ratio:
*   **STARR Ratio** = (Return - Rf) / CVaR
*   This metric informs investors exactly how much return they receive for each unit of risk during an extreme market downturn (Tail Risk).

## 📉 Results Summary

### 1. Algorithm Performance (Max STARR)
| Optimizer | Annual Return (%) | CVaR (Tail Risk) | STARR Ratio |
| :--- | :--- | :--- | :--- |
| **ACO** | 35.80% | 2.22% | **12.1169** |
| **EBGWO** | 35.40% | 2.29% | **11.8737** |
| **Hybrid (PSO+SLSQP)** | 32.93% | 2.40% | **9.5622** |

### 2. Temporal Robustness (Train vs Test)
Out-of-Sample testing confirms the high durability of the CVaR model:
*   **SLSQP Train Sharpe**: 2.048
*   **SLSQP Test Sharpe**: 1.738 (15.1% degradation - rated as Excellent/Stable).

## 🛠 Real-World Analysis & Implementation

### 1. Transaction Lot Adjustments
*   Weights are adjusted to comply with actual market lot sizes (Thai: 100 shares, US: 1 share, Crypto: 0.0001 units).
*   **RMSE (Tracking Error)**: 5.749% (minor deviation between theoretical and practical weights).
*   **Cash Reserves**: Approximately 23% cash is maintained to handle volatility and rounding during execution.

### 2. Short Sell and Leverage Strategy
*   **Long-Only (1.0x)**: Sharpe 2.602
*   **Long-Short (1.3x)**: Sharpe 2.857 (**Sweet Spot**)
*   Utilizing 1.3x leverage provides a better risk-adjusted return by using short selling to hedge high-volatility assets.

## 💡 Conclusions and Recommendations
1.  **Tail Risk Protection**: Managing CVaR provides better protection against "Black Swan" events compared to standard deviation methods.
2.  **STARR Ratio as the Decision Maker**: Assets selected for this portfolio are those with low "Fat Tail" risk (lower probability of extreme losses).
3.  **suitability**: Recommended for highly risk-averse investors who require strict control over drawdown levels during extreme market stress.
