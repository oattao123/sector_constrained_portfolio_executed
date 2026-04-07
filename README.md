# 📊 Global Multi-Asset Sector-Constrained Portfolio Optimization

> **State-of-the-Art Portfolio Engineering**: Combining Modern Portfolio Theory (MPT), Tail-Risk Management (CVaR), and Advanced Metaheuristic Optimizers (PSO, DE, ACO, EBGWO).

## 🌍 Overview

This project provides a comprehensive framework for optimizing a global multi-asset portfolio. It integrates standard **Mean-Variance Optimization (Sharpe Ratio)** with sophisticated **Conditional Value at Risk (CVaR)** models to manage extreme market events (Tail Risk). 

Across **4 specialized methodologies**, we analyze **50 global assets** across **10 sectors**, incorporating real-world constraints such as transaction lots, regional costs, and leverage limits.

---

## 📂 Project Structure & Methodologies

The repository is organized into four specialized optimization tracks:

| Track | Directory | Target Objective | Core Feature |
| :--- | :--- | :--- | :--- |
| **Simple** | `/sector_constrained_portfolio_executed_Simple` | Max Sharpe Ratio | Baseline MVO comparison |
| **Alternative** | `/sector_constrained_portfolio_executed_alternavie` | Heuristic Max Sharpe | Extended PSO/DE Hybrid analysis |
| **CVaR** | `/sector_constrained_portfolio_executed_CVaR` | Min Tail Risk (CVaR) | **STARR Ratio** optimization using ACO & EBGWO |
| **Constraints** | `/sector_constrained_portfolio_executed_Simple_contrains` | Execution Modeling | High-fidelity lot-size & cost accounting |

---

## 🍎 Asset Universe (50 Tickers, 10 Sectors)

All assets are screened for high risk-adjusted performance and converted to **THB Base currency**.

| Sector | Asset Tickers | Region |
| :--- | :--- | :--- |
| **US Tech** | AAPL, NVDA, META, GOOGL, MSFT | 🇺🇸 |
| **US Healthcare** | JNJ, ABBV, LLY, UNH, MRK | 🇺🇸 |
| **US Financial** | JPM, GS, V, MA, BAC | 🇺🇸 |
| **US Energy** | XOM, MPC, CVX, COP, EOG | 🇺🇸 |
| **US Consumer** | CAT, WMT, COST, MCD, HD | 🇺🇸 |
| **Thai Stocks** | DELTA.BK, ADVANC.BK, KBANK.BK, PTT.BK, AOT.BK | 🇹🇭 |
| **Chinese Stocks** | PDD, FUTU, BABA, JD, BIDU | 🇨🇳 |
| **Crypto** | BTC-USD, ETH-USD, BNB-USD, SOL-USD, XRP-USD | ₿ |
| **Bonds** | HYG, VCSH, EMLC, AGG, BND | 🌏 |
| **Commodities** | GLD, SLV, COPX, IAU, SGOL | 🪙 |

---

## 🤖 Advanced Optimization Algorithms

We deploy a suite of advanced algorithms to navigate the non-convex solution space created by sector constraints:

1.  **SLSQP**: Gradient-based local optimizer for high-precision refinement.
2.  **PSO (Particle Swarm)**: Swarm-based global search to avoid local optima.
3.  **DE (Differential Evolution)**: Population-based evolutionary strategy for robust convergence.
4.  **ACO (Ant Colony)**: Foraging logic used specifically in **CVaR** targets for path-based optimization.
5.  **EBGWO (Grey Wolf)**: Pack-hunting optimization for ultra-stable weight discovery.
6.  **NSGA-II**: Multi-objective mapping of the **Pareto Front** (Risk vs. Return).

---

## 📈 Performance Highlights (Consolidated)

| Metric | Max Sharpe (Simple) | Max STARR (CVaR - ACO) | Long-Short (1.3x) |
| :--- | :--- | :--- | :--- |
| **Annual Return** | 35.67% | 35.80% | 33.99% |
| **Risk Metric** | 18.19% (Vol) | 2.22% (CVaR) | 11.90% (Vol) |
| **Efficiency Ratio** | **1.738 (Sharpe)** | **12.1169 (STARR)** | **2.857 (Sharpe)** |
| **Max Drawdown** | -13.05% | -10.45% | -9.80% |

> [!TIP]
> **Key Finding**: The **1.3x Leverage** strategy identified a "Sweet Spot" where short selling significantly hedges market volatility without escalating gross risk.

---

## 🛠 Real-World Execution Constraints

- **Regional Lot Sizes**: Fixed 100-share lots for Thai stocks; fractional support for Crypto.
- **Transaction Costs**: 0.20% flat fee per trade incorporated into rebalancing frequency.
- **Tracking Error**: Maintained at **RMSE 5.749%** post lot-adjustment.
- **Out-of-Sample (OOS)**: Rigorous 75/25 Train-Test split showing highly durable performance (~11-15% degradation).

---

## 🚀 How to Run

```bash
# Install uv dependencies
uv sync

# Run the consolidated dashboard or specific notebooks
uv run jupyter notebook
```

---

## 📊 Dependencies

- `yfinance` / `pymarket` for live historical data.
- `scipy.optimize` / `pyswarm` / `pymoo` for the optimization engine.
- `pandas` / `numpy` for high-performance matrix operations.
- `matplotlib` / `seaborn` for analytical visualization.
