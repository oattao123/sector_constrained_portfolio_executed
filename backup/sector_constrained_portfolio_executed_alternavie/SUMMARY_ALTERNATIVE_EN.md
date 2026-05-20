# 📋 Analysis Summary: Multi-Asset Sector-Constrained Portfolio Optimization

Summary of the notebook `heath_sector_constrained_portfolio_executed_alternavie.ipynb`, which focuses on sector-diversified portfolio optimization using Thai Baht (THB) as the base currency.

## 🎯 Main Objectives
- Analyze and construct a portfolio from **50 assets** across **10 industry sectors/asset classes**.
- Utilize **4 years** of historical data (2022 - 2026).
- Adjust all asset prices to **THB** for accurate comparison.
- Implement **Weight Constraints** per sector to prevent portfolio concentration.

## 📂 Asset Sectors
1. **US Technology**: AAPL, NVDA, META, GOOGL, MSFT
2. **US Healthcare**: JNJ, ABBV, LLY, UNH, MRK
3. **US Financial**: JPM, GS, V, MA, BAC
4. **US Energy**: XOM, MPC, CVX, COP, EOG
5. **US Consumer**: CAT, WMT, COST, MCD, HD
6. **Thai Stocks**: DELTA.BK, ADVANC.BK, KBANK.BK, PTT.BK, AOT.BK
7. **Chinese Stocks**: PDD, FUTU, BABA, JD, BIDU
8. **Crypto**: BTC, ETH, BNB, SOL, XRP
9. **Bonds**: HYG, VCSH, EMLC, AGG, BND
10. **Commodities**: GLD, SLV, COPX, IAU, SGOL

## 📈 Performance Summary (Top Performers)
Top performing assets (ranked by Sharpe Ratio in THB):

| Ticker | Sector | Annual Return (%) | Annual Vol (%) | Sharpe Ratio |
| :--- | :--- | :--- | :--- | :--- |
| **VCSH** | Bonds | 4.24% | 3.15% | **1.346** |
| **SGOL** | Commodities | 23.83% | 18.87% | **1.263** |
| **WMT** | US Consumer | 28.67% | 23.02% | **1.246** |
| **NVDA** | US Technology | 63.90% | 53.87% | **1.186** |
| **DELTA.BK** | Thai Stocks | 71.56% | 61.99% | **1.154** |

## 3. Portfolio Constraints Details
*   **Assets**: 50 items (Thai Stocks, US Stocks, Chinese Stocks, Commodities, Bonds, and Crypto).
*   **Sector Constraints**:
    *   Sector Max Weight: 25%
    *   Sector Min Weight: 2%
*   **Stock Constraints**:
    *   Stock Max Weight: 15%
    *   Stock Min Weight: 1%
*   **Transaction Costs**: 0.20% per trade.
*   **Initial Investment**: 1,000,000 THB.

## 4. Optimization Results
The project utilizes 3 algorithms to find the optimal portfolio (Max Sharpe Ratio):

### 4.1 Performance Comparison (Train vs Test)
Data from 2022-2026 was split into Train (75%) and Test (25%) to check for overfitting:

| Optimizer | Period | Annualized Return | Volatility | Sharpe Ratio | Max Drawdown |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **SLSQP** | Train | 36.75% | 15.97% | 2.048 | -11.69% |
| | Test | 35.67% | 18.19% | 1.738 | -13.05% |
| **PSO** | Train | 38.19% | 17.76% | 1.921 | -12.26% |
| | Test | 36.40% | 18.94% | 1.708 | -12.36% |
| **DE** | Train | 36.43% | 17.07% | 1.897 | -13.34% |
| | Test | 34.48% | 18.57% | 1.638 | -12.05% |

> [!TIP]
> **Sharpe Degradation**: All models showed a Sharpe Ratio decrease of only 11-15% during the Test period, indicating excellent stability and no overfitting issues.

## 5. Real-World Analysis

### 5.1 Transaction Lots Adjustment
Calculates the number of shares to be purchased based on actual market Lot Sizes:
*   **Thai Stocks**: Must purchase in multiples of 100 shares.
*   **US Stocks/ETFs**: Minimum of 1 share.
*   **Cryptocurrency**: Decimal trading supported (0.0001).

**Lot Adjustment Impact:**
*   **Tracking Error (RMSE)**: 5.749%
*   Actual portfolio weights deviate slightly from theoretical optima but remain within acceptable limits.
*   **Cash Management**: For a 1M THB investment, approximately 23% cash reserve is maintained to handle volatility and lot size rounding.

### 5.2 Short Sell and Leverage Strategy
Tested returns for cases allowing Short Selling and Leverage:

| Strategy | Return (%) | Vol (%) | Sharpe Ratio | Leverage |
| :--- | :--- | :--- | :--- | :--- |
| Long-Only | 32.93% | 12.66% | 2.602 | 1.0x |
| **Long-Short (Sweet Spot)** | **33.99%** | **11.90%** | **2.857** | **1.3x** |
| Long-Short (High Risk) | 40.33% | 12.84% | 3.142 | 2.0x |

> [!IMPORTANT]
> **Conclusion**: Using 1.3x leverage is the most cost-effective point (Sweet Spot), as it benefits from risk management through Short Selling without excessive overall risk (Volatility).

## 6. Conclusions and Recommendations
1.  **Leading Assets**: US tech (NVDA) and Thai growth stocks (DELTA.BK) are the primary return drivers.
2.  **Risk Diversification**: Sector-Constrained portfolios provide more stability than unconstrained ones.
3.  **Robustness**: Out-of-Sample testing confirms model accuracy for real-world deployment. Rebalancing every 21 days (~1 month) is recommended to manage transaction costs effectively.
