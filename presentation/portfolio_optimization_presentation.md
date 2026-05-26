# Multi-Asset Portfolio Optimization Pipeline

Base currency: THB  
Source: `manual.md`, current config, latest PSO and ACO+EBGWO pipeline outputs.

---

## 1. Data Collection

The pipeline collects close-price data through Yahoo Finance and caches it locally. The current configured universe contains 30 assets from `config/assets.csv`, plus SPY as the benchmark appended by the data manager.

Current universe in the executed pipeline:

| Group | Examples | Notes |
| --- | --- | --- |
| Thai Stocks | `KTB.BK`, `SCB.BK`, `TISCO.BK` | THB-denominated |
| US Stocks | `PLTR`, `MU`, `GE`, `GILD`, `TJX` | Converted to THB |
| Bonds | `TIPS` | USD-denominated |
| Benchmark | `SPY` | S&P 500 proxy |

Requested topic groups such as Chinese assets, ETC, and Crypto can be added through `config/assets.csv`, but they are not present in the current executed asset file.

---

## 2. Train and Test Windows

Requested slide specification:

| Window | Start | End |
| --- | --- | --- |
| Training | `2015-01-01` | `2024-01-01` |
| Test | `2025-01-01` | `2026-01-01` |

Current `config/settings.json`:

| Setting | Value |
| --- | --- |
| `start_date` | `2015-01-01` |
| `end_date` | `2026-03-20` |
| `insample_end_date` | `2024-01-01` |
| `outsample_end_date` | `2026-03-20` |

The implemented walk-forward default in `backtest.py` uses `252*3` trading days as the rolling lookback and `21*3` trading days as the rebalance/test step. This differs from the requested 4-year / 1-year wording and should be aligned if required for final submission.

---

## 3. Data Preprocessing

Preprocessing steps:

- Remove assets with missing-data gaps longer than 365 days.
- Forward-fill and backward-fill remaining missing prices.
- Convert USD-denominated assets to THB using `USDTHB=X`; fallback is 35.0 THB/USD.
- Transform prices into log returns using `log(P_t / P_{t-1})`.
- Estimate covariance with Ledoit-Wolf shrinkage, implemented with PyTorch and CUDA auto-detection.

---

## 4. Asset Selection

The pipeline compares selection strategies before weight optimization:

| Strategy | Method |
| --- | --- |
| Sharpe Ratio | Select assets with strongest individual Sharpe |
| Diversification Score | Combine performance with lower average correlation |
| Ledoit-Wolf + Diversification | Use robust covariance / correlation estimates |
| HRP + Max Sharpe per Cluster | Cluster assets, choose strongest Sharpe representative |
| HRP + Max Diversification per Cluster | Cluster assets, choose best diversification representative |
| 2D-ACO | Discrete ACO path selects assets directly |
| 2D-ACO + Cluster Constraint | ACO selects representatives within HRP clusters |

---

## 5. Asset Weight Optimization

The current executed scripts focus on:

- PSO: `run_pipeline_pso.py`
- ACO+EBGWO: `run_pipeline_aco_ebgwo.py`
- Combined pipeline support in `run_pipeline.py`

Objective:

```text
maximize Sharpe Ratio + entropy reward - concentration / constraint penalties
```

Weight optimization uses floating-point weight vectors. ACO asset selection uses discrete/integer path encoding.

---

## 6. Constraints and Optional Features

Requested constraints:

| Constraint | Requested | Current implementation status |
| --- | --- | --- |
| Minimum assets | 5 | Current code uses `max(5, len(valid_stocks)//2)` |
| Maximum assets | 15 | Needs explicit enforcement if required |
| Sector constraint | Bonus | Available in dynamic ACO+EBGWO when sector labels are used |
| Short sell | Bonus | Not enabled in current long-only encoding |
| Transaction cost | Implemented | 0.1% turnover cost |

---

## 7. Algorithm Mechanism

| Algorithm | Mechanism | Role |
| --- | --- | --- |
| PSO | Particles update weight vectors from personal and global bests | Weight optimizer |
| ACO | Pheromone trails and heuristic Sharpe guide asset selection | Asset selector / co-evolution selector |
| EBGWO | Wolf population follows alpha/beta/delta solutions with entropy regularization | Weight optimizer |
| GWO / DE / ETC | Extensible families mentioned in topic | Not all are present in latest executed results |

---

## 8. Convergence and Stability

The latest optimizer runs report convergence plots and mean +- standard deviation over repeated trials.

Latest run folders:

- PSO: `output/pso_26_05_03_39`
- ACO+EBGWO: `output/aco_ebgwo_26_05_03_38`

Convergence output:

- `convergence.png`: best and average fitness across walk-forward windows.
- Fitness combines Sharpe, entropy, and penalty terms.

---

## 9. Metrics

Currently reported:

- Cumulative Return
- Annual Return
- Annual Volatility
- Sharpe Ratio
- Max Drawdown
- With-cost and no-cost comparison

Requested but not currently reported:

- Sortino Ratio

This should be added to `compute_metrics()` in `portfolio_optimization/backtest.py` if the final presentation must include it as a measured result.

---

## 10. Latest PSO Result Summary

Source: `output/pso_26_05_03_39/performance_report.md`

Best full-period with-cost result:

| Strategy | Cum Return | Ann Return | Ann Vol | Sharpe | Max DD |
| --- | ---: | ---: | ---: | ---: | ---: |
| PSO Portfolio | 50.85% +- 9.11% | 39.65% +- 4.98% | 24.32% +- 1.73% | 1.4564 +- 0.3083 | -21.97% +- 3.05% |
| SPY Benchmark | 7.43% | 8.10% | 18.48% | 0.1949 | -19.21% |

2025-only with-cost result:

| Strategy | Cum Return | Ann Return | Sharpe | Max DD |
| --- | ---: | ---: | ---: | ---: |
| PSO Portfolio | 34.53% +- 5.13% | 36.05% +- 3.73% | 1.2721 +- 0.2522 | -21.97% +- 3.05% |
| SPY Benchmark | 11.21% | 13.72% | 0.4674 | -19.21% |

---

## 11. Latest ACO+EBGWO Result Summary

Source: `output/aco_ebgwo_26_05_03_38/performance_report.md`

Best full-period with-cost result:

| Strategy | Cum Return | Ann Return | Ann Vol | Sharpe | Max DD |
| --- | ---: | ---: | ---: | ---: | ---: |
| ACO Selected | 40.90% +- 11.75% | 33.31% +- 6.88% | 23.61% +- 2.47% | 1.2425 +- 0.4215 | -17.75% +- 5.43% |
| Dynamic ACO+EBGWO | 16.43% +- 21.98% | 16.42% +- 16.96% | 26.91% +- 0.04% | 0.4436 +- 0.6312 | -21.68% +- 0.34% |
| SPY Benchmark | 7.43% | 8.10% | 18.48% | 0.1949 | -19.21% |

---

## 12. Key Findings

- PSO is the strongest latest executed optimizer, with full-period Sharpe of 1.4564 +- 0.3083.
- ACO Selected is the best ACO+EBGWO-run strategy, with full-period Sharpe of 1.2425 +- 0.4215.
- Both optimizer families beat the SPY benchmark on cumulative return and Sharpe in the latest run.
- Transaction costs reduce returns slightly but do not change the PSO leader.
- The current implementation should be adjusted for exact 4-year / 1-year walk-forward, Sortino reporting, max-15 selection, and short-selling if those rubric items are mandatory.

