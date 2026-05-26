# Source Notes for Presentation

## Files Used

- `manual.md`: pipeline description, dependencies, execution, configuration, outputs, data manager, covariance, selectors, optimizers, and backtester.
- `config/settings.json`: dates, base currency, transaction cost, risk-free-rate fallback, Monte Carlo count.
- `config/assets.csv`: actual executed asset universe.
- `output/pso_26_05_03_39/performance_report.md`: latest PSO metrics.
- `output/aco_ebgwo_26_05_03_38/performance_report.md`: latest ACO+EBGWO metrics.
- `portfolio_optimization/backtest.py`: walk-forward defaults and current metric implementation.

## Important Implementation Caveats

- The requested topic lists S&P 500, Thai, Chinese, ETC, and Crypto. The current `assets.csv` contains Thai stocks, US-listed stocks, and bonds. Chinese, ETC, and Crypto assets are not in the current executed universe.
- The requested test period is `2025-01-01` to `2026-01-01`. Current config ends out-of-sample on `2026-03-20`; the reports include 2025-only tables as well as full-period OOS tables.
- The requested walk-forward wording says 4 year / 1 year. Current `WalkForwardBacktester.run()` defaults to a 3-year lookback and quarterly step.
- Sortino Ratio is requested but not currently produced by `compute_metrics()`.
- Short selling is requested as a bonus feature but is not enabled in the current long-only optimizer encoding.

## Main Quantitative Results

### PSO Full Period, With Cost

| Strategy | Cum Return | Ann Return | Sharpe | Max DD |
| --- | ---: | ---: | ---: | ---: |
| PSO Portfolio | 50.85% +- 9.11% | 39.65% +- 4.98% | 1.4564 +- 0.3083 | -21.97% +- 3.05% |
| LW Div Cluster Selected | 45.72% +- 10.05% | 36.02% +- 6.50% | 1.4161 +- 0.1988 | -18.05% +- 0.97% |
| LW Sharpe Cluster Selected | 41.16% +- 15.18% | 33.40% +- 9.63% | 1.2159 +- 0.4073 | -21.01% +- 0.53% |
| SPY | 7.43% | 8.10% | 0.1949 | -19.21% |

### ACO+EBGWO Full Period, With Cost

| Strategy | Cum Return | Ann Return | Sharpe | Max DD |
| --- | ---: | ---: | ---: | ---: |
| ACO Selected | 40.90% +- 11.75% | 33.31% +- 6.88% | 1.2425 +- 0.4215 | -17.75% +- 5.43% |
| LW Sharpe Cluster Selected | 35.51% +- 11.71% | 29.30% +- 6.59% | 1.2577 +- 0.6371 | -16.86% +- 8.94% |
| LW Diversification Selected | 29.99% +- 0.22% | 25.00% +- 0.42% | 1.1720 +- 0.0771 | -13.79% +- 0.79% |
| SPY | 7.43% | 8.10% | 0.1949 | -19.21% |

