# Google Slides Outline: Portfolio Optimization Pipeline

Source material:
- `manual.md`
- `config/settings.json`
- `config/assets.csv`
- `output/pso_26_05_03_39/performance_report.md`
- `output/aco_ebgwo_26_05_03_38/performance_report.md`

## Slide 1 - Title
**Title:** Multi-Asset Portfolio Optimization Pipeline  
**Subtitle:** Data collection, preprocessing, asset selection, SI weight optimization, and walk-forward testing  
**Footer:** Base currency: THB | Pipeline run: 2026-05-26

Speaker note:
This presentation summarizes the implemented portfolio optimization pipeline and the latest generated PSO and ACO+EBGWO results.

## Slide 2 - Data Collection
**Main message:** The pipeline builds a multi-asset universe, converts prices into THB, and benchmarks against SPY.

Content:
- Configured universe: 30 assets from `config/assets.csv`.
- Markets currently present in the pipeline: Thai equities, US-listed equities, and bonds.
- Requested topic coverage also includes S&P 500, Chinese assets, ETC, and Crypto as extendable universes.
- Data source: Yahoo Finance through `yfinance`.
- Benchmark: SPY is appended automatically.

Visual suggestion:
Use a compact universe map/table: Thai Stocks, US Sectors, Bonds, Benchmark.

## Slide 3 - Train/Test Window
**Main message:** The experiment separates model fitting from out-of-sample evaluation.

Content:
- Requested training window: `2015-01-01` to `2024-01-01`.
- Requested test window: `2025-01-01` to `2026-01-01`.
- Current config: data starts `2015-01-01`; in-sample boundary `2024-01-01`; out-of-sample end `2026-03-20`.
- Current backtester default: rolling lookback `252*3` trading days and rebalance step `21*3` trading days.
- Transaction cost: 0.1% turnover cost per rebalance.

Speaker note:
The user-requested slide specification says 4-year / 1-year walk-forward. The current code uses 3-year lookback and quarterly steps unless changed in the backtester call.

## Slide 4 - Data Preprocessing
**Main message:** Price data is cleaned, normalized, and converted into return/covariance inputs.

Content:
- Missing data: drop assets with NaN gaps longer than 365 days.
- Remaining gaps: forward-fill then backward-fill.
- Currency conversion: USD assets converted to THB using `USDTHB=X`, fallback 35.0 THB/USD.
- Return transform: log return, `log(P_t / P_{t-1})`.
- Covariance: Ledoit-Wolf shrinkage covariance implemented with PyTorch and GPU auto-detection.

## Slide 5 - Asset Selection Strategies
**Main message:** Six selection strategies compare standalone scoring, robust covariance, clustering, and ACO.

Content:
- Top Sharpe Ratio.
- Diversification Score.
- Ledoit-Wolf + Diversification Score.
- HRP Clustering + Max Sharpe per Cluster.
- HRP Clustering + Max Diversification Score per Cluster.
- 2D-ACO independent selection.
- 2D-ACO + HRP cluster constraint.

Speaker note:
The pipeline uses selected representatives before running weight optimization, except the dynamic full-universe ACO+EBGWO/PSO cases.

## Slide 6 - Asset Weight Optimization
**Main message:** Weight optimizers search for high-Sharpe portfolios while controlling concentration and turnover costs.

Content:
- Implemented optimizers in current runs: PSO and ACO+EBGWO.
- EBGWO: grey-wolf style update, entropy regularization, Ledoit-Wolf volatility.
- PSO: particle positions represent portfolio weights; global and personal bests guide search.
- ACO+EBGWO: ACO selects assets and EBGWO allocates weights.
- Objective: maximize entropy-regularized Sharpe ratio with penalties.

Visual suggestion:
Show selection layer feeding optimizer layer, then walk-forward backtest.

## Slide 7 - Representation and Constraints
**Main message:** Asset selection is discrete, weight allocation is continuous.

Content:
- ACO asset selection: integer / discrete path encoding.
- PSO, EBGWO, and similar SI/DE optimizers: floating-point weight vectors.
- Portfolio size requested: minimum 5 assets, maximum 15 assets.
- Current code target count: `max(5, len(valid_stocks)//2)`.
- Weight cap: soft max-weight penalty, default 10%.
- Bonus feature: sector constraint is available for dynamic ACO+EBGWO, requiring sector labels during selection.
- Short selling: not enabled in the current long-only weight encoding.

## Slide 8 - Algorithm Mechanisms
**Main message:** Each optimizer balances exploration and exploitation differently.

Content:
- PSO: particles update positions using inertia, cognitive best, and social best.
- ACO: pheromone trails and heuristic Sharpe information bias future selections.
- EBGWO: population follows alpha/beta/delta wolves with exploration threshold and entropy regularization.
- GWO / DE / ETC: can be described as extensible optimizer families, but not all are present in the latest executed scripts.

Speaker note:
Avoid claiming GWO, DE, or short selling results unless those modules are added and executed.

## Slide 9 - Convergence and Stability
**Main message:** Stochastic optimizers are evaluated with convergence curves and repeated trials.

Content:
- Latest PSO run: 2 trials per strategy.
- Latest ACO+EBGWO run: 2 trials per strategy.
- Report tables use mean +- standard deviation.
- Convergence plot tracks best and average fitness.
- Fitness combines Sharpe, entropy, and penalty terms.

Visual:
Insert `output/pso_26_05_03_39/convergence.png`.

## Slide 10 - Evaluation Metrics
**Main message:** Performance is assessed using return, risk, and drawdown metrics.

Content:
- Sharpe Ratio: risk-adjusted return above risk-free rate.
- Sortino Ratio: requested metric; not currently reported by `compute_metrics`.
- Cumulative Return.
- Annual Return.
- Annual Volatility.
- Max Drawdown.
- With-cost and no-cost results are reported side by side.

Speaker note:
Sortino should be added to `portfolio_optimization/backtest.py` if required for final grading.

## Slide 11 - Latest PSO Results
**Main message:** PSO has the strongest latest full-period result.

Content:
- Best full-period with-cost result: PSO Portfolio.
- Cum Return: 50.85% +- 9.11%.
- Annual Return: 39.65% +- 4.98%.
- Sharpe Ratio: 1.4564 +- 0.3083.
- Max Drawdown: -21.97% +- 3.05%.
- Benchmark SPY: 7.43% cumulative return, Sharpe 0.1949.

Visual:
Insert `output/pso_26_05_03_39/performance.png` or `performance_2025.png`.

## Slide 12 - Latest ACO+EBGWO Results
**Main message:** ACO-selected portfolios performed best within the ACO+EBGWO weight-optimization run.

Content:
- Best full-period with-cost result: ACO Selected.
- Cum Return: 40.90% +- 11.75%.
- Annual Return: 33.31% +- 6.88%.
- Sharpe Ratio: 1.2425 +- 0.4215.
- Max Drawdown: -17.75% +- 5.43%.
- Dynamic ACO+EBGWO was more unstable: 16.43% +- 21.98% cumulative return.

Visual:
Insert `output/aco_ebgwo_26_05_03_38/performance.png`.

## Slide 13 - Walk-Forward Test Design
**Main message:** The OOS test re-optimizes weights through time, including turnover costs.

Content:
- Requested design: 4-year train / 1-year test walk-forward.
- Current executed design: rolling 3-year train / quarterly test step.
- Each window computes new weights, applies transaction cost to the first OOS day, then appends realized OOS returns.
- Cost formula: `sum(abs(w_new - w_prev)) * transaction_cost_rate`.

## Slide 14 - Key Findings
**Main message:** Robust selection plus SI weight optimization beats the S&P 500 benchmark in the latest run.

Content:
- PSO Portfolio delivered the highest full-period Sharpe in latest results.
- LW cluster methods remain competitive and often reduce drawdown.
- ACO+EBGWO results show higher variance, especially in the dynamic full-universe case.
- Transaction costs are small but consistently reduce returns.
- Missing final items for the requested rubric: Sortino metric, explicit 4-year/1-year WFO implementation, and short-sell support.

## Slide 15 - Recommendations / Next Steps
**Main message:** Align implementation with requested evaluation spec before final submission.

Content:
- Add Sortino Ratio to reported metrics.
- Change walk-forward parameters to 4-year lookback and 1-year test step if required.
- Enforce exact portfolio size min 5 / max 15 across all selectors.
- Add Chinese, ETC, and Crypto tickers to `config/assets.csv` if those markets must be tested.
- Keep PSO as the current leading weight optimizer baseline.

