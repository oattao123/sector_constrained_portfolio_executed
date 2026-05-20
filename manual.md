# Portfolio Optimization Pipeline — User Manual

A modular, production-grade pipeline implementing Ledoit-Wolf covariance estimation, Hierarchical Risk Parity (HRP), 2D Ant Colony Optimization (ACO), and Enhanced Binary Grey Wolf Optimizer (EBGWO) for multi-asset walk-forward portfolio optimization.

---

## 1. Prerequisites

Install dependencies via `uv`:

```bash
uv sync
```

Required packages: `torch`, `yfinance`, `pandas`, `numpy`, `scipy`, `rich`, `matplotlib`, `scikit-learn`.

GPU (CUDA) is auto-detected and used when available; falls back to CPU automatically.

---

## 2. Basic Execution

```bash
uv run python run_pipeline.py
```

---

## 3. Command Line Arguments

| Parameter | Type | Default | Description |
| :--- | :---: | :---: | :--- |
| `--settings` | `str` | `config/settings.json` | Path to settings JSON configuration file |
| `--assets` | `str` | `config/assets.csv` | Path to asset universe CSV definition file |
| `--cache` | `str` | `all_data.csv` | Cache file for downloaded price data |
| `--wolves` | `int` | `500` | EBGWO population size |
| `--iterations` | `int` | `1000` | Optimizer iterations (EBGWO & ACO) |
| `--agents` | `int` | `500` | Number of ants for ACO selectors |

### Fast Test Run
```bash
uv run python run_pipeline.py --wolves 10 --iterations 5 --agents 5
```

### Full Production Run
```bash
uv run python run_pipeline.py --wolves 1000 --iterations 1000 --agents 500
```

---

## 4. Configuration: `config/settings.json`

```json
{
  "base_currency": "THB",
  "data": {
    "start_date":          "2015-01-01",
    "end_date":            "2026-03-20",
    "insample_end_date":   "2024-01-01",
    "outsample_end_date":  "2026-03-20"
  },
  "risk_free_rate": {
    "ticker":   "^TNX",
    "fallback": 0.045
  },
  "portfolio": { "value": 1000000 },
  "currency":  { "fx_pairs": { "USD": "USDTHB=X" } },
  "quality_checks": {
    "max_abs_daily_return": 0.25,
    "max_annual_return":    1.5
  }
}
```

### Date Configuration

| Key | Purpose |
| :--- | :--- |
| `start_date` | Earliest date for data download |
| `end_date` | Latest date for data download |
| `insample_end_date` | Covariance & selector fitting boundary. Data **before** this date is used to train the selectors. |
| `outsample_end_date` | End of out-of-sample evaluation window for the backtester. |

> **Example**: `insample_end_date = 2024-01-01` means all 6 asset selectors and the Ledoit-Wolf covariance are fitted only on data from `start_date` to `2024-01-01`. The walk-forward backtester then rolls forward from `2024-01-01` to `outsample_end_date`.

---

## 5. Pipeline Outputs

Each run creates a **timestamped folder** `output/run_{DD_MM_HH_MM}/` containing:

| File | Description |
| :--- | :--- |
| `pipeline.log` | Full execution log with timestamps |
| `convergence.png` | Average EBGWO fitness convergence curves |
| `performance.png` | Cumulative return comparison (full period) |
| `performance_2025.png` | Cumulative return comparison (Year 2025 only) |
| `performance_report.md` | Backtest metrics table sorted by Cumulative Return |
| `candles/` | Subfolder containing candlestick charts for every selection strategy (6 files) |

---

## 5.5 Parameter Tuning & Sensitivity Analysis Pipeline

The system includes a dedicated `tune_pipeline.py` script to perform grid search and sensitivity analysis on hyperparameters. This allows developers to evaluate how internal algorithm parameters affect out-of-sample (OOS) performance.

### Tunable Parameters

The pipeline supports tuning any internal parameter for the optimizer, including:
- **EBGWO Parameters**: `ST` (Exploration threshold), `eps` (entropy regularizer epsilon), `wolves` (population size), `iterations` (EBGWO iterations).
- **ACO Parameters**: `Q` (Pheromone intensity), `evaporation_rate` (Pheromone decay), `alpha_aco`, `beta_aco`, `patience`, `agents` (number of ants).
- **Portfolio Objectives**: `lambda_ent` (Entropy regularization coefficient), `max_weight` (Allocation boundary).

### Command-Line Arguments

| Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--settings` | `str` | `"config/settings.json"` | Path to the settings JSON configuration file |
| `--assets` | `str` | `"config/assets.csv"` | Path to the target assets definition CSV file |
| `--cache` | `str` | `"data/all_data.csv"` | Path to local OHLCV price cache |
| `--wolves` | `int` | `100` | Number of wolves for the EBGWO optimizer (reduced default for tuning speed) |
| `--iterations` | `int` | `150` | Number of optimization iterations |
| `--agents` | `int` | `100` | Number of agents for the ACO optimizer |
| `--trials` | `int` | `10` | Number of trials to run per configuration to average stochastic optimization results |
| `--param` | `str` | `"ST"` | Name of the primary hyperparameter to tune |
| `--values` | `str` | `"0.1,0.2,0.3,0.4,0.5"` | Comma-separated values for the primary hyperparameter |
| `--param2` | `str` | `None` | Name of the optional secondary hyperparameter (enables 2D grid search) |
| `--values2` | `str` | `None` | Comma-separated values for the secondary hyperparameter |
| `--strategy` | `str` | `"ACO Cluster Selected"` | Base selection strategy (`Diversification Selected`, `LW Diversification Selected`, `LW Sharpe Cluster Selected`, `LW Div Cluster Selected`, `ACO Selected`, or `ACO Cluster Selected`) |

### Usage Examples

**1. 1D Sensitivity Search (Line Plot Output)**
Run a 1D sensitivity search for `ST` (exploration threshold):
```bash
uv run python tune_pipeline.py --param ST --values 0.1,0.2,0.3,0.4,0.5
```

**2. 2D Grid Search (Heatmap Output)**
Tune EBGWO exploration threshold `ST` vs weight entropy penalty `lambda_ent`:
```bash
uv run python tune_pipeline.py --param ST --values 0.2,0.3,0.4 --param2 lambda_ent --values2 0.01,0.05,0.1
```

### Outputs

Tuning results are stored in a unique `output/tuning_{DD_MM_HH_MM}/` directory:
1. `tuning.log`: Full execution log.
2. `tuning_report.md`: Markdown summary table sorted by Sharpe ratio. Metrics are reported in a `mean +- std` format (e.g., `20.20% +- 1.15%` / `0.6499 +- 0.0315`) across all successful trials to evaluate the stability of stochastic metaheuristics.
3. `sensitivity_analysis.png`:
   - Dual-axis line plot (for 1D sensitivity searches).
   - 2D Performance Heatmaps of Sharpe Ratio and Cumulative Return (for 2D grid searches).

---

## 6. Code Documentation: `portfolio_optimization` Package

---

### `data_manager.py` — `DataManager`

**Class: `DataManager(settings_path, assets_path)`**

Manages data ingestion, caching, cleaning, and currency conversion.

| Method | Signature | Description |
| :--- | :--- | :--- |
| `load_config` | `()` | Loads `settings.json` and `assets.csv`. Populates `sector_map`, `currency_map`, `tickers`. |
| `download_data` | `(start_date, end_date, cache_path, use_cache)` | Downloads OHLCV Close prices from Yahoo Finance in batches of 50. Reads from local CSV cache if available and all tickers match. Appends SPY automatically. |
| `clean_data` | `(df, max_gap_days=365)` | Drops columns with consecutive NaN gaps longer than `max_gap_days`. Forward-fills then backward-fills remaining gaps. |
| `convert_to_base_currency` | `(df, start_date, end_date)` | Downloads `USDTHB=X` FX rate and multiplies all USD-denominated assets. Falls back to fixed rate `35.0 THB/USD` on download failure. |
| `get_log_returns` | `(df)` | Computes `log(P_t / P_{t-1})` and drops the first NaN row. |

---

### `covariance.py` — Robust Covariance Estimation

**`get_best_device() → torch.device`**
Auto-detects CUDA GPU. Returns `cuda` if available, else `cpu`.

---

**`to_tensor(data, device, dtype) → torch.Tensor`**
Converts `np.ndarray`, `pd.DataFrame`, `pd.Series`, or `list` to a PyTorch tensor on the target device.

---

**`ledoit_wolf_covariance_gpu_dynamic(X) → (shrunk_cov, sample_cov, delta)`**

Computes Ledoit-Wolf analytical shrinkage covariance entirely in PyTorch (GPU-accelerated).

| Arg | Type | Description |
| :--- | :--- | :--- |
| `X` | `Tensor (n_samples, n_features)` | Log returns matrix |

**Returns:**

| Value | Description |
| :--- | :--- |
| `shrunk_cov` | `(n_features, n_features)` — Shrunk covariance `(1-δ)·S + δ·T` |
| `sample_cov` | `(n_features, n_features)` — Raw sample covariance |
| `delta` | `float` — Optimal shrinkage intensity `δ ∈ [0, 1]` |

**Algorithm:** Oracle Approximating Shrinkage (OAS) — computes `δ` analytically from `b²/d²` where `d²` is distance between sample and target, `b²` is the variance of the sample estimator.

---

**`compute_correlation_matrix(cov_matrix) → torch.Tensor`**

Converts covariance matrix to correlation matrix by normalising with outer product of standard deviations. Clamps to `[-1, 1]` to prevent floating-point boundary violations.

---

### `selectors.py` — Asset Selection Strategies

All selectors return a `pd.DataFrame` sorted by `Diversification Score` (descending) containing columns: `Ticker`, `Sector`, `Ann Return (%)`, `Ann Vol (%)`, `Sharpe`, `Avg Corr`, `Diversification Score`.

---

**`select_by_diversification(returns_df, all_sector_map, risk_free_rate, top_n, min_per_sector, min_sharpe_threshold) → DataFrame`**

**Strategy 1 — Standard Diversification Selection.**

Computes standard (non-robust) Sharpe ratio and pairwise correlations. Filters assets with `Sharpe > min_sharpe_threshold`. Scores each asset as `Sharpe × (1 − avg_corr)`. Applies stratified sector sampling: guarantees `min_per_sector` stocks from every sector, then fills remaining slots greedily by score.

---

**`select_by_lw_diversification(returns_df, shrunk_cov, corr_matrix, all_sector_map, risk_free_rate, top_n, min_per_sector, min_sharpe_threshold) → DataFrame`**

**Strategy 2 — Ledoit-Wolf Diversification Selection.**

Same as Strategy 1 but uses the pre-computed Ledoit-Wolf shrunk covariance for volatility and the LW correlation matrix. More robust to small sample sizes and noisy data.

---

**`select_by_hrp_sharpe(returns_df, shrunk_cov, corr_matrix, all_sector_map, risk_free_rate, num_clusters) → DataFrame`**

**Strategy 3 — HRP + Max Sharpe Cluster Selection.**

Builds a hierarchical clustering tree (Ward linkage on LW distance matrix `D = √(0.5·(1−ρ))`). Cuts the tree into `num_clusters` groups using `fcluster`. From each cluster, selects the asset with the **highest Sharpe ratio**.

---

**`select_by_hrp_div(returns_df, shrunk_cov, corr_matrix, all_sector_map, risk_free_rate, num_clusters) → DataFrame`**

**Strategy 4 — HRP + Max Diversification Cluster Selection.**

Same clustering approach as Strategy 3. From each cluster, selects the asset with the **highest Diversification Score** (`Sharpe × (1 − mean_corr)`).

---

**`select_by_aco(returns_df, shrunk_cov, corr_matrix, all_sector_map, risk_free_rate, target_assets, num_ants, num_iterations) → DataFrame`**

**Strategy 5 — 2D-ACO Direct Selection.**

Runs a vectorized 2D Ant Colony Optimization algorithm across all assets simultaneously. Each ant builds a portfolio of `target_assets` tickers by stepping through assets one at a time. Pheromone updates reinforce pairs of assets that co-occur in high-Sharpe portfolios.

| Param | Default | Description |
| :--- | :---: | :--- |
| `target_assets` | `30` | Number of assets to select |
| `num_ants` | `500` | Colony size (population) |
| `num_iterations` | `1000` | ACO training epochs |
| `alpha` | `1.0` | Pheromone weight |
| `beta` | `2.0` | Heuristic (Sharpe) weight |
| `evaporation_rate` | `0.1` | Trail evaporation per iteration |

---

**`select_by_aco_cluster(returns_df, shrunk_cov, corr_matrix, all_sector_map, risk_free_rate, num_clusters, num_ants, num_epochs) → DataFrame`**

**Strategy 6 — 2D-ACO Cluster Selection.**

Combines HRP clustering with 2D-ACO. First clusters assets (Ward linkage), then runs ACO where each ant picks **one representative from each cluster** per step, learning inter-cluster synergies via 2D pheromone trails.

---

### `optimizers.py` — Weight Allocation

**`optimize_weights_ebgwo_monte_carlo_entropy(train_returns_gpu, max_weight, lambda_ent, num_wolves, iterations) → np.ndarray`**

EBGWO (Enhanced Binary Grey Wolf Optimizer) with Monte Carlo bootstrapping and entropy regularization.

| Param | Default | Description |
| :--- | :---: | :--- |
| `max_weight` | `0.1` | Maximum weight per asset (concentration limit) |
| `lambda_ent` | `0.05` | Entropy regularization coefficient (higher = more equal-weight) |
| `num_wolves` | `500` | Wolf population size |
| `iterations` | `1000` | Training epochs |

**Fitness function:** `Sharpe + λ·H(w)/log(n) − penalty` where `H(w)` is normalised portfolio weight entropy and `penalty` penalises `max_weight` violations.

**Returns:** `np.ndarray (n_assets,)` — optimal normalised weights summing to 1.

---

**`optimize_weights_aco_ebgwo(train_returns_gpu, target_assets, heuristic_tensor, sector_labels, num_iterations, num_agents) → (weights, best_conv, avg_conv)`**

Co-evolutionary ACO + EBGWO: ACO selects which assets to include; EBGWO allocates weights among the selected assets. Supports optional sector-constraint enforcement via `sector_labels`.

**Returns:**
- `weights` — `np.ndarray (n_assets,)` final portfolio weights
- `best_conv` — `list[float]` best fitness per iteration (convergence curve)
- `avg_conv` — `list[float]` average population fitness per iteration

---

### `backtest.py` — Walk-Forward Engine

**`compute_metrics(returns, risk_free_rate) → dict`**

Computes portfolio performance metrics from a daily returns array.

| Output Key | Description |
| :--- | :--- |
| `cum_return` | Total cumulative return over the period |
| `ann_return` | Annualised mean return (×252) |
| `ann_vol` | Annualised volatility (×√252) |
| `sharpe` | Annualised Sharpe ratio |
| `max_dd` | Maximum drawdown (negative value) |
| `cum_returns_arr` | Full cumulative return time series |

---

**Class: `WalkForwardBacktester(full_returns, spy_full_returns, sector_map, risk_free_rate)`**

Executes rolling window walk-forward optimization.

**`.run(portfolio_name, selected_stocks, lookback_window, step_size, num_iterations, num_agents, use_sector_constraints) → dict`**

| Param | Default | Description |
| :--- | :---: | :--- |
| `lookback_window` | `252×3` | In-sample training window (trading days) |
| `step_size` | `21×3` | Out-of-sample test window (quarterly rebalance) |
| `num_iterations` | `1000` | Optimizer iterations per window |
| `num_agents` | `500` | Optimizer population per window |
| `use_sector_constraints` | `False` | Enforce sector diversification via ACO+EBGWO |

**Returns dict keys:** `Strategy`, `Cum Return`, `Ann Return`, `Ann Volatility`, `Sharpe Ratio`, `Max Drawdown`, `OOS_Returns_Array`, `OOS_Cum_Returns_Array`, `Avg_Best_Convergence`, `Avg_Avg_Convergence`, `Dates`.

---

**`save_convergence_plot(results_list, output_path) → None`**
Saves average EBGWO fitness convergence curves for all strategies to a PNG file.

**`save_performance_plot(results_list, spy_cum_returns, trade_dates, output_path) → None`**
Saves cumulative return comparison chart (full period) to a PNG file.

**`save_performance_plot_2025(results_list, spy_returns_daily, trade_dates, output_path) → None`**
Saves cumulative return comparison chart filtered to Year 2025 only.

---

### `cli.py` — Rich Terminal Output

| Function | Description |
| :--- | :--- |
| `print_welcome()` | Renders a styled welcome panel with project title and base currency |
| `print_asset_summary(assets_df)` | Renders a table of loaded tickers, sectors, and currencies |
| `print_selection_summary(strategy_name, df_selected)` | Renders selected asset table with return/vol/Sharpe/diversification metrics |
| `print_backtest_table(results, title)` | Renders walk-forward performance comparison table. Expects list of result dicts. |

All output uses `Console(width=120)` to prevent column truncation in redirected terminals.

---

### `charts.py` — Candlestick Grid Plotting

**`save_candlestick_grid(strategy_name, selected_df, start_date, end_date, output_path, max_cols, candle_period, lookback_bars, sector_map) → None`**

Downloads OHLCV price data for all tickers in a selection result and plots them as a dark-themed candlestick grid, saved as a PNG file.

| Parameter | Type | Default | Description |
| :--- | :---: | :---: | :--- |
| `strategy_name` | `str` | — | Chart title (strategy label) |
| `selected_df` | `DataFrame` | — | Selector output — must contain a `Ticker` column |
| `start_date` | `str` | — | OHLCV download start (`YYYY-MM-DD`) — typically `insample_end_date` |
| `end_date` | `str` | — | OHLCV download end (`YYYY-MM-DD`) — typically `outsample_end_date` |
| `output_path` | `str` | — | Full path to the output PNG file |
| `max_cols` | `int` | `5` | Maximum subplot columns in the grid |
| `candle_period` | `str` | `'W'` | Resample period: `'D'` (daily), `'W'` (weekly), `'ME'` (monthly) |
| `lookback_bars` | `int` | `52` | Number of candles to display per ticker (e.g. 52 = 1 year of weekly candles) |
| `sector_map` | `dict` | `None` | Maps `ticker → sector` string for colour-coded subplot borders and legend |

**Visual features:**
- Dark `#0d1117` background (GitHub dark theme)
- Green (`#00e676`) up-candles / Red (`#ff1744`) down-candles
- Subplot border colour-coded by sector with a bottom legend
- Title of each subplot shows ticker and total % return over the window
- Grid saved at 150 DPI

**Pipeline integration:**  
Called automatically in `run_pipeline.py` after all 6 selectors complete. Generates one file per strategy inside a dedicated subfolder:

```
output/run_{DD_MM_HH_MM}/
└── candles/
    ├── diversification_selected.png
    ├── lw_diversification_selected.png
    ├── lw_sharpe_cluster_selected.png
    ├── lw_div_cluster_selected.png
    ├── aco_selected.png
    └── aco_cluster_selected.png
```
