# Parameter Tuning and Sensitivity Analysis Pipeline

We will create a new pipeline script `tune_pipeline.py` to systematically evaluate how key hyperparameters affect portfolio optimization performance, and plot sensitivity graphs.

## Proposed Design & Architecture

### Key Parameters to Support for Tuning/Sensitivity
1. **Internal EBGWO Parameters**:
   - `ST` (Exploration threshold, default `0.4`): Controls exploitation vs exploration.
   - `eps` (Entropy epsilon, default `1e-10`): Numerical stability parameter in weight entropy.
2. **Internal ACO Parameters**:
   - `Q` (Pheromone intensity coefficient, default `1.0`): Pheromone deposit amount.
   - `evaporation_rate` (Pheromone evaporation rate, default `0.1`): Decay rate.
   - `alpha` (Pheromone importance, default `1.0`) & `beta` (Heuristic importance, default `2.0`).
3. **Portfolio Parameters**:
   - `lambda_ent` (default `0.05`): Entropy regularization coefficient.
   - `max_weight` (default `0.1`): Allocation boundary.

### Features of `tune_pipeline.py`
- Grid/Sensitivity search execution.
- Evaluates the core `Dynamic ACO+EBGWO Portfolio` strategy out-of-sample (OOS) or a specific selection strategy.
- Saves results to a timestamped folder: `output/tuning_{DD_MM_HH_MM}/`.
- Generates 2D heatmaps (when testing two parameters) or line charts (when testing a single parameter).
- Outputs a comprehensive markdown table summary (`tuning_report.md`).

---

## Proposed Changes

### [Component Name: Parameter Tuning Module]

#### [NEW] [tune_pipeline.py](file:///D:/Code/sector_constrained_portfolio_executed/tune_pipeline.py)
New pipeline script to coordinate the grid execution:
- Parses arguments for parameter grids (ranges, step sizes).
- Slices data using settings from `config/settings.json`.
- Runs rolling backtests for each parameter configuration.
- Aggregates performance metrics (Cumulative Return, Sharpe, Max Drawdown).
- Uses `matplotlib` to plot sensitivity charts (line plots or 2D heatmaps).
- Outputs a sorted performance table.

#### [MODIFY] [manual.md](file:///D:/Code/sector_constrained_portfolio_executed/manual.md)
Update documentation to explain how to use the tuning pipeline.

---

## Verification Plan

### Automated Tests
- Run a fast test tuning run with small iteration sizes:
  ```bash
  uv run python tune_pipeline.py --lambda-grid 0.01,0.05,0.1 --wolves 10 --iterations 5 --agents 5
  ```
- Verify output folder generation, charts, and report generation in `output/tuning_{DD_MM_HH_MM}/`.
