# Project Rules and Conventions

## Workspace Rules
- Always use `uv` for python dependencies and execution (e.g. `uv run python ...`).
- Base currency is THB (Thailand Baht).
- Use `torch` for computations where GPU/CUDA fallback to CPU is automatically supported.
- Console logs and outputs should use `rich` library styled tables with a standard console width of 120 to prevent wrapping/unicode clipping in Windows terminals.

## Data & Date Configurations
- Date boundaries are managed dynamically in `config/settings.json`:
  - `start_date` and `end_date`: Outer boundaries for downloading asset data.
  - `insample_end_date`: Cutoff date for fitting selector strategies and covariance matrix estimation.
  - `outsample_end_date`: End date of backtest validation.
- All returns sorting pinned SPY (S&P 500 Benchmark) to the bottom of the table. Other strategies are sorted descending by Cumulative Return (`Cum Return`).

## Pipeline & Outputs Structure
- Running `run_pipeline.py` creates a timestamped subdirectory: `output/run_{DD_MM_HH_MM}/`.
- Output assets created per run:
  - `pipeline.log`: Execution log.
  - `convergence.png`: Optimization convergence chart.
  - `performance.png` / `performance_2025.png`: Performance charts.
  - `performance_report.md`: Markdown report table.
  - `candles/`: Subfolder containing strategy-wise candlestick grid charts (`candles/*.png`).
