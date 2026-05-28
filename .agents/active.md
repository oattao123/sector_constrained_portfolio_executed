# Active Task Status

## Current Goal
Implement continuous domain and adaptive optimizers (LAPSO, ACOR, CIAC), integrate them into the backtest framework and main pipeline, configure subfolder outputs per optimizer, establish dedicated run pipeline scripts, and document implementation in English and Thai user manuals.

## Progress
- [x] Configure transaction cost rate in `config/settings.json` under `portfolio.transaction_cost_rate`
- [x] Implement two-way weight turnover calculation and cost deduction at each rebalancing step in `portfolio_optimization/backtest.py`
- [x] Modify `run_pipeline.py` and `tune_pipeline.py` to pass transaction cost rate to backtester
- [x] Update `run_pipeline.py` to compare with-cost and no-cost portfolios, outputting side-by-side comparisons in `performance_report.md`
- [x] Implement `--trials` argument in `run_pipeline.py` to execute multiple runs per strategy and display results as `Mean +- Std`
- [x] Create dedicated pipeline scripts for EBGWO (`run_pipeline_aco_ebgwo.py`) and PSO (`run_pipeline_pso.py`)
- [x] Implement CLPSO and APSO weight optimizers in `portfolio_optimization/optimizers.py`
- [x] Add CLPSO and APSO run branches in `WalkForwardBacktester` dispatch
- [x] Create dedicated run scripts `run_pipeline_clpso.py` and `run_pipeline_apso.py`
- [x] Implement Landscape-Aware Adaptive PSO (LAPSO) weight optimizer (with FDC and Mirrored Boundary Handling) in `portfolio_optimization/optimizers.py`
- [x] Create dedicated run script `run_pipeline_lapso.py`
- [x] Implement Ant Colony Optimization for Continuous Domains ($ACO_{\mathbb{R}}$ or ACOR) weight optimizer in `portfolio_optimization/optimizers.py`
- [x] Create dedicated run script `run_pipeline_acor.py`
- [x] Implement Continuous Interacting Ant Colony (CIAC) weight optimizer in `portfolio_optimization/optimizers.py`
- [x] Fix index out of bounds error in CIAC optimizer when particle count < archive size by dynamically clamping archive size
- [x] Create dedicated run script `run_pipeline_ciac.py`
- [x] Integrate LAPSO, ACOR, and CIAC into `portfolio_optimization/backtest.py` dispatch
- [x] Save weight optimization results (average optimized asset weights per rebalancing date) as CSV files inside the output run and optimizer-specific folders
- [x] Integrate LAPSO, ACOR, and CIAC comparisons in the main `run_pipeline.py` pipeline (totaling 7 optimizers evaluated on all 6 selection strategies + full universe)
- [x] Configure main pipeline to save optimizer-specific subdirectories containing reports + convergence/performance plots for each optimizer
- [x] Update `tune_pipeline.py` to support selecting and tuning all 7 optimizers (by adding `--optimizer` option and passing it to the backtester)
- [x] Verify all pipelines (main, EBGWO, PSO, CLPSO, APSO, LAPSO, ACOR, and CIAC) with fast test runs
- [x] Update English and Thai manuals (`manual.md`, `manual_TH.md`) to document all new optimizers, parameter details, run commands, and pipeline reports

## Next Steps
- Await next instructions from Master.
