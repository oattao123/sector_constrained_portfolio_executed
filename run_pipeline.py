import argparse
import logging
import math
import os
import warnings
from datetime import datetime

import matplotlib
matplotlib.use("Agg")  # force non-interactive backend before any other mpl import

import numpy as np
import pandas as pd
import torch

# Import package components
from portfolio_optimization import (
    DataManager,
    WalkForwardBacktester,
    compute_correlation_matrix,
    compute_metrics,
    ledoit_wolf_covariance_gpu_dynamic,
    optimize_weights_pso,
    print_asset_summary,
    print_backtest_table,
    print_selection_summary,
    print_welcome,
    save_candlestick_grid,
    save_convergence_plot,
    save_performance_plot,
    save_performance_plot_2025,
    select_by_aco,
    select_by_aco_cluster,
    select_by_diversification,
    select_by_hrp_div,
    select_by_hrp_sharpe,
    select_by_lw_diversification,
)

warnings.filterwarnings("ignore")

# Logger configured inside main() after output dir is known
logger = logging.getLogger("run_pipeline")


def parse_args():
    parser = argparse.ArgumentParser(description="Multi-Asset Portfolio Optimization Pipeline")
    parser.add_argument("--settings", type=str, default="config/settings.json", help="Path to settings.json")
    parser.add_argument("--assets", type=str, default="config/assets.csv", help="Path to assets.csv")
    parser.add_argument("--cache", type=str, default="data/all_data.csv", help="Path to cache price CSV file")
    parser.add_argument("--wolves", type=int, default=500, help="Number of wolves for EBGWO")
    parser.add_argument("--iterations", type=int, default=1000, help="Number of iterations for optimizers")
    parser.add_argument("--agents", type=int, default=500, help="Number of agents for ACO")
    parser.add_argument("--particles", type=int, default=500, help="Number of particles for PSO")
    return parser.parse_args()


def main():
    args = parse_args()

    # 0. Create timestamped output directory: output/run_{DD_M_HH_MM}
    now = datetime.now()
    run_tag = now.strftime("run_%d_%m_%H_%M")
    out_dir = os.path.join("output", run_tag)
    os.makedirs(out_dir, exist_ok=True)

    # Configure logging now that we know the output path
    log_path = os.path.join(out_dir, "pipeline.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_path, encoding="utf-8"), logging.StreamHandler()],
    )
    logger.info(f"Run output directory: {out_dir}")

    # 1. Print Welcome and config info nya~!
    print_welcome()

    # 2. Ingest, Clean, and Convert Currency
    manager = DataManager(settings_path=args.settings, assets_path=args.assets)
    print_asset_summary(manager.assets_df)

    # Read in-sample / out-of-sample date boundaries from settings
    data_cfg = manager.settings.get("data", {})
    insample_end_date = data_cfg.get("insample_end_date", None)
    outsample_end_date = data_cfg.get("outsample_end_date", None)
    logger.info(f"In-sample period  : {data_cfg.get('start_date')} -> {insample_end_date or 'full'}")
    logger.info(f"Out-of-sample period: {insample_end_date or 'start'} -> {outsample_end_date or 'full'}")

    raw_data = manager.download_data(cache_path=args.cache)
    clean_data = manager.clean_data(raw_data)
    converted_data = manager.convert_to_base_currency(clean_data)

    # Compute log returns (full range for the backtester)
    returns = manager.get_log_returns(converted_data)

    # Handle SPY Benchmark separately
    spy_full_returns = returns["SPY"] if "SPY" in returns.columns else pd.Series(0.0, index=returns.index)

    # Remove SPY from optimization universe
    opt_universe_returns = returns.drop(columns=["SPY"]) if "SPY" in returns.columns else returns
    valid_tickers = opt_universe_returns.columns

    # Slice in-sample window for covariance & selector fitting
    insample_returns = opt_universe_returns.loc[:insample_end_date] if insample_end_date else opt_universe_returns
    logger.info(f"In-sample returns shape : {insample_returns.shape}")
    logger.info(f"Full returns shape      : {opt_universe_returns.shape}")

    # 3. Compute Robust Ledoit-Wolf Covariance (fit on in-sample only)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    returns_gpu = torch.tensor(insample_returns.values, dtype=torch.float32, device=device)

    shrunk_cov_gpu, sample_cov_gpu, delta_val = ledoit_wolf_covariance_gpu_dynamic(returns_gpu)
    corr_matrix_gpu = compute_correlation_matrix(shrunk_cov_gpu)

    # Expose risk free rate
    risk_free_rate = manager.settings.get("risk_free_rate", {}).get("fallback", 0.0434)

    # 4. Run Asset Selectors (fit on in-sample returns only)
    logger.info("Running Selection Strategies...")

    # S1. Diversification Selected (Standard)
    div_sel = select_by_diversification(insample_returns, manager.sector_map, risk_free_rate, top_n=30)
    print_selection_summary("Diversification Selected", div_sel)

    # S2. LW Diversification Selected
    lw_div_sel = select_by_lw_diversification(
        insample_returns, shrunk_cov_gpu, corr_matrix_gpu, manager.sector_map, risk_free_rate, top_n=30
    )
    print_selection_summary("LW Diversification Selected", lw_div_sel)

    # S3. LW Sharpe Cluster Selected
    lw_sharpe_cl = select_by_hrp_sharpe(
        insample_returns, shrunk_cov_gpu, corr_matrix_gpu, manager.sector_map, risk_free_rate, num_clusters=30
    )
    print_selection_summary("LW Sharpe Cluster Selected", lw_sharpe_cl)

    # S4. LW Div Cluster Selected
    lw_div_cl = select_by_hrp_div(
        insample_returns, shrunk_cov_gpu, corr_matrix_gpu, manager.sector_map, risk_free_rate, num_clusters=30
    )
    print_selection_summary("LW Div Cluster Selected", lw_div_cl)

    # S5. ACO Selected
    aco_sel = select_by_aco(
        insample_returns,
        shrunk_cov_gpu,
        corr_matrix_gpu,
        manager.sector_map,
        risk_free_rate,
        target_assets=30,
        num_ants=args.agents,
        num_iterations=args.iterations,
    )
    print_selection_summary("ACO Selected", aco_sel)

    # S6. ACO Cluster Selected
    aco_cl_sel = select_by_aco_cluster(
        insample_returns,
        shrunk_cov_gpu,
        corr_matrix_gpu,
        manager.sector_map,
        risk_free_rate,
        num_clusters=30,
        num_ants=args.agents,
        num_epochs=args.iterations,
    )
    print_selection_summary("ACO Cluster Selected", aco_cl_sel)

    # Create portfolios dict
    portfolio_dfs = {
        "Diversification Selected": div_sel,
        "LW Diversification Selected": lw_div_sel,
        "LW Sharpe Cluster Selected": lw_sharpe_cl,
        "LW Div Cluster Selected": lw_div_cl,
        "ACO Selected": aco_sel,
        "ACO Cluster Selected": aco_cl_sel,
    }

    # 4b. Save candlestick charts for each selection method
    logger.info("Generating candlestick charts for each selection strategy...")
    candle_start = data_cfg.get("insample_end_date") or data_cfg.get("start_date", "2015-01-01")
    candle_end   = data_cfg.get("outsample_end_date") or data_cfg.get("end_date", "2026-03-20")
    
    # Create subfolder inside out_dir
    candle_dir = os.path.join(out_dir, "candles")
    os.makedirs(candle_dir, exist_ok=True)
    
    for strat_name, strat_df in portfolio_dfs.items():
        if strat_df.empty:
            continue
        safe_name = strat_name.lower().replace(" ", "_")
        chart_path = os.path.join(candle_dir, f"{safe_name}.png")
        save_candlestick_grid(
            strategy_name=strat_name,
            selected_df=strat_df,
            start_date=candle_start,
            end_date=candle_end,
            output_path=chart_path,
            max_cols=5,
            candle_period="W",
            lookback_bars=52,
            sector_map=manager.sector_map,
        )

    # 5. Walk-Forward Backtester
    logger.info("Initializing Walk-Forward Backtester...")
    backtester = WalkForwardBacktester(
        full_returns=returns,
        spy_full_returns=spy_full_returns,
        sector_map=manager.sector_map,
        risk_free_rate=risk_free_rate,
    )

    all_results = []

    # A. Run walk-forward with EBGWO weight optimization for each static selection strategy
    for name, df in portfolio_dfs.items():
        if df.empty:
            logger.warning(f"Portfolio {name} is empty. Skipping backtest.")
            continue
        tickers_list = df["Ticker"].tolist()
        res = backtester.run(
            portfolio_name=name,
            selected_stocks=tickers_list,
            num_iterations=args.iterations,
            num_agents=args.wolves,
            use_sector_constraints=False,
        )
        all_results.append(res)

    # B. Run dynamic co-evolutionary ACO + EBGWO optimization (Full Universe)
    logger.info("Running dynamic co-evolutionary ACO+EBGWO walk-forward optimization...")
    dynamic_res = backtester.run(
        portfolio_name="Dynamic ACO+EBGWO Portfolio",
        selected_stocks=valid_tickers.tolist(),
        num_iterations=args.iterations,
        num_agents=args.wolves,
        use_sector_constraints=True,
        optimizer='aco_ebgwo',
    )
    all_results.append(dynamic_res)

    # C. Run PSO walk-forward optimization (Full Universe)
    logger.info("Running PSO walk-forward optimization...")
    pso_res = backtester.run(
        portfolio_name="PSO Portfolio",
        selected_stocks=valid_tickers.tolist(),
        num_iterations=args.iterations,
        num_agents=args.particles,
        use_sector_constraints=False,
        optimizer='pso',
    )
    all_results.append(pso_res)

    # C. Calculate SPY Benchmark Metrics
    lookback_window = 252 * 3
    test_len = len(dynamic_res["OOS_Returns_Array"])
    trade_dates = returns.index[lookback_window : lookback_window + test_len]
    spy_returns_arr = spy_full_returns.iloc[lookback_window : lookback_window + test_len].values

    spy_metrics = compute_metrics(spy_returns_arr, risk_free_rate)
    spy_res = {
        "Strategy": "SPY (S&P 500 Benchmark)",
        "Cum Return": spy_metrics["cum_return"],
        "Ann Return": spy_metrics["ann_return"],
        "Ann Volatility": spy_metrics["ann_vol"],
        "Sharpe Ratio": spy_metrics["sharpe"],
        "Max Drawdown": spy_metrics["max_dd"],
        "OOS_Returns_Array": spy_returns_arr,
        "OOS_Cum_Returns_Array": spy_metrics["cum_returns_arr"],
    }

    # 6. Format and Print Backtest Table (sorted by Cum Return, SPY pinned at bottom)
    sorted_all = sorted(all_results, key=lambda x: x.get("Cum Return", 0.0), reverse=True)
    print_backtest_table(sorted_all + [spy_res], title="Walk-Forward Performance Comparison (Full Period)")

    # 7. Print Year 2025 Specific Table
    mask_2025 = trade_dates.year == 2025
    if mask_2025.any():
        results_2025 = []
        for r in all_results:
            port_ret_25 = r["OOS_Returns_Array"][mask_2025]
            m25 = compute_metrics(port_ret_25, risk_free_rate)
            results_2025.append(
                {
                    "Strategy": r["Strategy"],
                    "Cum Return": m25["cum_return"],
                    "Ann Return": m25["ann_return"],
                    "Ann Volatility": m25["ann_vol"],
                    "Sharpe Ratio": m25["sharpe"],
                    "Max Drawdown": m25["max_dd"],
                }
            )
        spy_ret_25 = spy_returns_arr[mask_2025]
        sm25 = compute_metrics(spy_ret_25, risk_free_rate)
        spy_res_25 = {
            "Strategy": "SPY (S&P 500 Benchmark)",
            "Cum Return": sm25["cum_return"],
            "Ann Return": sm25["ann_return"],
            "Ann Volatility": sm25["ann_vol"],
            "Sharpe Ratio": sm25["sharpe"],
            "Max Drawdown": sm25["max_dd"],
        }
        sorted_2025 = sorted(results_2025, key=lambda x: x.get("Cum Return", 0.0), reverse=True)
        print_backtest_table(sorted_2025 + [spy_res_25], title="Walk-Forward Performance Comparison (Year 2025 Only)")

    # 8. Save PNG Plots
    save_convergence_plot(all_results, output_path=os.path.join(out_dir, "convergence.png"))
    save_performance_plot(
        all_results, spy_metrics["cum_returns_arr"], trade_dates, output_path=os.path.join(out_dir, "performance.png")
    )
    if mask_2025.any():
        save_performance_plot_2025(
            all_results, spy_returns_arr, trade_dates, output_path=os.path.join(out_dir, "performance_2025.png")
        )

    # 9. Save Markdown Performance Report
    save_markdown_report(
        all_results,
        spy_res,
        results_2025 if mask_2025.any() else None,
        spy_res_25 if mask_2025.any() else None,
        output_path=os.path.join(out_dir, "performance_report.md"),
    )

    logger.info("Pipeline executed successfully nya~! (=^･ω･^=)")


def save_markdown_report(all_results, spy_res, results_2025=None, spy_res_25=None, output_path="performance_report.md"):
    """Saves the backtest performance metrics tables to a Markdown file, sorted by Cum Return (excluding SPY)."""
    # Sort all_results by 'Cum Return' in descending order
    sorted_all = sorted(all_results, key=lambda x: x.get("Cum Return", 0.0), reverse=True)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("# Performance Report: Walk-Forward Backtesting Metrics\n\n")
        f.write("This report summarizes the walk-forward out-of-sample (OOS) simulation performance metrics.\n\n")

        f.write("## 1. Walk-Forward Performance Comparison (Full Period)\n\n")
        f.write("| Strategy Name | Cum Return | Ann Return | Ann Vol | Sharpe Ratio | Max Drawdown |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: | :---: |\n")

        for r in sorted_all:
            vol_val = r.get("Ann Volatility", r.get("Ann Vol", 0.0))
            sharpe_val = r.get("Sharpe Ratio", r.get("Sharpe", 0.0))
            max_dd_val = r.get("Max Drawdown", r.get("Max DD", 0.0))
            f.write(
                f"| **{r['Strategy']}** | {r['Cum Return']:.2%} | {r['Ann Return']:.2%} | {vol_val:.2%} | {sharpe_val:.4f} | {max_dd_val:.2%} |\n"
            )

        spy_vol_val = spy_res.get("Ann Volatility", spy_res.get("Ann Vol", 0.0))
        spy_sharpe_val = spy_res.get("Sharpe Ratio", spy_res.get("Sharpe", 0.0))
        spy_max_dd_val = spy_res.get("Max Drawdown", spy_res.get("Max DD", 0.0))
        f.write(
            f"| *{spy_res['Strategy']}* | {spy_res['Cum Return']:.2%} | {spy_res['Ann Return']:.2%} | {spy_vol_val:.2%} | {spy_sharpe_val:.4f} | {spy_max_dd_val:.2%} |\n\n"
        )

        if results_2025 and spy_res_25:
            # Sort results_2025 by 'Cum Return' in descending order
            sorted_2025 = sorted(results_2025, key=lambda x: x.get("Cum Return", 0.0), reverse=True)

            f.write("## 2. Walk-Forward Performance Comparison (Year 2025 Only)\n\n")
            f.write("| Strategy Name | Cum Return | Ann Return | Ann Vol | Sharpe Ratio | Max Drawdown |\n")
            f.write("| :--- | :---: | :---: | :---: | :---: | :---: |\n")
            for r in sorted_2025:
                vol_val = r.get("Ann Volatility", r.get("Ann Vol", 0.0))
                sharpe_val = r.get("Sharpe Ratio", r.get("Sharpe", 0.0))
                max_dd_val = r.get("Max Drawdown", r.get("Max DD", 0.0))
                f.write(
                    f"| **{r['Strategy']}** | {r['Cum Return']:.2%} | {r['Ann Return']:.2%} | {vol_val:.2%} | {sharpe_val:.4f} | {max_dd_val:.2%} |\n"
                )

            spy_vol_25 = spy_res_25.get("Ann Volatility", spy_res_25.get("Ann Vol", 0.0))
            spy_sharpe_25 = spy_res_25.get("Sharpe Ratio", spy_res_25.get("Sharpe", 0.0))
            spy_max_dd_25 = spy_res_25.get("Max Drawdown", spy_res_25.get("Max DD", 0.0))
            f.write(
                f"| *{spy_res_25['Strategy']}* | {spy_res_25['Cum Return']:.2%} | {spy_res_25['Ann Return']:.2%} | {spy_vol_25:.2%} | {spy_sharpe_25:.4f} | {spy_max_dd_25:.2%} |\n"
            )
    logger.info(f"Saved performance report to {output_path}")


if __name__ == "__main__":
    main()
