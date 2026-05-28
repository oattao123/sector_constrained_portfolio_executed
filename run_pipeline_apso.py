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

logger = logging.getLogger("run_pipeline_apso")


def parse_args():
    parser = argparse.ArgumentParser(description="Multi-Asset Portfolio Optimization Pipeline (APSO)")
    parser.add_argument("--settings", type=str, default="config/settings.json", help="Path to settings.json")
    parser.add_argument("--assets", type=str, default="config/assets.csv", help="Path to assets.csv")
    parser.add_argument("--cache", type=str, default="data/all_data.csv", help="Path to cache price CSV file")
    parser.add_argument("--particles", type=int, default=500, help="Number of particles for APSO")
    parser.add_argument("--iterations", type=int, default=1000, help="Number of iterations for optimizers")
    parser.add_argument("--trials", type=int, default=5, help="Number of trials per strategy to run and average")
    return parser.parse_args()


def run_backtest_with_trials(backtester, portfolio_name, selected_stocks, trials, **run_kwargs):
    """Runs a backtest multiple times, logs each trial, and returns averaged results."""
    logger.info(f"[{portfolio_name}] Running {trials} trials to compute mean and error...")
    
    trial_results = []
    for trial in range(trials):
        logger.info(f"  --> Trial {trial + 1}/{trials}...")
        res = backtester.run(
            portfolio_name=f"{portfolio_name}_trial_{trial}",
            selected_stocks=selected_stocks,
            **run_kwargs
        )
        trial_results.append(res)
        
    ddof = 1 if trials > 1 else 0
    
    # Aggregate results
    cums = [r["Cum Return"] for r in trial_results]
    anns = [r["Ann Return"] for r in trial_results]
    vols = [r["Ann Volatility"] for r in trial_results]
    sharpes = [r["Sharpe Ratio"] for r in trial_results]
    dds = [r["Max Drawdown"] for r in trial_results]
    
    # No cost fields
    cums_nc = [r["Cum Return (No Cost)"] for r in trial_results]
    anns_nc = [r["Ann Return (No Cost)"] for r in trial_results]
    vols_nc = [r["Ann Volatility (No Cost)"] for r in trial_results]
    sharpes_nc = [r["Sharpe Ratio (No Cost)"] for r in trial_results]
    dds_nc = [r["Max Drawdown (No Cost)"] for r in trial_results]
    
    # Arrays - we average them element-wise
    avg_oos_returns = np.mean([r["OOS_Returns_Array"] for r in trial_results], axis=0)
    avg_oos_cum_returns = np.mean([r["OOS_Cum_Returns_Array"] for r in trial_results], axis=0)
    
    avg_oos_returns_nc = np.mean([r["OOS_Returns_Array_No_Cost"] for r in trial_results], axis=0)
    avg_oos_cum_returns_nc = np.mean([r["OOS_Cum_Returns_Array_No_Cost"] for r in trial_results], axis=0)
    
    # Convergence curves
    avg_best_conv = np.mean([r["Avg_Best_Convergence"] for r in trial_results], axis=0)
    avg_avg_conv = np.mean([r["Avg_Avg_Convergence"] for r in trial_results], axis=0)
    
    # Average the weights across trials
    avg_rebalance_history = []
    if trial_results and "Rebalance_History" in trial_results[0]:
        num_rebalance_dates = len(trial_results[0]["Rebalance_History"])
        for idx in range(num_rebalance_dates):
            date_str = trial_results[0]["Rebalance_History"][idx]["date"]
            all_trial_weights = [r["Rebalance_History"][idx]["weights"] for r in trial_results]
            mean_weights = np.mean(all_trial_weights, axis=0)
            avg_rebalance_history.append({
                "date": date_str,
                "weights": mean_weights
            })

    return {
        "Strategy": portfolio_name,
        "Cum Return": np.mean(cums),
        "Cum Return Std": np.std(cums, ddof=ddof),
        "Ann Return": np.mean(anns),
        "Ann Return Std": np.std(anns, ddof=ddof),
        "Ann Volatility": np.mean(vols),
        "Ann Volatility Std": np.std(vols, ddof=ddof),
        "Sharpe Ratio": np.mean(sharpes),
        "Sharpe Ratio Std": np.std(sharpes, ddof=ddof),
        "Max Drawdown": np.mean(dds),
        "Max Drawdown Std": np.std(dds, ddof=ddof),
        
        "OOS_Returns_Array": avg_oos_returns,
        "OOS_Cum_Returns_Array": avg_oos_cum_returns,
        
        # No cost fields
        "Cum Return (No Cost)": np.mean(cums_nc),
        "Cum Return (No Cost) Std": np.std(cums_nc, ddof=ddof),
        "Ann Return (No Cost)": np.mean(anns_nc),
        "Ann Return (No Cost) Std": np.std(anns_nc, ddof=ddof),
        "Ann Volatility (No Cost)": np.mean(vols_nc),
        "Ann Volatility (No Cost) Std": np.std(vols_nc, ddof=ddof),
        "Sharpe Ratio (No Cost)": np.mean(sharpes_nc),
        "Sharpe Ratio (No Cost) Std": np.std(sharpes_nc, ddof=ddof),
        "Max Drawdown (No Cost)": np.mean(dds_nc),
        "Max Drawdown (No Cost) Std": np.std(dds_nc, ddof=ddof),
        
        "OOS_Returns_Array_No_Cost": avg_oos_returns_nc,
        "OOS_Cum_Returns_Array_No_Cost": avg_oos_cum_returns_nc,
        
        "Avg_Best_Convergence": avg_best_conv,
        "Avg_Avg_Convergence": avg_avg_conv,
        "Dates": trial_results[0]["Dates"],
        "Rebalance_History": avg_rebalance_history,
        "Tickers": trial_results[0]["Tickers"] if trial_results else [],
        "trial_results": trial_results
    }


def main():
    args = parse_args()

    # 0. Create timestamped output directory: output/apso_{DD_M_HH_MM}
    now = datetime.now()
    run_tag = now.strftime("apso_%d_%m_%H_%M")
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
        num_ants=100, # ACO is only used for asset selection here, so hardcode a fast baseline
        num_iterations=100,
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
        num_ants=100,
        num_epochs=100,
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

    # 4a. Save selected stocks to CSV in output directory
    selections_dir = os.path.join(out_dir, "selections")
    os.makedirs(selections_dir, exist_ok=True)
    for name, df in portfolio_dfs.items():
        filename = name.lower().replace(" ", "_") + ".csv"
        df.to_csv(os.path.join(selections_dir, filename), index=False)
    logger.info(f"Saved selected asset lists to {selections_dir}/")

    # 4b. Save candlestick charts for each selection method
    logger.info("Generating candlestick charts for each selection strategy...")
    candle_start = data_cfg.get("insample_end_date") or data_cfg.get("start_date", "2015-01-01")
    candle_end   = data_cfg.get("outsample_end_date") or data_cfg.get("end_date", "2026-03-20")
    
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
    portfolio_cfg = manager.settings.get("portfolio", {})
    transaction_cost_rate = portfolio_cfg.get("transaction_cost_rate", 0.0)
    backtester = WalkForwardBacktester(
        full_returns=returns,
        spy_full_returns=spy_full_returns,
        sector_map=manager.sector_map,
        risk_free_rate=risk_free_rate,
        transaction_cost_rate=transaction_cost_rate,
    )

    all_results = []

    # A. Run walk-forward with APSO weight optimization for each static selection strategy
    for name, df in portfolio_dfs.items():
        if df.empty:
            logger.warning(f"Portfolio {name} is empty. Skipping backtest.")
            continue
        tickers_list = df["Ticker"].tolist()
        res = run_backtest_with_trials(
            backtester=backtester,
            portfolio_name=name,
            selected_stocks=tickers_list,
            trials=args.trials,
            num_iterations=args.iterations,
            num_agents=args.particles,
            use_sector_constraints=False,
            optimizer='apso',
        )
        all_results.append(res)

    # B. Run APSO walk-forward optimization (Full Universe)
    logger.info("Running APSO walk-forward optimization...")
    apso_res = run_backtest_with_trials(
        backtester=backtester,
        portfolio_name="APSO Portfolio",
        selected_stocks=valid_tickers.tolist(),
        trials=args.trials,
        num_iterations=args.iterations,
        num_agents=args.particles,
        use_sector_constraints=False,
        optimizer='apso',
    )
    all_results.append(apso_res)

    # C. Calculate SPY Benchmark Metrics
    lookback_window = 252 * 3
    test_len = len(apso_res["OOS_Returns_Array"])
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
        ddof = 1 if args.trials > 1 else 0
        for r in all_results:
            trials_2025 = []
            if "trial_results" in r:
                for t_res in r["trial_results"]:
                    t_ret_25 = t_res["OOS_Returns_Array"][mask_2025]
                    t_m25 = compute_metrics(t_ret_25, risk_free_rate)
                    
                    t_ret_25_nc = t_res["OOS_Returns_Array_No_Cost"][mask_2025]
                    t_m25_nc = compute_metrics(t_ret_25_nc, risk_free_rate)
                    
                    trials_2025.append((t_m25, t_m25_nc))
            else:
                # single trial (e.g. SPY benchmark)
                t_ret_25 = r["OOS_Returns_Array"][mask_2025]
                t_m25 = compute_metrics(t_ret_25, risk_free_rate)
                trials_2025.append((t_m25, t_m25))

            cums = [t[0]["cum_return"] for t in trials_2025]
            anns = [t[0]["ann_return"] for t in trials_2025]
            vols = [t[0]["ann_vol"] for t in trials_2025]
            sharpes = [t[0]["sharpe"] for t in trials_2025]
            dds = [t[0]["max_dd"] for t in trials_2025]
            
            cums_nc = [t[1]["cum_return"] for t in trials_2025]
            anns_nc = [t[1]["ann_return"] for t in trials_2025]
            vols_nc = [t[1]["ann_vol"] for t in trials_2025]
            sharpes_nc = [t[1]["sharpe"] for t in trials_2025]
            dds_nc = [t[1]["max_dd"] for t in trials_2025]

            results_2025.append(
                {
                    "Strategy": r["Strategy"],
                    "Cum Return": np.mean(cums),
                    "Cum Return Std": np.std(cums, ddof=ddof) if "trial_results" in r else 0.0,
                    "Ann Return": np.mean(anns),
                    "Ann Return Std": np.std(anns, ddof=ddof) if "trial_results" in r else 0.0,
                    "Ann Volatility": np.mean(vols),
                    "Ann Volatility Std": np.std(vols, ddof=ddof) if "trial_results" in r else 0.0,
                    "Sharpe Ratio": np.mean(sharpes),
                    "Sharpe Ratio Std": np.std(sharpes, ddof=ddof) if "trial_results" in r else 0.0,
                    "Max Drawdown": np.mean(dds),
                    "Max Drawdown Std": np.std(dds, ddof=ddof) if "trial_results" in r else 0.0,
                    
                    "Cum Return (No Cost)": np.mean(cums_nc),
                    "Cum Return (No Cost) Std": np.std(cums_nc, ddof=ddof) if "trial_results" in r else 0.0,
                    "Ann Return (No Cost)": np.mean(anns_nc),
                    "Ann Return (No Cost) Std": np.std(anns_nc, ddof=ddof) if "trial_results" in r else 0.0,
                    "Ann Volatility (No Cost)": np.mean(vols_nc),
                    "Ann Volatility (No Cost) Std": np.std(vols_nc, ddof=ddof) if "trial_results" in r else 0.0,
                    "Sharpe Ratio (No Cost)": np.mean(sharpes_nc),
                    "Sharpe Ratio (No Cost) Std": np.std(sharpes_nc, ddof=ddof) if "trial_results" in r else 0.0,
                    "Max Drawdown (No Cost)": np.mean(dds_nc),
                    "Max Drawdown (No Cost) Std": np.std(dds_nc, ddof=ddof) if "trial_results" in r else 0.0,
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

    # 9b. Save average optimized weights for all strategies in the main run directory
    weights_dir = os.path.join(out_dir, "weights")
    for r in all_results:
        save_weights_to_csv(r, weights_dir)

    # 9b. Save average optimized weights for all strategies in the main run directory
    weights_dir = os.path.join(out_dir, "weights")
    for r in all_results:
        save_weights_to_csv(r, weights_dir)

    logger.info("Pipeline executed successfully nya~! (=^･ω･^=)")


def save_markdown_report(all_results, spy_res, results_2025=None, spy_res_25=None, output_path="performance_report.md"):
    """Saves the backtest performance metrics tables to a Markdown file, comparing with and without transaction costs."""
    # Sort all_results by 'Cum Return' (with cost) in descending order
    sorted_all = sorted(all_results, key=lambda x: x.get("Cum Return", 0.0), reverse=True)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("# Performance Report: Walk-Forward Backtesting Metrics (APSO)\n\n")
        f.write("This report summarizes the walk-forward out-of-sample (OOS) simulation performance metrics using APSO weight optimization, comparing results with and without transaction costs. Metrics display Mean ± Standard Deviation across trials.\n\n")

        f.write("## 1. Walk-Forward Performance Comparison (Full Period)\n\n")
        f.write("| Strategy Name | Scenario | Cum Return | Ann Return | Ann Vol | Sharpe Ratio | Max Drawdown |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: | :---: | :---: |\n")

        for r in sorted_all:
            cum_std = r.get("Cum Return Std", 0.0)
            ann_std = r.get("Ann Return Std", 0.0)
            vol_std = r.get("Ann Volatility Std", 0.0)
            sharpe_std = r.get("Sharpe Ratio Std", 0.0)
            max_dd_std = r.get("Max Drawdown Std", 0.0)
            
            cum_nc_std = r.get("Cum Return (No Cost) Std", 0.0)
            ann_nc_std = r.get("Ann Return (No Cost) Std", 0.0)
            vol_nc_std = r.get("Ann Volatility (No Cost) Std", 0.0)
            sharpe_nc_std = r.get("Sharpe Ratio (No Cost) Std", 0.0)
            max_dd_nc_std = r.get("Max Drawdown (No Cost) Std", 0.0)
            
            vol_val = r.get("Ann Volatility", r.get("Ann Vol", 0.0))
            sharpe_val = r.get("Sharpe Ratio", r.get("Sharpe", 0.0))
            max_dd_val = r.get("Max Drawdown", r.get("Max DD", 0.0))
            
            vol_nc = r.get("Ann Volatility (No Cost)", r.get("Ann Vol (No Cost)", 0.0))
            sharpe_nc = r.get("Sharpe Ratio (No Cost)", r.get("Sharpe (No Cost)", 0.0))
            max_dd_nc = r.get("Max Drawdown (No Cost)", r.get("Max DD (No Cost)", 0.0))
            
            cum_str = f"{r['Cum Return']:.2%}" + (f" ± {cum_std:.2%}" if cum_std > 0 else "")
            ann_str = f"{r['Ann Return']:.2%}" + (f" ± {ann_std:.2%}" if ann_std > 0 else "")
            vol_str = f"{vol_val:.2%}" + (f" ± {vol_std:.2%}" if vol_std > 0 else "")
            sharpe_str = f"{sharpe_val:.4f}" + (f" ± {sharpe_std:.4f}" if sharpe_std > 0 else "")
            max_dd_str = f"{max_dd_val:.2%}" + (f" ± {max_dd_std:.2%}" if max_dd_std > 0 else "")
            
            cum_nc_str = f"{r['Cum Return (No Cost)']:.2%}" + (f" ± {cum_nc_std:.2%}" if cum_nc_std > 0 else "")
            ann_nc_str = f"{r['Ann Return (No Cost)']:.2%}" + (f" ± {ann_nc_std:.2%}" if ann_nc_std > 0 else "")
            vol_nc_str = f"{vol_nc:.2%}" + (f" ± {vol_nc_std:.2%}" if vol_nc_std > 0 else "")
            sharpe_nc_str = f"{sharpe_nc:.4f}" + (f" ± {sharpe_nc_std:.4f}" if sharpe_nc_std > 0 else "")
            max_dd_nc_str = f"{max_dd_nc:.2%}" + (f" ± {max_dd_nc_std:.2%}" if max_dd_nc_std > 0 else "")
            
            f.write(
                f"| **{r['Strategy']}** | **With Cost** | **{cum_str}** | **{ann_str}** | {vol_str} | **{sharpe_str}** | {max_dd_str} |\n"
            )
            f.write(
                f"| | *No Cost* | {cum_nc_str} | {ann_nc_str} | {vol_nc_str} | {sharpe_nc_str} | {max_dd_nc_str} |\n"
            )

        spy_vol_val = spy_res.get("Ann Volatility", spy_res.get("Ann Vol", 0.0))
        spy_sharpe_val = spy_res.get("Sharpe Ratio", spy_res.get("Sharpe", 0.0))
        spy_max_dd_val = spy_res.get("Max Drawdown", spy_res.get("Max DD", 0.0))
        f.write(
            f"| *{spy_res['Strategy']}* | *N/A (No Cost)* | {spy_res['Cum Return']:.2%} | {spy_res['Ann Return']:.2%} | {spy_vol_val:.2%} | {spy_sharpe_val:.4f} | {spy_max_dd_val:.2%} |\n\n"
        )

        if results_2025 and spy_res_25:
            # Sort results_2025 by 'Cum Return' in descending order
            sorted_2025 = sorted(results_2025, key=lambda x: x.get("Cum Return", 0.0), reverse=True)

            f.write("## 2. Walk-Forward Performance Comparison (Year 2025 Only)\n\n")
            f.write("| Strategy Name | Scenario | Cum Return | Ann Return | Ann Vol | Sharpe Ratio | Max Drawdown |\n")
            f.write("| :--- | :---: | :---: | :---: | :---: | :---: | :---: |\n")
            
            for r in sorted_2025:
                cum_std = r.get("Cum Return Std", 0.0)
                ann_std = r.get("Ann Return Std", 0.0)
                vol_std = r.get("Ann Volatility Std", 0.0)
                sharpe_std = r.get("Sharpe Ratio Std", 0.0)
                max_dd_std = r.get("Max Drawdown Std", 0.0)
                
                cum_nc_std = r.get("Cum Return (No Cost) Std", 0.0)
                ann_nc_std = r.get("Ann Return (No Cost) Std", 0.0)
                vol_nc_std = r.get("Ann Volatility (No Cost) Std", 0.0)
                sharpe_nc_std = r.get("Sharpe Ratio (No Cost) Std", 0.0)
                max_dd_nc_std = r.get("Max Drawdown (No Cost) Std", 0.0)
                
                vol_val = r.get("Ann Volatility", r.get("Ann Vol", 0.0))
                sharpe_val = r.get("Sharpe Ratio", r.get("Sharpe", 0.0))
                max_dd_val = r.get("Max Drawdown", r.get("Max DD", 0.0))
                
                vol_nc = r.get("Ann Volatility (No Cost)", r.get("Ann Vol (No Cost)", 0.0))
                sharpe_nc = r.get("Sharpe Ratio (No Cost)", r.get("Sharpe (No Cost)", 0.0))
                max_dd_nc = r.get("Max Drawdown (No Cost)", r.get("Max DD (No Cost)", 0.0))
                
                cum_str = f"{r['Cum Return']:.2%}" + (f" ± {cum_std:.2%}" if cum_std > 0 else "")
                ann_str = f"{r['Ann Return']:.2%}" + (f" ± {ann_std:.2%}" if ann_std > 0 else "")
                vol_str = f"{vol_val:.2%}" + (f" ± {vol_std:.2%}" if vol_std > 0 else "")
                sharpe_str = f"{sharpe_val:.4f}" + (f" ± {sharpe_std:.4f}" if sharpe_std > 0 else "")
                max_dd_str = f"{max_dd_val:.2%}" + (f" ± {max_dd_std:.2%}" if max_dd_std > 0 else "")
                
                cum_nc_str = f"{r['Cum Return (No Cost)']:.2%}" + (f" ± {cum_nc_std:.2%}" if cum_nc_std > 0 else "")
                ann_nc_str = f"{r['Ann Return (No Cost)']:.2%}" + (f" ± {ann_nc_std:.2%}" if ann_nc_std > 0 else "")
                vol_nc_str = f"{vol_nc:.2%}" + (f" ± {vol_nc_std:.2%}" if vol_nc_std > 0 else "")
                sharpe_nc_str = f"{sharpe_nc:.4f}" + (f" ± {sharpe_nc_std:.4f}" if sharpe_nc_std > 0 else "")
                max_dd_nc_str = f"{max_dd_nc:.2%}" + (f" ± {max_dd_nc_std:.2%}" if max_dd_nc_std > 0 else "")
                
                f.write(
                    f"| **{r['Strategy']}** | **With Cost** | **{cum_str}** | **{ann_str}** | {vol_str} | **{sharpe_str}** | {max_dd_str} |\n"
                )
                f.write(
                    f"| | *No Cost* | {cum_nc_str} | {ann_nc_str} | {vol_nc_str} | {sharpe_nc_str} | {max_dd_nc_str} |\n"
                )

            spy_vol_25 = spy_res_25.get("Ann Volatility", spy_res_25.get("Ann Vol", 0.0))
            spy_sharpe_25 = spy_res_25.get("Sharpe Ratio", spy_res_25.get("Sharpe", 0.0))
            spy_max_dd_25 = spy_res_25.get("Max Drawdown", spy_res_25.get("Max DD", 0.0))
            f.write(
                f"| *{spy_res_25['Strategy']}* | *N/A (No Cost)* | {spy_res_25['Cum Return']:.2%} | {spy_res_25['Ann Return']:.2%} | {spy_vol_25:.2%} | {spy_sharpe_25:.4f} | {spy_max_dd_25:.2%} |\n"
            )
    logger.info(f"Saved performance report to {output_path}")


def save_weights_to_csv(result, output_dir):
    """Saves average optimized weights of a strategy to a CSV file."""
    if "Rebalance_History" not in result or not result["Rebalance_History"]:
        return
    
    os.makedirs(output_dir, exist_ok=True)
    strategy_name_clean = result["Strategy"].lower().replace(" ", "_").replace("(", "").replace(")", "")
    filename = f"{strategy_name_clean}_weights.csv"
    filepath = os.path.join(output_dir, filename)
    
    tickers = result["Tickers"]
    history = result["Rebalance_History"]
    
    # Construct a DataFrame where index is Date, and columns are Tickers
    dates = [h["date"] for h in history]
    weights_data = [h["weights"] for h in history]
    
    df = pd.DataFrame(weights_data, index=dates, columns=tickers)
    df.index.name = "Date"
    df.to_csv(filepath)
    logger.info(f"Saved optimized weights to {filepath}")


if __name__ == "__main__":
    main()
