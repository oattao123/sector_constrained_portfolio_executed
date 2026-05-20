import os
import argparse
import logging
import warnings
from datetime import datetime
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text

from portfolio_optimization import (
    DataManager,
    WalkForwardBacktester,
    ledoit_wolf_covariance_gpu_dynamic,
    compute_correlation_matrix,
    select_by_diversification,
    select_by_lw_diversification,
    select_by_hrp_sharpe,
    select_by_hrp_div,
    select_by_aco,
    select_by_aco_cluster,
    print_asset_summary
)

warnings.filterwarnings("ignore")
console = Console(width=120)
logger = logging.getLogger("tune_pipeline")

def parse_args():
    parser = argparse.ArgumentParser(description="Parameter Tuning & Sensitivity Analysis Pipeline")
    parser.add_argument("--settings", type=str, default="config/settings.json", help="Path to settings.json")
    parser.add_argument("--assets", type=str, default="config/assets.csv", help="Path to assets.csv")
    parser.add_argument("--cache", type=str, default="data/all_data.csv", help="Path to cache price CSV file")
    
    # Fast Defaults for tuning speed
    parser.add_argument("--wolves", type=int, default=100, help="EBGWO population size (tuning base)")
    parser.add_argument("--iterations", type=int, default=150, help="EBGWO iterations (tuning base)")
    parser.add_argument("--agents", type=int, default=100, help="ACO agents (tuning base)")
    parser.add_argument("--trials", type=int, default=10, help="Number of trials to run per configuration to average stochastic results")
    
    # Tuning Targets
    parser.add_argument("--param", type=str, default="ST", help="First hyperparameter to tune (e.g. ST, Q, evaporation_rate, lambda_ent)")
    parser.add_argument("--values", type=str, default="0.1,0.2,0.3,0.4,0.5", help="Comma-separated values for the first parameter")
    parser.add_argument("--param2", type=str, default=None, help="Optional second hyperparameter for 2D Grid Tuning")
    parser.add_argument("--values2", type=str, default=None, help="Comma-separated values for the second parameter")
    
    parser.add_argument("--strategy", type=str, default="ACO Cluster Selected", help="Asset Selection Strategy to use as base")
    
    return parser.parse_args()

def cast_value(param_name, val):
    int_params = {"patience", "num_wolves", "iterations", "agents"}
    try:
        if param_name in int_params:
            return int(float(val))
        return float(val)
    except ValueError:
        return val

def run_tuning():
    print_welcome_panel()
    args = parse_args()
    
    # 0. Output directory creation
    now = datetime.now()
    run_tag = now.strftime("tuning_%d_%m_%H_%M")
    out_dir = os.path.join("output", run_tag)
    os.makedirs(out_dir, exist_ok=True)
    
    # Setup Logging
    log_file = os.path.join(out_dir, "tuning.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler()
        ]
    )
    
    logger.info(f"Tuning folder created: {out_dir}")
    logger.info(f"Tuning target: {args.param} = [{args.values}]")
    if args.param2:
        logger.info(f"2D Grid target: {args.param2} = [{args.values2}]")
        
    # 1. Load Data
    manager = DataManager(settings_path=args.settings, assets_path=args.assets)
    print_asset_summary(manager.assets_df)
    
    data_cfg = manager.settings.get("data", {})
    insample_end_date = data_cfg.get("insample_end_date", None)
    
    raw_data = manager.download_data(cache_path=args.cache)
    clean_data = manager.clean_data(raw_data)
    converted_data = manager.convert_to_base_currency(clean_data)
    returns = manager.get_log_returns(converted_data)
    
    spy_full_returns = returns['SPY'] if 'SPY' in returns.columns else pd.Series(0.0, index=returns.index)
    opt_universe_returns = returns.drop(columns=['SPY']) if 'SPY' in returns.columns else returns
    
    insample_returns = (
        opt_universe_returns.loc[:insample_end_date]
        if insample_end_date else opt_universe_returns
    )
    
    # 2. Get Covariance & Select assets using chosen Strategy
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    returns_gpu = torch.tensor(insample_returns.values, dtype=torch.float32, device=device)
    shrunk_cov_gpu, _, _ = ledoit_wolf_covariance_gpu_dynamic(returns_gpu)
    corr_matrix_gpu = compute_correlation_matrix(shrunk_cov_gpu)
    
    risk_free_rate = manager.settings.get("risk_free_rate", {}).get("fallback", 0.0434)
    
    logger.info(f"Running selection strategy: {args.strategy}")
    if args.strategy == "Diversification Selected":
        selected_df = select_by_diversification(insample_returns, manager.sector_map, risk_free_rate)
    elif args.strategy == "LW Diversification Selected":
        selected_df = select_by_lw_diversification(insample_returns, shrunk_cov_gpu, corr_matrix_gpu, manager.sector_map, risk_free_rate)
    elif args.strategy == "LW Sharpe Cluster Selected":
        selected_df = select_by_hrp_sharpe(insample_returns, shrunk_cov_gpu, corr_matrix_gpu, manager.sector_map, risk_free_rate)
    elif args.strategy == "LW Div Cluster Selected":
        selected_df = select_by_hrp_div(insample_returns, shrunk_cov_gpu, corr_matrix_gpu, manager.sector_map, risk_free_rate)
    elif args.strategy == "ACO Selected":
        selected_df = select_by_aco(insample_returns, shrunk_cov_gpu, corr_matrix_gpu, manager.sector_map, risk_free_rate, num_ants=args.agents, num_iterations=args.iterations)
    else:
        # Default: ACO Cluster Selected
        selected_df = select_by_aco_cluster(insample_returns, shrunk_cov_gpu, corr_matrix_gpu, manager.sector_map, risk_free_rate, num_ants=args.agents, num_epochs=args.iterations)
        
    selected_stocks = selected_df['Ticker'].tolist() if not selected_df.empty else []
    if not selected_stocks:
        logger.error("No stocks selected! Cannot proceed with tuning.")
        return
        
    # 3. Parse grid values
    vals1 = [cast_value(args.param, v) for v in args.values.split(",")]
    vals2 = [cast_value(args.param2, v) for v in args.values2.split(",")] if args.param2 and args.values2 else [None]
    
    results = []
    
    backtester = WalkForwardBacktester(
        full_returns=returns,
        spy_full_returns=spy_full_returns,
        sector_map=manager.sector_map,
        risk_free_rate=risk_free_rate
    )
    
    lookback = data_cfg.get("lookback_window", 252*3)
    step = data_cfg.get("step_size", 21*3)
    
    # 4. Run Grid Loop
    total_runs = len(vals1) * len(vals2)
    current_run = 0
    
    for v1 in vals1:
        for v2 in vals2:
            current_run += 1
            opt_kwargs = {args.param: v1}
            run_name = f"{args.param}={v1}"
            if v2 is not None:
                opt_kwargs[args.param2] = v2
                run_name += f", {args.param2}={v2}"
                
            logger.info(f"[{current_run}/{total_runs}] Running test configuration: {run_name} (Averaging over {args.trials} trials)...")
            
            trial_metrics = []
            for trial in range(args.trials):
                try:
                    res = backtester.run(
                        portfolio_name=f"{run_name}_trial_{trial}",
                        selected_stocks=selected_stocks,
                        lookback_window=lookback,
                        step_size=step,
                        num_iterations=args.iterations,
                        num_agents=args.wolves,
                        use_sector_constraints=True,
                        **opt_kwargs
                    )
                    trial_metrics.append({
                        "cum_ret": res["Cum Return"],
                        "ann_ret": res["Ann Return"],
                        "ann_vol": res["Ann Volatility"],
                        "sharpe": res["Sharpe Ratio"],
                        "max_dd": res["Max Drawdown"]
                    })
                except Exception as e:
                    logger.error(f"Trial {trial+1}/{args.trials} failed for {run_name}: {e}")
            
            if trial_metrics:
                mean_cum_ret = np.mean([m["cum_ret"] for m in trial_metrics])
                std_cum_ret = np.std([m["cum_ret"] for m in trial_metrics])
                
                mean_ann_ret = np.mean([m["ann_ret"] for m in trial_metrics])
                std_ann_ret = np.std([m["ann_ret"] for m in trial_metrics])
                
                mean_ann_vol = np.mean([m["ann_vol"] for m in trial_metrics])
                std_ann_vol = np.std([m["ann_vol"] for m in trial_metrics])
                
                mean_sharpe = np.mean([m["sharpe"] for m in trial_metrics])
                std_sharpe = np.std([m["sharpe"] for m in trial_metrics])
                
                mean_max_dd = np.mean([m["max_dd"] for m in trial_metrics])
                std_max_dd = np.std([m["max_dd"] for m in trial_metrics])
                
                results.append({
                    "val1": v1,
                    "val2": v2,
                    "name": run_name,
                    "cum_ret": mean_cum_ret,
                    "cum_ret_std": std_cum_ret,
                    "ann_ret": mean_ann_ret,
                    "ann_ret_std": std_ann_ret,
                    "ann_vol": mean_ann_vol,
                    "ann_vol_std": std_ann_vol,
                    "sharpe": mean_sharpe,
                    "sharpe_std": std_sharpe,
                    "max_dd": mean_max_dd,
                    "max_dd_std": std_max_dd
                })
            else:
                logger.error(f"All trials failed for configuration {run_name}")
                
    # 5. Output Table & Save report
    results_df = pd.DataFrame(results)
    if results_df.empty:
        logger.error("No results obtained from grid search.")
        return
        
    results_sorted = results_df.sort_values(by="sharpe", ascending=False)
    
    print_tuning_results_table(results_sorted, args.param, args.param2)
    save_tuning_report(results_sorted, args.param, args.param2, out_dir)
    
    # 6. Plot sensitivity charts
    plot_path = os.path.join(out_dir, "sensitivity_analysis.png")
    generate_sensitivity_plots(results_df, args.param, args.param2, plot_path)
    
    logger.info(f"Tuning finished successfully nya~! (=^.^=)")
    logger.info(f"Saved results and plots to: {out_dir}")

def print_welcome_panel():
    welcome_text = Text()
    welcome_text.append("**  Cristina's Hyperparameter Tuning Pipeline  **\n", style="bold magenta")
    welcome_text.append("Grid Search and OOS Sensitivity Evaluation desu~!\n", style="italic cyan")
    panel = Panel(
        welcome_text,
        title="[bold green]HYPERPARAMETER SENSITIVITY PIPELINE[/bold green]",
        border_style="magenta",
        expand=False
    )
    console.print(panel)

def print_tuning_results_table(df, p1, p2):
    title = f"[bold green]Grid Search Performance Report ({p1}"
    if p2:
        title += f" vs {p2}"
    title += ")[/bold green]"
    
    table = Table(title=title, border_style="green")
    table.add_column(p1, justify="right", style="cyan")
    if p2:
        table.add_column(p2, justify="right", style="magenta")
    table.add_column("Cum Return", justify="right", style="green")
    table.add_column("Ann Return", justify="right", style="cyan")
    table.add_column("Ann Vol", justify="right", style="red")
    table.add_column("Sharpe Ratio", justify="right", style="bold yellow")
    table.add_column("Max DD", justify="right", style="bold red")
    
    for _, row in df.iterrows():
        row_data = [f"{row['val1']:.4g}" if isinstance(row['val1'], float) else str(row['val1'])]
        if p2:
            row_data.append(f"{row['val2']:.4g}" if isinstance(row['val2'], float) else str(row['val2']))
            
        row_data.extend([
            f"{row['cum_ret'] * 100:.2f}% +- {row['cum_ret_std'] * 100:.2f}%",
            f"{row['ann_ret'] * 100:.2f}% +- {row['ann_ret_std'] * 100:.2f}%",
            f"{row['ann_vol'] * 100:.2f}% +- {row['ann_vol_std'] * 100:.2f}%",
            f"{row['sharpe']:.4f} +- {row['sharpe_std']:.4f}",
            f"{row['max_dd'] * 100:.2f}% +- {row['max_dd_std'] * 100:.2f}%"
        ])
        table.add_row(*row_data)
        
    console.print(table)

def save_tuning_report(df, p1, p2, out_dir):
    report_path = os.path.join(out_dir, "tuning_report.md")
    
    header = f"# Hyperparameter Tuning & Sensitivity Analysis Report\n\n"
    header += f"**Tuned parameter(s):** `{p1}`"
    if p2:
        header += f" vs `{p2}`"
    header += f"\n**Evaluation Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    
    table_header = f"| {p1} |"
    table_sep = "|---|"
    if p2:
        table_header += f" {p2} |"
        table_sep += "---|"
    table_header += " Cum Return | Ann Return | Ann Vol | Sharpe Ratio | Max DD |\n"
    table_sep += "---|---|---|---|---|\n"
    
    rows = ""
    for _, row in df.iterrows():
        v1_str = f"{row['val1']:.4g}" if isinstance(row['val1'], float) else str(row['val1'])
        r_str = f"| {v1_str} |"
        if p2:
            v2_str = f"{row['val2']:.4g}" if isinstance(row['val2'], float) else str(row['val2'])
            r_str += f" {v2_str} |"
        r_str += f" {row['cum_ret']*100:.2f}% +- {row['cum_ret_std']*100:.2f}% | {row['ann_ret']*100:.2f}% +- {row['ann_ret_std']*100:.2f}% | {row['ann_vol']*100:.2f}% +- {row['ann_vol_std']*100:.2f}% | {row['sharpe']:.4f} +- {row['sharpe_std']:.4f} | {row['max_dd']*100:.2f}% +- {row['max_dd_std']*100:.2f}% |\n"
        rows += r_str
        
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(header + table_header + table_sep + rows)

def generate_sensitivity_plots(df, p1, p2, output_path):
    plt.style.use("dark_background")
    
    if p2:
        # 2D Grid Tuning: Generate Heatmaps
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        fig.suptitle(f"Grid Search Sensitivity Map: {p1} vs {p2}", color="white", fontsize=14, fontweight="bold")
        
        # Pivot tables for Heatmaps
        pivot_sharpe = df.pivot(index="val1", columns="val2", values="sharpe")
        pivot_cum_ret = df.pivot(index="val1", columns="val2", values="cum_ret") * 100
        
        # Plot Sharpe Heatmap
        sns.heatmap(pivot_sharpe, annot=True, fmt=".3f", cmap="viridis", ax=axes[0], cbar_kws={'label': 'Sharpe Ratio'})
        axes[0].set_title("OOS Sharpe Ratio", fontweight="bold", color="white")
        axes[0].set_xlabel(p2, color="white")
        axes[0].set_ylabel(p1, color="white")
        
        # Plot Cum Return Heatmap
        sns.heatmap(pivot_cum_ret, annot=True, fmt=".1f", cmap="magma", ax=axes[1], cbar_kws={'label': 'Cumulative Return (%)'})
        axes[1].set_title("OOS Cumulative Return (%)", fontweight="bold", color="white")
        axes[1].set_xlabel(p2, color="white")
        axes[1].set_ylabel(p1, color="white")
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, facecolor="#0d1117")
        plt.close()
    else:
        # 1D Tuning: Generate line/bar charts
        fig, ax1 = plt.subplots(figsize=(10, 6))
        fig.suptitle(f"OOS Performance Sensitivity to {p1}", color="white", fontsize=14, fontweight="bold")
        
        ax1.set_facecolor("#0d1117")
        fig.patch.set_facecolor("#0d1117")
        
        x_vals = df["val1"].astype(str).tolist()
        sharpe_vals = df["sharpe"].tolist()
        cum_ret_vals = (df["cum_ret"] * 100).tolist()
        
        # Line 1: Sharpe Ratio
        color = '#ff9800'
        ax1.set_xlabel(p1, color="white", fontweight="bold")
        ax1.set_ylabel("OOS Sharpe Ratio", color=color, fontweight="bold")
        ax1.plot(x_vals, sharpe_vals, color=color, marker='o', linewidth=2.5, label="Sharpe Ratio")
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.grid(color="#1f2937", linestyle="--")
        
        # Line 2: Cumulative Return on secondary y-axis
        ax2 = ax1.twinx()
        color = '#00e676'
        ax2.set_ylabel("OOS Cumulative Return (%)", color=color, fontweight="bold")
        ax2.plot(x_vals, cum_ret_vals, color=color, marker='s', linewidth=2.5, linestyle="--", label="Cum Return")
        ax2.tick_params(axis='y', labelcolor=color)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, facecolor="#0d1117")
        plt.close()

if __name__ == "__main__":
    run_tuning()
