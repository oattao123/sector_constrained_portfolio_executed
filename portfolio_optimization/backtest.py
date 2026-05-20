import math
import logging
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from .covariance import get_best_device, to_tensor
from .optimizers import optimize_weights_aco_ebgwo

logger = logging.getLogger(__name__)

def compute_metrics(returns, risk_free_rate=0.0434):
    """Computes key performance metrics for a series of returns."""
    returns = np.array(returns)
    if len(returns) == 0:
        return {
            "cum_return": 0.0,
            "ann_return": 0.0,
            "ann_vol": 0.0,
            "sharpe": 0.0,
            "max_dd": 0.0
        }
        
    cum_returns = np.cumprod(1 + returns) - 1
    cum_ret = cum_returns[-1]
    
    ann_ret = np.mean(returns) * 252
    ann_vol = np.std(returns) * math.sqrt(252)
    sharpe = (ann_ret - risk_free_rate) / ann_vol if ann_vol > 0 else 0.0
    
    rolling_max = np.maximum.accumulate(1 + cum_returns)
    drawdowns = (1 + cum_returns) / rolling_max - 1
    max_dd = np.min(drawdowns)
    
    return {
        "cum_return": cum_ret,
        "ann_return": ann_ret,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "max_dd": max_dd,
        "cum_returns_arr": cum_returns
    }

class WalkForwardBacktester:
    """Executes Walk-Forward Optimization across rolling windows."""
    
    def __init__(self, full_returns, spy_full_returns, sector_map=None, risk_free_rate=0.0434):
        self.full_returns = full_returns
        self.spy_full_returns = spy_full_returns
        self.sector_map = sector_map or {}
        self.risk_free_rate = risk_free_rate
        self.device = get_best_device()
        
    def run(self, portfolio_name, selected_stocks, lookback_window=252*3, step_size=21*3, 
            num_iterations=1000, num_agents=500, use_sector_constraints=False, **optimizer_kwargs):
        """Runs rolling window Walk-Forward optimization."""
        logger.info(f"[{portfolio_name}] Starting Walk-Forward Backtest...")
        
        valid_stocks = [t for t in selected_stocks if t in self.full_returns.columns]
        if not valid_stocks:
            raise ValueError(f"No valid stocks found in returns columns for portfolio {portfolio_name}")
            
        port_returns = self.full_returns[valid_stocks]
        returns_gpu_full = to_tensor(port_returns, self.device)
        
        total_days = returns_gpu_full.shape[0]
        target_assets_count = max(5, len(valid_stocks) // 2)
        
        # Prepare sector labels if requested
        sector_labels_tensor = None
        if use_sector_constraints and self.sector_map:
            unique_sectors = sorted(list(set(self.sector_map.values())))
            sector_to_id = {sec: idx for idx, sec in enumerate(unique_sectors)}
            sector_ids = [sector_to_id[self.sector_map.get(t, 'Unknown')] for t in valid_stocks]
            sector_labels_tensor = torch.tensor(sector_ids, dtype=torch.int64, device=self.device)
            
        out_of_sample_returns = []
        all_best_conv = []
        all_avg_conv = []
        
        for start_idx in range(0, total_days - lookback_window, step_size):
            train_end = start_idx + lookback_window
            test_end = min(train_end + step_size, total_days)
            
            if test_end <= train_end:
                break
                
            train_returns = returns_gpu_full[start_idx:train_end]
            
            # Run ACO + EBGWO weight optimization
            best_weights, conv_b, conv_a = optimize_weights_aco_ebgwo(
                train_returns_gpu=train_returns,
                target_assets=target_assets_count,
                heuristic_tensor=None,
                sector_labels=sector_labels_tensor,
                num_iterations=num_iterations,
                num_agents=num_agents,
                **optimizer_kwargs
            )
            
            all_best_conv.append(conv_b)
            all_avg_conv.append(conv_a)
            
            test_returns_cpu = port_returns.iloc[train_end:test_end].values
            period_returns = (test_returns_cpu * best_weights).sum(axis=1)
            out_of_sample_returns.extend(period_returns)
            
        # Compute metrics
        port_returns_arr = np.array(out_of_sample_returns)
        metrics = compute_metrics(port_returns_arr, self.risk_free_rate)
        
        # Compute average convergence curves
        max_len = max([len(c) for c in all_best_conv]) if all_best_conv else 0
        if max_len > 0:
            padded_best = [np.pad(c, (0, max_len - len(c)), 'edge') for c in all_best_conv]
            avg_best_conv = np.mean(padded_best, axis=0)
            
            padded_avg = [np.pad(c, (0, max_len - len(c)), 'edge') for c in all_avg_conv]
            avg_avg_conv = np.mean(padded_avg, axis=0)
        else:
            avg_best_conv = np.array([])
            avg_avg_conv = np.array([])
            
        # Get dates matching the Out-Of-Sample test period
        trade_dates = self.full_returns.index[lookback_window : lookback_window + len(port_returns_arr)]
        
        return {
            "Strategy": portfolio_name,
            "Cum Return": metrics["cum_return"],
            "Ann Return": metrics["ann_return"],
            "Ann Volatility": metrics["ann_vol"],
            "Sharpe Ratio": metrics["sharpe"],
            "Max Drawdown": metrics["max_dd"],
            "OOS_Returns_Array": port_returns_arr,
            "OOS_Cum_Returns_Array": metrics["cum_returns_arr"],
            "Avg_Best_Convergence": avg_best_conv,
            "Avg_Avg_Convergence": avg_avg_conv,
            "Dates": trade_dates
        }

def save_convergence_plot(results_list, output_path="convergence.png"):
    """Saves average convergence curves of all portfolios to a PNG file."""
    plt.figure(figsize=(16, 8))
    colors = plt.cm.tab10(np.linspace(0, 1, len(results_list)))
    
    for idx, res in enumerate(results_list):
        strat_name = res["Strategy"]
        best_conv = res["Avg_Best_Convergence"]
        avg_conv = res["Avg_Avg_Convergence"]
        
        if len(best_conv) > 0:
            plt.plot(best_conv, label=f"{strat_name} (Best Fitness)", color=colors[idx], linewidth=2.5)
        if len(avg_conv) > 0:
            plt.plot(avg_conv, label=f"{strat_name} (Avg Fitness)", color=colors[idx], linewidth=1.5, linestyle='--', alpha=0.6)
            
    plt.title('EBGWO Optimizer Convergence Curve\n(Averaged across all Walk-Forward windows)', fontsize=16, pad=15)
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('Fitness Value (Sharpe + Entropy - Penalty)', fontsize=12)
    plt.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    logger.info(f"Saved convergence plot to {output_path}")

def save_performance_plot(results_list, spy_cum_returns, trade_dates, output_path="performance.png"):
    """Saves cumulative performance comparison plot to a PNG file."""
    plt.figure(figsize=(16, 8))
    
    for res in results_list:
        plt.plot(trade_dates, res["OOS_Cum_Returns_Array"] * 100, label=res["Strategy"], linewidth=2)
        
    plt.plot(trade_dates, spy_cum_returns * 100, label='SPY (S&P 500 Benchmark)', color='black', linestyle='--', linewidth=2.5)
    
    plt.title('Walk-Forward EBGWO Optimization Comparison', fontsize=16, pad=15)
    plt.xlabel('Date')
    plt.ylabel('Cumulative Return (%)')
    plt.legend(loc='upper left', bbox_to_anchor=(1.02, 1))
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    logger.info(f"Saved performance comparison plot to {output_path}")

def save_performance_plot_2025(results_list, spy_returns_daily, trade_dates, output_path="performance_2025.png"):
    """Saves Year 2025 specific cumulative performance comparison plot to a PNG file."""
    mask_2025 = trade_dates.year == 2025
    if not mask_2025.any():
        logger.warning("No data found for Year 2025 in trade_dates.")
        return
        
    dates_2025 = trade_dates[mask_2025]
    spy_returns_2025 = spy_returns_daily[mask_2025]
    cum_spy_2025 = np.cumprod(1 + spy_returns_2025) - 1
    
    plt.figure(figsize=(16, 8))
    for res in results_list:
        port_returns_2025 = res["OOS_Returns_Array"][mask_2025]
        cum_port_2025 = np.cumprod(1 + port_returns_2025) - 1
        plt.plot(dates_2025, cum_port_2025 * 100, label=res["Strategy"], linewidth=2)
        
    plt.plot(dates_2025, cum_spy_2025 * 100, label='SPY (S&P 500 Benchmark)', color='black', linestyle='--', linewidth=2.5)
    
    plt.title('Walk-Forward Optimization: True Out-of-Sample (Year 2025 Only)', fontsize=16, pad=15)
    plt.xlabel('Date')
    plt.ylabel('Cumulative Return (%)')
    plt.legend(loc='upper left', bbox_to_anchor=(1.02, 1))
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    logger.info(f"Saved 2025 performance plot to {output_path}")
