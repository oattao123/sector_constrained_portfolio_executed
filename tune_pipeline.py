import os
import sys
import argparse
import logging
import warnings
import itertools
from datetime import datetime

import numpy as np
import pandas as pd
import yaml
import torch
import matplotlib
matplotlib.use("Agg")
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
    print_asset_summary,
)

warnings.filterwarnings("ignore")
console = Console(width=120)
logger = logging.getLogger("tune_pipeline")

TUNE_YAML = "tune_param.ymal"

# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(
        description="Hyperparameter Tuning & Sensitivity Analysis Pipeline (N-param)"
    )
    # ── Path args ──────────────────────────────────────────────────────────
    parser.add_argument("--settings", type=str, default="config/settings.json")
    parser.add_argument("--assets",   type=str, default="config/assets.csv")
    parser.add_argument("--cache",    type=str, default="data/all_data.csv")
    parser.add_argument("--yaml",     type=str, default=TUNE_YAML,
                        help="Path to the tuning YAML config file")

    # ── Base optimiser / selection args (override YAML) ───────────────────
    parser.add_argument("--optimizer",  type=str, default=None,
                        choices=["aco_ebgwo", "pso", "clpso", "apso", "lapso", "acor", "ciac"],
                        help="Optimizer to tune (overrides YAML)")
    parser.add_argument("--strategy",   type=str, default=None,
                        help="Asset-selection strategy (overrides YAML)")
    parser.add_argument("--trials",     type=int, default=None,
                        help="Trials per config (overrides YAML)")
    parser.add_argument("--wolves",     type=int, default=None,
                        help="num_agents / wolves (overrides YAML)")
    parser.add_argument("--iterations", type=int, default=None,
                        help="num_iterations (overrides YAML)")
    parser.add_argument("--agents",     type=int, default=None,
                        help="ACO agents (overrides YAML)")

    # ── Legacy 1-D / 2-D tuning shorthand (still work) ────────────────────
    parser.add_argument("--param",   type=str, default=None)
    parser.add_argument("--values",  type=str, default=None)
    parser.add_argument("--param2",  type=str, default=None)
    parser.add_argument("--values2", type=str, default=None)

    return parser.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# YAML loading & merging
# ──────────────────────────────────────────────────────────────────────────────
YAML_DEFAULTS = {
    "optimizer":  "aco_ebgwo",
    "strategy":   "ACO Cluster Selected",
    "trials":     3,
    "wolves":     30,
    "iterations": 50,
    "agents":     30,
    "parameters": {},
}

def load_yaml_config(path: str) -> dict:
    if not os.path.exists(path):
        logger.warning(f"YAML config not found at '{path}'. Using defaults.")
        return dict(YAML_DEFAULTS)
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    # fill missing keys with defaults
    for k, v in YAML_DEFAULTS.items():
        cfg.setdefault(k, v)
    return cfg


def merge_config(cfg: dict, args) -> dict:
    """CLI args override YAML values when explicitly provided."""
    if args.optimizer  is not None: cfg["optimizer"]  = args.optimizer
    if args.strategy   is not None: cfg["strategy"]   = args.strategy
    if args.trials     is not None: cfg["trials"]     = args.trials
    if args.wolves     is not None: cfg["wolves"]     = args.wolves
    if args.iterations is not None: cfg["iterations"] = args.iterations
    if args.agents     is not None: cfg["agents"]     = args.agents

    # Legacy 1-D / 2-D shorthand: inject into parameters dict
    if args.param and args.values:
        vals = [cast_value(args.param, v) for v in args.values.split(",")]
        cfg["parameters"][args.param] = vals
    if args.param2 and args.values2:
        vals2 = [cast_value(args.param2, v) for v in args.values2.split(",")]
        cfg["parameters"][args.param2] = vals2

    return cfg


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────
INT_PARAMS = {"patience", "num_wolves", "iterations", "agents", "num_particles",
              "archive_size", "num_ants", "num_epochs"}

def cast_value(param_name, val):
    try:
        if param_name in INT_PARAMS:
            return int(float(val))
        return float(val)
    except (ValueError, TypeError):
        return val


def build_grid(param_dict: dict) -> list[dict]:
    """Return list of dicts representing every combination in the Cartesian product."""
    if not param_dict:
        return [{}]
    keys = list(param_dict.keys())
    value_lists = [param_dict[k] for k in keys]
    combinations = list(itertools.product(*value_lists))
    return [dict(zip(keys, combo)) for combo in combinations]


def combo_name(combo: dict) -> str:
    return ", ".join(f"{k}={v}" for k, v in combo.items())


# ──────────────────────────────────────────────────────────────────────────────
# Rich UI
# ──────────────────────────────────────────────────────────────────────────────
def print_welcome_panel(cfg: dict):
    t = Text()
    t.append("  Cristina's Hyperparameter Tuning Pipeline  \n", style="bold magenta")
    t.append("Grid Search · OOS Sensitivity · N-param desu~!\n", style="italic cyan")
    t.append(f"  Optimizer : {cfg['optimizer'].upper()}\n", style="yellow")
    t.append(f"  Strategy  : {cfg['strategy']}\n", style="yellow")
    t.append(f"  Trials    : {cfg['trials']}   Wolves/Agents: {cfg['wolves']}   Iterations: {cfg['iterations']}\n",
             style="yellow")
    params_str = " × ".join(
        f"{k}({len(v)})" for k, v in cfg.get("parameters", {}).items()
    ) or "(none)"
    t.append(f"  Grid      : {params_str}\n", style="bold white")
    panel = Panel(t, title="[bold green]HYPERPARAMETER SENSITIVITY PIPELINE[/bold green]",
                  border_style="magenta", expand=False)
    console.print(panel)


def print_tuning_results_table(results_df: pd.DataFrame, param_keys: list[str]):
    n_params = len(param_keys)
    title = f"[bold green]Grid Search Results — {' × '.join(param_keys) or 'No params'}[/bold green]"
    table = Table(title=title, border_style="green", show_lines=True)

    colors = ["cyan", "magenta", "yellow", "blue", "white", "red"]
    for i, k in enumerate(param_keys):
        table.add_column(k, justify="right", style=colors[i % len(colors)])

    table.add_column("Cum Return",   justify="right", style="green")
    table.add_column("Ann Return",   justify="right", style="cyan")
    table.add_column("Ann Vol",      justify="right", style="red")
    table.add_column("Sharpe Ratio", justify="right", style="bold yellow")
    table.add_column("Max DD",       justify="right", style="bold red")

    for _, row in results_df.iterrows():
        cells = []
        for k in param_keys:
            v = row.get(k, "")
            cells.append(f"{v:.4g}" if isinstance(v, float) else str(v))
        cells += [
            f"{row['cum_ret']*100:.2f}% ± {row['cum_ret_std']*100:.2f}%",
            f"{row['ann_ret']*100:.2f}% ± {row['ann_ret_std']*100:.2f}%",
            f"{row['ann_vol']*100:.2f}% ± {row['ann_vol_std']*100:.2f}%",
            f"{row['sharpe']:.4f} ± {row['sharpe_std']:.4f}",
            f"{row['max_dd']*100:.2f}% ± {row['max_dd_std']*100:.2f}%",
        ]
        table.add_row(*cells)

    console.print(table)


# ──────────────────────────────────────────────────────────────────────────────
# Report saver
# ──────────────────────────────────────────────────────────────────────────────
def save_tuning_report(df: pd.DataFrame, param_keys: list[str], out_dir: str,
                       optimizer_name: str, cfg: dict):
    report_path = os.path.join(out_dir, "tuning_report.md")

    lines = ["# Hyperparameter Tuning & Sensitivity Analysis Report\n"]
    lines.append(f"**Optimizer:** `{optimizer_name.upper()}`  \n")
    lines.append(f"**Strategy:** `{cfg['strategy']}`  \n")
    lines.append(f"**Trials per config:** {cfg['trials']}  \n")
    lines.append(f"**wolves/agents:** {cfg['wolves']}   **iterations:** {cfg['iterations']}  \n")
    lines.append(f"**Evaluation Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

    # Parameter grid summary
    lines.append("## Tuning Grid\n")
    for k, v in cfg.get("parameters", {}).items():
        lines.append(f"- `{k}`: {v}\n")
    lines.append("\n")

    # Table header
    col_heads = " | ".join(param_keys) + " | Cum Return | Ann Return | Ann Vol | Sharpe Ratio | Max DD"
    sep = " | ".join(["---"] * len(param_keys)) + " | --- | --- | --- | --- | ---"
    lines.append(f"| {col_heads} |\n")
    lines.append(f"| {sep} |\n")

    for _, row in df.iterrows():
        p_vals = " | ".join(
            f"{row[k]:.4g}" if isinstance(row[k], float) else str(row[k])
            for k in param_keys
        )
        lines.append(
            f"| {p_vals} "
            f"| {row['cum_ret']*100:.2f}%±{row['cum_ret_std']*100:.2f}% "
            f"| {row['ann_ret']*100:.2f}%±{row['ann_ret_std']*100:.2f}% "
            f"| {row['ann_vol']*100:.2f}%±{row['ann_vol_std']*100:.2f}% "
            f"| {row['sharpe']:.4f}±{row['sharpe_std']:.4f} "
            f"| {row['max_dd']*100:.2f}%±{row['max_dd_std']*100:.2f}% |\n"
        )

    with open(report_path, "w", encoding="utf-8") as f:
        f.writelines(lines)

    logger.info(f"Tuning report saved → {report_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Sensitivity Plots
# ──────────────────────────────────────────────────────────────────────────────
DARK_BG   = "#0d1117"
GRID_COL  = "#1f2937"
METRICS   = [
    ("sharpe",  "Sharpe Ratio",         "#ff9800"),
    ("cum_ret", "Cum Return (%)",        "#00e676"),
    ("ann_vol", "Ann Volatility (%)",    "#f06292"),
    ("max_dd",  "Max Drawdown (%)",      "#ef5350"),
]


def _format_ax(ax):
    ax.set_facecolor(DARK_BG)
    ax.grid(color=GRID_COL, linestyle="--", linewidth=0.7)
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID_COL)


def generate_sensitivity_plots(df: pd.DataFrame, param_keys: list[str], out_dir: str):
    """Dispatch to the right plot type based on number of tuned parameters."""
    n = len(param_keys)
    plt.style.use("dark_background")

    if n == 0:
        logger.warning("No parameters to plot.")
        return

    if n == 1:
        _plot_1d(df, param_keys[0], out_dir)
    elif n == 2:
        _plot_2d_heatmaps(df, param_keys, out_dir)
    else:
        # N-D: one subplot row per metric, columns = parameters
        _plot_nd_marginal(df, param_keys, out_dir)
        _plot_parallel_coords(df, param_keys, out_dir)


# ── 1-D ───────────────────────────────────────────────────────────────────────
def _plot_1d(df: pd.DataFrame, p1: str, out_dir: str):
    x_vals = df[p1].astype(str).tolist()

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.patch.set_facecolor(DARK_BG)
    fig.suptitle(f"OOS Sensitivity — {p1}", color="white", fontsize=15, fontweight="bold")

    plot_specs = [
        (axes[0, 0], "sharpe",  "Sharpe Ratio",      "#ff9800"),
        (axes[0, 1], "cum_ret", "Cum Return (%)",     "#00e676"),
        (axes[1, 0], "ann_vol", "Ann Volatility (%)", "#f06292"),
        (axes[1, 1], "max_dd",  "Max Drawdown (%)",   "#ef5350"),
    ]

    for ax, col, label, color in plot_specs:
        vals = df[col].tolist()
        if col in ("cum_ret", "ann_vol", "max_dd"):
            vals = [v * 100 for v in vals]
        std_col = col + "_std"
        errs = (df[std_col] * (100 if col in ("cum_ret", "ann_vol", "max_dd") else 1)).tolist()

        ax.plot(x_vals, vals, color=color, marker="o", linewidth=2.5, label=label)
        ax.fill_between(range(len(x_vals)),
                        [v - e for v, e in zip(vals, errs)],
                        [v + e for v, e in zip(vals, errs)],
                        color=color, alpha=0.15)
        ax.set_title(label, color=color, fontweight="bold")
        ax.set_xlabel(p1, color="white")
        ax.set_xticks(range(len(x_vals)))
        ax.set_xticklabels(x_vals, rotation=30, color="white", fontsize=8)
        _format_ax(ax)

    plt.tight_layout()
    path = os.path.join(out_dir, "sensitivity_1d.png")
    plt.savefig(path, dpi=150, facecolor=DARK_BG)
    plt.close()
    logger.info(f"1-D sensitivity plot saved → {path}")


# ── 2-D ───────────────────────────────────────────────────────────────────────
def _plot_2d_heatmaps(df: pd.DataFrame, param_keys: list[str], out_dir: str):
    p1, p2 = param_keys

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.patch.set_facecolor(DARK_BG)
    fig.suptitle(f"Grid Search Heatmaps: {p1} vs {p2}", color="white",
                 fontsize=15, fontweight="bold")

    heatmap_specs = [
        (axes[0, 0], "sharpe",  "Sharpe Ratio",      "viridis"),
        (axes[0, 1], "cum_ret", "Cum Return (%)",     "magma"),
        (axes[1, 0], "ann_vol", "Ann Volatility (%)", "coolwarm_r"),
        (axes[1, 1], "max_dd",  "Max Drawdown (%)",   "Reds"),
    ]

    for ax, col, label, cmap in heatmap_specs:
        scale = 100 if col in ("cum_ret", "ann_vol", "max_dd") else 1
        pivot = df.pivot(index=p1, columns=p2, values=col) * scale
        sns.heatmap(pivot, annot=True, fmt=".2f", cmap=cmap, ax=ax,
                    cbar_kws={"label": label},
                    annot_kws={"size": 8, "color": "white"})
        ax.set_title(label, color="white", fontweight="bold")
        ax.set_xlabel(p2, color="white")
        ax.set_ylabel(p1, color="white")
        ax.tick_params(colors="white")

    plt.tight_layout()
    path = os.path.join(out_dir, "sensitivity_2d.png")
    plt.savefig(path, dpi=150, facecolor=DARK_BG)
    plt.close()
    logger.info(f"2-D heatmap plot saved → {path}")


# ── N-D marginal sensitivity ───────────────────────────────────────────────────
def _plot_nd_marginal(df: pd.DataFrame, param_keys: list[str], out_dir: str):
    """
    For each metric, plot a marginal line: fix all other params at their median value,
    vary one param. Produces a grid of (n_metrics × n_params) subplots.
    """
    n_metrics = len(METRICS)
    n_params  = len(param_keys)

    fig, axes = plt.subplots(n_metrics, n_params,
                             figsize=(max(5 * n_params, 12), max(4 * n_metrics, 10)),
                             squeeze=False)
    fig.patch.set_facecolor(DARK_BG)
    fig.suptitle(f"Marginal Sensitivity ({', '.join(param_keys)})",
                 color="white", fontsize=14, fontweight="bold")

    # Median index per param (for fixing)
    median_vals = {k: float(np.median(df[k].unique())) for k in param_keys}

    for row_idx, (col, label, color) in enumerate(METRICS):
        scale = 100 if col in ("cum_ret", "ann_vol", "max_dd") else 1

        for col_idx, target_param in enumerate(param_keys):
            ax = axes[row_idx][col_idx]
            _format_ax(ax)

            # Filter: fix all other params to their median
            mask = pd.Series([True] * len(df), index=df.index)
            for k in param_keys:
                if k == target_param:
                    continue
                closest = df[k].unique()[
                    np.argmin(np.abs(df[k].unique() - median_vals[k]))
                ]
                mask &= (df[k] == closest)

            sub = df[mask].sort_values(target_param)
            if sub.empty:
                ax.set_visible(False)
                continue

            x_raw = sub[target_param].tolist()
            x_str = [str(v) for v in x_raw]
            y     = (sub[col] * scale).tolist()
            std_col = col + "_std"
            errs  = (sub[std_col] * scale).tolist() if std_col in sub.columns else [0] * len(y)

            ax.plot(x_str, y, color=color, marker="o", linewidth=2.0)
            ax.fill_between(range(len(x_str)),
                            [v - e for v, e in zip(y, errs)],
                            [v + e for v, e in zip(y, errs)],
                            color=color, alpha=0.15)
            ax.set_xlabel(target_param, color="white", fontsize=8)
            if col_idx == 0:
                ax.set_ylabel(label, color=color, fontsize=8)
            ax.set_title(f"{label} vs {target_param}", color=color, fontsize=8, fontweight="bold")
            ax.set_xticks(range(len(x_str)))
            ax.set_xticklabels(x_str, rotation=30, fontsize=7, color="white")

    plt.tight_layout()
    path = os.path.join(out_dir, "sensitivity_nd_marginal.png")
    plt.savefig(path, dpi=150, facecolor=DARK_BG)
    plt.close()
    logger.info(f"N-D marginal plot saved → {path}")


# ── Parallel coordinates ───────────────────────────────────────────────────────
def _plot_parallel_coords(df: pd.DataFrame, param_keys: list[str], out_dir: str):
    """
    Parallel coordinate plot of all grid configurations, coloured by Sharpe.
    Shows interplay between N parameters simultaneously.
    """
    cols_to_plot = param_keys + ["sharpe", "cum_ret"]
    plot_df = df[cols_to_plot].copy()
    plot_df["cum_ret"] = plot_df["cum_ret"] * 100

    # Normalise each column to [0, 1] for parallel coords
    norm_df = plot_df.copy()
    for c in cols_to_plot:
        mn, mx = norm_df[c].min(), norm_df[c].max()
        norm_df[c] = (norm_df[c] - mn) / (mx - mn + 1e-9)

    fig, ax = plt.subplots(figsize=(max(12, 2 * len(cols_to_plot)), 6))
    fig.patch.set_facecolor(DARK_BG)
    ax.set_facecolor(DARK_BG)

    sharpe_raw = plot_df["sharpe"].values
    sharpe_norm = norm_df["sharpe"].values
    cmap = plt.get_cmap("plasma")

    for i in range(len(norm_df)):
        ys = norm_df.iloc[i][cols_to_plot].values
        color = cmap(sharpe_norm[i])
        ax.plot(range(len(cols_to_plot)), ys, color=color, alpha=0.7, linewidth=1.5)

    ax.set_xticks(range(len(cols_to_plot)))
    ax.set_xticklabels(cols_to_plot, color="white", fontsize=9, rotation=30)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["min", "", "mid", "", "max"], color="white")
    ax.set_title("Parallel Coordinate Sensitivity (coloured by Sharpe)",
                 color="white", fontsize=12, fontweight="bold")
    ax.grid(axis="x", color=GRID_COL, linestyle="--", linewidth=0.7)

    sm = plt.cm.ScalarMappable(cmap=cmap,
                                norm=plt.Normalize(vmin=sharpe_raw.min(), vmax=sharpe_raw.max()))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.01)
    cbar.set_label("Sharpe Ratio", color="white")
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")

    plt.tight_layout()
    path = os.path.join(out_dir, "sensitivity_parallel_coords.png")
    plt.savefig(path, dpi=150, facecolor=DARK_BG)
    plt.close()
    logger.info(f"Parallel coords plot saved → {path}")


# ──────────────────────────────────────────────────────────────────────────────
# Asset selection dispatcher
# ──────────────────────────────────────────────────────────────────────────────
def run_selection(strategy, insample_returns, shrunk_cov_gpu, corr_matrix_gpu,
                  sector_map, risk_free_rate, agents, iterations):
    if strategy == "Diversification Selected":
        return select_by_diversification(insample_returns, sector_map, risk_free_rate)
    elif strategy == "LW Diversification Selected":
        return select_by_lw_diversification(insample_returns, shrunk_cov_gpu, corr_matrix_gpu,
                                            sector_map, risk_free_rate)
    elif strategy == "LW Sharpe Cluster Selected":
        return select_by_hrp_sharpe(insample_returns, shrunk_cov_gpu, corr_matrix_gpu,
                                    sector_map, risk_free_rate)
    elif strategy == "LW Div Cluster Selected":
        return select_by_hrp_div(insample_returns, shrunk_cov_gpu, corr_matrix_gpu,
                                 sector_map, risk_free_rate)
    elif strategy == "ACO Selected":
        return select_by_aco(insample_returns, shrunk_cov_gpu, corr_matrix_gpu,
                             sector_map, risk_free_rate,
                             num_ants=agents, num_iterations=iterations)
    else:
        return select_by_aco_cluster(insample_returns, shrunk_cov_gpu, corr_matrix_gpu,
                                     sector_map, risk_free_rate,
                                     num_ants=agents, num_epochs=iterations)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def run_tuning():
    args = parse_args()

    # Load & merge config
    cfg = load_yaml_config(args.yaml)
    cfg = merge_config(cfg, args)

    print_welcome_panel(cfg)

    # Output directory
    now     = datetime.now()
    run_tag = now.strftime("tuning_%d_%m_%H_%M_%S")
    out_dir = os.path.join("output", run_tag)
    os.makedirs(out_dir, exist_ok=True)

    # Logging
    log_file = os.path.join(out_dir, "tuning.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )

    param_dict = cfg.get("parameters", {})
    param_keys = list(param_dict.keys())
    grid       = build_grid(param_dict)
    total_runs = len(grid)

    logger.info(f"Tuning folder: {out_dir}")
    logger.info(f"Optimizer    : {cfg['optimizer']}")
    logger.info(f"Strategy     : {cfg['strategy']}")
    logger.info(f"Grid size    : {total_runs} combos × {cfg['trials']} trials "
                f"= {total_runs * cfg['trials']} backtests")
    for k, v in param_dict.items():
        logger.info(f"  {k}: {v}")

    # ── 1. Load Data ──────────────────────────────────────────────────────
    manager = DataManager(settings_path=args.settings, assets_path=args.assets)
    print_asset_summary(manager.assets_df)

    data_cfg         = manager.settings.get("data", {})
    insample_end_date = data_cfg.get("insample_end_date", None)

    raw_data       = manager.download_data(cache_path=args.cache)
    clean_data     = manager.clean_data(raw_data)
    converted_data = manager.convert_to_base_currency(clean_data)
    returns        = manager.get_log_returns(converted_data)

    spy_full_returns    = returns["SPY"] if "SPY" in returns.columns else pd.Series(0.0, index=returns.index)
    opt_universe_returns = returns.drop(columns=["SPY"]) if "SPY" in returns.columns else returns
    insample_returns    = (opt_universe_returns.loc[:insample_end_date]
                           if insample_end_date else opt_universe_returns)

    # ── 2. Covariance & Selection ─────────────────────────────────────────
    device         = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    returns_gpu    = torch.tensor(insample_returns.values, dtype=torch.float32, device=device)
    shrunk_cov_gpu, _, _ = ledoit_wolf_covariance_gpu_dynamic(returns_gpu)
    corr_matrix_gpu = compute_correlation_matrix(shrunk_cov_gpu)
    risk_free_rate  = manager.settings.get("risk_free_rate", {}).get("fallback", 0.0434)

    logger.info(f"Running selection strategy: {cfg['strategy']}")
    selected_df = run_selection(
        cfg["strategy"], insample_returns, shrunk_cov_gpu, corr_matrix_gpu,
        manager.sector_map, risk_free_rate, cfg["agents"], cfg["iterations"]
    )
    selected_stocks = selected_df["Ticker"].tolist() if not selected_df.empty else []
    if not selected_stocks:
        logger.error("No stocks selected. Cannot proceed.")
        return

    # ── 3. Backtester ─────────────────────────────────────────────────────
    portfolio_cfg        = manager.settings.get("portfolio", {})
    transaction_cost_rate = portfolio_cfg.get("transaction_cost_rate", 0.0)
    backtester = WalkForwardBacktester(
        full_returns=returns,
        spy_full_returns=spy_full_returns,
        sector_map=manager.sector_map,
        risk_free_rate=risk_free_rate,
        transaction_cost_rate=transaction_cost_rate,
    )
    lookback = data_cfg.get("lookback_window", 252 * 3)
    step     = data_cfg.get("step_size", 21 * 3)

    # ── 4. Grid Search ────────────────────────────────────────────────────
    results = []

    for run_idx, combo in enumerate(grid, start=1):
        name = combo_name(combo) if combo else "baseline"
        logger.info(f"[{run_idx}/{total_runs}] Config: {name}  (×{cfg['trials']} trials)")

        trial_metrics = []
        for trial in range(cfg["trials"]):
            try:
                res = backtester.run(
                    portfolio_name=f"{name}_t{trial}",
                    selected_stocks=selected_stocks,
                    lookback_window=lookback,
                    step_size=step,
                    num_iterations=cfg["iterations"],
                    num_agents=cfg["wolves"],
                    use_sector_constraints=True,
                    optimizer=cfg["optimizer"],
                    **combo,
                )
                trial_metrics.append({
                    "cum_ret": res["Cum Return"],
                    "ann_ret": res["Ann Return"],
                    "ann_vol": res["Ann Volatility"],
                    "sharpe":  res["Sharpe Ratio"],
                    "max_dd":  res["Max Drawdown"],
                })
            except Exception as e:
                logger.error(f"  Trial {trial+1} failed: {e}")

        if not trial_metrics:
            logger.error(f"  All trials failed for {name}")
            continue

        row = {k: v for k, v in combo.items()}
        row["name"] = name
        for metric in ("cum_ret", "ann_ret", "ann_vol", "sharpe", "max_dd"):
            vals = [m[metric] for m in trial_metrics]
            row[metric]          = float(np.mean(vals))
            row[f"{metric}_std"] = float(np.std(vals))
        results.append(row)

    if not results:
        logger.error("No results from grid search. Exiting.")
        return

    results_df = pd.DataFrame(results)
    results_df_sorted = results_df.sort_values("sharpe", ascending=False).reset_index(drop=True)

    # ── 5. Output ─────────────────────────────────────────────────────────
    print_tuning_results_table(results_df_sorted, param_keys)
    save_tuning_report(results_df_sorted, param_keys, out_dir, cfg["optimizer"], cfg)

    # Save raw CSV
    csv_path = os.path.join(out_dir, "tuning_results.csv")
    results_df_sorted.to_csv(csv_path, index=False)
    logger.info(f"Raw results CSV saved → {csv_path}")

    # ── 6. Plots ──────────────────────────────────────────────────────────
    generate_sensitivity_plots(results_df, param_keys, out_dir)

    logger.info("Tuning finished successfully nya~! (=^.^=)")
    logger.info(f"All outputs saved to: {out_dir}")


if __name__ == "__main__":
    run_tuning()
