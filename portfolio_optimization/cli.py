import logging
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich.live import Live
from rich.spinner import Spinner

console = Console(width=120)
logger = logging.getLogger(__name__)

def print_welcome():
    """Prints a beautiful welcome panel to the console nya~! (=^･ω･^=)"""
    welcome_text = Text()
    welcome_text.append("**  Cristina's Advanced Portfolio Optimiser  **\n", style="bold magenta")
    welcome_text.append("Integrating Ledoit-Wolf Covariance, 2D-ACO, and EBGWO Metaheuristics desu~! nya~!\n\n", style="italic cyan")
    welcome_text.append("Current base currency: ", style="white")
    welcome_text.append("THB\n", style="bold green")
    welcome_text.append("Optimization Framework: Walk-Forward Rolling OOS", style="white")
    
    panel = Panel(
        welcome_text,
        title="[bold green]MULTIVARIATE PORTFOLIO PIPELINE[/bold green]",
        border_style="magenta",
        expand=False
    )
    console.print(panel)

def print_asset_summary(assets_df):
    """Prints a summarized table of loaded assets."""
    table = Table(title="[bold cyan]Target Asset Universe[/bold cyan]", border_style="cyan")
    table.add_column("Ticker", style="bold yellow")
    table.add_column("Sector", style="magenta")
    table.add_column("Currency", style="blue")
    table.add_column("Lot Size", justify="right", style="green")
    
    for _, row in assets_df.iterrows():
        table.add_row(
            str(row['Ticker']),
            str(row['Sector']),
            str(row['Currency']),
            str(row['LotSize'])
        )
    console.print(table)

def print_selection_summary(strategy_name, df_selected):
    """Prints the selected assets for a strategy in a rich table."""
    table = Table(title=f"[bold green]Selected Assets: {strategy_name}[/bold green]", border_style="green")
    table.add_column("Ticker", style="bold yellow")
    table.add_column("Sector", style="magenta")
    
    # Check column names dynamically
    ret_col = 'LW Annual Return (%)' if 'LW Annual Return (%)' in df_selected.columns else ('Annual Return (%)' if 'Annual Return (%)' in df_selected.columns else None)
    vol_col = 'LW Annual Vol (%)' if 'LW Annual Vol (%)' in df_selected.columns else ('Annual Vol (%)' if 'Annual Vol (%)' in df_selected.columns else None)
    sharpe_col = 'LW Sharpe' if 'LW Sharpe' in df_selected.columns else ('Sharpe' if 'Sharpe' in df_selected.columns else None)
    
    table.add_column("Ann Return", justify="right", style="cyan")
    table.add_column("Ann Vol", justify="right", style="red")
    table.add_column("Sharpe Ratio", justify="right", style="bold green")
    table.add_column("Div Score", justify="right", style="bold yellow")
    
    for _, row in df_selected.iterrows():
        ret_val = f"{row[ret_col]:.2f}%" if ret_col else "N/A"
        vol_val = f"{row[vol_col]:.2f}%" if vol_col else "N/A"
        sharpe_val = f"{row[sharpe_col]:.3f}" if sharpe_col else "N/A"
        div_val = f"{row['Diversification Score']:.3f}" if 'Diversification Score' in row else "N/A"
        
        table.add_row(
            str(row['Ticker']),
            str(row['Sector']),
            ret_val,
            vol_val,
            sharpe_val,
            div_val
        )
    console.print(table)

def print_backtest_table(results, title="Walk-Forward Performance Metrics"):
    """Prints comparison table for backtest results."""
    table = Table(title=f"[bold yellow]{title}[/bold yellow]", border_style="yellow")
    table.add_column("Strategy Name", style="bold white", width=35)
    table.add_column("Cum Return", justify="right", style="green")
    table.add_column("Ann Return", justify="right", style="cyan")
    table.add_column("Ann Vol", justify="right", style="red")
    table.add_column("Sharpe", justify="right", style="bold green")
    table.add_column("Max DD", justify="right", style="bold red")
    
    for r in results:
        table.add_row(
            r["Strategy"],
            f"{r['Cum Return']:>9.2%}",
            f"{r['Ann Return']:>7.2%}",
            f"{r['Ann Volatility'] if 'Ann Volatility' in r else r['Ann Vol']:>7.2%}",
            f"{r['Sharpe Ratio'] if 'Sharpe Ratio' in r else r['Sharpe']:>7.4f}",
            f"{r['Max Drawdown'] if 'Max Drawdown' in r else r['Max DD']:>7.2%}"
        )
    console.print(table)
