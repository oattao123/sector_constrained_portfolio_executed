import math
import logging
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib
matplotlib.use("Agg")  # force non-interactive backend; must precede mplfinance/pyplot import
import mplfinance as mpf
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


def _download_ohlcv(tickers: list, start_date: str, end_date: str) -> dict:
    """Downloads OHLCV data for a list of tickers. Returns dict {ticker: DataFrame}."""
    result = {}
    batch_size = 20
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i : i + batch_size]
        try:
            raw = yf.download(
                batch, start=start_date, end=end_date,
                auto_adjust=True, threads=True, progress=False
            )
            for ticker in batch:
                try:
                    if isinstance(raw.columns, pd.MultiIndex):
                        df = raw.xs(ticker, axis=1, level=1).dropna()
                    else:
                        df = raw.copy().dropna()
                    if not df.empty and all(c in df.columns for c in ['Open', 'High', 'Low', 'Close']):
                        result[ticker] = df
                except Exception:
                    pass
        except Exception as e:
            logger.warning(f"Failed to download OHLCV batch {batch}: {e}")
    return result


def save_candlestick_grid(
    strategy_name: str,
    selected_df: pd.DataFrame,
    start_date: str,
    end_date: str,
    output_path: str,
    max_cols: int = 5,
    candle_period: str = "W",          # 'D' daily | 'W' weekly | 'ME' monthly
    lookback_bars: int = 52,            # how many candles to show
    sector_map: dict = None  ,  # ty:ignore[invalid-parameter-default]
):
    """
    Plots a candlestick grid for all assets selected by a given strategy and saves as PNG.

    Parameters
    ----------
    strategy_name : str
        Title displayed at the top of the figure.
    selected_df : pd.DataFrame
        DataFrame with at least a 'Ticker' column (output of any selector function).
    start_date : str
        Download start date ('YYYY-MM-DD').
    end_date : str
        Download end date ('YYYY-MM-DD').
    output_path : str
        Full path to the output PNG file.
    max_cols : int
        Maximum number of subplot columns in the grid.
    candle_period : str
        Resample period for candles — 'D' (daily), 'W' (weekly), 'ME' (monthly).
    lookback_bars : int
        Number of candles to display per chart.
    sector_map : dict, optional
        Maps ticker -> sector string for colour-coding subplot borders.
    """
    tickers = selected_df['Ticker'].tolist() if 'Ticker' in selected_df.columns else list(selected_df.index)
    if not tickers:
        logger.warning(f"[{strategy_name}] No tickers to plot.")
        return

    logger.info(f"[{strategy_name}] Downloading OHLCV for {len(tickers)} tickers ({start_date} -> {end_date})...")
    ohlcv_data = _download_ohlcv(tickers, start_date, end_date)

    # Only plot tickers that actually have data
    valid_tickers = [t for t in tickers if t in ohlcv_data]
    if not valid_tickers:
        logger.warning(f"[{strategy_name}] No OHLCV data available to plot.")
        return

    n = len(valid_tickers)
    ncols = min(max_cols, n)
    nrows = math.ceil(n / ncols)

    # Assign a colour to each sector for border highlighting
    sectors = sorted(set((sector_map or {}).get(t, "Unknown") for t in valid_tickers))
    sector_palette = plt.cm.tab20(np.linspace(0, 1, max(len(sectors), 1)))
    sector_color = {s: sector_palette[i] for i, s in enumerate(sectors)}

    fig = plt.figure(figsize=(ncols * 4, nrows * 3.5), facecolor="#0d1117")
    fig.suptitle(
        f"Candlestick Charts — {strategy_name}",
        fontsize=16, fontweight="bold", color="white", y=1.01
    )

    gs = gridspec.GridSpec(nrows, ncols, figure=fig, hspace=0.55, wspace=0.35)

    mc = mpf.make_marketcolors(
        up="#00e676", down="#ff1744",
        wick={"up": "#00e676", "down": "#ff1744"},
        edge={"up": "#00e676", "down": "#ff1744"},
        volume={"up": "#00e676", "down": "#ff1744"},
    )
    style = mpf.make_mpf_style(
        marketcolors=mc,
        facecolor="#0d1117",
        figcolor="#0d1117",
        gridcolor="#1f2937",
        gridstyle="--",
        rc={
            "axes.labelcolor": "white",
            "xtick.color": "#9ca3af",
            "ytick.color": "#9ca3af",
            "axes.titlecolor": "white",
        }
    )

    for idx, ticker in enumerate(valid_tickers):
        row, col = divmod(idx, ncols)
        ax = fig.add_subplot(gs[row, col])

        df = ohlcv_data[ticker].copy()

        # Resample to requested period
        if candle_period != "D":
            ohlc_map = {"Open": "first", "High": "max", "Low": "min", "Close": "last"}
            vol_map = {"Volume": "sum"} if "Volume" in df.columns else {}
            df = df.resample(candle_period).agg({**ohlc_map, **vol_map}).dropna()

        # Trim to last `lookback_bars` candles
        df = df.tail(lookback_bars)

        if df.empty or len(df) < 3:
            ax.set_facecolor("#0d1117")
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                    color="#6b7280", transform=ax.transAxes)
            ax.set_title(ticker, color="white", fontsize=9, pad=4)
            ax.axis("off")
            continue

        # Plot candlestick into the existing axes
        mpf.plot(
            df,
            type="candle",
            style=style,
            ax=ax,
            volume=False,
            show_nontrading=False,
        )

        # Colour-coded sector border
        sector = (sector_map or {}).get(ticker, "Unknown")
        border_color = sector_color.get(sector, "#6b7280")
        for spine in ax.spines.values():
            spine.set_edgecolor(border_color)
            spine.set_linewidth(2)

        # Compute simple return % over the shown window
        pct = (df["Close"].iloc[-1] / df["Close"].iloc[0] - 1) * 100
        pct_str = f"+{pct:.1f}%" if pct >= 0 else f"{pct:.1f}%"
        pct_color = "#00e676" if pct >= 0 else "#ff1744"

        ax.set_title(
            f"{ticker}  {pct_str}",
            color="white", fontsize=9, pad=4,
            fontweight="bold"
        )
        ax.tick_params(colors="#9ca3af", labelsize=6)

    # Hide any unused subplots
    for idx in range(n, nrows * ncols):
        row, col = divmod(idx, ncols)
        ax = fig.add_subplot(gs[row, col])
        ax.set_visible(False)

    # Sector legend
    if sector_map:
        legend_elements = [
            Patch(facecolor=sector_color[s], edgecolor="white", label=s)
            for s in sectors
        ]
        fig.legend(
            handles=legend_elements,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.03),
            ncol=min(len(sectors), 6),
            fontsize=7,
            facecolor="#1f2937",
            edgecolor="#374151",
            labelcolor="white",
        )

    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="#0d1117")
    plt.close(fig)
    logger.info(f"[{strategy_name}] Saved candlestick grid to {output_path}")
