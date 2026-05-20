import json
import logging
import math
import os
import warnings

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


class DataManager:
    """Manages downloading, caching, cleaning, and currency conversion of asset data."""

    def __init__(self, settings_path="config/settings.json", assets_path="config/assets.csv"):
        self.settings_path = settings_path
        self.assets_path = assets_path
        self.settings = {}
        self.assets_df = pd.DataFrame()
        self.tickers = []
        self.sector_map = {}
        self.currency_map = {}

        self.load_config()

    def load_config(self):
        """Loads configuration settings and asset metadata."""
        # Load settings.json
        if os.path.exists(self.settings_path):
            with open(self.settings_path, "r", encoding="utf-8") as f:
                self.settings = json.load(f)
            logger.info(f"Loaded settings from {self.settings_path}")
        else:
            logger.warning(f"Settings file not found at {self.settings_path}. Using defaults.")
            self.settings = {
                "base_currency": "THB",
                "data": {"start_date": "2015-01-01", "end_date": "2025-12-31"},
                "risk_free_rate": {"ticker": "^TNX", "fallback": 0.0434},
            }

        # Load assets.csv
        if os.path.exists(self.assets_path):
            self.assets_df = pd.read_csv(self.assets_path)
            self.tickers = self.assets_df["Ticker"].tolist()
            self.sector_map = dict(zip(self.assets_df["Ticker"], self.assets_df["Sector"]))
            self.currency_map = dict(zip(self.assets_df["Ticker"], self.assets_df["Currency"]))
            logger.info(f"Loaded {len(self.tickers)} assets from {self.assets_path}")
        else:
            logger.error(f"Assets file not found at {self.assets_path}!")
            raise FileNotFoundError(f"Assets CSV not found at {self.assets_path}")

    def download_data(self, start_date=None, end_date=None, cache_path="data/all_data.csv", use_cache=True):
        """Downloads historical price data for the tickers, using local cache if available."""
        if start_date is None:
            start_date = self.settings.get("data", {}).get("start_date", "2015-01-01")
        if end_date is None:
            end_date = self.settings.get("data", {}).get("end_date", "2025-12-31")

        df = pd.DataFrame()

        # Try loading from cache
        if use_cache and os.path.exists(cache_path):
            logger.info(f"Loading data from cache: {cache_path}")
            df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
            # Find missing tickers
            missing_tickers = [t for t in self.tickers if t not in df.columns]
            if not missing_tickers:
                # Crop to desired dates
                df = df.loc[start_date:end_date]
                logger.info("Cache hit for all tickers.")
                return df
            else:
                logger.info(f"Cache missing {len(missing_tickers)} tickers. Downloading missing ones...")
                download_list = missing_tickers
        else:
            download_list = self.tickers

        if download_list:
            logger.info(f"Downloading data for {len(download_list)} tickers from {start_date} to {end_date}...")
            batch_size = 50
            downloaded_dfs = []

            # Add SPY benchmark if not in download list
            tickers_to_get = list(download_list)
            if "SPY" not in tickers_to_get and "SPY" not in df.columns:
                tickers_to_get.append("SPY")

            for i in range(0, len(tickers_to_get), batch_size):
                batch = tickers_to_get[i : i + batch_size]
                try:
                    batch_data = yf.download(batch, start=start_date, end=end_date, auto_adjust=True, threads=True)
                    if isinstance(batch_data.columns, pd.MultiIndex):
                        batch_close = batch_data["Close"]
                    else:
                        # Single ticker case
                        batch_close = batch_data[["Close"]].rename(columns={"Close": batch[0]})
                    downloaded_dfs.append(batch_close)
                except Exception as e:
                    logger.error(f"Error downloading batch {batch}: {e}")

            if downloaded_dfs:
                new_df = pd.concat(downloaded_dfs, axis=1)
                # Keep only unique columns
                new_df = new_df.loc[:, ~new_df.columns.duplicated()]
                if not df.empty:
                    # Merge on index
                    df = df.combine_first(new_df)
                else:
                    df = new_df

                # Save to cache
                df.to_csv(cache_path)
                logger.info(f"Cached full dataset to {cache_path}")

        # Crop to desired dates
        df = df.loc[start_date:end_date]
        return df

    def clean_data(self, df, max_gap_days=365):
        """Cleans missing data by forward-filling, backward-filling and filtering large gaps."""
        # Find columns that have a gap of NaNs larger than max_gap_days
        mask = df.isnull().rolling(window=max_gap_days).sum()
        cols_with_long_strips = (mask == max_gap_days).any()

        # Filter them out
        df_filtered = df.loc[:, ~cols_with_long_strips].copy()

        # Forward fill and backward fill remaining missing values
        df_clean = df_filtered.ffill().bfill()

        removed_cols = [c for c in df.columns if c not in df_clean.columns]
        if removed_cols:
            logger.warning(f"Removed columns due to >{max_gap_days} days gap: {removed_cols}")

        return df_clean

    def convert_to_base_currency(self, df, start_date=None, end_date=None):
        """Converts non-THB assets to THB using exchange rates."""
        base_currency = self.settings.get("base_currency", "THB")
        if base_currency != "THB":
            # Current implementation only supports THB as base currency as per original notebook.
            logger.warning(f"Base currency {base_currency} requested, but conversion is set to THB.")

        if start_date is None:
            start_date = df.index.min().strftime("%Y-%m-%d")
        if end_date is None:
            end_date = df.index.max().strftime("%Y-%m-%d")

        logger.info("Downloading USD/THB exchange rate for currency conversion...")
        fx_ticker = self.settings.get("currency", {}).get("fx_pairs", {}).get("USD", "USDTHB=X")

        try:
            usdthb_raw = yf.download(fx_ticker, start=start_date, end=end_date, auto_adjust=True)
            if isinstance(usdthb_raw.columns, pd.MultiIndex):
                usdthb = usdthb_raw["Close"].iloc[:, 0]
            else:
                usdthb = usdthb_raw["Close"]
            # Reindex to match the asset prices DataFrame index
            usdthb = usdthb.reindex(df.index).ffill().bfill()
        except Exception as e:
            fallback_rate = float(self.settings.get("currency", {}).get("fallback_rate", 35.0))
            logger.warning(f"Could not download FX rate ({fx_ticker}): {e}. Using fallback rate {fallback_rate}")
            usdthb = pd.Series(fallback_rate, index=df.index)

        # Convert all USD assets to THB
        df_converted = pd.DataFrame(index=df.index)
        for ticker in df.columns:
            # We assume non-Thai stocks (not ending with .BK) and not benchmark SPY are USD-denominated
            # or we check currency mapping from assets.csv if available.
            currency = self.currency_map.get(ticker, "THB" if (ticker.endswith(".BK") or ticker == "SPY") else "USD")

            if currency == "USD" and ticker != "SPY":
                df_converted[ticker] = df[ticker] * usdthb.values.flatten()
            else:
                df_converted[ticker] = df[ticker]

        return df_converted

    def get_log_returns(self, df):
        """Computes log returns for the price dataframe."""
        return np.log(df / df.shift(1)).dropna()
