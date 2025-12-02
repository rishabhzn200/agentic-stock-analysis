import logging
import yfinance as yf
import pandas as pd

logger = logging.getLogger(__name__)


def get_stock_data(ticker, period="1y"):
    """
    Download historical OHLCV data for a given ticker using yfinance.
    """
    logger.info(f"Downloading {ticker} data for period={period}...")
    df = yf.download(ticker, period=period, progress=False)
    print(f"[INFO] Download complete. Rows: {len(df)}")

    # Ensure we have a DataFrame with data
    if df is None or df.empty:
        raise ValueError(f"No data returned for ticker {ticker}")

    # Flatten MultiIndex columns (e.g. ('Close','') -> 'Close')
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    logger.info(f"Download complete. Rows: {len(df)}")
    return df
