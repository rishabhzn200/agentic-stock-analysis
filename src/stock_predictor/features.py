import numpy as np


def compute_features(df):
    df.copy()

    # Price change
    delta = df["Close"].diff()

    # Compute gains and losses
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    # Average gain/loss for RSI over 14 periods
    roll_up = gain.rolling(window=14).mean()
    roll_down = loss.rolling(window=14).mean()
    rs = roll_up / roll_down
    rs = rs.replace([np.inf, -np.inf], np.nan)

    df["RSI"] = 100 - (100 / (1 + rs))

    # EMA
    df["EMA_10"] = df["Close"].ewm(span=10, adjust=False).mean()
    df["EMA_50"] = df["Close"].ewm(span=50, adjust=False).mean()

    # MACD
    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26

    # Target: 1 if next day's close is higher than today's close, else 0
    df["Target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)

    df = df.dropna()
    return df
