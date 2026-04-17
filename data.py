import yfinance as yf
import pandas as pd

def get_data(symbol="BBCA.JK", interval="5m"):
    df = yf.download(
        symbol,
        interval=interval,
        period="5d",
        progress=False
    )

    if df is None or df.empty:
        return pd.DataFrame()

    # reset index
    df = df.reset_index()

    # paksa rename manual
    df.columns = [
        "datetime",
        "open",
        "high",
        "low",
        "close",
        "volume"
    ]

    return df
