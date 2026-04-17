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

    df = df.reset_index()

    # rename aman
    df.columns = [str(c).lower() for c in df.columns]

    return df
