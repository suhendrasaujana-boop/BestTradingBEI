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

    # flatten kalau multi column
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ['_'.join(col).lower() for col in df.columns]
    else:
        df.columns = [str(c).lower() for c in df.columns]

    return df
