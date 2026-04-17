import yfinance as yf

def get_data(symbol="BBCA.JK", interval="5m"):
    df = yf.download(
        symbol,
        interval=interval,
        period="5d",
        progress=False
    )

    df = df.reset_index()
    df.columns = [c.lower() for c in df.columns]

    return df
