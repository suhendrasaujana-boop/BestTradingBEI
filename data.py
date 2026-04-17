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
def add_indicators(df):
    # EMA
    df['ema20'] = df['close'].ewm(span=20).mean()
    df['ema50'] = df['close'].ewm(span=50).mean()

    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))

    return df
# MACD
exp1 = df['close'].ewm(span=12).mean()
exp2 = df['close'].ewm(span=26).mean()
df['macd'] = exp1 - exp2
df['macd_signal'] = df['macd'].ewm(span=9).mean()
def calculate_score(df):
    score = 0

    # EMA trend
    if df['ema20'].iloc[-1] > df['ema50'].iloc[-1]:
        score += 25

    # RSI
    if df['rsi'].iloc[-1] > 50:
        score += 25

    # MACD
    if df['macd'].iloc[-1] > df['macd_signal'].iloc[-1]:
        score += 25

    # Momentum tambahan
    if df['close'].iloc[-1] > df['ema20'].iloc[-1]:
        score += 25

    return score
