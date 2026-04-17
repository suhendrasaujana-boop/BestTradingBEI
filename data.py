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
    df.columns = ["datetime","open","high","low","close","volume"]

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

    # MACD (HARUS di dalam fungsi)
    exp1 = df['close'].ewm(span=12).mean()
    exp2 = df['close'].ewm(span=26).mean()
    df['macd'] = exp1 - exp2
    df['macd_signal'] = df['macd'].ewm(span=9).mean()

    return df


def calculate_score(df):
    score = 0

    if df['ema20'].iloc[-1] > df['ema50'].iloc[-1]:
        score += 25

    if df['rsi'].iloc[-1] > 50:
        score += 25

    if df['macd'].iloc[-1] > df['macd_signal'].iloc[-1]:
        score += 25

    if df['close'].iloc[-1] > df['ema20'].iloc[-1]:
        score += 25

    return score
def multi_timeframe_analysis(symbol):

    timeframes = {
        "5m": "5m",
        "15m": "15m",
        "30m": "30m",
        "1h": "60m",
        "1d": "1d"
    }

    results = {}

    for label, tf in timeframes.items():
        df = get_data(symbol, tf)

        if df.empty:
            results[label] = 0
            continue

        df = add_indicators(df)
        score = calculate_score(df)

        results[label] = score

    return results
def scan_saham():

    saham_list = [
        "BBCA.JK",
        "BBRI.JK",
        "BMRI.JK",
        "TLKM.JK",
        "ASII.JK",
        "ANTM.JK",
        "MDKA.JK",
        "GOTO.JK",
        "BRPT.JK"
    ]

    results = []

    for saham in saham_list:
        mtf = multi_timeframe_analysis(saham)
        avg_score = sum(mtf.values()) / len(mtf)

        results.append({
            "saham": saham,
            "score": round(avg_score,2)
        })

    return sorted(results, key=lambda x: x['score'], reverse=True)
from data import scan_saham
st.subheader("Scanner Saham Terbaik")

if st.button("Scan Market"):
    results = scan_saham()
    st.write(results)
