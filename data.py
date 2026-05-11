import pandas as pd
import yfinance as yf
import numpy as np
import time
from datetime import datetime

_data_cache = {}
_last_request_time = 0
_MIN_REQUEST_INTERVAL = 2

def _wait_for_rate_limit():
    global _last_request_time
    now = time.time()
    elapsed = now - _last_request_time
    if elapsed < _MIN_REQUEST_INTERVAL:
        time.sleep(_MIN_REQUEST_INTERVAL - elapsed)
    _last_request_time = time.time()

def get_data(symbol, timeframe="1d"):
    global _data_cache
    
    _wait_for_rate_limit()
    
    cache_key = f"{symbol}_{timeframe}"
    
    if cache_key in _data_cache:
        cached_time, cached_data = _data_cache[cache_key]
        if (datetime.now() - cached_time).seconds < 30:
            return cached_data
    
    try:
        interval_map = {"5m": "5m", "15m": "15m", "30m": "30m", "60m": "60m", "1d": "1d"}
        
        if timeframe in ["5m", "15m", "30m", "60m"]:
            period = "5d"
        else:
            period = "1mo"
        
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period, interval=interval_map.get(timeframe, "1d"))
        
        if df.empty:
            return pd.DataFrame()
        
        df = df.reset_index()
        df.columns = [col.lower() for col in df.columns]
        
        _data_cache[cache_key] = (datetime.now(), df)
        return df
        
    except Exception as e:
        print(f"Error get_data {symbol}: {e}")
        return pd.DataFrame()

def add_indicators(df):
    if df.empty:
        return df
    df = df.copy()
    df['ema20'] = df['close'].ewm(span=20, adjust=False).mean()
    df['ema50'] = df['close'].ewm(span=50, adjust=False).mean()
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    exp1 = df['close'].ewm(span=12, adjust=False).mean()
    exp2 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = exp1 - exp2
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_histogram'] = df['macd'] - df['macd_signal']
    df['volume_ma20'] = df['volume'].rolling(window=20).mean()
    df['support'] = df['low'].rolling(window=20).min()
    df['resistance'] = df['high'].rolling(window=20).max()
    return df

def calculate_score(df):
    if df.empty or len(df) < 2:
        return 50
    last = df.iloc[-1]
    score = 50
    if 'rsi' in last and not pd.isna(last['rsi']):
        if last['rsi'] < 30: score += 20
        elif last['rsi'] < 40: score += 10
        elif last['rsi'] > 70: score -= 20
        elif last['rsi'] > 60: score -= 10
    if 'macd_histogram' in last and not pd.isna(last['macd_histogram']):
        if last['macd_histogram'] > 0: score += 15
        else: score -= 15
    if 'ema20' in last and 'ema50' in last:
        if not pd.isna(last['ema20']) and not pd.isna(last['ema50']):
            if last['close'] > last['ema20'] > last['ema50']: score += 15
            elif last['close'] < last['ema20'] < last['ema50']: score -= 15
    return max(0, min(100, score))

def get_signal_label(score):
    if score >= 70: return ("STRONG BUY", "green", "🔥")
    elif score >= 60: return ("BUY", "lightgreen", "📈")
    elif score <= 30: return ("STRONG SELL", "red", "🔴")
    elif score <= 40: return ("SELL", "orange", "📉")
    else: return ("NEUTRAL", "gray", "⏸️")

def get_confidence_level(score):
    if score >= 80: return ("Sangat Tinggi", "green")
    elif score >= 65: return ("Tinggi", "lightgreen")
    elif score >= 45: return ("Sedang", "yellow")
    elif score >= 30: return ("Rendah", "orange")
    else: return ("Sangat Rendah", "red")

def multi_timeframe_analysis(symbol):
    timeframes = ["5m", "15m", "30m", "60m", "1d"]
    scores = {}
    
    for tf in timeframes:
        df = get_data(symbol, tf)
        if not df.empty and len(df) > 5:
            df = add_indicators(df)
            scores[tf] = calculate_score(df)
        else:
            scores[tf] = 50
        
        time.sleep(1)  # Delay biar tidak kena rate limit
    
    weights = {"5m": 0.1, "15m": 0.15, "30m": 0.2, "60m": 0.25, "1d": 0.3}
    weighted = sum(scores[tf] * weights.get(tf, 0.2) for tf in timeframes if tf in scores)
    
    return {**scores, "weighted": weighted, "filtered": False}

def scan_saham():
    stocks = ["BBCA.JK", "BBRI.JK", "BMRI.JK", "BBNI.JK", "TLKM.JK", "ASII.JK"]
    results = []
    
    for stock in stocks:
        try:
            df = get_data(stock, "1d")
            if not df.empty and len(df) > 5:
                df = add_indicators(df)
                score = calculate_score(df)
                signal, _, emoji = get_signal_label(score)
                results.append({
                    "Kode": stock,
                    "Score": f"{score:.0f}",
                    "Sinyal": f"{emoji} {signal}",
                    "Harga": f"Rp{df.iloc[-1]['close']:,.0f}"
                })
            time.sleep(1)
        except Exception as e:
            print(f"Error scan {stock}: {e}")
            continue
    
    results.sort(key=lambda x: int(x['Score']), reverse=True)
    return results

def get_trading_recommendation(score, df):
    if score >= 70: return "✅ AKSI: BELI AGGRESIF - Semua indikator mendukung uptrend"
    elif score >= 60: return "📈 AKSI: BELI - Momentum positif, pantau konfirmasi"
    elif score <= 30: return "❌ AKSI: JUAL AGGRESIF - Semua indikator menunjukkan downtrend"
    elif score <= 40: return "📉 AKSI: JUAL - Tekanan bearish, cut loss jika perlu"
    else: return "⏸️ AKSI: HOLD/TUNGGU - Kondisi sideways, tunggu sinyal jelas"

def backtest_strategy(df):
    if df.empty or len(df) < 10:
        return {"return": 0, "winrate": 0, "trades": 0, "final_capital": 100000000}
    df_test = df.copy()
    df_test = add_indicators(df_test)
    capital = 100000000
    position = 0
    trades = []
    entry_price = 0
    for i in range(5, len(df_test)):
        score = calculate_score(df_test.iloc[:i+1])
        if score >= 60 and position == 0:
            position = capital / df_test.iloc[i]['close']
            capital = 0
            entry_price = df_test.iloc[i]['close']
        elif score <= 40 and position > 0:
            capital = position * df_test.iloc[i]['close']
            position = 0
            pnl = (df_test.iloc[i]['close'] - entry_price) / entry_price * 100
            trades.append(pnl)
    if position > 0:
        capital = position * df_test.iloc[-1]['close']
    winrate = len([t for t in trades if t > 0]) / len(trades) * 100 if trades else 0
    total_return = ((capital - 100000000) / 100000000) * 100
    return {"return": round(total_return, 2), "winrate": round(winrate, 2), "trades": len(trades), "final_capital": capital}
