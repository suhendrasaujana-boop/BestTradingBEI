import yfinance as yf
import pandas as pd
import numpy as np

def get_data(symbol="BBCA.JK", interval="5m"):
    """Ambil data dari Yahoo Finance"""
    try:
        if interval in ["5m", "15m", "30m"]:
            period = "5d"
        elif interval == "60m":
            period = "1mo"
        else:
            period = "3mo"
            
        df = yf.download(symbol, interval=interval, period=period, progress=False)
        
        if df is None or df.empty:
            return pd.DataFrame()
        
        df = df.reset_index()
        df.columns = ["datetime", "open", "high", "low", "close", "volume"]
        df = df.dropna()
        
        return df
    except Exception:
        return pd.DataFrame()


def add_indicators(df):
    """Tambah indikator lengkap"""
    if df.empty:
        return df
    
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
    
    df['bb_middle'] = df['close'].rolling(window=20).mean()
    bb_std = df['close'].rolling(window=20).std()
    df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
    df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
    
    df['volume_ma20'] = df['volume'].rolling(window=20).mean()
    
    high_low = df['high'] - df['low']
    high_close = abs(df['high'] - df['close'].shift())
    low_close = abs(df['low'] - df['close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr'] = tr.rolling(window=14).mean()
    
    df['resistance'] = df['high'].rolling(window=20).max()
    df['support'] = df['low'].rolling(window=20).min()
    
    return df


def calculate_score(df):
    """Hitung skor sinyal (0-100)"""
    if df.empty or len(df) < 50:
        return 0
    
    score = 0
    last = df.iloc[-1]
    prev = df.iloc[-2] if len(df) > 1 else last
    
    if last['ema20'] > last['ema50']:
        score += 15
    if last['close'] > last['ema20']:
        score += 5
        
    if last['rsi'] < 30:
        score += 20
    elif last['rsi'] > 70:
        score += 0
    elif last['rsi'] > 50:
        score += 10
        
    if last['macd'] > last['macd_signal']:
        score += 15
    if last['macd_histogram'] > 0:
        score += 5
        
    if last['close'] <= last['bb_lower']:
        score += 15
    elif last['close'] >= last['bb_upper']:
        score += 0
    elif last['close'] > last['bb_middle']:
        score += 8
        
    if last['volume'] > last['volume_ma20'] * 1.2:
        score += 10
    if last['close'] > prev['close'] and last['volume'] > prev['volume']:
        score += 5
        
    if last['close'] <= last['support'] * 1.02:
        score += 10
    elif last['close'] >= last['resistance'] * 0.98:
        score += 0
    else:
        score += 5
        
    return min(100, max(0, score))


def get_signal_label(score):
    """Konversi skor ke label sinyal"""
    if score >= 75:
        return "🔥 STRONG BUY", "green", "🟢"
    elif score >= 60:
        return "✅ BUY", "lightgreen", "🟢"
    elif score >= 40:
        return "⏳ WAIT / HOLD", "orange", "🟡"
    elif score >= 25:
        return "⚠️ SELL", "salmon", "🔴"
    else:
        return "🔴 STRONG SELL", "red", "🔴"


def multi_timeframe_analysis(symbol):
    """Analisis multi timeframe dengan bobot berbeda"""
    timeframes = {
        "5m": "5m",
        "15m": "15m", 
        "30m": "30m",
        "1h": "60m",
        "1d": "1d"
    }
    
    weights = {
        "5m": 0.10,
        "15m": 0.15,
        "30m": 0.20,
        "1h": 0.25,
        "1d": 0.30
    }
    
    results = {}
    weighted_score = 0
    
    for label, tf in timeframes.items():
        df = get_data(symbol, tf)
        if df.empty:
            results[label] = 0
            continue
        
        df = add_indicators(df)
        score = calculate_score(df)
        results[label] = score
        weighted_score += score * weights[label]
    
    results['weighted'] = round(weighted_score, 2)
    return results


def scan_saham():
    """Scan saham"""
    saham_list = [
        "BBCA.JK", "BBRI.JK", "BMRI.JK", "TLKM.JK", "ASII.JK",
        "ANTM.JK", "MDKA.JK", "GOTO.JK", "BRPT.JK", "ADRO.JK",
        "ICBP.JK", "INDF.JK", "UNVR.JK", "HMSP.JK", "ARTO.JK",
        "BUKA.JK", "PTBA.JK", "SMGR.JK", "CPIN.JK", "JPFA.JK"
    ]
    
    results = []
    for saham in saham_list:
        mtf = multi_timeframe_analysis(saham)
        avg_score = mtf.get('weighted', 0)
        
        if avg_score >= 60:
            signal = "BUY"
        elif avg_score >= 40:
            signal = "WAIT"
        else:
            signal = "SELL"
            
        results.append({
            "Kode": saham.replace(".JK", ""),
            "Score": round(avg_score, 2),
            "Sinyal": signal
        })
    
    return sorted(results, key=lambda x: x["Score"], reverse=True)


def get_trading_recommendation(score, df):
    """Buat rekomendasi trading lengkap dengan Support & Resistance"""
    if df.empty:
        return "Tidak ada data"
    
    last = df.iloc[-1]
    support = last['support']
    resistance = last['resistance']
    atr = last['atr'] if last['atr'] > 0 else last['close'] * 0.02
    
    if score >= 60:
        entry = last['close']
        stop_loss = last['close'] - (atr * 1.5)
        target1 = last['close'] + (atr * 1.5)
        target2 = last['close'] + (atr * 3)
        return f"""📈 **REKOMENDASI BUY**
━━━━━━━━━━━━━━━━━━━━
💰 **Entry:** Rp{entry:,.0f}
🛑 **Stop Loss:** Rp{stop_loss:,.0f}
🎯 **Target 1:** Rp{target1:,.0f}
🎯 **Target 2:** Rp{target2:,.0f}
━━━━━━━━━━━━━━━━━━━━
🛡️ **Support:** Rp{support:,.0f}
🚧 **Resistance:** Rp{resistance:,.0f}"""
    
    elif score <= 35:
        entry = last['close']
        stop_loss = last['close'] + (atr * 1.5)
        target1 = last['close'] - (atr * 1.5)
        target2 = last['close'] - (atr * 3)
        return f"""📉 **REKOMENDASI SELL**
━━━━━━━━━━━━━━━━━━━━
💰 **Entry:** Rp{entry:,.0f}
🛑 **Stop Loss:** Rp{stop_loss:,.0f}
🎯 **Target 1:** Rp{target1:,.0f}
🎯 **Target 2:** Rp{target2:,.0f}
━━━━━━━━━━━━━━━━━━━━
🛡️ **Support:** Rp{support:,.0f}
🚧 **Resistance:** Rp{resistance:,.0f}"""
    
    else:
        return f"""⏳ **REKOMENDASI WAIT/HOLD**
━━━━━━━━━━━━━━━━━━━━
📊 **Harga saat ini:** Rp{last['close']:,.0f}
━━━━━━━━━━━━━━━━━━━━
🛡️ **Support:** Rp{support:,.0f}
🚧 **Resistance:** Rp{resistance:,.0f}
💡 **Saran:** Tunggu konfirmasi lebih lanjut
   (Harga mendekati Support → potensi BUY)
   (Harga mendekati Resistance → potensi SELL)"""
