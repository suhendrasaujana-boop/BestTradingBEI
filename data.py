import yfinance as yf
import pandas as pd
import numpy as np

def get_data(symbol="BBCA.JK", interval="5m"):
    """Ambil data dari Yahoo Finance"""
    try:
        if interval in ["5m", "15m", "30m"]:
            period = "7d"
        elif interval == "60m":
            period = "5d"
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
    """Tambah indikator lengkap + ADX + Stochastic RSI"""
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
    
    # ADX
    high = df['high']
    low = df['low']
    close = df['close']
    
    plus_dm = high.diff()
    minus_dm = low.diff()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm > 0] = 0
    minus_dm = abs(minus_dm)
    
    atr_period = 14
    tr_adx = pd.concat([high - low, abs(high - close.shift()), abs(low - close.shift())], axis=1).max(axis=1)
    atr_adx = tr_adx.rolling(window=atr_period).mean()
    
    plus_di = 100 * (plus_dm.ewm(span=atr_period).mean() / atr_adx)
    minus_di = abs(100 * (minus_dm.ewm(span=atr_period).mean() / atr_adx))
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    df['adx'] = dx.rolling(window=atr_period).mean()
    
    # Stochastic RSI
    rsi = df['rsi']
    stoch_rsi = (rsi - rsi.rolling(14).min()) / (rsi.rolling(14).max() - rsi.rolling(14).min())
    df['stoch_rsi'] = stoch_rsi * 100
    df['stoch_rsi_k'] = df['stoch_rsi'].rolling(3).mean()
    df['stoch_rsi_d'] = df['stoch_rsi_k'].rolling(3).mean()
    
    return df


def calculate_score(df):
    """Hitung skor sinyal (0-100)"""
    if df.empty or len(df) < 50:
        return 0
    
    score = 0
    last = df.iloc[-1]
    prev = df.iloc[-2] if len(df) > 1 else last
    
    if last['ema20'] > last['ema50']:
        score += 10
    if last['close'] > last['ema20']:
        score += 5
        
    if last['rsi'] < 30:
        score += 25
    elif last['rsi'] > 70:
        score += 0
    elif last['rsi'] > 50:
        score += 12
        
    if last['macd'] > last['macd_signal']:
        score += 15
    if last['macd_histogram'] > 0:
        score += 5
        
    if last['close'] <= last['bb_lower']:
        score += 10
    elif last['close'] >= last['bb_upper']:
        score += 0
    elif last['close'] > last['bb_middle']:
        score += 5
        
    if last['volume'] > last['volume_ma20'] * 1.2:
        score += 12
    if last['close'] > prev['close'] and last['volume'] > prev['volume']:
        score += 8
        
    if last['close'] <= last['support'] * 1.02:
        score += 10
    elif last['close'] >= last['resistance'] * 0.98:
        score += 0
    else:
        score += 5
    
    if not pd.isna(last['adx']):
        if last['adx'] >= 25:
            score += 10
        elif last['adx'] >= 20:
            score += 5
    
    if not pd.isna(last['stoch_rsi_k']):
        if last['stoch_rsi_k'] < 20:
            score += 10
        elif last['stoch_rsi_k'] > 80:
            score += 0
        elif last['stoch_rsi_k'] > 50:
            score += 5
    
    if last['volume'] < last['volume_ma20'] * 0.8:
        score -= 15
    
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


def get_confidence_level(score):
    """Hitung level keyakinan sinyal (FITUR BARU)"""
    if score >= 80:
        return "🔥 HIGH CONFIDENCE", "green"
    elif score >= 60:
        return "✅ MEDIUM CONFIDENCE", "lightgreen"
    elif score >= 40:
        return "⚠️ LOW CONFIDENCE", "orange"
    else:
        return "❌ VERY LOW CONFIDENCE", "red"


def multi_timeframe_analysis(symbol):
    """Analisis multi timeframe dengan bobot baru"""
    timeframes = {
        "5m": "5m",
        "15m": "15m", 
        "30m": "30m",
        "1h": "60m",
        "1d": "1d"
    }
    
    weights = {
        "5m": 0.05,
        "15m": 0.10,
        "30m": 0.20,
        "1h": 0.30,
        "1d": 0.35
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
    
    daily_score = results.get("1d", 0)
    
    if weighted_score >= 60 and daily_score < 40:
        weighted_score = weighted_score * 0.7
        results['filtered'] = True
        results['filter_message'] = "⚠️ Daily masih SELL, skor dikurangi 30%"
    else:
        results['filtered'] = False
        results['filter_message'] = ""
    
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
    """Buat rekomendasi trading lengkap"""
    if df.empty:
        return "Tidak ada data"
    
    last = df.iloc[-1]
    support = last['support']
    resistance = last['resistance']
    atr = last['atr'] if last['atr'] > 0 else last['close'] * 0.02
    
    adx_text = ""
    if 'adx' in last and not pd.isna(last['adx']):
        if last['adx'] >= 25:
            adx_text = f"\n📊 **ADX:** {last['adx']:.1f} (Tren Kuat ✅)"
        elif last['adx'] >= 20:
            adx_text = f"\n📊 **ADX:** {last['adx']:.1f} (Tren Mulai)"
        else:
            adx_text = f"\n📊 **ADX:** {last['adx']:.1f} (Tren Lemah ⚠️)"
    
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
🎯 **Target 2:** Rp{target2:,.0f}{adx_text}
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
🎯 **Target 2:** Rp{target2:,.0f}{adx_text}
━━━━━━━━━━━━━━━━━━━━
🛡️ **Support:** Rp{support:,.0f}
🚧 **Resistance:** Rp{resistance:,.0f}"""
    
    else:
        return f"""⏳ **REKOMENDASI WAIT/HOLD**
━━━━━━━━━━━━━━━━━━━━
📊 **Harga saat ini:** Rp{last['close']:,.0f}{adx_text}
━━━━━━━━━━━━━━━━━━━━
🛡️ **Support:** Rp{support:,.0f}
🚧 **Resistance:** Rp{resistance:,.0f}
💡 **Saran:** Tunggu konfirmasi lebih lanjut"""
