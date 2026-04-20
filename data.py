import yfinance as yf
import pandas as pd
import numpy as np

def get_data(symbol="BBCA.JK", interval="5m"):
    """Ambil data dari Yahoo Finance"""
    try:
        if interval in ["5m", "15m", "30m"]:
            period = "7d"
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
    """Tambah indikator lengkap + ADX + Stochastic RSI + Breakout Detector"""
    if df.empty:
        return df
    
    # EMA
    df['ema20'] = df['close'].ewm(span=20, adjust=False).mean()
    df['ema50'] = df['close'].ewm(span=50, adjust=False).mean()
    
    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # MACD
    exp1 = df['close'].ewm(span=12, adjust=False).mean()
    exp2 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = exp1 - exp2
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_histogram'] = df['macd'] - df['macd_signal']
    
    # Bollinger Bands
    df['bb_middle'] = df['close'].rolling(window=20).mean()
    bb_std = df['close'].rolling(window=20).std()
    df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
    df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
    
    # Volume Average
    df['volume_ma20'] = df['volume'].rolling(window=20).mean()
    
    # ATR
    high_low = df['high'] - df['low']
    high_close = abs(df['high'] - df['close'].shift())
    low_close = abs(df['low'] - df['close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr'] = tr.rolling(window=14).mean()
    
    # Support & Resistance
    df['resistance'] = df['high'].rolling(window=20).max()
    df['support'] = df['low'].rolling(window=20).min()
    
    # ========== FITUR BARU: BREAKOUT DETECTOR ==========
    df['breakout_high'] = df['close'] > df['high'].rolling(20).max().shift(1)
    df['breakout_low'] = df['close'] < df['low'].rolling(20).min().shift(1)
    
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
    """Hitung skor sinyal (0-100) dengan optimasi akurasi maksimal"""
    if df.empty or len(df) < 50:
        return 0
    
    score = 0
    last = df.iloc[-1]
    prev = df.iloc[-2] if len(df) > 1 else last
    
    # ========== 1. TREND MASTER FILTER ==========
    is_uptrend = last['ema20'] > last['ema50']
    is_price_above_ema = last['close'] > last['ema20']
    
    # ========== 2. TREND STRENGTH FILTER (SIDEWAYS PENALTY) ==========
    trend_strength = abs(last['ema20'] - last['ema50']) / last['close']
    if trend_strength < 0.002:  # Sideways market
        score -= 15
    
    if not is_uptrend:
        score -= 20  # Penalti downtrend
    
    if is_price_above_ema:
        score += 8
    else:
        score -= 8
    
    # ========== 3. RSI (DENGAN KONFIRMASI TREND) ==========
    if last['rsi'] < 30:  # Oversold
        if is_uptrend:
            score += 8  # Hanya +8 jika uptrend
        else:
            score += 2  # +2 saja jika downtrend (hindari falling knife)
    elif last['rsi'] > 70:  # Overbought
        score -= 10
    elif last['rsi'] > 50:
        score += 5
        
    # ========== 4. MACD ==========
    if last['macd'] > last['macd_signal']:
        score += 12
    if last['macd_histogram'] > 0:
        score += 5
        
    # ========== 5. BOLLINGER BANDS (DENGAN KONFIRMASI RSI) ==========
    if last['close'] <= last['bb_lower']:
        if last['rsi'] > 30:  # Hindari breakdown
            score += 6
    elif last['close'] >= last['bb_upper']:
        score -= 10
    elif last['close'] > last['bb_middle']:
        score += 5
        
    # ========== 6. VOLUME KONFIRMASI ==========
    volume_surge = last['volume'] > last['volume_ma20'] * 1.2
    volume_price_up = last['close'] > prev['close'] and last['volume'] > prev['volume']
    
    # ========== 7. BREAKOUT DETECTOR (FITUR BARU) ==========
    breakout_high = last['breakout_high'] if 'breakout_high' in last else False
    breakout_low = last['breakout_low'] if 'breakout_low' in last else False
    
    if volume_surge and breakout_high:
        score += 15  # Breakout dengan volume tinggi
    elif volume_surge and breakout_low:
        score -= 15  # Breakdown dengan volume tinggi
    elif volume_surge:
        score += 12
    else:
        score -= 5
    
    if volume_price_up:
        score += 5
        
    # ========== 8. SUPPORT/RESISTANCE ==========
    if last['close'] <= last['support'] * 1.02:
        score += 12
    elif last['close'] >= last['resistance'] * 0.98:
        score -= 12
    else:
        score += 3
    
    # ========== 9. ADX DENGAN CEK ARAH TREND ==========
    if not pd.isna(last['adx']):
        if last['adx'] >= 25:
            if is_uptrend:
                score += 15  # Tren naik kuat
            else:
                score -= 10  # Tren turun kuat
        elif last['adx'] >= 20:
            if is_uptrend:
                score += 8
            else:
                score -= 5
        elif last['adx'] < 18:
            score -= 15  # Tren lemah
    
    # ========== 10. STOCHASTIC RSI ==========
    if not pd.isna(last['stoch_rsi_k']):
        if last['stoch_rsi_k'] < 20 and is_uptrend:
            score += 8
        elif last['stoch_rsi_k'] > 80:
            score -= 8
    
    # ========== 11. SELL PROTECTION ==========
    if last['close'] < last['ema50']:
        score -= 15  # Harga di bawah EMA50, hati-hati
    
    # ========== 12. TREND DIRECTION RULE ==========
    if not is_uptrend and score > 50:
        score = 50  # Tidak boleh BUY di downtrend
    
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
    """Hitung level keyakinan sinyal"""
    if score >= 80:
        return "🔥 HIGH CONFIDENCE", "green"
    elif score >= 60:
        return "✅ MEDIUM CONFIDENCE", "lightgreen"
    elif score >= 40:
        return "⚠️ LOW CONFIDENCE", "orange"
    else:
        return "❌ VERY LOW CONFIDENCE", "red"


def multi_timeframe_analysis(symbol):
    """Analisis multi timeframe dengan bobot baru (5m dihilangkan)"""
    timeframes = {
        "15m": "15m", 
        "30m": "30m",
        "1h": "60m",
        "1d": "1d"
    }
    
    # Bobot baru: 5m dihilangkan untuk mengurangi noise
    weights = {
        "15m": 0.15,
        "30m": 0.25,
        "1h": 0.30,
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
    """Scan 20 saham likuid"""
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
    harga = last['close']
    
    adx_text = ""
    if 'adx' in last and not pd.isna(last['adx']):
        if last['adx'] >= 25:
            adx_text = f"✅ ADX: {last['adx']:.1f} (Tren Kuat)"
        elif last['adx'] >= 20:
            adx_text = f"📈 ADX: {last['adx']:.1f} (Tren Mulai)"
        else:
            adx_text = f"⚠️ ADX: {last['adx']:.1f} (Tren Lemah)"
    
    if score >= 60:  # BUY
        entry = harga
        stop_loss = harga - (atr * 1.5)
        target1 = harga + (atr * 1.5)
        target2 = harga + (atr * 3)
        return f"""
📈 **REKOMENDASI BUY** (Harga diprediksi NAIK)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
💰 **Entry (Beli):** Rp{entry:,.0f}
🛑 **Stop Loss (Rugi):** Rp{stop_loss:,.0f} (turun {((entry - stop_loss)/entry*100):.1f}%)
🎯 **Target 1 (Jual):** Rp{target1:,.0f} (naik {((target1 - entry)/entry*100):.1f}%)
🎯 **Target 2 (Jual):** Rp{target2:,.0f} (naik {((target2 - entry)/entry*100):.1f}%)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{adx_text}
🛡️ **Support:** Rp{support:,.0f} (harga terendah)
🚧 **Resistance:** Rp{resistance:,.0f} (harga tertinggi)
"""
    
    elif score <= 35:  # SELL
        entry = harga
        stop_loss = harga + (atr * 1.5)
        target1 = harga - (atr * 1.5)
        target2 = harga - (atr * 3)
        return f"""
📉 **REKOMENDASI SELL** (Harga diprediksi TURUN)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
💰 **Entry (Jual):** Rp{entry:,.0f}
🛑 **Stop Loss (Rugi):** Rp{stop_loss:,.0f} (naik {((stop_loss - entry)/entry*100):.1f}%)
🎯 **Target 1 (Beli Kembali):** Rp{target1:,.0f} (turun {((entry - target1)/entry*100):.1f}%)
🎯 **Target 2 (Beli Kembali):** Rp{target2:,.0f} (turun {((entry - target2)/entry*100):.1f}%)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{adx_text}
🛡️ **Support:** Rp{support:,.0f} (harga terendah)
🚧 **Resistance:** Rp{resistance:,.0f} (harga tertinggi)
"""
    
    else:
        return f"""
⏳ **REKOMENDASI WAIT/HOLD** (Tunggu sinyal lebih jelas)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📊 **Harga saat ini:** Rp{harga:,.0f}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{adx_text}
🛡️ **Support:** Rp{support:,.0f}
🚧 **Resistance:** Rp{resistance:,.0f}
💡 **Saran:** Harga mendekati Support → potensi BUY
         Harga mendekati Resistance → potensi SELL
"""
