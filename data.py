import yfinance as yf
import pandas as pd
import numpy as np

def get_data(symbol="BBCA.JK", interval="5m"):
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
    if df.empty:
        return df

    # EMA
    df['ema20'] = df['close'].ewm(span=20, adjust=False).mean()
    df['ema50'] = df['close'].ewm(span=50, adjust=False).mean()
    df['ema20_slope'] = df['ema20'].diff(5) / df['ema20'].shift(5) * 100
    df['ema50_slope'] = df['ema50'].diff(5) / df['ema50'].shift(5) * 100

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

    # Volume MA
    df['volume_ma20'] = df['volume'].rolling(window=20).mean()

    # ATR
    high_low = df['high'] - df['low']
    high_close = abs(df['high'] - df['close'].shift())
    low_close = abs(df['low'] - df['close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr'] = tr.rolling(window=14).mean()

    # Support & Resistance (rolling 20)
    df['resistance'] = df['high'].rolling(window=20).max()
    df['support'] = df['low'].rolling(window=20).min()

    # Breakout detector
    df['breakout_high'] = df['close'] > df['high'].rolling(20).max().shift(1)
    df['breakout_low'] = df['close'] < df['low'].rolling(20).min().shift(1)

    # Candle body & range
    df['candle_body'] = abs(df['close'] - df['open'])
    df['candle_range'] = df['high'] - df['low']
    df['candle_strength'] = df['candle_body'] / df['candle_range'].replace(0, np.nan)

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

    # Upgrade: Trend Quality
    df['trend_strength'] = abs(df['ema20'] - df['ema50']) / df['close']
    df['trend_quality'] = df['trend_strength'] / df['atr'].replace(0, np.nan)

    # Upgrade: Upper/Lower wick (liquidity trap)
    df['upper_wick'] = df['high'] - df[['close', 'open']].max(axis=1)
    df['lower_wick'] = df[['close', 'open']].min(axis=1) - df['low']

    return df

def detect_market_regime(df):
    if df.empty or len(df) < 30:
        return "sideways", 0
    last = df.iloc[-1]
    adx = last['adx'] if not pd.isna(last['adx']) else 20
    volatility = last['atr'] / last['close'] if last['close'] > 0 else 0.02
    avg_range = (df['high'] - df['low']).tail(20).mean()
    current_range = last['high'] - last['low']
    range_ratio = current_range / avg_range if avg_range > 0 else 1

    if adx >= 25 and volatility < 0.025:
        regime = "trending_strong"
        regime_score = 1.2
    elif adx >= 20 and volatility < 0.03:
        regime = "trending_weak"
        regime_score = 1.1
    elif adx < 20 and volatility < 0.02:
        regime = "sideways"
        regime_score = 0.7
    elif range_ratio > 1.5 or volatility > 0.035:
        regime = "volatile"
        regime_score = 0.5
    else:
        regime = "neutral"
        regime_score = 1.0
    return regime, regime_score

def get_market_regime_text(regime):
    regime_text = {
        "trending_strong": "📈 Trending Kuat (Agresif)",
        "trending_weak": "📈 Trending Lemah (Normal)",
        "sideways": "🔄 Sideways (Hati-hati)",
        "volatile": "🎢 Volatile (Kurangi Eksposur)",
        "neutral": "⚖️ Netral"
    }
    return regime_text.get(regime, "⚖️ Netral")

def calculate_score(df):
    if df.empty or len(df) < 50:
        return 0
    score = 0
    last = df.iloc[-1]
    prev = df.iloc[-2] if len(df) > 1 else last

    # Market regime
    regime, regime_score = detect_market_regime(df)

    # Volatility Regime Fix
    volatility = last['atr'] / last['close'] if last['close'] > 0 else 0.02
    if volatility < 0.007:
        score -= 10
    elif volatility > 0.04:
        score -= 8

    # Trend basics
    is_uptrend = last['ema20'] > last['ema50']
    is_price_above_ema = last['close'] > last['ema20']
    if not is_uptrend:
        score -= 20
    if is_price_above_ema:
        score += 8
    else:
        score -= 8

    # EMA slope
    ema20_slope = last['ema20_slope'] if not pd.isna(last['ema20_slope']) else 0
    ema50_slope = last['ema50_slope'] if not pd.isna(last['ema50_slope']) else 0
    if ema20_slope > 0.1 and ema50_slope > 0.05:
        score += 10
    elif ema20_slope < -0.1:
        score -= 10

    # Trend Quality
    if not pd.isna(last.get('trend_quality', np.nan)):
        if last['trend_quality'] > 1.5:
            score += 12
        elif last['trend_quality'] < 0.7:
            score -= 12

    # Smart RSI
    if last['rsi'] < 30:
        if is_uptrend:
            score += 10
        else:
            score += 2
    elif last['rsi'] > 70:
        score -= 10
    elif last['rsi'] > 50:
        score += 5

    # MACD
    if last['macd'] > last['macd_signal']:
        score += 12
    if last['macd_histogram'] > 0:
        score += 5

    # Bollinger
    if last['close'] <= last['bb_lower']:
        if last['rsi'] > 30:
            score += 6
    elif last['close'] >= last['bb_upper']:
        score -= 10
    elif last['close'] > last['bb_middle']:
        score += 5

    # Volume Validity
    volume_surge = last['volume'] > last['volume_ma20'] * 1.2
    volume_valid = (last['volume'] > last['volume_ma20'] * 1.3) and (last['close'] > prev['close'])
    if volume_valid:
        score += 10
    else:
        score -= 5

    volume_price_up = last['close'] > prev['close'] and last['volume'] > prev['volume']
    if volume_price_up:
        score += 5

    # Breakout & Fake Breakout Killer
    breakout_high = last.get('breakout_high', False)
    breakout_low = last.get('breakout_low', False)
    candle_strength = last.get('candle_strength', 0.5)
    if pd.isna(candle_strength):
        candle_strength = 0.5

    valid_breakout = breakout_high and candle_strength > 0.6 and volume_surge and last['close'] > last['ema20']
    valid_breakdown = breakout_low and candle_strength > 0.6 and volume_surge and last['close'] < last['ema20']

    fake_breakout = breakout_high and (last['close'] < last['ema20'])
    if fake_breakout:
        score -= 15

    if valid_breakout:
        score += 18
    elif breakout_high and volume_surge:
        score += 12
    elif breakout_high:
        score += 5
    elif valid_breakdown:
        score -= 18
    elif breakout_low and volume_surge:
        score -= 12

    if breakout_high and last['close'] < last['ema20']:
        score -= 12
    if breakout_low and last['close'] > last['ema20']:
        score += 5

    # Risk-Reward Filter
    support = last['support']
    resistance = last['resistance']
    close = last['close']
    rr_denom = (close - support) if (close - support) > 0 else 0.01
    rr = (resistance - close) / rr_denom
    if rr < 1.3:
        score -= 15
    elif rr > 2:
        score += 10

    # Liquidity Trap (Upper Wick)
    upper_wick = last.get('upper_wick', 0)
    candle_body = last.get('candle_body', 0.01)
    if not pd.isna(upper_wick) and not pd.isna(candle_body) and candle_body > 0:
        if upper_wick > candle_body * 2:
            score -= 10

    # Support/Resistance proximity
    if last['close'] <= last['support'] * 1.02:
        score += 12
    elif last['close'] >= last['resistance'] * 0.98:
        score -= 12
    else:
        score += 3

    # ADX
    if not pd.isna(last['adx']):
        if last['adx'] >= 25:
            if is_uptrend:
                score += 15
            else:
                score -= 10
        elif last['adx'] >= 20:
            if is_uptrend:
                score += 8
            else:
                score -= 5
        elif last['adx'] < 18:
            score -= 15

    if not pd.isna(last['adx']):
        if last['adx'] > 30 and is_uptrend and last['close'] > last['ema20']:
            score += 10

    # Stochastic RSI
    if not pd.isna(last.get('stoch_rsi_k', np.nan)):
        if last['stoch_rsi_k'] < 20 and is_uptrend:
            score += 8
        elif last['stoch_rsi_k'] > 80:
            score -= 8

    # Sell protection
    if last['close'] < last['ema50']:
        score -= 15

    # Regime multiplier
    score = score * regime_score

    # Trend direction rule
    if not is_uptrend and score > 50:
        score = 50
    if regime == "sideways" and score > 60:
        score = 60
    elif regime == "volatile" and score > 40:
        score = 40

    return min(100, max(0, int(score)))

def get_signal_label(score):
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
    if score >= 80:
        return "🔥 HIGH CONFIDENCE", "green"
    elif score >= 60:
        return "✅ MEDIUM CONFIDENCE", "lightgreen"
    elif score >= 40:
        return "⚠️ LOW CONFIDENCE", "orange"
    else:
        return "❌ VERY LOW CONFIDENCE", "red"

def multi_timeframe_analysis(symbol):
    timeframes = {"15m": "15m", "30m": "30m", "1h": "60m", "1d": "1d"}
    weights = {"15m": 0.15, "30m": 0.25, "1h": 0.30, "1d": 0.30}
    results = {}
    weighted_score = 0
    all_scores = []
    for label, tf in timeframes.items():
        df = get_data(symbol, tf)
        if df.empty:
            results[label] = 0
            continue
        df = add_indicators(df)
        score = calculate_score(df)
        results[label] = score
        weighted_score += score * weights[label]
        all_scores.append(score)
    if len(all_scores) >= 3:
        buy_count = sum(1 for s in all_scores if s >= 60)
        sell_count = sum(1 for s in all_scores if s <= 35)
        if buy_count >= 3:
            weighted_score *= 1.15
        elif sell_count >= 3:
            weighted_score *= 0.85
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
        signal = "BUY" if avg_score >= 60 else "WAIT" if avg_score >= 40 else "SELL"
        results.append({"Kode": saham.replace(".JK", ""), "Score": round(avg_score, 2), "Sinyal": signal})
    return sorted(results, key=lambda x: x["Score"], reverse=True)

def get_trading_recommendation(score, df):
    if df.empty:
        return "Tidak ada data"
    last = df.iloc[-1]
    support = last['support']
    resistance = last['resistance']
    atr = last['atr'] if last['atr'] > 0 else last['close'] * 0.02
    harga = last['close']
    regime, _ = detect_market_regime(df)
    regime_text = get_market_regime_text(regime)
    adx_text = ""
    if 'adx' in last and not pd.isna(last['adx']):
        if last['adx'] >= 25:
            adx_text = f"✅ ADX: {last['adx']:.1f} (Tren Kuat)"
        elif last['adx'] >= 20:
            adx_text = f"📈 ADX: {last['adx']:.1f} (Tren Mulai)"
        else:
            adx_text = f"⚠️ ADX: {last['adx']:.1f} (Tren Lemah)"
    if score >= 60:
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
📊 **Market Regime:** {regime_text}
🛡️ **Support:** Rp{support:,.0f}
🚧 **Resistance:** Rp{resistance:,.0f}
"""
    elif score <= 35:
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
📊 **Market Regime:** {regime_text}
🛡️ **Support:** Rp{support:,.0f}
🚧 **Resistance:** Rp{resistance:,.0f}
"""
    else:
        return f"""
⏳ **REKOMENDASI WAIT/HOLD** (Tunggu sinyal lebih jelas)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📊 **Harga saat ini:** Rp{harga:,.0f}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{adx_text}
📊 **Market Regime:** {regime_text}
🛡️ **Support:** Rp{support:,.0f}
🚧 **Resistance:** Rp{resistance:,.0f}
💡 **Saran:** Harga mendekati Support → potensi BUY
         Harga mendekati Resistance → potensi SELL
"""

def backtest_strategy(df, capital_initial=100_000_000, buy_threshold=60, sell_threshold=40):
    """Backtest sederhana berdasarkan skor."""
    if df.empty or len(df) < 50:
        return {"return": 0, "winrate": 0, "trades": 0, "final_capital": capital_initial}
    capital = capital_initial
    position = 0
    entry_price = 0
    trades = []
    for i in range(50, len(df)):
        sub_df = df.iloc[:i+1].copy()
        score = calculate_score(sub_df)
        price = df.iloc[i]['close']
        if score >= buy_threshold and position == 0:
            position = capital / price
            entry_price = price
            capital = 0
        elif score <= sell_threshold and position > 0:
            pnl = (price - entry_price) / entry_price
            trades.append(pnl)
            capital = position * price
            position = 0
    if position > 0:
        capital = position * df.iloc[-1]['close']
    total_return_pct = (capital - capital_initial) / capital_initial * 100
    winrate = (sum(1 for t in trades if t > 0) / len(trades) * 100) if trades else 0
    return {
        "return": round(total_return_pct, 2),
        "winrate": round(winrate, 2),
        "trades": len(trades),
        "final_capital": round(capital, 0)
    }
