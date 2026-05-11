import pandas as pd
import yfinance as yf
import numpy as np
import time
from datetime import datetime

_data_cache = {}
_last_request_time = 0
_MIN_REQUEST_INTERVAL = 1

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
    
    if symbol == "^JKSE" or symbol == "JKSE":
        symbol = "^JKSE"
    elif not symbol.endswith('.JK') and symbol not in ["^JKSE"]:
        symbol = f"{symbol}.JK"
    
    cache_key = f"{symbol}_{timeframe}"
    
    if cache_key in _data_cache:
        cached_time, cached_data = _data_cache[cache_key]
        if (datetime.now() - cached_time).seconds < 60:
            return cached_data
    
    try:
        interval_map = {"5m": "5m", "15m": "15m", "30m": "30m", "60m": "60m", "1d": "1d"}
        
        if timeframe in ["5m", "15m", "30m", "60m"]:
            period = "7d"
        else:
            period = "3mo"
        
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
    
    # Basic EMA
    df['ema10'] = df['close'].ewm(span=10, adjust=False).mean()
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
    
    # Volume
    df['volume_ma20'] = df['volume'].rolling(window=20).mean()
    df['volume_ratio'] = df['volume'] / df['volume_ma20']
    
    # ATR
    high_low = df['high'] - df['low']
    high_close = abs(df['high'] - df['close'].shift())
    low_close = abs(df['low'] - df['close'].shift())
    df['tr'] = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr'] = df['tr'].rolling(window=14).mean()
    
    # ADX
    plus_dm = df['high'].diff()
    minus_dm = df['low'].diff()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm > 0] = 0
    
    plus_di = 100 * (plus_dm.ewm(alpha=1/14).mean() / df['atr'])
    minus_di = 100 * (abs(minus_dm).ewm(alpha=1/14).mean() / df['atr'])
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    df['adx'] = dx.rolling(window=14).mean()
    
    # Supertrend
    atr_supertrend = df['atr'].rolling(window=10).mean()
    hl_avg = (df['high'] + df['low']) / 2
    df['upper_band'] = hl_avg + (3 * atr_supertrend)
    df['lower_band'] = hl_avg - (3 * atr_supertrend)
    
    df['supertrend'] = 0
    df['supertrend_direction'] = 1
    for i in range(1, len(df)):
        if df['close'].iloc[i] > df['upper_band'].iloc[i-1]:
            df.loc[df.index[i], 'supertrend_direction'] = 1
        elif df['close'].iloc[i] < df['lower_band'].iloc[i-1]:
            df.loc[df.index[i], 'supertrend_direction'] = -1
        else:
            df.loc[df.index[i], 'supertrend_direction'] = df['supertrend_direction'].iloc[i-1]
        
        if df['supertrend_direction'].iloc[i] == 1:
            df.loc[df.index[i], 'supertrend'] = df['lower_band'].iloc[i]
        else:
            df.loc[df.index[i], 'supertrend'] = df['upper_band'].iloc[i]
    
    # VWAP
    df['vwap'] = (df['volume'] * (df['high'] + df['low'] + df['close']) / 3).cumsum() / df['volume'].cumsum()
    
    return df

# ========== 1. MARKET STRUCTURE (BOS) ==========
def detect_market_structure(df):
    """Deteksi Break of Structure (BOS) - Smart Money Concept"""
    if df.empty or len(df) < 20:
        return "RANGE", 0, "Data tidak cukup"
    
    last = df.iloc[-1]
    prev_5 = df.iloc[-6:-1]
    prev_10 = df.iloc[-11:-1]
    prev_20 = df.iloc[-21:-1]
    
    # Swing High/Low detection
    recent_high = prev_10['high'].max()
    recent_low = prev_10['low'].min()
    higher_high = prev_20['high'].max()
    lower_low = prev_20['low'].min()
    
    structure = "RANGE"
    confidence = 0
    signals = []
    
    # Bullish BOS (Higher High)
    if last['close'] > recent_high and last['close'] > higher_high:
        structure = "BULLISH_BOS"
        confidence += 40
        signals.append("Break of Structure UP")
    
    # Bearish BOS (Lower Low)
    elif last['close'] < recent_low and last['close'] < lower_low:
        structure = "BEARISH_BOS"
        confidence += 40
        signals.append("Break of Structure DOWN")
    
    # Higher Low (bullish structure)
    if last['low'] > prev_5['low'].min():
        confidence += 15
        signals.append("Higher Low")
    
    # Lower High (bearish structure)
    if last['high'] < prev_5['high'].max():
        confidence += 15
        signals.append("Lower High")
    
    # Trend strength via Supertrend
    if 'supertrend_direction' in last:
        if last['supertrend_direction'] == 1 and structure in ["BULLISH_BOS", "RANGE"]:
            structure = "BULLISH_TREND"
            confidence += 20
        elif last['supertrend_direction'] == -1 and structure in ["BEARISH_BOS", "RANGE"]:
            structure = "BEARISH_TREND"
            confidence += 20
    
    desc = " | ".join(signals) if signals else "No structure signal"
    return structure, min(100, confidence), desc

# ========== 2. SMART MONEY VOLUME ==========
def detect_smart_money_volume(df):
    """Deteksi Accumulation / Distribution (Smart Money)"""
    if df.empty or len(df) < 30:
        return "NEUTRAL", 0, "Data tidak cukup"
    
    last = df.iloc[-1]
    prev_5 = df.iloc[-6:-1]
    prev_10 = df.iloc[-11:-1]
    prev_20 = df.iloc[-21:-1]
    
    signals = []
    confidence = 0
    result = "NEUTRAL"
    
    # 1. Accumulation: Harga sideways + Volume naik
    price_range = (prev_10['high'].max() - prev_10['low'].min()) / prev_10['close'].mean()
    volume_increasing = prev_5['volume'].mean() > prev_20['volume'].mean() * 1.2
    price_sideways = price_range < 0.03  # 3% range
    
    if price_sideways and volume_increasing:
        result = "ACCUMULATION"
        confidence += 40
        signals.append("Volume naik, harga sideways = Akumulasi")
    
    # 2. Distribution: Harga stagnan + Volume tinggi
    volume_high = last['volume_ratio'] > 1.5
    price_stagnant = abs(last['close'] - prev_10['close'].mean()) / prev_10['close'].mean() < 0.02
    
    if volume_high and price_stagnant:
        result = "DISTRIBUTION"
        confidence += 35
        signals.append("Volume tinggi, harga stagnan = Distribusi")
    
    # 3. Volume Spike + Close near low = Distribution
    if last['volume_ratio'] > 2.0 and last['close'] < (last['high'] + last['low']) / 2:
        confidence += 20
        signals.append("Volume spike + close bawah = tekanan jual")
    
    # 4. Volume Spike + Close near high = Accumulation
    if last['volume_ratio'] > 2.0 and last['close'] > (last['high'] + last['low']) / 2:
        confidence += 20
        signals.append("Volume spike + close atas = tekanan beli")
    
    # 5. OBV divergence (jika ada OBV)
    if 'obv' in df.columns:
        obv_up = df['obv'].iloc[-1] > df['obv'].iloc[-5]
        price_up = last['close'] > prev_5['close'].mean()
        if obv_up and not price_up:
            signals.append("OBV bullish divergence")
            confidence += 15
    
    desc = " | ".join(signals) if signals else "No smart money signal"
    return result, min(100, confidence), desc

# ========== 3. LIQUIDITY SWEEP / SFP ==========
def detect_liquidity_sweep(df, lookback=20):
    """Deteksi Liquidity Sweep / Swing Failure Pattern (SFP)"""
    if df.empty or len(df) < lookback + 5:
        return False, 0, "NONE"
    
    last = df.iloc[-1]
    prev = df.iloc[-2]
    prev_highs = df.iloc[-lookback:-1]['high']
    prev_lows = df.iloc[-lookback:-1]['low']
    resistance = prev_highs.max()
    support = prev_lows.min()
    
    signals = []
    confidence = 0
    sweep_type = "NONE"
    
    # Bullish SFP: Sweep below support, then close above
    if last['low'] < support and last['close'] > support:
        sweep_type = "BULLISH_SFP"
        confidence += 50
        signals.append("Liquidity sweep bawah + reversal")
    
    # Bearish SFP: Sweep above resistance, then close below
    elif last['high'] > resistance and last['close'] < resistance:
        sweep_type = "BEARISH_SFP"
        confidence += 50
        signals.append("Liquidity sweep atas + reversal")
    
    # Fake breakout detection
    if prev['high'] > resistance and last['close'] < resistance:
        confidence += 30
        signals.append("Fake breakout (false breakout)")
        if sweep_type == "NONE":
            sweep_type = "FAKE_BREAKOUT"
    
    desc = " | ".join(signals) if signals else "No sweep detected"
    return sweep_type != "NONE", confidence, sweep_type, desc

# ========== 4. CANDLESTICK PATTERNS ==========
def detect_candlestick_pattern(df):
    """Deteksi candlestick pattern (Engulfing, Pinbar, Marubozu)"""
    if df.empty or len(df) < 3:
        return "NONE", 0, "Data tidak cukup"
    
    last = df.iloc[-1]
    prev = df.iloc[-2]
    
    body = abs(last['close'] - last['open'])
    upper_wick = last['high'] - max(last['open'], last['close'])
    lower_wick = min(last['open'], last['close']) - last['low']
    candle_range = last['high'] - last['low']
    
    signals = []
    confidence = 0
    pattern = "NONE"
    
    # Marubozu (no wick)
    if body / candle_range > 0.85:
        if last['close'] > last['open']:
            pattern = "BULLISH_MARUBOZU"
            confidence += 30
            signals.append("Bullish Marubozu")
        else:
            pattern = "BEARISH_MARUBOZU"
            confidence -= 30
            signals.append("Bearish Marubozu")
    
    # Pinbar / Hammer
    if lower_wick > body * 2 and upper_wick < body:
        pattern = "HAMMER"
        confidence += 35
        signals.append("Hammer / Pinbar")
    
    # Shooting Star
    if upper_wick > body * 2 and lower_wick < body:
        pattern = "SHOOTING_STAR"
        confidence -= 35
        signals.append("Shooting Star")
    
    # Bullish Engulfing
    if (last['close'] > last['open'] and 
        prev['close'] < prev['open'] and
        last['close'] > prev['open'] and 
        last['open'] < prev['close']):
        pattern = "BULLISH_ENGULFING"
        confidence += 40
        signals.append("Bullish Engulfing")
    
    # Bearish Engulfing
    if (last['close'] < last['open'] and 
        prev['close'] > prev['open'] and
        last['open'] > prev['close'] and 
        last['close'] < prev['open']):
        pattern = "BEARISH_ENGULFING"
        confidence -= 40
        signals.append("Bearish Engulfing")
    
    desc = " | ".join(signals) if signals else "No pattern"
    return pattern, confidence, desc

# ========== 5. PIVOT SUPPORT RESISTANCE ==========
def get_pivot_sr(df, lookback=20):
    """Hitung Support & Resistance berbasis pivot (lebih akurat)"""
    if df.empty or len(df) < lookback + 5:
        return df.iloc[-1]['support'] if 'support' in df.columns else 0, df.iloc[-1]['resistance'] if 'resistance' in df.columns else 0, 0, 0
    
    last = df.iloc[-1]
    
    # Pivot Points
    pivot = (last['high'] + last['low'] + last['close']) / 3
    r1 = (2 * pivot) - last['low']
    r2 = pivot + (last['high'] - last['low'])
    s1 = (2 * pivot) - last['high']
    s2 = pivot - (last['high'] - last['low'])
    
    # Rolling support/resistance
    rolling_support = df['low'].rolling(window=lookback).min().iloc[-1]
    rolling_resistance = df['high'].rolling(window=lookback).max().iloc[-1]
    
    # Fibonacci levels
    high_20 = df['high'].iloc[-20:].max()
    low_20 = df['low'].iloc[-20:].min()
    range_20 = high_20 - low_20
    fib_382 = low_20 + (range_20 * 0.382)
    fib_618 = low_20 + (range_20 * 0.618)
    
    return rolling_support, rolling_resistance, pivot, r1, r2, s1, s2, fib_382, fib_618

# ========== 6. MARKET REGIME ==========
def detect_market_regime(df):
    """Deteksi regime pasar: Trending, Sideways, Volatile, Panic"""
    if df.empty or len(df) < 30:
        return "UNKNOWN", 0, "Data tidak cukup"
    
    last = df.iloc[-1]
    prev_5 = df.iloc[-6:-1]
    prev_20 = df.iloc[-21:-1]
    
    adx = last.get('adx', 0)
    atr_pct = (last['atr'] / last['close']) * 100 if last['close'] > 0 else 0
    volume_ratio = last['volume_ratio']
    daily_change = (last['close'] - prev_20['close'].iloc[0]) / prev_20['close'].iloc[0] * 100
    
    regime = "NEUTRAL"
    confidence = 50
    desc = ""
    
    # Trending
    if adx >= 25:
        regime = "TRENDING"
        confidence = 70 + min(10, adx - 25)
        desc = f"Tren kuat (ADX {adx:.0f})"
    
    # Sideways
    elif adx < 20:
        regime = "SIDEWAYS"
        confidence = 40
        desc = f"Pasar ranging (ADX {adx:.0f})"
    
    # Volatile
    if atr_pct > 4:
        regime = "VOLATILE"
        confidence = 60
        desc = f"Volatilitas tinggi (ATR {atr_pct:.1f}%)"
    
    # Panic
    if daily_change < -5 or volume_ratio > 2.5:
        regime = "PANIC"
        confidence = 80
        desc = "Kondisi panic selling! Hati-hati!"
    
    # Strong trending
    if adx >= 35 and atr_pct > 3:
        regime = "STRONG_TRENDING"
        confidence = 85
        desc = f"Tren sangat kuat (ADX {adx:.0f}, Vol {atr_pct:.1f}%)"
    
    return regime, confidence, desc

# ========== 7. ENTRY, SL, TP DARI INDIKATOR (LENGKAP) ==========
def calculate_entry_sl_tp(df, capital=100000000, risk_percent=2):
    """Entry, Stop Loss, Take Profit dari Market Structure, Smart Money, dll"""
    if df.empty or len(df) < 30:
        return None, None, None, 0, 0, "NO_SETUP", 0
    
    last = df.iloc[-1]
    atr = last.get('atr', last['close'] * 0.02)
    
    # Get all detections
    structure, struct_conf, struct_desc = detect_market_structure(df)
    smart_money, sm_conf, sm_desc = detect_smart_money_volume(df)
    is_sweep, sweep_conf, sweep_type, sweep_desc = detect_liquidity_sweep(df)
    pattern, pattern_conf, pattern_desc = detect_candlestick_pattern(df)
    regime, regime_conf, regime_desc = detect_market_regime(df)
    support, resistance, pivot, r1, r2, s1, s2, fib_382, fib_618 = get_pivot_sr(df)
    
    trend_up = last['ema20'] > last['ema50']
    momentum_bullish = last['macd_histogram'] > 0 and last['rsi'] > 50
    volume_spike = last['volume_ratio'] >= 1.5
    strong_trend = last['adx'] >= 25
    supertrend_bullish = last.get('supertrend_direction', 0) == 1
    
    entry_price = None
    stop_loss = None
    take_profit = None
    setup_name = "NO_SETUP"
    confidence = 0
    signals = []
    
    # ========== PRIORITY 1: BULLISH BOS + SMART MONEY + SWEEP ==========
    if structure in ["BULLISH_BOS", "BULLISH_TREND"] and smart_money == "ACCUMULATION" and is_sweep:
        entry_price = last['close'] * 1.001
        stop_loss = min(last['low'], s1) - (0.5 * atr)
        take_profit = last['close'] + (3 * atr)
        setup_name = "SMART_MONEY_SNIPER_BUY"
        confidence = 85 + min(15, sm_conf//2)
        signals = [struct_desc, sm_desc, sweep_desc]
    
    # ========== PRIORITY 2: BULLISH BOS + VOLUME SPIKE ==========
    elif structure in ["BULLISH_BOS", "BULLISH_TREND"] and volume_spike:
        entry_price = last['close'] * 1.001
        stop_loss = entry_price - (2 * atr)
        take_profit = entry_price + (3 * atr)
        setup_name = "BREAKOUT_BUY"
        confidence = struct_conf + 10
        signals = [struct_desc]
    
    # ========== PRIORITY 3: BULLISH CANDLESTICK PATTERN ==========
    elif pattern in ["BULLISH_ENGULFING", "HAMMER"] and trend_up:
        entry_price = last['close']
        stop_loss = entry_price - (1.5 * atr)
        take_profit = entry_price + (3 * atr)
        setup_name = f"{pattern}_BUY"
        confidence = 65 + abs(pattern_conf)//2
        signals = [pattern_desc]
    
    # ========== PRIORITY 4: BULLISH SFP (Liquidity Sweep) ==========
    elif is_sweep and sweep_type == "BULLISH_SFP":
        entry_price = last['close']
        stop_loss = entry_price - (1.5 * atr)
        take_profit = entry_price + (2.5 * atr)
        setup_name = "LIQUIDITY_SWEEP_BUY"
        confidence = sweep_conf
        signals = [sweep_desc]
    
    # ========== PRIORITY 5: TREND BUY ==========
    elif trend_up and strong_trend and momentum_bullish:
        entry_price = last['close']
        stop_loss = entry_price - (2 * atr)
        take_profit = entry_price + (3 * atr)
        setup_name = "TREND_BUY"
        confidence = 70
        signals = ["Strong uptrend"]
    
    # ========== PRIORITY 6: PULLBACK TO SUPPORT ==========
    elif trend_up and last['close'] <= support * 1.02:
        entry_price = support
        stop_loss = entry_price - (1.5 * atr)
        take_profit = entry_price + (2.5 * atr)
        setup_name = "SUPPORT_BOUNCE_BUY"
        confidence = 65
        signals = ["Pullback to support"]
    
    # ========== BEARISH SIGNALS ==========
    elif structure in ["BEARISH_BOS", "BEARISH_TREND"] and smart_money == "DISTRIBUTION":
        entry_price = last['close'] * 0.999
        stop_loss = entry_price + (2 * atr)
        take_profit = entry_price - (3 * atr)
        setup_name = "SMART_MONEY_SELL"
        confidence = 80
        signals = [struct_desc, sm_desc]
    
    elif pattern in ["BEARISH_ENGULFING", "SHOOTING_STAR"]:
        entry_price = last['close']
        stop_loss = entry_price + (1.5 * atr)
        take_profit = entry_price - (3 * atr)
        setup_name = f"{pattern}_SELL"
        confidence = 65
        signals = [pattern_desc]
    
    else:
        return None, None, None, 0, 0, "NO_SETUP", 0
    
    if entry_price:
        risk = abs(entry_price - stop_loss)
        reward = abs(take_profit - entry_price)
        rr_ratio = reward / risk if risk > 0 else 0
        
        risk_amount = capital * (risk_percent / 100)
        risk_per_share = risk
        shares = int(risk_amount / risk_per_share) if risk_per_share > 0 else 0
        max_shares = int((capital * 0.5) / entry_price)
        shares = min(shares, max_shares)
        
        return entry_price, stop_loss, take_profit, shares, rr_ratio, setup_name, confidence, signals
    
    return None, None, None, 0, 0, "NO_SETUP", 0, []

# ========== 8. MULTI TIMEFRAME ALIGNMENT ==========
def get_multi_timeframe_alignment(symbol, capital=100000000, risk_percent=2):
    """Analisis multi timeframe dengan alignment"""
    timeframes = {
        "daily": "1d",
        "hourly": "60m",
        "fifteen": "15m"
    }
    
    results = {}
    alignment_score = 0
    signals = []
    
    for name, tf in timeframes.items():
        df = get_data(symbol, tf)
        if not df.empty and len(df) > 30:
            df = add_indicators(df)
            entry, sl, tp, shares, rr, setup, conf, sigs = calculate_entry_sl_tp(df, capital, risk_percent)
            
            # Deteksi direction
            last = df.iloc[-1]
            direction = "BULLISH" if last['ema20'] > last['ema50'] else "BEARISH"
            structure, struct_conf, _ = detect_market_structure(df)
            regime, _, _ = detect_market_regime(df)
            
            results[name] = {
                "direction": direction,
                "structure": structure,
                "regime": regime,
                "setup": setup,
                "confidence": conf,
                "entry": entry,
                "stop_loss": sl,
                "take_profit": tp,
                "rr": rr,
                "score": conf
            }
            
            if direction == "BULLISH":
                signals.append(f"{name}: BULLISH")
            else:
                signals.append(f"{name}: BEARISH")
    
    # Hitung alignment
    if len(results) == 3:
        directions = [results[tf]["direction"] for tf in results]
        if all(d == "BULLISH" for d in directions):
            alignment_score = 90
            alignment = "FULL_BULLISH_ALIGNMENT"
        elif all(d == "BEARISH" for d in directions):
            alignment_score = 90
            alignment = "FULL_BEARISH_ALIGNMENT"
        elif directions[0] == "BULLISH" and directions[1] == "BULLISH":
            alignment_score = 75
            alignment = "BULLISH_DAILY_HOURLY"
        elif directions[0] == "BEARISH" and directions[1] == "BEARISH":
            alignment_score = 75
            alignment = "BEARISH_DAILY_HOURLY"
        else:
            alignment_score = 40
            alignment = "MIXED_ALIGNMENT"
    else:
        alignment_score = 50
        alignment = "INSUFFICIENT_DATA"
    
    return results, alignment, alignment_score, signals

def get_ihsg_trend():
    try:
        ihsg = get_data("^JKSE", "1d")
        if ihsg.empty or len(ihsg) < 20:
            return "NEUTRAL", 50, "Data IHSG tidak cukup"
        
        ihsg = add_indicators(ihsg)
        last = ihsg.iloc[-1]
        prev = ihsg.iloc[-2]
        
        if last['close'] > last['ema20'] > last['ema50']:
            trend = "BULLISH"
            score = 70
        elif last['close'] < last['ema20'] < last['ema50']:
            trend = "BEARISH"
            score = 30
        else:
            trend = "SIDEWAYS"
            score = 50
        
        daily_change = ((last['close'] - prev['close']) / prev['close']) * 100
        return trend, score, f"IHSG {trend} ({daily_change:+.1f}%)"
    except:
        return "NEUTRAL", 50, "IHSG data unavailable"

def calculate_confidence_score(df, ihsg_score=50):
    if df.empty or len(df) < 30:
        return 50, [], "NORMAL"
    
    last = df.iloc[-1]
    factors = []
    total = 50
    
    # Structure (20%)
    structure, struct_conf, _ = detect_market_structure(df)
    if "BULLISH" in structure:
        total += struct_conf * 0.2
        factors.append(("Structure", struct_conf * 0.2, structure))
    elif "BEARISH" in structure:
        total -= struct_conf * 0.2
        factors.append(("Structure", -struct_conf * 0.2, structure))
    
    # Smart Money (15%)
    sm, sm_conf, _ = detect_smart_money_volume(df)
    if sm == "ACCUMULATION":
        total += sm_conf * 0.15
        factors.append(("Smart Money", sm_conf * 0.15, sm))
    elif sm == "DISTRIBUTION":
        total -= sm_conf * 0.15
        factors.append(("Smart Money", -sm_conf * 0.15, sm))
    
    # Trend (15%)
    if last['ema20'] > last['ema50']:
        total += 15
        factors.append(("Trend", 15, "Bullish"))
    else:
        total -= 15
        factors.append(("Trend", -15, "Bearish"))
    
    # ADX (10%)
    if last['adx'] >= 25:
        total += 10
        factors.append(("ADX", 10, f"Strong ({last['adx']:.0f})"))
    elif last['adx'] < 20:
        total -= 10
        factors.append(("ADX", -10, f"Sideways"))
    
    # Volume (10%)
    if last['volume_ratio'] >= 1.5:
        total += 10
        factors.append(("Volume", 10, f"Spike ({last['volume_ratio']:.1f}x)"))
    elif last['volume_ratio'] < 0.6:
        total -= 10
        factors.append(("Volume", -10, f"Sepi"))
    
    # Momentum (15%)
    if last['macd_histogram'] > 0 and last['rsi'] > 50:
        total += 15
        factors.append(("Momentum", 15, "Bullish"))
    elif last['macd_histogram'] < 0 and last['rsi'] < 50:
        total -= 15
        factors.append(("Momentum", -15, "Bearish"))
    
    # Market regime (15%)
    regime, regime_conf, _ = detect_market_regime(df)
    if regime == "TRENDING" or regime == "STRONG_TRENDING":
        total += regime_conf * 0.15
        factors.append(("Regime", regime_conf * 0.15, regime))
    elif regime == "PANIC":
        total -= regime_conf * 0.15
        factors.append(("Regime", -regime_conf * 0.15, regime))
    
    total += (ihsg_score - 50) * 0.2
    final = max(0, min(100, total))
    
    if final >= 80:
        grade = "SNIPER"
    elif final >= 65:
        grade = "HIGH"
    elif final >= 50:
        grade = "NORMAL"
    else:
        grade = "AVOID"
    
    return final, factors, grade

def detect_high_quality_setup(df):
    entry, sl, tp, shares, rr, setup, conf, signals = calculate_entry_sl_tp(df)
    if entry:
        if conf >= 80:
            return f"{setup} (SNIPER)", conf, " | ".join(signals)
        elif conf >= 65:
            return f"{setup} (HIGH)", conf, " | ".join(signals)
        else:
            return setup, conf, " | ".join(signals)
    return "NO_SETUP", 0, "Tidak ada setup"

def get_trading_recommendation(df):
    entry, sl, tp, shares, rr, setup, conf, signals = calculate_entry_sl_tp(df)
    if entry:
        if conf >= 80:
            return f"🎯 {setup} - Sniper eksekusi dengan RR 1:{rr:.1f}"
        elif conf >= 70:
            return f"📈 {setup} - Setup bagus, pastikan konfirmasi"
        elif conf >= 60:
            return f"⏸️ {setup} - Tunggu konfirmasi lebih lanjut"
        else:
            return f"📊 {setup} - Setup medium, hati-hati"
    return "⛔ NO TRADE ZONE - Hindari entry"

def scan_saham():
    stocks = ["BBCA.JK", "BBRI.JK", "BMRI.JK", "TLKM.JK", "ASII.JK", "UNVR.JK", "INDF.JK"]
    results = []
    for stock in stocks:
        try:
            df = get_data(stock, "1d")
            if not df.empty and len(df) > 30:
                df = add_indicators(df)
                setup, quality, msg = detect_high_quality_setup(df)
                if quality >= 60:
                    if "BUY" in setup:
                        signal = "🔥 BUY"
                    elif "SELL" in setup:
                        signal = "🔴 SELL"
                    else:
                        signal = "⏸️ HOLD"
                    results.append({
                        "Kode": stock,
                        "Score": f"{quality:.0f}",
                        "Setup": msg[:40],
                        "Sinyal": signal,
                        "Harga": f"Rp{df.iloc[-1]['close']:,.0f}"
                    })
            time.sleep(0.5)
        except:
            continue
    results.sort(key=lambda x: int(x['Score']), reverse=True)
    return results[:7]

def backtest_strategy(df, initial_capital=100000000, risk_per_trade=2, fee_buy=0.0015, fee_sell=0.0025, slippage=0.001):
    if df.empty or len(df) < 30:
        return {"return": 0, "winrate": 0, "trades": 0, "final_capital": initial_capital, "max_drawdown": 0, "profit_factor": 0, "sharpe_ratio": 0, "expectancy": 0}
    
    df_test = df.copy()
    df_test = add_indicators(df_test)
    capital = initial_capital
    position = 0
    trades = []
    equity_curve = [initial_capital]
    peak = initial_capital
    max_dd = 0
    entry_price_used = 0
    stop_price = 0
    take_price = 0
    
    for i in range(20, len(df_test)):
        entry, sl, tp, shares, rr, setup, conf, _ = calculate_entry_sl_tp(df_test.iloc[:i+1], capital, risk_per_trade)
        close = df_test.iloc[i]['close']
        
        if entry and "BUY" in setup and position == 0 and shares > 0:
            entry_price_used = entry * (1 + slippage)
            cost = shares * entry_price_used * (1 + fee_buy)
            if cost <= capital:
                position = shares
                capital -= cost
                stop_price = sl
                take_price = tp
        
        elif position > 0:
            if close <= stop_price:
                exit_p = close * (1 - slippage)
                capital += position * exit_p * (1 - fee_sell)
                pnl = (exit_p - entry_price_used) / entry_price_used * 100
                trades.append(pnl)
                position = 0
            elif close >= take_price:
                exit_p = close * (1 - slippage)
                capital += position * exit_p * (1 - fee_sell)
                pnl = (exit_p - entry_price_used) / entry_price_used * 100
                trades.append(pnl)
                position = 0
        
        current = capital + (position * close) if position else capital
        equity_curve.append(current)
        if current > peak:
            peak = current
        dd = (peak - current) / peak * 100
        max_dd = max(max_dd, dd)
    
    if position > 0:
        exit_p = df_test.iloc[-1]['close'] * (1 - slippage)
        capital += position * exit_p * (1 - fee_sell)
        equity_curve.append(capital)
    
    winrate = len([t for t in trades if t > 0]) / len(trades) * 100 if trades else 0
    gross_profit = sum([t for t in trades if t > 0])
    gross_loss = abs(sum([t for t in trades if t < 0]))
    pf = gross_profit / gross_loss if gross_loss > 0 else 0
    ret = ((capital - initial_capital) / initial_capital) * 100
    
    # Calculate Sharpe Ratio
    returns = [equity_curve[i] / equity_curve[i-1] - 1 for i in range(1, len(equity_curve)) if equity_curve[i-1] > 0]
    sharpe = (np.mean(returns) / np.std(returns) * np.sqrt(252)) if len(returns) > 0 and np.std(returns) > 0 else 0
    
    # Calculate Expectancy
    expectancy = (np.mean(trades) * winrate / 100) if trades else 0
    
    return {
        "return": round(ret, 2),
        "winrate": round(winrate, 2),
        "trades": len(trades),
        "final_capital": round(capital, 0),
        "max_drawdown": round(max_dd, 2),
        "profit_factor": round(pf, 2),
        "sharpe_ratio": round(sharpe, 2),
        "expectancy": round(expectancy, 2),
        "equity_curve": equity_curve[-100:]  # Last 100 points
    }
