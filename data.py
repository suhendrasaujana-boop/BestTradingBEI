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
    
    # Normalisasi symbol
    symbol = symbol.upper()
    if symbol == "IHSG":
        symbol = "^JKSE"
    elif symbol != "^JKSE" and not symbol.endswith('.JK'):
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
    
    # Support & Resistance (rolling)
    df['support'] = df['low'].rolling(window=20).min()
    df['resistance'] = df['high'].rolling(window=20).max()
    
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
    
    # Fill NaN awal (backfill dan forward fill) - cara baru untuk pandas >= 2.2
    df = df.bfill().ffill()
    
    return df

def get_ihsg_trend():
    try:
        ihsg = get_data("^JKSE", "1d")
        if ihsg.empty or len(ihsg) < 20:
            # Fallback: langsung dari yfinance tanpa cache
            ticker = yf.Ticker("^JKSE")
            df_fallback = ticker.history(period="1mo")
            if df_fallback.empty:
                return "NEUTRAL", 50, "IHSG data unavailable"
            df_fallback = df_fallback.reset_index()
            df_fallback.columns = [col.lower() for col in df_fallback.columns]
            ihsg = add_indicators(df_fallback)
            if ihsg.empty or len(ihsg) < 20:
                return "NEUTRAL", 50, "IHSG data insufficient"
        
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
    except Exception as e:
        return "NEUTRAL", 50, "IHSG error"

# ========== FITUR LAMA (BOTTOM, BREAKOUT, REVERSAL) ==========
def detect_bottom_pattern(df):
    if df.empty or len(df) < 30:
        return False, 0, "Data tidak cukup"
    
    last = df.iloc[-1]
    prev_5 = df.iloc[-5:]
    prev_10 = df.iloc[-10:]
    prev_20 = df.iloc[-20:]
    
    signals = []
    confidence = 0
    
    recent_lows = prev_20['low'].nsmallest(2).values
    if len(recent_lows) >= 2:
        double_bottom = abs(recent_lows[0] - recent_lows[1]) / recent_lows[0] < 0.02
        if double_bottom:
            signals.append("Double Bottom")
            confidence += 25
    
    price_lower = prev_10['close'].min() < prev_20['close'].min()
    rsi_higher = prev_5['rsi'].mean() > prev_20['rsi'].mean()
    if price_lower and rsi_higher:
        signals.append("Bullish Divergence")
        confidence += 30
    
    if last['rsi'] < 30 and last['volume_ratio'] > 1.5:
        signals.append("Oversold + Volume")
        confidence += 25
    
    support = prev_20['low'].min()
    if abs(last['close'] - support) / support < 0.01 and last['close'] > support:
        signals.append("Support Bounce")
        confidence += 20
    
    is_bottom = confidence >= 40
    desc = " | ".join(signals) if signals else "Tidak ada sinyal"
    return is_bottom, min(100, confidence), desc

def detect_valid_breakout(df, lookback=20):
    if df.empty or len(df) < lookback + 5:
        return False, "NONE", 0, "Data tidak cukup", 0
    
    last = df.iloc[-1]
    prev = df.iloc[-2]
    prev_highs = df.iloc[-lookback:-1]['high']
    resistance = prev_highs.max()
    
    signals = []
    confidence = 0
    
    if last['close'] > resistance:
        signals.append("Break resistance")
        confidence += 30
    else:
        return False, "NONE", 0, "No breakout", resistance
    
    if last['volume_ratio'] >= 2.0:
        signals.append("High volume")
        confidence += 30
    elif last['volume_ratio'] >= 1.5:
        signals.append("Good volume")
        confidence += 20
    else:
        return False, "FAKE_BREAKOUT", 10, "Volume rendah - fake breakout!", resistance
    
    if last['macd_histogram'] > 0 and last['macd_histogram'] > prev['macd_histogram']:
        signals.append("MACD bullish")
        confidence += 15
    
    if last['adx'] >= 25:
        signals.append("Strong trend")
        confidence += 15
    
    if confidence >= 70:
        breakout_type = "STRONG_BREAKOUT"
    elif confidence >= 50:
        breakout_type = "VALID_BREAKOUT"
    else:
        breakout_type = "WEAK_BREAKOUT"
    
    desc = " | ".join(signals)
    return confidence >= 50, breakout_type, min(100, confidence), desc, resistance

def detect_reversal(df):
    if df.empty or len(df) < 30:
        return "NONE", 0, "Data tidak cukup"
    
    last = df.iloc[-1]
    prev_5 = df.iloc[-6:-1]
    prev_20 = df.iloc[-21:-1]
    
    signals = []
    confidence = 0
    reversal_type = "NONE"
    
    prev_trend_down = prev_5['close'].mean() < prev_20['close'].mean()
    current_trend_up = last['close'] > last['ema20'] > last['ema50']
    if prev_trend_down and current_trend_up:
        signals.append("DOWN → UP")
        confidence += 35
        reversal_type = "BULLISH"
    
    prev_macd_bearish = prev_5['macd_histogram'].mean() < 0
    current_macd_bullish = last['macd_histogram'] > 0
    if prev_macd_bearish and current_macd_bullish:
        signals.append("MACD crossover")
        confidence += 25
    
    prev_trend_up = prev_5['close'].mean() > prev_20['close'].mean()
    current_trend_down = last['close'] < last['ema20'] < last['ema50']
    if prev_trend_up and current_trend_down:
        signals.append("UP → DOWN")
        confidence += 35
        reversal_type = "BEARISH"
    
    prev_macd_bullish = prev_5['macd_histogram'].mean() > 0
    current_macd_bearish = last['macd_histogram'] < 0
    if prev_macd_bullish and current_macd_bearish:
        signals.append("MACD cross below")
        confidence += 25
    
    desc = " | ".join(signals) if signals else "No reversal"
    
    if confidence >= 50 and reversal_type != "NONE":
        return reversal_type, min(100, confidence), desc
    return "NONE", confidence, desc

# ========== FITUR BARU 1: SWING STRUCTURE (BOS, CHOCH, LIQUIDITY ZONES) ==========
def detect_swing_points(df, lookback=5):
    """Deteksi valid swing high dan swing low"""
    if df.empty or len(df) < lookback * 2 + 1:
        return [], []
    
    swing_highs = []
    swing_lows = []
    
    for i in range(lookback, len(df) - lookback):
        is_swing_high = True
        is_swing_low = True
        
        for j in range(1, lookback + 1):
            if df['high'].iloc[i] <= df['high'].iloc[i - j] or df['high'].iloc[i] <= df['high'].iloc[i + j]:
                is_swing_high = False
            if df['low'].iloc[i] >= df['low'].iloc[i - j] or df['low'].iloc[i] >= df['low'].iloc[i + j]:
                is_swing_low = False
        
        if is_swing_high:
            swing_highs.append((i, df['high'].iloc[i]))
        if is_swing_low:
            swing_lows.append((i, df['low'].iloc[i]))
    
    return swing_highs, swing_lows

def detect_bos_choch(df):
    """Deteksi Break of Structure (BOS) dan Change of Character (CHOCH)"""
    if df.empty or len(df) < 30:
        return "NEUTRAL", 0, "No BOS/CHOCH"
    
    swing_highs, swing_lows = detect_swing_points(df)
    
    if len(swing_highs) < 2 or len(swing_lows) < 2:
        return "NEUTRAL", 0, "Insufficient swing points"
    
    last_high = swing_highs[-1][1]
    prev_high = swing_highs[-2][1]
    last_low = swing_lows[-1][1]
    prev_low = swing_lows[-2][1]
    
    last_close = df.iloc[-1]['close']
    
    signals = []
    confidence = 0
    result = "NEUTRAL"
    
    # Bullish BOS: higher high
    if last_high > prev_high and last_close > prev_high:
        result = "BULLISH_BOS"
        confidence += 40
        signals.append("Break of Structure UP")
    
    # Bearish BOS: lower low
    elif last_low < prev_low and last_close < prev_low:
        result = "BEARISH_BOS"
        confidence += 40
        signals.append("Break of Structure DOWN")
    
    # CHOCH (Change of Character) - reversal
    if len(swing_highs) >= 3 and len(swing_lows) >= 3:
        if swing_highs[-1][1] < swing_highs[-2][1] and swing_lows[-1][1] > swing_lows[-2][1]:
            result = "BULLISH_CHOCH"
            confidence += 35
            signals.append("Change of Character UP")
        elif swing_highs[-1][1] > swing_highs[-2][1] and swing_lows[-1][1] < swing_lows[-2][1]:
            result = "BEARISH_CHOCH"
            confidence += 35
            signals.append("Change of Character DOWN")
    
    desc = " | ".join(signals) if signals else "No BOS/CHOCH"
    return result, min(100, confidence), desc

def detect_liquidity_zones(df, lookback=20):
    """Deteksi liquidity zones (equal highs/lows)"""
    if df.empty or len(df) < lookback:
        return [], []
    
    recent_highs = df['high'].iloc[-lookback:].tolist()
    recent_lows = df['low'].iloc[-lookback:].tolist()
    
    liquidity_highs = []
    liquidity_lows = []
    
    # Detect equal highs (liquidity above)
    for i in range(len(recent_highs)):
        count = 1
        for j in range(i+1, len(recent_highs)):
            if abs(recent_highs[i] - recent_highs[j]) / recent_highs[i] < 0.005:
                count += 1
        if count >= 2:
            liquidity_highs.append(recent_highs[i])
    
    # Detect equal lows (liquidity below)
    for i in range(len(recent_lows)):
        count = 1
        for j in range(i+1, len(recent_lows)):
            if abs(recent_lows[i] - recent_lows[j]) / recent_lows[i] < 0.005:
                count += 1
        if count >= 2:
            liquidity_lows.append(recent_lows[i])
    
    return list(set(liquidity_highs)), list(set(liquidity_lows))

# ========== FITUR BARU 2: FAIR VALUE GAP (FVG) ==========
def detect_fair_value_gap(df):
    """Deteksi Fair Value Gap (Imbalance) - Smart Money Concept"""
    if df.empty or len(df) < 3:
        return [], []
    
    fvg_bullish = []
    fvg_bearish = []
    
    for i in range(2, len(df)):
        # Bullish FVG: low candle i > high candle i-2 (gap)
        if df['low'].iloc[i] > df['high'].iloc[i-2]:
            fvg_bullish.append({
                'index': i,
                'upper': df['low'].iloc[i],
                'lower': df['high'].iloc[i-2],
                'strength': df['volume_ratio'].iloc[i] if 'volume_ratio' in df.columns else 1
            })
        
        # Bearish FVG: high candle i < low candle i-2 (gap down)
        if df['high'].iloc[i] < df['low'].iloc[i-2]:
            fvg_bearish.append({
                'index': i,
                'upper': df['low'].iloc[i-2],
                'lower': df['high'].iloc[i],
                'strength': df['volume_ratio'].iloc[i] if 'volume_ratio' in df.columns else 1
            })
    
    return fvg_bullish, fvg_bearish

def is_fvg_still_valid(df, fvg, current_idx):
    """FVG masih valid jika harga belum menyentuh area gap (antara lower dan upper)"""
    if fvg is None:
        return False
    # Cek dari candle setelah FVG hingga current_idx
    for i in range(fvg['index']+1, min(current_idx+1, len(df))):
        low = df['low'].iloc[i]
        high = df['high'].iloc[i]
        if low <= fvg['upper'] and high >= fvg['lower']:
            return False  # sudah terisi
    return True

def get_nearest_fvg(df):
    """Dapatkan FVG terdekat dengan harga saat ini (hanya yang masih valid)"""
    last_close = df.iloc[-1]['close']
    fvg_bullish_raw, fvg_bearish_raw = detect_fair_value_gap(df)
    
    # Filter yang masih valid
    current_idx = len(df) - 1
    fvg_bullish = [f for f in fvg_bullish_raw if is_fvg_still_valid(df, f, current_idx)]
    fvg_bearish = [f for f in fvg_bearish_raw if is_fvg_still_valid(df, f, current_idx)]
    
    nearest_bullish = None
    nearest_bearish = None
    dist_bullish = float('inf')
    dist_bearish = float('inf')
    
    for fvg in fvg_bullish:
        if fvg['upper'] > last_close:
            dist = fvg['upper'] - last_close
            if dist < dist_bullish:
                dist_bullish = dist
                nearest_bullish = fvg
    
    for fvg in fvg_bearish:
        if fvg['lower'] < last_close:
            dist = last_close - fvg['lower']
            if dist < dist_bearish:
                dist_bearish = dist
                nearest_bearish = fvg
    
    return nearest_bullish, nearest_bearish

# ========== FITUR BARU 3: ORDER BLOCK ==========
def detect_order_blocks(df):
    """Deteksi Order Block (Supply/Demand Zone)"""
    if df.empty or len(df) < 5:
        return [], []
    
    bullish_blocks = []
    bearish_blocks = []
    
    for i in range(2, len(df) - 2):
        # Bullish Order Block: bearish candle sebelum bullish move
        if df['close'].iloc[i] > df['open'].iloc[i] and df['close'].iloc[i-1] < df['open'].iloc[i-1]:
            if df['close'].iloc[i] > df['high'].iloc[i-1]:
                bullish_blocks.append({
                    'index': i-1,
                    'high': df['high'].iloc[i-1],
                    'low': df['low'].iloc[i-1],
                    'strength': 1
                })
        
        # Bearish Order Block: bullish candle sebelum bearish move
        if df['close'].iloc[i] < df['open'].iloc[i] and df['close'].iloc[i-1] > df['open'].iloc[i-1]:
            if df['close'].iloc[i] < df['low'].iloc[i-1]:
                bearish_blocks.append({
                    'index': i-1,
                    'high': df['high'].iloc[i-1],
                    'low': df['low'].iloc[i-1],
                    'strength': 1
                })
    
    return bullish_blocks, bearish_blocks

def get_nearest_order_block(df):
    """Dapatkan Order Block terdekat dengan harga saat ini"""
    last_close = df.iloc[-1]['close']
    bullish_blocks, bearish_blocks = detect_order_blocks(df)
    
    nearest_bullish = None
    nearest_bearish = None
    dist_bullish = float('inf')
    dist_bearish = float('inf')
    
    for block in bullish_blocks:
        if block['high'] > last_close:
            dist = block['high'] - last_close
            if dist < dist_bullish:
                dist_bullish = dist
                nearest_bullish = block
    
    for block in bearish_blocks:
        if block['low'] < last_close:
            dist = last_close - block['low']
            if dist < dist_bearish:
                dist_bearish = dist
                nearest_bearish = block
    
    return nearest_bullish, nearest_bearish

# ========== MARKET STRUCTURE (LAMA, TAPI DIPERBAIKI) ==========
def detect_market_structure(df):
    """Market structure dengan swing points"""
    if df.empty or len(df) < 20:
        return "RANGE", 0, "Data tidak cukup"
    
    bos_cho, bos_conf, bos_desc = detect_bos_choch(df)
    liquidity_highs, liquidity_lows = detect_liquidity_zones(df)
    
    signals = []
    confidence = bos_conf
    
    if "BULLISH" in bos_cho:
        signals.append(bos_desc)
    elif "BEARISH" in bos_cho:
        signals.append(bos_desc)
    
    if liquidity_highs:
        signals.append(f"Liquidity above: Rp{min(liquidity_highs):,.0f}")
        confidence += 10
    
    if liquidity_lows:
        signals.append(f"Liquidity below: Rp{max(liquidity_lows):,.0f}")
        confidence += 10
    
    result = bos_cho if "BOS" in bos_cho or "CHOCH" in bos_cho else "RANGE"
    
    desc = " | ".join(signals) if signals else "No structure signal"
    return result, min(100, confidence), desc

# ========== MARKET REGIME ==========
def detect_market_regime(df):
    if df.empty or len(df) < 30:
        return "UNKNOWN", 0, "Data tidak cukup"
    
    last = df.iloc[-1]
    prev_20 = df.iloc[-21:-1]
    
    adx = last.get('adx', 0)
    if pd.isna(adx):
        adx = 0
    atr_val = last.get('atr', 0)
    if pd.isna(atr_val):
        atr_val = 0
    atr_pct = (atr_val / last['close']) * 100 if last['close'] > 0 else 0
    volume_ratio = last.get('volume_ratio', 1)
    if pd.isna(volume_ratio):
        volume_ratio = 1
    daily_change = (last['close'] - prev_20['close'].iloc[0]) / prev_20['close'].iloc[0] * 100 if prev_20['close'].iloc[0] > 0 else 0
    
    regime = "NEUTRAL"
    confidence = 50
    desc = ""
    
    if adx >= 25:
        regime = "TRENDING"
        confidence = 70 + min(10, adx - 25)
        desc = f"Tren kuat (ADX {adx:.0f})"
    elif adx < 20:
        regime = "SIDEWAYS"
        confidence = 40
        desc = f"Pasar ranging (ADX {adx:.0f})"
    
    if atr_pct > 4:
        regime = "VOLATILE"
        confidence = 60
        desc = f"Volatilitas tinggi (ATR {atr_pct:.1f}%)"
    
    if daily_change < -5 or volume_ratio > 2.5:
        regime = "PANIC"
        confidence = 80
        desc = "Kondisi panic selling! Hati-hati!"
    
    if adx >= 35 and atr_pct > 3:
        regime = "STRONG_TRENDING"
        confidence = 85
        desc = f"Tren sangat kuat (ADX {adx:.0f}, Vol {atr_pct:.1f}%)"
    
    return regime, confidence, desc

# ========== SMART MONEY VOLUME ==========
def detect_smart_money_volume(df):
    if df.empty or len(df) < 30:
        return "NEUTRAL", 0, "Data tidak cukup"
    
    last = df.iloc[-1]
    prev_5 = df.iloc[-6:-1]
    prev_10 = df.iloc[-11:-1]
    prev_20 = df.iloc[-21:-1]
    
    signals = []
    confidence = 0
    result = "NEUTRAL"
    
    price_range = (prev_10['high'].max() - prev_10['low'].min()) / prev_10['close'].mean()
    volume_increasing = prev_5['volume'].mean() > prev_20['volume'].mean() * 1.2
    price_sideways = price_range < 0.03
    
    if price_sideways and volume_increasing:
        result = "ACCUMULATION"
        confidence += 40
        signals.append("Volume naik, harga sideways = Akumulasi")
    
    volume_high = last['volume_ratio'] > 1.5
    price_stagnant = abs(last['close'] - prev_10['close'].mean()) / prev_10['close'].mean() < 0.02
    
    if volume_high and price_stagnant:
        result = "DISTRIBUTION"
        confidence += 35
        signals.append("Volume tinggi, harga stagnan = Distribusi")
    
    if last['volume_ratio'] > 2.0 and last['close'] < (last['high'] + last['low']) / 2:
        confidence += 20
        signals.append("Volume spike + close bawah = tekanan jual")
    
    if last['volume_ratio'] > 2.0 and last['close'] > (last['high'] + last['low']) / 2:
        confidence += 20
        signals.append("Volume spike + close atas = tekanan beli")
    
    desc = " | ".join(signals) if signals else "No smart money signal"
    return result, min(100, confidence), desc

# ========== LIQUIDITY SWEEP ==========
def detect_liquidity_sweep(df, lookback=20):
    if df.empty or len(df) < lookback + 5:
        return False, 0, "NONE", "Data tidak cukup"
    
    last = df.iloc[-1]
    prev = df.iloc[-2]
    prev_highs = df.iloc[-lookback:-1]['high']
    prev_lows = df.iloc[-lookback:-1]['low']
    resistance = prev_highs.max()
    support = prev_lows.min()
    
    signals = []
    confidence = 0
    sweep_type = "NONE"
    
    if last['low'] < support and last['close'] > support:
        sweep_type = "BULLISH_SFP"
        confidence += 50
        signals.append("Liquidity sweep bawah + reversal")
    elif last['high'] > resistance and last['close'] < resistance:
        sweep_type = "BEARISH_SFP"
        confidence += 50
        signals.append("Liquidity sweep atas + reversal")
    
    if prev['high'] > resistance and last['close'] < resistance:
        confidence += 30
        signals.append("Fake breakout")
        if sweep_type == "NONE":
            sweep_type = "FAKE_BREAKOUT"
    
    desc = " | ".join(signals) if signals else "No sweep detected"
    return sweep_type != "NONE", confidence, sweep_type, desc

# ========== CANDLESTICK PATTERN ==========
def detect_candlestick_pattern(df):
    if df.empty or len(df) < 3:
        return "NONE", 0, "Data tidak cukup"
    
    last = df.iloc[-1]
    prev = df.iloc[-2]
    
    body = abs(last['close'] - last['open'])
    candle_range = last['high'] - last['low']
    
    if candle_range <= 0:
        return "NONE", 0, "Candle range zero"
    
    upper_wick = last['high'] - max(last['open'], last['close'])
    lower_wick = min(last['open'], last['close']) - last['low']
    
    signals = []
    confidence = 0
    pattern = "NONE"
    
    if body / candle_range > 0.85:
        if last['close'] > last['open']:
            pattern = "BULLISH_MARUBOZU"
            confidence += 30
            signals.append("Bullish Marubozu")
        else:
            pattern = "BEARISH_MARUBOZU"
            confidence -= 30
            signals.append("Bearish Marubozu")
    
    if lower_wick > body * 2 and upper_wick < body:
        pattern = "HAMMER"
        confidence += 35
        signals.append("Hammer / Pinbar")
    
    if upper_wick > body * 2 and lower_wick < body:
        pattern = "SHOOTING_STAR"
        confidence -= 35
        signals.append("Shooting Star")
    
    if (last['close'] > last['open'] and prev['close'] < prev['open'] and
        last['close'] > prev['open'] and last['open'] < prev['close']):
        pattern = "BULLISH_ENGULFING"
        confidence += 40
        signals.append("Bullish Engulfing")
    
    if (last['close'] < last['open'] and prev['close'] > prev['open'] and
        last['open'] > prev['close'] and last['close'] < prev['open']):
        pattern = "BEARISH_ENGULFING"
        confidence -= 40
        signals.append("Bearish Engulfing")
    
    desc = " | ".join(signals) if signals else "No pattern"
    return pattern, confidence, desc

# ========== PIVOT SUPPORT RESISTANCE ==========
def get_pivot_sr(df, lookback=20):
    if df.empty or len(df) < lookback + 5:
        last = df.iloc[-1] if not df.empty else None
        if last is not None:
            return last.get('support', 0), last.get('resistance', 0), 0, 0, 0, 0, 0, 0, 0
        return 0, 0, 0, 0, 0, 0, 0, 0, 0
    
    last = df.iloc[-1]
    
    pivot = (last['high'] + last['low'] + last['close']) / 3
    r1 = (2 * pivot) - last['low']
    r2 = pivot + (last['high'] - last['low'])
    s1 = (2 * pivot) - last['high']
    s2 = pivot - (last['high'] - last['low'])
    
    rolling_support = df['low'].rolling(window=lookback).min().iloc[-1]
    rolling_resistance = df['high'].rolling(window=lookback).max().iloc[-1]
    
    high_20 = df['high'].iloc[-20:].max()
    low_20 = df['low'].iloc[-20:].min()
    range_20 = high_20 - low_20
    fib_382 = low_20 + (range_20 * 0.382)
    fib_618 = low_20 + (range_20 * 0.618)
    
    return rolling_support, rolling_resistance, pivot, r1, r2, s1, s2, fib_382, fib_618

# ========== CONFIDENCE SCORE DENGAN BOBOT BARU ==========
def calculate_confidence_score(df, ihsg_score=50):
    if df.empty or len(df) < 30:
        return 50, [], "NORMAL"
    
    last = df.iloc[-1]
    factors = []
    total = 50
    
    # 1. STRUCTURE SCORE (20%)
    structure, struct_conf, struct_desc = detect_market_structure(df)
    if "BULLISH" in structure:
        total += struct_conf * 0.20
        factors.append(("Structure", struct_conf * 0.20, structure))
    elif "BEARISH" in structure:
        total -= struct_conf * 0.20
        factors.append(("Structure", -struct_conf * 0.20, structure))
    else:
        factors.append(("Structure", 0, "Neutral"))
    
    # 2. FVG SCORE (10%)
    nearest_bullish_fvg, nearest_bearish_fvg = get_nearest_fvg(df)
    fvg_score = 0
    if nearest_bullish_fvg:
        dist_pct = (nearest_bullish_fvg['upper'] - last['close']) / last['close'] * 100
        if dist_pct < 2:
            fvg_score = 15
            factors.append(("FVG", 15, "Bullish FVG dekat"))
        else:
            fvg_score = 5
            factors.append(("FVG", 5, "Bullish FVG ada"))
    elif nearest_bearish_fvg:
        dist_pct = (last['close'] - nearest_bearish_fvg['lower']) / last['close'] * 100
        if dist_pct < 2:
            fvg_score = -15
            factors.append(("FVG", -15, "Bearish FVG dekat"))
        else:
            fvg_score = -5
            factors.append(("FVG", -5, "Bearish FVG ada"))
    total += fvg_score
    
    # 3. ORDER BLOCK SCORE (10%)
    nearest_bullish_ob, nearest_bearish_ob = get_nearest_order_block(df)
    ob_score = 0
    if nearest_bullish_ob:
        dist_pct = (nearest_bullish_ob['high'] - last['close']) / last['close'] * 100
        if dist_pct < 2:
            ob_score = 15
            factors.append(("Order Block", 15, "Bullish OB dekat"))
        else:
            ob_score = 5
            factors.append(("Order Block", 5, "Bullish OB ada"))
    elif nearest_bearish_ob:
        dist_pct = (last['close'] - nearest_bearish_ob['low']) / last['close'] * 100
        if dist_pct < 2:
            ob_score = -15
            factors.append(("Order Block", -15, "Bearish OB dekat"))
        else:
            ob_score = -5
            factors.append(("Order Block", -5, "Bearish OB ada"))
    total += ob_score
    
    # 4. TREND SCORE (10%)
    if last['ema20'] > last['ema50']:
        total += 10
        factors.append(("Trend", 10, "Bullish"))
    else:
        total -= 10
        factors.append(("Trend", -10, "Bearish"))
    
    # 5. MOMENTUM SCORE (5%)
    if last['macd_histogram'] > 0 and last['rsi'] > 50:
        total += 5
        factors.append(("Momentum", 5, "Bullish"))
    elif last['macd_histogram'] < 0 and last['rsi'] < 50:
        total -= 5
        factors.append(("Momentum", -5, "Bearish"))
    
    # 6. VOLUME SCORE (15%)
    if last['volume_ratio'] >= 1.5:
        total += 15
        factors.append(("Volume", 15, f"Spike ({last['volume_ratio']:.1f}x)"))
    elif last['volume_ratio'] < 0.6:
        total -= 10
        factors.append(("Volume", -10, f"Sepi"))
    
    # 7. VOLATILITY SCORE (5%)
    if last['adx'] >= 25:
        total += 5
        factors.append(("ADX", 5, f"Strong ({last['adx']:.0f})"))
    elif last['adx'] < 20:
        total -= 5
        factors.append(("ADX", -5, f"Sideways"))
    
    # 8. REGIME SCORE (10%)
    regime, regime_conf, _ = detect_market_regime(df)
    if regime == "TRENDING" or regime == "STRONG_TRENDING":
        total += regime_conf * 0.10
        factors.append(("Regime", regime_conf * 0.10, regime))
    elif regime == "PANIC":
        total -= regime_conf * 0.10
        factors.append(("Regime", -regime_conf * 0.10, regime))
    
    # 9. MARKET SCORE (5%)
    total += (ihsg_score - 50) * 0.10
    factors.append(("IHSG", (ihsg_score - 50) * 0.10, f"Score {ihsg_score:.0f}"))
    
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

# ========== ENTRY, SL, TP DARI INDIKATOR (DENGAN FITUR BARU) ==========
def calculate_entry_sl_tp(df, capital=100000000, risk_percent=2):
    if df.empty or len(df) < 30:
        return None, None, None, 0, 0, "NO_SETUP", 0, []
    
    last = df.iloc[-1]
    atr = last.get('atr', last['close'] * 0.02) if last['close'] > 0 else 100
    if pd.isna(atr) or atr <= 0:
        atr = last['close'] * 0.02 if last['close'] > 0 else 100
    
    structure, struct_conf, struct_desc = detect_market_structure(df)
    smart_money, sm_conf, sm_desc = detect_smart_money_volume(df)
    is_sweep, sweep_conf, sweep_type, sweep_desc = detect_liquidity_sweep(df)
    pattern, pattern_conf, pattern_desc = detect_candlestick_pattern(df)
    regime, regime_conf, regime_desc = detect_market_regime(df)
    support, resistance, pivot, r1, r2, s1, s2, fib_382, fib_618 = get_pivot_sr(df)
    
    # Fitur baru
    nearest_bullish_fvg, nearest_bearish_fvg = get_nearest_fvg(df)
    nearest_bullish_ob, nearest_bearish_ob = get_nearest_order_block(df)
    
    trend_up = last['ema20'] > last['ema50']
    momentum_bullish = last['macd_histogram'] > 0 and last['rsi'] > 50
    volume_spike = last['volume_ratio'] >= 1.5
    strong_trend = last['adx'] >= 25
    
    entry_price = None
    stop_loss = None
    take_profit = None
    setup_name = "NO_SETUP"
    confidence = 0
    signals = []
    
    # PRIORITY 1: BULLISH BOS + FVG + ORDER BLOCK (Smart Money Combo)
    if "BULLISH" in structure and nearest_bullish_fvg and nearest_bullish_ob:
        entry_price = last['close']
        stop_loss = entry_price - (1.5 * atr)
        take_profit = entry_price + (3 * atr)
        setup_name = "SMART_MONEY_COMBO_BUY"
        confidence = 85
        signals = [struct_desc, "FVG detected", "OB detected"]
    
    # PRIORITY 2: BULLISH BOS + VOLUME SPIKE
    elif "BULLISH" in structure and volume_spike:
        entry_price = last['close']
        stop_loss = entry_price - (2 * atr)
        take_profit = entry_price + (3 * atr)
        setup_name = "BREAKOUT_BUY"
        confidence = struct_conf
        signals = [struct_desc]
    
    # PRIORITY 3: BULLISH CANDLESTICK PATTERN
    elif pattern in ["BULLISH_ENGULFING", "HAMMER"] and trend_up:
        entry_price = last['close']
        stop_loss = entry_price - (1.5 * atr)
        take_profit = entry_price + (3 * atr)
        setup_name = f"{pattern}_BUY"
        confidence = 65 + abs(pattern_conf)//2
        signals = [pattern_desc]
    
    # PRIORITY 4: BULLISH LIQUIDITY SWEEP
    elif is_sweep and sweep_type == "BULLISH_SFP":
        entry_price = last['close']
        stop_loss = entry_price - (1.5 * atr)
        take_profit = entry_price + (2.5 * atr)
        setup_name = "LIQUIDITY_SWEEP_BUY"
        confidence = sweep_conf
        signals = [sweep_desc]
    
    # PRIORITY 5: FVG BUY (harga mendekati bullish FVG)
    elif nearest_bullish_fvg and (nearest_bullish_fvg['upper'] - last['close']) / last['close'] < 0.02:
        entry_price = last['close']
        stop_loss = entry_price - (1.5 * atr)
        take_profit = entry_price + (2.5 * atr)
        setup_name = "FVG_BUY"
        confidence = 70
        signals = ["Near Fair Value Gap"]
    
    # PRIORITY 6: ORDER BLOCK BUY
    elif nearest_bullish_ob and (nearest_bullish_ob['high'] - last['close']) / last['close'] < 0.02:
        entry_price = last['close']
        stop_loss = entry_price - (1.5 * atr)
        take_profit = entry_price + (2.5 * atr)
        setup_name = "ORDER_BLOCK_BUY"
        confidence = 70
        signals = ["Near Order Block"]
    
    # PRIORITY 7: TREND BUY
    elif trend_up and strong_trend and momentum_bullish:
        entry_price = last['close']
        stop_loss = entry_price - (2 * atr)
        take_profit = entry_price + (3 * atr)
        setup_name = "TREND_BUY"
        confidence = 60
        signals = ["Strong uptrend"]
    
    else:
        return None, None, None, 0, 0, "NO_SETUP", 0, []
    
    if entry_price:
        risk = abs(entry_price - stop_loss)
        reward = abs(take_profit - entry_price)
        rr_ratio = reward / risk if risk > 0 else 0
        
        risk_amount = capital * (risk_percent / 100)
        risk_per_share = risk
        shares = int(risk_amount / risk_per_share) if risk_per_share > 0 else 0
        max_shares = int((capital * 0.5) / entry_price) if entry_price > 0 else 0
        shares = min(shares, max_shares)
        
        return entry_price, stop_loss, take_profit, shares, rr_ratio, setup_name, confidence, signals
    
    return None, None, None, 0, 0, "NO_SETUP", 0, []

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

# ========== MULTI TIMEFRAME ALIGNMENT ==========
def get_multi_timeframe_alignment(symbol, capital=100000000, risk_percent=2):
    timeframes = {"daily": "1d", "hourly": "60m", "fifteen": "15m"}
    
    results = {}
    alignment_score = 0
    signals = []
    
    for name, tf in timeframes.items():
        df = get_data(symbol, tf)
        if not df.empty and len(df) > 30:
            df = add_indicators(df)
            entry, sl, tp, shares, rr, setup, conf, sigs = calculate_entry_sl_tp(df, capital, risk_percent)
            
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

def scan_saham():
    stocks = ["BBCA.JK", "BBRI.JK", "BMRI.JK", "TLKM.JK", "ASII.JK"]
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
        return {"return": 0, "winrate": 0, "trades": 0, "final_capital": initial_capital, "max_drawdown": 0, "profit_factor": 0, "sharpe_ratio": 0, "expectancy": 0, "equity_curve": []}
    
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
        dd = (peak - current) / peak * 100 if peak > 0 else 0
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
    
    returns = [equity_curve[i] / equity_curve[i-1] - 1 for i in range(1, len(equity_curve)) if equity_curve[i-1] > 0]
    sharpe = (np.mean(returns) / np.std(returns) * np.sqrt(252)) if len(returns) > 0 and np.std(returns) > 0 else 0
    
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
        "equity_curve": equity_curve[-100:]
    }
