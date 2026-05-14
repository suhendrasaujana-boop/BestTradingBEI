import pandas as pd
import numpy as np
import yfinance as yf
import time
from datetime import datetime
from scipy.signal import argrelextrema
import ta
import warnings
warnings.filterwarnings('ignore')

# ========== CACHE & RATE LIMIT ==========
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

# ========== DATA FETCHING ==========
ddef get_data(symbol, timeframe="1d"):
    """
    Ambil data dari database lokal dulu.
    Kalau tidak ada atau butuh update, fetch dari Yahoo Finance lalu simpan ke database.
    """
    global _data_cache
    _wait_for_rate_limit()
    
    symbol = symbol.upper()
    if symbol == "IHSG":
        symbol = "^JKSE"
    elif symbol != "^JKSE" and not symbol.endswith('.JK'):
        symbol = f"{symbol}.JK"
    
    cache_key = f"{symbol}_{timeframe}"
    if cache_key in _data_cache:
        cached_time, cached_data = _data_cache[cache_key]
        if (datetime.now() - cached_time).seconds < 60:
            return cached_data.copy()
    
    try:
        # Coba ambil dari database dulu (hanya untuk timeframe 1d)
        if timeframe == "1d":
            try:
                from database import load_data as db_load, store_data, get_last_date
                today_str = datetime.now().strftime('%Y-%m-%d')
                last_date = get_last_date(symbol)
                
                # Kalau data di database ada dan sudah mencakup hari ini, pakai database
                if last_date and last_date >= today_str:
                    df = db_load(symbol)
                    if not df.empty:
                        df = df.reset_index()
                        df.columns = [col.lower() for col in df.columns]
                        if 'datetime' not in df.columns and 'date' in df.columns:
                            df.rename(columns={'date': 'datetime'}, inplace=True)
                        _data_cache[cache_key] = (datetime.now(), df.copy())
                        return df
            except ImportError:
                pass  # database.py belum ada, lanjut ke Yahoo
        
        # Fallback: ambil dari Yahoo Finance
        interval_map = {"5m": "5m", "15m": "15m", "30m": "30m", "60m": "60m", "1d": "1d"}
        period = "7d" if timeframe in ["5m", "15m", "30m", "60m"] else "3mo"
        
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period, interval=interval_map.get(timeframe, "1d"))
        if df.empty:
            return pd.DataFrame()
        
        df = df.reset_index()
        df.columns = [col.lower() for col in df.columns]
        if 'datetime' not in df.columns and 'date' in df.columns:
            df.rename(columns={'date': 'datetime'}, inplace=True)
        
        # Simpan ke database (hanya timeframe 1d)
        if timeframe == "1d":
            try:
                from database import store_data
                store_data(symbol, df)
            except ImportError:
                pass
        
        _data_cache[cache_key] = (datetime.now(), df.copy())
        return df
        
    except Exception as e:
        print(f"Error get_data {symbol}: {e}")
        return pd.DataFrame()
# ========== INDIKATOR (pakai library ta) ==========
def add_indicators(df):
    if df.empty or len(df) < 2:
        return df
    df = df.copy()
    
    # EMA
    df['ema10'] = ta.trend.ema_indicator(df['close'], window=10)
    df['ema20'] = ta.trend.ema_indicator(df['close'], window=20)
    df['ema50'] = ta.trend.ema_indicator(df['close'], window=50)
    df['ema200'] = ta.trend.ema_indicator(df['close'], window=200)
    
    # RSI
    df['rsi'] = ta.momentum.rsi(df['close'], window=14)
    
    # MACD
    df['macd'] = ta.trend.macd(df['close'], window_slow=26, window_fast=12)
    df['macd_signal'] = ta.trend.macd_signal(df['close'], window_slow=26, window_fast=12, window_sign=9)
    df['macd_histogram'] = ta.trend.macd_diff(df['close'], window_slow=26, window_fast=12, window_sign=9)
    
    # Volume
    df['volume_ma20'] = df['volume'].rolling(window=20).mean()
    df['volume_ratio'] = df['volume'] / df['volume_ma20']
    
    # ATR
    df['atr'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=14)
    
    # ADX
    df['adx'] = ta.trend.adx(df['high'], df['low'], df['close'], window=14)
    
    # Supertrend (manual karena library 'ta' tidak punya)
    atr_st = ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=10)
    hl_avg = (df['high'] + df['low']) / 2
    multiplier = 3.0
    upper_band = hl_avg + (multiplier * atr_st)
    lower_band = hl_avg - (multiplier * atr_st)
    
    df['supertrend'] = 0.0
    df['supertrend_direction'] = 1
    for i in range(1, len(df)):
        prev_close = df['close'].iloc[i-1]
        prev_upper = upper_band.iloc[i-1]
        prev_lower = lower_band.iloc[i-1]
        curr_upper = upper_band.iloc[i]
        curr_lower = lower_band.iloc[i]
        
        if df['close'].iloc[i] > prev_upper:
            df.loc[df.index[i], 'supertrend_direction'] = 1
        elif df['close'].iloc[i] < prev_lower:
            df.loc[df.index[i], 'supertrend_direction'] = -1
        else:
            df.loc[df.index[i], 'supertrend_direction'] = df['supertrend_direction'].iloc[i-1]
        
        if df['supertrend_direction'].iloc[i] == 1:
            df.loc[df.index[i], 'supertrend'] = curr_lower
        else:
            df.loc[df.index[i], 'supertrend'] = curr_upper
    
    # VWAP reset harian
    if 'datetime' in df.columns:
        df['date'] = pd.to_datetime(df['datetime']).dt.date
        df['vwap'] = df.groupby('date').apply(
            lambda g: (g['volume'] * (g['high'] + g['low'] + g['close']) / 3).cumsum() / g['volume'].cumsum()
        ).reset_index(level=0, drop=True)
        df.drop('date', axis=1, inplace=True)
    else:
        df['vwap'] = (df['volume'] * (df['high'] + df['low'] + df['close']) / 3).cumsum() / df['volume'].cumsum()
    
    df = df.ffill().bfill().fillna(0)
    return df

# ========== SWING POINTS (no repaint) ==========
def detect_swing_points(df, lookback=5, confirmation=2):
    if df.empty or len(df) < lookback*2 + confirmation:
        return [], []
    high = df['high'].values
    low = df['low'].values
    safe_end = len(df) - confirmation
    local_high_idx = argrelextrema(high[:safe_end], np.greater, order=lookback)[0]
    local_low_idx = argrelextrema(low[:safe_end], np.less, order=lookback)[0]
    swing_highs = [(int(i), high[i]) for i in local_high_idx]
    swing_lows = [(int(i), low[i]) for i in local_low_idx]
    return swing_highs, swing_lows

# ========== BOS & CHOCH ==========
def detect_bos_choch(df):
    if df.empty or len(df) < 30:
        return "NEUTRAL", 0, "Insufficient data"
    swing_highs, swing_lows = detect_swing_points(df, lookback=5, confirmation=1)
    if len(swing_highs) < 2 or len(swing_lows) < 2:
        return "NEUTRAL", 0, "Insufficient swing points"
    last_close = df.iloc[-1]['close']
    h1, h2 = swing_highs[-1], swing_highs[-2]
    l1, l2 = swing_lows[-1], swing_lows[-2]
    result = "NEUTRAL"
    confidence = 0
    signals = []
    if h1[1] > h2[1] and last_close > h2[1]:
        result = "BULLISH_BOS"
        confidence += 40
        signals.append("BOS UP")
    elif l1[1] < l2[1] and last_close < l2[1]:
        result = "BEARISH_BOS"
        confidence += 40
        signals.append("BOS DOWN")
    if len(swing_highs) >= 3 and len(swing_lows) >= 3:
        h3 = swing_highs[-3]
        l3 = swing_lows[-3]
        if h1[1] < h2[1] and l1[1] > l2[1]:
            if last_close > df.iloc[h1[0]]['close']:
                result = "BULLISH_CHOCH"
                confidence += 35
                signals.append("CHOCH UP")
            elif last_close < df.iloc[l1[0]]['close']:
                result = "BEARISH_CHOCH"
                confidence += 35
                signals.append("CHOCH DOWN")
    desc = " | ".join(signals) if signals else "No BOS/CHOCH"
    return result, min(100, confidence), desc

# ========== FVG ==========
def detect_fair_value_gap(df):
    fvg_bullish, fvg_bearish = [], []
    if df.empty or len(df) < 3:
        return [], []
    for i in range(2, len(df)):
        vol_ratio = df['volume_ratio'].iloc[i] if 'volume_ratio' in df.columns else 1.0
        if df['low'].iloc[i] > df['high'].iloc[i-2]:
            body = abs(df['close'].iloc[i] - df['open'].iloc[i])
            candle_range = df['high'].iloc[i] - df['low'].iloc[i]
            if candle_range > 0 and body/candle_range > 0.5 and vol_ratio > 1.2:
                gap = df['low'].iloc[i] - df['high'].iloc[i-2]
                if gap / df['high'].iloc[i-2] > 0.002:
                    fvg_bullish.append({
                        'index': i,
                        'upper': df['low'].iloc[i],
                        'lower': df['high'].iloc[i-2],
                        'strength': vol_ratio
                    })
        if df['high'].iloc[i] < df['low'].iloc[i-2]:
            body = abs(df['close'].iloc[i] - df['open'].iloc[i])
            candle_range = df['high'].iloc[i] - df['low'].iloc[i]
            if candle_range > 0 and body/candle_range > 0.5 and vol_ratio > 1.2:
                gap = df['low'].iloc[i-2] - df['high'].iloc[i]
                if gap / df['low'].iloc[i-2] > 0.002:
                    fvg_bearish.append({
                        'index': i,
                        'upper': df['low'].iloc[i-2],
                        'lower': df['high'].iloc[i],
                        'strength': vol_ratio
                    })
    return fvg_bullish, fvg_bearish

def is_fvg_still_valid(df, fvg, current_idx):
    if fvg is None:
        return False
    for i in range(fvg['index']+1, min(current_idx+1, len(df))):
        if df['low'].iloc[i] <= fvg['upper'] and df['high'].iloc[i] >= fvg['lower']:
            return False
    return True

def get_nearest_fvg(df):
    last_close = df.iloc[-1]['close']
    fvg_bull, fvg_bear = detect_fair_value_gap(df)
    current_idx = len(df)-1
    valid_bull = [f for f in fvg_bull if is_fvg_still_valid(df, f, current_idx) and f['upper'] > last_close]
    valid_bear = [f for f in fvg_bear if is_fvg_still_valid(df, f, current_idx) and f['lower'] < last_close]
    nearest_bull = min(valid_bull, key=lambda x: x['upper'] - last_close) if valid_bull else None
    nearest_bear = min(valid_bear, key=lambda x: last_close - x['lower']) if valid_bear else None
    return nearest_bull, nearest_bear

# ========== ORDER BLOCK ==========
def detect_order_blocks(df):
    bullish_blocks, bearish_blocks = [], []
    if df.empty or len(df) < 5:
        return [], []
    for i in range(2, len(df)-1):
        atr_i = df['atr'].iloc[i] if pd.notna(df['atr'].iloc[i]) else df['close'].iloc[i]*0.02
        if df['close'].iloc[i] > df['open'].iloc[i] and df['close'].iloc[i-1] < df['open'].iloc[i-1]:
            if df['close'].iloc[i] > df['high'].iloc[i-1]:
                displacement = (df['high'].iloc[i] - df['low'].iloc[i]) > 1.5 * atr_i
                vol_spike = df['volume_ratio'].iloc[i] > 1.5 if 'volume_ratio' in df.columns else True
                strength = 2 if (displacement and vol_spike) else 1
                bullish_blocks.append({
                    'index': i-1,
                    'high': df['high'].iloc[i-1],
                    'low': df['low'].iloc[i-1],
                    'strength': strength
                })
        if df['close'].iloc[i] < df['open'].iloc[i] and df['close'].iloc[i-1] > df['open'].iloc[i-1]:
            if df['close'].iloc[i] < df['low'].iloc[i-1]:
                displacement = (df['high'].iloc[i] - df['low'].iloc[i]) > 1.5 * atr_i
                vol_spike = df['volume_ratio'].iloc[i] > 1.5 if 'volume_ratio' in df.columns else True
                strength = 2 if (displacement and vol_spike) else 1
                bearish_blocks.append({
                    'index': i-1,
                    'high': df['high'].iloc[i-1],
                    'low': df['low'].iloc[i-1],
                    'strength': strength
                })
    return bullish_blocks, bearish_blocks

def is_ob_still_valid(df, ob, current_idx):
    if ob is None:
        return False
    for i in range(ob['index']+1, min(current_idx+1, len(df))):
        if df['high'].iloc[i] >= ob['high'] and df['low'].iloc[i] <= ob['low']:
            return False
    return True

def get_nearest_order_block(df):
    last_close = df.iloc[-1]['close']
    bull_ob, bear_ob = detect_order_blocks(df)
    current_idx = len(df)-1
    valid_bull = [ob for ob in bull_ob if is_ob_still_valid(df, ob, current_idx) and ob['high'] > last_close]
    valid_bear = [ob for ob in bear_ob if is_ob_still_valid(df, ob, current_idx) and ob['low'] < last_close]
    nearest_bull = min(valid_bull, key=lambda x: x['high'] - last_close) if valid_bull else None
    nearest_bear = min(valid_bear, key=lambda x: last_close - x['low']) if valid_bear else None
    return nearest_bull, nearest_bear

# ========== MARKET STRUCTURE ==========
def detect_market_structure(df):
    if df.empty or len(df) < 20:
        return "RANGE", 0, "Data tidak cukup"
    bos_cho, conf, desc = detect_bos_choch(df)
    liq_highs, liq_lows = detect_liquidity_zones(df)
    signals = [desc]
    if liq_highs:
        signals.append(f"Liq High: {min(liq_highs):.0f}")
    if liq_lows:
        signals.append(f"Liq Low: {max(liq_lows):.0f}")
    return bos_cho if "BOS" in bos_cho or "CHOCH" in bos_cho else "RANGE", min(100, conf+10*len(liq_highs+liq_lows)), " | ".join(signals)

# ========== LIQUIDITY ZONES ==========
def detect_liquidity_zones(df, lookback=20):
    if df.empty or len(df) < lookback:
        return [], []
    recent_highs = df['high'].iloc[-lookback:].tolist()
    recent_lows = df['low'].iloc[-lookback:].tolist()
    liq_highs, liq_lows = [], []
    for h in set(recent_highs):
        if recent_highs.count(h) >= 2:
            liq_highs.append(h)
    for l in set(recent_lows):
        if recent_lows.count(l) >= 2:
            liq_lows.append(l)
    return liq_highs, liq_lows

# ========== LIQUIDITY SWEEP ==========
def detect_liquidity_sweep(df, lookback=20):
    if df.empty or len(df) < lookback+5:
        return False, 0, "NONE", "Data tidak cukup"
    last = df.iloc[-1]
    prev_highs = df['high'].iloc[-lookback:-1]
    prev_lows = df['low'].iloc[-lookback:-1]
    resistance = prev_highs.max()
    support = prev_lows.min()
    sweep_type = "NONE"
    confidence = 0
    signals = []
    if last['low'] < support and last['close'] > support:
        sweep_type = "BULLISH_SFP"
        confidence += 50
        signals.append("Sweep bawah + reversal")
    elif last['high'] > resistance and last['close'] < resistance:
        sweep_type = "BEARISH_SFP"
        confidence += 50
        signals.append("Sweep atas + reversal")
    desc = " | ".join(signals) if signals else "No sweep"
    return sweep_type != "NONE", confidence, sweep_type, desc

# ========== MARKET REGIME ==========
def detect_market_regime(df):
    if df.empty or len(df) < 30:
        return "UNKNOWN", 0, "Data tidak cukup"
    last = df.iloc[-1]
    adx = last.get('adx', 0) or 0
    atr_pct = (last.get('atr', 0)/last['close']*100) if last['close'] != 0 else 0
    vol_ratio = last.get('volume_ratio', 1) or 1
    ret_20 = (last['close'] - df['close'].iloc[-21])/df['close'].iloc[-21]*100 if len(df)>20 else 0
    if adx > 35 and atr_pct > 3:
        return "STRONG_TRENDING", 85, f"ADX {adx:.0f}, ATR {atr_pct:.1f}%"
    if adx >= 25:
        return "TRENDING", 70, f"ADX {adx:.0f}"
    if ret_20 < -5 or vol_ratio > 2.5:
        return "PANIC", 80, "Panic selling detected"
    if adx < 20:
        return "SIDEWAYS", 40, "Ranging market"
    return "NEUTRAL", 50, "Normal"

# ========== CANDLESTICK ==========
def detect_candlestick_pattern(df):
    if df.empty or len(df) < 3:
        return "NONE", 0, "Data tidak cukup"
    last = df.iloc[-1]
    prev = df.iloc[-2]
    body = abs(last['close'] - last['open'])
    candle_range = last['high'] - last['low']
    upper_wick = last['high'] - max(last['open'], last['close'])
    lower_wick = min(last['open'], last['close']) - last['low']
    pattern = "NONE"
    conf = 0
    signals = []
    if candle_range > 0:
        if body/candle_range > 0.85:
            if last['close'] > last['open']:
                pattern, conf = "BULLISH_MARUBOZU", 30
                signals.append("Marubozu bullish")
            else:
                pattern, conf = "BEARISH_MARUBOZU", -30
                signals.append("Marubozu bearish")
        if lower_wick > body*2 and upper_wick < body:
            pattern, conf = "HAMMER", 35
            signals.append("Hammer")
        if upper_wick > body*2 and lower_wick < body:
            pattern, conf = "SHOOTING_STAR", -35
            signals.append("Shooting star")
    if last['close'] > last['open'] and prev['close'] < prev['open'] and last['close'] > prev['open'] and last['open'] < prev['close']:
        pattern, conf = "BULLISH_ENGULFING", 40
        signals.append("Bullish engulfing")
    elif last['close'] < last['open'] and prev['close'] > prev['open'] and last['open'] > prev['close'] and last['close'] < prev['open']:
        pattern, conf = "BEARISH_ENGULFING", -40
        signals.append("Bearish engulfing")
    desc = " | ".join(signals) if signals else "No pattern"
    return pattern, conf, desc

# ========== PIVOT SR ==========
def get_pivot_sr(df, lookback=20):
    if df.empty or len(df) < lookback+5:
        last = df.iloc[-1] if not df.empty else None
        if last is not None:
            return last.get('support',0), last.get('resistance',0), 0, 0, 0, 0, 0, 0, 0
        return 0,0,0,0,0,0,0,0,0
    last = df.iloc[-1]
    pivot = (last['high'] + last['low'] + last['close'])/3
    r1 = 2*pivot - last['low']
    r2 = pivot + (last['high'] - last['low'])
    s1 = 2*pivot - last['high']
    s2 = pivot - (last['high'] - last['low'])
    roll_sup = df['low'].rolling(lookback).min().iloc[-1]
    roll_res = df['high'].rolling(lookback).max().iloc[-1]
    high20 = df['high'].iloc[-20:].max()
    low20 = df['low'].iloc[-20:].min()
    rng = high20 - low20
    fib382 = low20 + rng*0.382
    fib618 = low20 + rng*0.618
    return roll_sup, roll_res, pivot, r1, r2, s1, s2, fib382, fib618

# ========== CONFIDENCE SCORE ==========
def calculate_confidence_score(df, ihsg_score=50):
    if df.empty or len(df) < 30:
        return 50, [], "NORMAL"
    last = df.iloc[-1]
    factors = []
    total = 50
    structure, struct_conf, struct_desc = detect_market_structure(df)
    if "BULLISH" in structure:
        total += struct_conf * 0.20
        factors.append(("Structure", struct_conf*0.20, structure))
    elif "BEARISH" in structure:
        total -= struct_conf * 0.20
        factors.append(("Structure", -struct_conf*0.20, structure))
    else:
        factors.append(("Structure", 0, "Neutral"))
    nearest_bull_fvg, nearest_bear_fvg = get_nearest_fvg(df)
    fvg_score = 0
    if nearest_bull_fvg:
        dist = (nearest_bull_fvg['upper'] - last['close']) / last['close'] * 100
        fvg_score = 15 if dist < 2 else 5
        factors.append(("FVG", fvg_score, "Bullish FVG dekat" if dist<2 else "Bullish FVG"))
    elif nearest_bear_fvg:
        dist = (last['close'] - nearest_bear_fvg['lower']) / last['close'] * 100
        fvg_score = -15 if dist < 2 else -5
        factors.append(("FVG", fvg_score, "Bearish FVG dekat" if dist<2 else "Bearish FVG"))
    total += fvg_score
    nearest_bull_ob, nearest_bear_ob = get_nearest_order_block(df)
    ob_score = 0
    if nearest_bull_ob:
        dist = (nearest_bull_ob['high'] - last['close']) / last['close'] * 100
        ob_score = 15 if dist < 2 else 5
        factors.append(("OrderBlock", ob_score, "Bullish OB near" if dist<2 else "Bullish OB"))
    elif nearest_bear_ob:
        dist = (last['close'] - nearest_bear_ob['low']) / last['close'] * 100
        ob_score = -15 if dist < 2 else -5
        factors.append(("OrderBlock", ob_score, "Bearish OB near" if dist<2 else "Bearish OB"))
    total += ob_score
    if last['ema20'] > last['ema50']:
        total += 10
        factors.append(("Trend", 10, "Bullish"))
    else:
        total -= 10
        factors.append(("Trend", -10, "Bearish"))
    mom_score = 5 if last['macd_histogram'] > 0 and last['rsi'] > 50 else (-5 if last['macd_histogram'] < 0 and last['rsi'] < 50 else 0)
    total += mom_score
    factors.append(("Momentum", mom_score, "MACD+RSI"))
    vol_score = 10 if last['volume_ratio'] >= 1.5 else (-10 if last['volume_ratio'] < 0.6 else 0)
    total += vol_score
    factors.append(("Volume", vol_score, f"Vol ratio {last['volume_ratio']:.1f}"))
    adx_val = last.get('adx',0) or 0
    adx_score = 5 if adx_val >= 25 else (-5 if adx_val < 20 else 0)
    total += adx_score
    factors.append(("ADX", adx_score, f"ADX {adx_val:.0f}"))
    regime, regime_conf, _ = detect_market_regime(df)
    if "TRENDING" in regime or "STRONG" in regime:
        total += 10
        factors.append(("Regime", 10, regime))
    elif regime == "PANIC":
        total -= 10
        factors.append(("Regime", -10, regime))
    else:
        factors.append(("Regime", 0, regime))
    market_score = (ihsg_score - 50) * 0.10
    total += market_score
    factors.append(("IHSG", market_score, f"Score {ihsg_score:.0f}"))
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

# ========== ENTRY, SL, TP ==========
def calculate_entry_sl_tp(df, capital=100000000, risk_percent=2):
    if df.empty or len(df) < 30:
        return None, None, None, 0, 0, "NO_SETUP", 0, []
    last = df.iloc[-1]
    atr = last.get('atr', last['close']*0.02)
    if pd.isna(atr) or atr <= 0:
        atr = last['close'] * 0.02
    structure, struct_conf, struct_desc = detect_market_structure(df)
    smart_money, sm_conf, sm_desc = detect_smart_money_volume(df)
    is_sweep, sweep_conf, sweep_type, sweep_desc = detect_liquidity_sweep(df)
    pattern, pattern_conf, pattern_desc = detect_candlestick_pattern(df)
    regime, regime_conf, regime_desc = detect_market_regime(df)
    support, resistance, _, _, _, _, _, _, _ = get_pivot_sr(df)
    nearest_bull_fvg, nearest_bear_fvg = get_nearest_fvg(df)
    nearest_bull_ob, nearest_bear_ob = get_nearest_order_block(df)
    trend_up = last['ema20'] > last['ema50']
    momentum_bullish = last['macd_histogram'] > 0 and last['rsi'] > 50
    volume_spike = last['volume_ratio'] >= 1.5
    strong_trend = last.get('adx', 0) >= 25
    entry_price = None
    stop_loss = None
    take_profit = None
    setup_name = "NO_SETUP"
    confidence = 0
    signals = []
    if "BULLISH" in structure and nearest_bull_fvg and nearest_bull_ob:
        entry_price = last['close']
        stop_loss = entry_price - 1.5 * atr
        take_profit = entry_price + 3 * atr
        setup_name = "SMART_MONEY_COMBO_BUY"
        confidence = 85
        signals = [struct_desc, "Bullish FVG", "Bullish OB"]
    elif "BULLISH" in structure and volume_spike:
        entry_price = last['close']
        stop_loss = entry_price - 2 * atr
        take_profit = entry_price + 3 * atr
        setup_name = "BREAKOUT_BUY"
        confidence = struct_conf
        signals = [struct_desc]
    elif pattern in ["BULLISH_ENGULFING", "HAMMER"] and trend_up:
        entry_price = last['close']
        stop_loss = entry_price - 1.5 * atr
        take_profit = entry_price + 3 * atr
        setup_name = f"{pattern}_BUY"
        confidence = 65 + abs(pattern_conf)//2
        signals = [pattern_desc]
    elif is_sweep and sweep_type == "BULLISH_SFP":
        entry_price = last['close']
        stop_loss = entry_price - 1.5 * atr
        take_profit = entry_price + 2.5 * atr
        setup_name = "LIQUIDITY_SWEEP_BUY"
        confidence = sweep_conf
        signals = [sweep_desc]
    elif nearest_bull_fvg and (nearest_bull_fvg['upper'] - last['close']) / last['close'] < 0.02:
        entry_price = last['close']
        stop_loss = entry_price - 1.5 * atr
        take_profit = entry_price + 2.5 * atr
        setup_name = "FVG_BUY"
        confidence = 70
        signals = ["Near FVG"]
    elif nearest_bull_ob and (nearest_bull_ob['high'] - last['close']) / last['close'] < 0.02:
        entry_price = last['close']
        stop_loss = entry_price - 1.5 * atr
        take_profit = entry_price + 2.5 * atr
        setup_name = "ORDER_BLOCK_BUY"
        confidence = 70
        signals = ["Near OB"]
    elif trend_up and strong_trend and momentum_bullish:
        entry_price = last['close']
        stop_loss = entry_price - 2 * atr
        take_profit = entry_price + 3 * atr
        setup_name = "TREND_BUY"
        confidence = 60
        signals = ["Strong uptrend"]
    else:
        return None, None, None, 0, 0, "NO_SETUP", 0, []
    risk = abs(entry_price - stop_loss)
    reward = abs(take_profit - entry_price)
    rr = reward / risk if risk > 0 else 0
    risk_amount = capital * (risk_percent / 100)
    shares = int(risk_amount / risk) if risk > 0 else 0
    max_shares = int((capital * 0.5) / entry_price) if entry_price > 0 else 0
    shares = min(shares, max_shares)
    return entry_price, stop_loss, take_profit, shares, rr, setup_name, confidence, signals

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

# ========== MULTI TIMEFRAME ==========
def get_multi_timeframe_alignment(symbol, capital=100000000, risk_percent=2):
    timeframes = {"daily": "1d", "hourly": "60m", "fifteen": "15m"}
    results = {}
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
            signals.append(f"{name}: {direction}")
    if len(results) == 3:
        dirs = [results[t]['direction'] for t in results]
        if all(d == "BULLISH" for d in dirs):
            alignment, score = "FULL_BULLISH", 90
        elif all(d == "BEARISH" for d in dirs):
            alignment, score = "FULL_BEARISH", 90
        elif dirs[0] == "BULLISH" and dirs[1] == "BULLISH":
            alignment, score = "BULLISH_DAILY_HOURLY", 75
        elif dirs[0] == "BEARISH" and dirs[1] == "BEARISH":
            alignment, score = "BEARISH_DAILY_HOURLY", 75
        else:
            alignment, score = "MIXED", 40
    else:
        alignment, score = "INSUFFICIENT_DATA", 50
    return results, alignment, score, signals

# ========== IHSG TREND ==========
def get_ihsg_trend():
    try:
        ihsg = get_data("^JKSE", "1d")
        if ihsg.empty or len(ihsg) < 20:
            return "NEUTRAL", 50, "IHSG data unavailable"
        ihsg = add_indicators(ihsg)
        last = ihsg.iloc[-1]
        if last['close'] > last['ema20'] > last['ema50']:
            trend = "BULLISH"
            score = 70
        elif last['close'] < last['ema20'] < last['ema50']:
            trend = "BEARISH"
            score = 30
        else:
            trend = "SIDEWAYS"
            score = 50
        change = ((last['close'] - ihsg.iloc[-2]['close']) / ihsg.iloc[-2]['close']) * 100
        return trend, score, f"IHSG {trend} ({change:+.1f}%)"
    except:
        return "NEUTRAL", 50, "IHSG error"

# ========== SMART MONEY VOLUME ==========
def detect_smart_money_volume(df):
    if df.empty or len(df) < 30:
        return "NEUTRAL", 0, "Data tidak cukup"
    last = df.iloc[-1]
    prev_5 = df.iloc[-6:-1]
    prev_20 = df.iloc[-21:-1]
    signals = []
    conf = 0
    result = "NEUTRAL"
    price_range = (prev_20['high'].max() - prev_20['low'].min()) / prev_20['close'].mean()
    vol_increasing = prev_5['volume'].mean() > prev_20['volume'].mean() * 1.2
    if price_range < 0.03 and vol_increasing:
        result = "ACCUMULATION"
        conf += 40
        signals.append("Akumulasi")
    if last['volume_ratio'] > 1.5 and abs(last['close'] - prev_20['close'].mean())/prev_20['close'].mean() < 0.02:
        result = "DISTRIBUTION"
        conf += 35
        signals.append("Distribusi")
    if last['volume_ratio'] > 2.0 and last['close'] < (last['high']+last['low'])/2:
        conf += 20
        signals.append("Volume spike jual")
    if last['volume_ratio'] > 2.0 and last['close'] > (last['high']+last['low'])/2:
        conf += 20
        signals.append("Volume spike beli")
    desc = " | ".join(signals) if signals else "No smart money volume"
    return result, min(100, conf), desc

# ========== BACKTEST (no lookahead) ==========
def backtest_strategy(df, initial_capital=100000000, risk_per_trade=2, fee_buy=0.0015, fee_sell=0.0025, slippage=0.001):
    if df.empty or len(df) < 30:
        return {"return": 0, "winrate": 0, "trades": 0, "final_capital": initial_capital, "max_drawdown": 0, "profit_factor": 0, "sharpe_ratio": 0, "expectancy": 0, "equity_curve": []}
    df_test = df.copy()
    df_test = add_indicators(df_test)
    capital = initial_capital
    position = 0
    trades = []
    equity = [initial_capital]
    peak = initial_capital
    max_dd = 0
    entry_price_used = 0
    stop_price = 0
    take_price = 0
    for i in range(20, len(df_test)-1):
        snapshot = df_test.iloc[:i+1]
        entry, sl, tp, shares, rr, setup, conf, _ = calculate_entry_sl_tp(snapshot, capital, risk_per_trade)
        open_next = df_test.iloc[i+1]['open']
        close_next = df_test.iloc[i+1]['close']
        low_next = df_test.iloc[i+1]['low']
        high_next = df_test.iloc[i+1]['high']
        if entry and "BUY" in setup and position == 0 and shares > 0:
            entry_price_used = open_next * (1 + slippage)
            cost = shares * entry_price_used * (1 + fee_buy)
            if cost <= capital:
                position = shares
                capital -= cost
                stop_price = sl
                take_price = tp
        elif position > 0:
            if low_next <= stop_price:
                exit_p = stop_price * (1 - slippage)
                capital += position * exit_p * (1 - fee_sell)
                pnl = (exit_p - entry_price_used) / entry_price_used * 100
                trades.append(pnl)
                position = 0
            elif high_next >= take_price:
                exit_p = take_price * (1 - slippage)
                capital += position * exit_p * (1 - fee_sell)
                pnl = (exit_p - entry_price_used) / entry_price_used * 100
                trades.append(pnl)
                position = 0
        current_eq = capital + (position * close_next if position else 0)
        equity.append(current_eq)
        if current_eq > peak:
            peak = current_eq
        dd = (peak - current_eq) / peak * 100 if peak > 0 else 0
        max_dd = max(max_dd, dd)
    if position > 0:
        last_close = df_test.iloc[-1]['close']
        capital += position * last_close * (1 - fee_sell)
        equity.append(capital)
    win_trades = [t for t in trades if t > 0]
    loss_trades = [t for t in trades if t < 0]
    winrate = len(win_trades)/len(trades)*100 if trades else 0
    gross_profit = sum(win_trades) if win_trades else 0
    gross_loss = abs(sum(loss_trades)) if loss_trades else 0
    pf = gross_profit/gross_loss if gross_loss else 0
    total_return = (capital - initial_capital)/initial_capital*100
    returns = np.diff(equity)/equity[:-1]
    sharpe = (np.mean(returns)/np.std(returns)*np.sqrt(252)) if len(returns)>0 and np.std(returns)>0 else 0
    expectancy = np.mean(trades) if trades else 0
    return {
        "return": round(total_return, 2),
        "winrate": round(winrate, 2),
        "trades": len(trades),
        "final_capital": round(capital, 0),
        "max_drawdown": round(max_dd, 2),
        "profit_factor": round(pf, 2),
        "sharpe_ratio": round(sharpe, 2),
        "expectancy": round(expectancy, 2),
        "equity_curve": equity[-100:]
    }

# ========== SCANNER ==========
def scan_saham():
    try:
        from scanner_engine import scan_saham_fast
        return scan_saham_fast()
    except ImportError:
        # Fallback ke cara lama jika scanner_engine.py tidak ditemukan
        stocks = [
            # LEVEL 1: MSCI Global Standard (17 saham)
            "BBCA.JK", "BBRI.JK", "BMRI.JK", "BBNI.JK",
            "TLKM.JK", "ASII.JK", "UNTR.JK", "ICBP.JK",
            "INDF.JK", "KLBF.JK", "SMGR.JK", "CTRA.JK",
            "SMRA.JK", "PTBA.JK", "CPIN.JK", "GOTO.JK",
            "MDKA.JK",
            # LEVEL 2: LQ45 Likuid Lainnya (13 saham)
            "ADRO.JK", "ANTM.JK", "AKRA.JK", "BRIS.JK",
            "INCO.JK", "ITMG.JK", "JPFA.JK", "MAPI.JK",
            "MEDC.JK", "PGAS.JK", "TOWR.JK", "EXCL.JK",
            "ISAT.JK",
            # LEVEL 3: Energi & Metal (10 saham)
            "AMMN.JK", "BYAN.JK", "TPIA.JK", "DSSA.JK",
            "CUAN.JK", "ADMR.JK", "AADI.JK", "PGEO.JK",
            "BRPT.JK", "ESSA.JK",
        ]
        
        results = []
        for stock in stocks:
            try:
                df = get_data(stock, "1d")
                if not df.empty and len(df) > 30:
                    df = add_indicators(df)
                    setup, quality, msg = detect_high_quality_setup(df)
                    if quality >= 60:
                        signal = "🔥 BUY" if "BUY" in setup else "⏸️ HOLD"
                        results.append({
                            "Kode": stock,
                            "Score": f"{quality:.0f}",
                            "Setup": msg[:40],
                            "Sinyal": signal,
                            "Harga": f"Rp{df.iloc[-1]['close']:,.0f}"
                        })
                time.sleep(0.3)
            except:
                continue
        
        results.sort(key=lambda x: int(x['Score']), reverse=True)
        return results[:10]
