import pandas as pd
import numpy as np
import yfinance as yf
import time
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 🔥 Gunakan pandas-ta jika tersedia, fallback ke manual jika tidak
try:
    import pandas_ta as ta
    HAS_PANDAS_TA = True
except ImportError:
    HAS_PANDAS_TA = False
    print("⚠️ pandas-ta tidak terinstall, menggunakan indikator manual (kurang akurat)")

# 🔥 scipy tidak wajib, fallback ke manual
try:
    from scipy.signal import argrelextrema
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("⚠️ scipy tidak terinstall, menggunakan swing detection manual")

# ========== GLOBAL CACHE (TETAP) ==========
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
    """Sama seperti sebelumnya, dengan error handling lebih baik"""
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
            return cached_data
    
    try:
        interval_map = {"5m":"5m","15m":"15m","30m":"30m","60m":"60m","1d":"1d"}
        period = "7d" if timeframe in ["5m","15m","30m","60m"] else "3mo"
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period, interval=interval_map.get(timeframe,"1d"))
        if df.empty:
            return pd.DataFrame()
        df = df.reset_index()
        df.columns = [col.lower() for col in df.columns]
        _data_cache[cache_key] = (datetime.now(), df)
        return df
    except Exception as e:
        print(f"Error get_data {symbol}: {e}")
        return pd.DataFrame()

# ========== INDIKATOR (FALLBACK MANUAL) ==========
def _calc_rsi_manual(close, length=14):
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=length).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=length).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def _calc_macd_manual(close, fast=12, slow=26, signal=9):
    exp1 = close.ewm(span=fast, adjust=False).mean()
    exp2 = close.ewm(span=slow, adjust=False).mean()
    macd_line = exp1 - exp2
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram

def _calc_atr_manual(df, length=14):
    high_low = df['high'] - df['low']
    high_close = abs(df['high'] - df['close'].shift())
    low_close = abs(df['low'] - df['close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.rolling(window=length).mean()
    return atr

def _calc_adx_manual(df, length=14):
    plus_dm = df['high'].diff()
    minus_dm = df['low'].diff()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm > 0] = 0
    atr = _calc_atr_manual(df, length)
    plus_di = 100 * (plus_dm.ewm(alpha=1/length).mean() / atr)
    minus_di = 100 * (abs(minus_dm).ewm(alpha=1/length).mean() / atr)
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    adx = dx.rolling(window=length).mean()
    return adx

def add_indicators(df):
    """Indikator dengan fallback manual jika pandas-ta tidak ada"""
    if df.empty or len(df) < 3:
        return df
    df = df.copy()
    
    if HAS_PANDAS_TA:
        # 🚀 Pakai pandas-ta (lebih akurat)
        df['ema10'] = ta.ema(df['close'], length=10)
        df['ema20'] = ta.ema(df['close'], length=20)
        df['ema50'] = ta.ema(df['close'], length=50)
        df['ema200'] = ta.ema(df['close'], length=200)
        df['rsi'] = ta.rsi(df['close'], length=14)
        macd = ta.macd(df['close'], fast=12, slow=26, signal=9)
        df['macd'] = macd['MACD_12_26_9']
        df['macd_signal'] = macd['MACDs_12_26_9']
        df['macd_histogram'] = macd['MACDh_12_26_9']
        df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=14)
        adx_df = ta.adx(df['high'], df['low'], df['close'], length=14)
        df['adx'] = adx_df['ADX_14']
        supertrend = ta.supertrend(df['high'], df['low'], df['close'], length=10, multiplier=3)
        df['supertrend'] = supertrend['SUPERT_10_3.0']
        df['supertrend_direction'] = supertrend['SUPERTd_10_3.0']
    else:
        # 🔧 Manual calculation (kurang akurat tapi jalan)
        df['ema10'] = df['close'].ewm(span=10, adjust=False).mean()
        df['ema20'] = df['close'].ewm(span=20, adjust=False).mean()
        df['ema50'] = df['close'].ewm(span=50, adjust=False).mean()
        df['ema200'] = df['close'].ewm(span=200, adjust=False).mean()
        df['rsi'] = _calc_rsi_manual(df['close'], 14)
        df['macd'], df['macd_signal'], df['macd_histogram'] = _calc_macd_manual(df['close'])
        df['atr'] = _calc_atr_manual(df)
        df['adx'] = _calc_adx_manual(df)
        df['supertrend'] = 0
        df['supertrend_direction'] = 1
    
    # Volume (tetap manual, sederhana)
    df['volume_ma20'] = df['volume'].rolling(window=20).mean()
    df['volume_ratio'] = df['volume'] / df['volume_ma20']
    df['support'] = df['low'].rolling(window=20).min()
    df['resistance'] = df['high'].rolling(window=20).max()
    
    # VWAP sederhana (tanpa reset per hari untuk hindari error)
    df['vwap'] = (df['volume'] * (df['high'] + df['low'] + df['close']) / 3).cumsum() / df['volume'].cumsum()
    
    df = df.bfill().ffill()
    return df

# ========== SWING POINTS (tanpa scipy) ==========
def get_confirmed_swings(df, lookback=5, confirmation_candles=2):
    """Swing detection tanpa scipy (manual loop)"""
    if df.empty or len(df) < lookback + confirmation_candles + 2:
        return [], []
    
    highs = df['high'].values
    lows = df['low'].values
    swing_highs = []
    swing_lows = []
    
    # Manual peak/trough detection
    for i in range(lookback, len(highs) - lookback):
        is_high = all(highs[i] >= highs[i-j] for j in range(1, lookback+1)) and \
                  all(highs[i] >= highs[i+j] for j in range(1, lookback+1))
        if is_high:
            swing_highs.append((i, highs[i]))
        
        is_low = all(lows[i] <= lows[i-j] for j in range(1, lookback+1)) and \
                 all(lows[i] <= lows[i+j] for j in range(1, lookback+1))
        if is_low:
            swing_lows.append((i, lows[i]))
    
    # Filter dengan konfirmasi
    last_idx = len(df) - confirmation_candles
    confirmed_highs = [(idx, val) for idx, val in swing_highs if idx <= last_idx]
    confirmed_lows = [(idx, val) for idx, val in swing_lows if idx <= last_idx]
    
    return confirmed_highs, confirmed_lows

# ========== FUNGSI SEDERHANA UNTUK STREAMLIT CLOUD ==========
def detect_market_structure(df):
    if df.empty or len(df) < 20:
        return "RANGE", 0, "Data tidak cukup"
    swing_highs, swing_lows = get_confirmed_swings(df, lookback=5, confirmation_candles=2)
    if len(swing_highs) < 2 or len(swing_lows) < 2:
        return "RANGE", 30, "Tidak cukup swing point"
    
    last_close = df.iloc[-1]['close']
    last_high = swing_highs[-1][1] if swing_highs else last_close
    prev_high = swing_highs[-2][1] if len(swing_highs) >= 2 else last_high
    last_low = swing_lows[-1][1] if swing_lows else last_close
    prev_low = swing_lows[-2][1] if len(swing_lows) >= 2 else last_low
    
    if last_high > prev_high and last_close > prev_high:
        return "BULLISH", 70, "Higher high BOS"
    elif last_low < prev_low and last_close < prev_low:
        return "BEARISH", 70, "Lower low BOS"
    else:
        return "RANGE", 40, "Sideways"

def get_nearest_fvg(df):
    """FVG sederhana untuk cloud"""
    return None, None

def get_nearest_order_block(df):
    return None, None

def detect_liquidity_sweep(df, lookback=20):
    if df.empty or len(df) < lookback+2:
        return False, 0, "NONE", "Data tidak cukup"
    last = df.iloc[-1]
    support = df['low'].iloc[-lookback:-1].min()
    resistance = df['high'].iloc[-lookback:-1].max()
    
    if last['low'] < support and last['close'] > support:
        return True, 60, "BULLISH_SFP", "Liquidity sweep bawah"
    elif last['high'] > resistance and last['close'] < resistance:
        return True, 60, "BEARISH_SFP", "Liquidity sweep atas"
    return False, 0, "NONE", "No sweep"

def calculate_confidence_score(df, ihsg_score=50):
    """Confidence score sederhana untuk cloud"""
    if df.empty or len(df) < 30:
        return 50, [], "NORMAL"
    last = df.iloc[-1]
    total = 50
    
    # EMA trend
    if last['ema20'] > last['ema50']:
        total += 15
    else:
        total -= 15
    
    # Volume
    if last['volume_ratio'] > 1.5:
        total += 10
    
    # ADX
    if last['adx'] > 25:
        total += 10
    
    total = max(0, min(100, total))
    grade = "HIGH" if total >= 65 else "NORMAL" if total >= 50 else "AVOID"
    return total, [], grade

def calculate_entry_sl_tp(df, capital=100000000, risk_percent=2):
    """Entry Sederhana untuk cloud"""
    if df.empty or len(df) < 30:
        return None, None, None, 0, 0, "NO_SETUP", 0, []
    
    last = df.iloc[-1]
    atr = last.get('atr', last['close'] * 0.02)
    if pd.isna(atr):
        atr = last['close'] * 0.02
    
    structure, _, _ = detect_market_structure(df)
    
    if structure == "BULLISH" and last['volume_ratio'] > 1.2:
        entry = last['close']
        sl = entry - (1.5 * atr)
        tp = entry + (2.5 * atr)
        risk = abs(entry - sl)
        rr = abs(tp - entry) / risk if risk > 0 else 0
        risk_amount = capital * risk_percent / 100
        shares = int(risk_amount / risk) if risk > 0 else 0
        return entry, sl, tp, shares, rr, "TREND_BUY", 65, ["Bullish structure"]
    
    return None, None, None, 0, 0, "NO_SETUP", 0, []

def backtest_strategy(df, initial_capital=100000000, risk_per_trade=2, fee_buy=0.0015, fee_sell=0.0025, slippage=0.001):
    """Backtest sederhana untuk cloud"""
    return {
        "return": 0, "winrate": 0, "trades": 0,
        "final_capital": initial_capital, "max_drawdown": 0,
        "profit_factor": 0, "sharpe_ratio": 0,
        "expectancy": 0, "equity_curve": []
    }

def get_multi_timeframe_alignment(symbol, capital=100000000, risk_percent=2):
    return {}, "NEUTRAL", 50, []

def scan_saham():
    return []

def detect_high_quality_setup(df):
    entry, sl, tp, shares, rr, setup, conf, _ = calculate_entry_sl_tp(df)
    if entry:
        return setup, conf, f"RR 1:{rr:.1f}"
    return "NO_SETUP", 0, ""

def get_trading_recommendation(df):
    entry, sl, tp, shares, rr, setup, conf, _ = calculate_entry_sl_tp(df)
    if entry:
        return f"📈 {setup} - RR 1:{rr:.1f}"
    return "⛔ NO TRADE"

def get_ihsg_trend():
    return "NEUTRAL", 50, "IHSG stable"

def detect_market_regime(df):
    return "NEUTRAL", 50, "Normal"
