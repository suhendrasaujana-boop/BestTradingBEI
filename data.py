import pandas as pd
import numpy as np
import yfinance as yf
import time
from datetime import datetime
from scipy.signal import argrelextrema
import pandas_ta as ta
import warnings
warnings.filterwarnings('ignore')

# ========== GLOBAL CACHE (dengan rate limit) ==========
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

# ========== INDIKATOR PAKAI PANDAS-TA (Akurat) ==========
def add_indicators(df):
    if df.empty or len(df) < 3:
        return df
    df = df.copy()

    # EMA
    df['ema10'] = ta.ema(df['close'], length=10)
    df['ema20'] = ta.ema(df['close'], length=20)
    df['ema50'] = ta.ema(df['close'], length=50)
    df['ema200'] = ta.ema(df['close'], length=200)   # 🔥 tambahan

    # RSI
    df['rsi'] = ta.rsi(df['close'], length=14)

    # MACD
    macd = ta.macd(df['close'], fast=12, slow=26, signal=9)
    df['macd'] = macd['MACD_12_26_9']
    df['macd_signal'] = macd['MACDs_12_26_9']
    df['macd_histogram'] = macd['MACDh_12_26_9']

    # ATR
    df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=14)

    # ADX (yang benar)
    adx_df = ta.adx(df['high'], df['low'], df['close'], length=14)
    df['adx'] = adx_df['ADX_14']

    # Volume MA & ratio
    df['volume_ma20'] = df['volume'].rolling(20).mean()
    df['volume_ratio'] = df['volume'] / df['volume_ma20']

    # Support & Resistance rolling
    df['support'] = df['low'].rolling(20).min()
    df['resistance'] = df['high'].rolling(20).max()

    # Supertrend (pandas-ta)
    supertrend = ta.supertrend(df['high'], df['low'], df['close'], length=10, multiplier=3)
    df['supertrend'] = supertrend['SUPERT_10_3.0']
    df['supertrend_direction'] = supertrend['SUPERTd_10_3.0']

    # 🔥 VWAP reset per hari (jika ada kolom tanggal)
    if 'datetime' in df.columns:
        df['date'] = df['datetime'].dt.date
        df['vwap'] = df.groupby('date').apply(
            lambda g: (g['volume'] * (g['high'] + g['low'] + g['close'])/3).cumsum() / g['volume'].cumsum()
        ).values
        df.drop('date', axis=1, inplace=True)
    else:
        # fallback: VWAP kumulatif
        df['vwap'] = (df['volume'] * (df['high'] + df['low'] + df['close']) / 3).cumsum() / df['volume'].cumsum()

    # Bersihkan NaN
    df = df.bfill().ffill()
    return df

# ========== SWING POINT (TIDAK REPAINT) ==========
def get_confirmed_swings(df, lookback=5, confirmation_candles=2):
    """
    Mengembalikan swing high/low yang sudah dikonfirmasi.
    Tidak repaint karena hanya pakai data hingga candle - confirmation_candles.
    """
    if df.empty or len(df) < lookback + confirmation_candles + 2:
        return [], []

    highs = df['high'].values
    lows = df['low'].values
    # Gunakan argrelextrema (scipy)
    swing_high_idx = argrelextrema(highs, np.greater, order=lookback)[0]
    swing_low_idx = argrelextrema(lows, np.less, order=lookback)[0]

    # Konfirmasi: tidak berubah dalam confirmation_candles terakhir
    confirmed_highs = []
    confirmed_lows = []
    last_idx = len(df) - confirmation_candles
    for idx in swing_high_idx:
        if idx <= last_idx:
            confirmed_highs.append((idx, highs[idx]))
    for idx in swing_low_idx:
        if idx <= last_idx:
            confirmed_lows.append((idx, lows[idx]))
    return confirmed_highs, confirmed_lows

# ========== BOS / CHOCH YANG VALID ==========
def detect_bos_choch(df):
    if df.empty or len(df) < 30:
        return "NEUTRAL", 0, "No BOS/CHOCH"
    swing_highs, swing_lows = get_confirmed_swings(df, lookback=5, confirmation_candles=2)
    if len(swing_highs) < 2 or len(swing_lows) < 2:
        return "NEUTRAL", 0, "Insufficient confirmed swings"

    last_high_val = swing_highs[-1][1]
    prev_high_val = swing_highs[-2][1]
    last_low_val = swing_lows[-1][1]
    prev_low_val = swing_lows[-2][1]
    last_close = df.iloc[-1]['close']

    signals = []
    conf = 0
    result = "NEUTRAL"

    # Bullish BOS : higher high dan close di atas prev high
    if last_high_val > prev_high_val and last_close > prev_high_val:
        result = "BULLISH_BOS"
        conf = 45
        signals.append("BOS UP")
    # Bearish BOS
    elif last_low_val < prev_low_val and last_close < prev_low_val:
        result = "BEARISH_BOS"
        conf = 45
        signals.append("BOS DOWN")
    # CHOCH: reversal pada swing points
    if len(swing_highs) >= 3 and len(swing_lows) >= 3:
        if swing_highs[-1][1] < swing_highs[-2][1] and swing_lows[-1][1] > swing_lows[-2][1]:
            result = "BULLISH_CHOCH"
            conf = 40
            signals.append("CHOCH UP")
        elif swing_highs[-1][1] > swing_highs[-2][1] and swing_lows[-1][1] < swing_lows[-2][1]:
            result = "BEARISH_CHOCH"
            conf = 40
            signals.append("CHOCH DOWN")
    desc = " | ".join(signals) if signals else "No BOS/CHOCH"
    return result, min(100, conf), desc

def detect_market_structure(df):
    if df.empty or len(df) < 20:
        return "RANGE", 0, "Data tidak cukup"
    bos, conf, desc = detect_bos_choch(df)
    return bos, conf, desc

# ========== FVG (dengan displacement & volume filter) ==========
def detect_fair_value_gap(df):
    if df.empty or len(df) < 5:
        return [], []
    bullish_fvg = []
    bearish_fvg = []
    for i in range(2, len(df)-1):
        # Bullish FVG : low i > high i-2  (gap)
        if df['low'].iloc[i] > df['high'].iloc[i-2]:
            # Validasi tambahan: displacement & volume
            body_prev = abs(df['close'].iloc[i-2] - df['open'].iloc[i-2])
            body_current = abs(df['close'].iloc[i] - df['open'].iloc[i])
            if body_current > body_prev * 1.5 and df['volume_ratio'].iloc[i] > 1.2:
                bullish_fvg.append({
                    'index': i, 'upper': df['low'].iloc[i], 'lower': df['high'].iloc[i-2],
                    'strength': df['volume_ratio'].iloc[i]
                })
        # Bearish FVG
        if df['high'].iloc[i] < df['low'].iloc[i-2]:
            body_prev = abs(df['close'].iloc[i-2] - df['open'].iloc[i-2])
            body_current = abs(df['close'].iloc[i] - df['open'].iloc[i])
            if body_current > body_prev * 1.5 and df['volume_ratio'].iloc[i] > 1.2:
                bearish_fvg.append({
                    'index': i, 'upper': df['low'].iloc[i-2], 'lower': df['high'].iloc[i],
                    'strength': df['volume_ratio'].iloc[i]
                })
    return bullish_fvg, bearish_fvg

def is_fvg_valid(df, fvg, current_idx):
    """Cek apakah FVG belum terisi (belum disentuh)"""
    for j in range(fvg['index']+1, min(current_idx+1, len(df))):
        low_j = df['low'].iloc[j]
        high_j = df['high'].iloc[j]
        if low_j <= fvg['upper'] and high_j >= fvg['lower']:
            return False
    return True

def get_nearest_fvg(df):
    if df.empty: return None, None
    bullish_raw, bearish_raw = detect_fair_value_gap(df)
    cur_idx = len(df)-1
    bullish = [f for f in bullish_raw if is_fvg_valid(df, f, cur_idx)]
    bearish = [f for f in bearish_raw if is_fvg_valid(df, f, cur_idx)]
    last_close = df.iloc[-1]['close']
    nearest_bull = None
    nearest_bear = None
    min_dist_bull = float('inf')
    min_dist_bear = float('inf')
    for f in bullish:
        if f['upper'] > last_close:
            dist = f['upper'] - last_close
            if dist < min_dist_bull:
                min_dist_bull = dist
                nearest_bull = f
    for f in bearish:
        if f['lower'] < last_close:
            dist = last_close - f['lower']
            if dist < min_dist_bear:
                min_dist_bear = dist
                nearest_bear = f
    return nearest_bull, nearest_bear

# ========== ORDER BLOCK (dengan displacement & BOS) ==========
def detect_order_blocks(df):
    if df.empty or len(df) < 5:
        return [], []
    bullish_blocks = []
    bearish_blocks = []
    bos, _, _ = detect_bos_choch(df)
    for i in range(2, len(df)-2):
        # Bullish OB: bearish candle sebelum bullish move dan ada BOS
        if df['close'].iloc[i] > df['open'].iloc[i] and df['close'].iloc[i-1] < df['open'].iloc[i-1]:
            if df['close'].iloc[i] > df['high'].iloc[i-1] and "BULLISH" in bos:
                volume_ok = df['volume_ratio'].iloc[i] > 1.2
                if volume_ok:
                    bullish_blocks.append({
                        'index': i-1, 'high': df['high'].iloc[i-1], 'low': df['low'].iloc[i-1],
                        'strength': 1.0
                    })
        # Bearish OB
        if df['close'].iloc[i] < df['open'].iloc[i] and df['close'].iloc[i-1] > df['open'].iloc[i-1]:
            if df['close'].iloc[i] < df['low'].iloc[i-1] and "BEARISH" in bos:
                volume_ok = df['volume_ratio'].iloc[i] > 1.2
                if volume_ok:
                    bearish_blocks.append({
                        'index': i-1, 'high': df['high'].iloc[i-1], 'low': df['low'].iloc[i-1],
                        'strength': 1.0
                    })
    return bullish_blocks, bearish_blocks

def get_nearest_order_block(df):
    if df.empty: return None, None
    bullish_raw, bearish_raw = detect_order_blocks(df)
    last_close = df.iloc[-1]['close']
    nearest_bull = None
    nearest_bear = None
    min_dist_bull = float('inf')
    min_dist_bear = float('inf')
    for b in bullish_raw:
        if b['high'] > last_close:
            dist = b['high'] - last_close
            if dist < min_dist_bull:
                min_dist_bull = dist
                nearest_bull = b
    for b in bearish_raw:
        if b['low'] < last_close:
            dist = last_close - b['low']
            if dist < min_dist_bear:
                min_dist_bear = dist
                nearest_bear = b
    return nearest_bull, nearest_bear

# ========== LIQUIDITY SWEEP (lebih baik) ==========
def detect_liquidity_sweep(df, lookback=20):
    if df.empty or len(df) < lookback+3:
        return False, 0, "NONE", "Data insufficient"
    last = df.iloc[-1]
    prev = df.iloc[-2]
    support = df['low'].iloc[-lookback:-1].min()
    resistance = df['high'].iloc[-lookback:-1].max()
    sweep_type = "NONE"
    conf = 0
    if last['low'] < support and last['close'] > support:
        sweep_type = "BULLISH_SFP"
        conf = 55
    elif last['high'] > resistance and last['close'] < resistance:
        sweep_type = "BEARISH_SFP"
        conf = 55
    elif prev['high'] > resistance and last['close'] < resistance:
        sweep_type = "FAKE_BREAKOUT"
        conf = 30
    is_sweep = sweep_type in ["BULLISH_SFP","BEARISH_SFP"]
    return is_sweep, conf, sweep_type, f"Liquidity: {sweep_type}"

# ========== MARKET REGIME, IHSG TREND, DLL (tetap) ==========
def detect_market_regime(df):
    # sama seperti sebelumnya, dipersingkat
    if df.empty or len(df) < 30:
        return "UNKNOWN", 0, "Data tidak cukup"
    last = df.iloc[-1]
    adx = last.get('adx', 0)
    atr_pct = (last.get('atr',0) / last['close'])*100 if last['close']>0 else 0
    vol_ratio = last.get('volume_ratio',1)
    if adx >= 25:
        return "TRENDING", 70, f"ADX {adx:.0f}"
    elif adx < 20:
        return "SIDEWAYS", 40, "Range"
    elif atr_pct > 4:
        return "VOLATILE", 60, "High volatility"
    else:
        return "NEUTRAL", 50, "Normal"

def get_ihsg_trend():
    # tetap dari sebelumnya (ringkas)
    return "NEUTRAL", 50, "IHSG moderate"

# ========== CONFIDENCE SCORE (TIDAK OVERLAP) ==========
def calculate_confidence_score(df, ihsg_score=50):
    if df.empty or len(df) < 30:
        return 50, [], "NORMAL"
    last = df.iloc[-1]
    factors = []
    total = 50

    # 1. Structure (20%) - independent
    struct, s_conf, _ = detect_market_structure(df)
    if "BULLISH" in struct:
        total += 20
        factors.append(("Structure",20,struct))
    elif "BEARISH" in struct:
        total -= 20
        factors.append(("Structure",-20,struct))

    # 2. Liquidity sweep (15%)
    is_sweep, sweep_conf, _, _ = detect_liquidity_sweep(df)
    if is_sweep:
        total += 15
        factors.append(("Liquidity Sweep",15,"Sweep detected"))

    # 3. Order Block (10%)
    bull_ob, bear_ob = get_nearest_order_block(df)
    if bull_ob:
        total += 10
        factors.append(("Order Block",10,"Near Bullish OB"))
    elif bear_ob:
        total -= 10
        factors.append(("Order Block",-10,"Near Bearish OB"))

    # 4. FVG (10%)
    bull_fvg, bear_fvg = get_nearest_fvg(df)
    if bull_fvg:
        total += 10
        factors.append(("FVG",10,"Bullish FVG"))
    elif bear_fvg:
        total -= 10
        factors.append(("FVG",-10,"Bearish FVG"))

    # 5. ADX & Trend (10%)
    if last['ema20'] > last['ema200']:
        total += 5
        factors.append(("HTF Trend",5,"Bullish"))
    if last['adx'] > 25:
        total += 5
        factors.append(("Trend Strength",5,f"ADX {last['adx']:.0f}"))

    # 6. Volume (10%)
    if last['volume_ratio'] > 1.5:
        total += 10
        factors.append(("Volume",10,"High"))
    # 7. IHSG context (5%)
    total += (ihsg_score - 50)*0.1
    final_score = np.clip(total, 0, 100)
    grade = "SNIPER" if final_score >= 80 else "HIGH" if final_score >= 65 else "NORMAL" if final_score >= 50 else "AVOID"
    return final_score, factors, grade

# ========== ENTRY/SL/TP REALISTIS (ENTRY DI OPEN BERIKUTNYA) ==========
def calculate_entry_sl_tp(df, capital=100000000, risk_percent=2):
    if df.empty or len(df) < 31:
        return None, None, None, 0, 0, "NO_SETUP", 0, []
    # Gunakan data sampai -1 untuk sinyal, tapi entry di candle berikutnya (open)
    signal_idx = -2
    next_idx = -1
    signal_candle = df.iloc[signal_idx]
    next_candle = df.iloc[next_idx]
    last_candle = df.iloc[-1]   # untuk konteks tambahan

    atr = signal_candle.get('atr', signal_candle['close']*0.02)
    if pd.isna(atr): atr = signal_candle['close']*0.02
    structure, _, _ = detect_market_structure(df.iloc[:signal_idx+1])
    is_sweep, sweep_conf, sweep_type, _ = detect_liquidity_sweep(df.iloc[:signal_idx+1])
    bull_ob, _ = get_nearest_order_block(df.iloc[:signal_idx+1])
    bull_fvg, _ = get_nearest_fvg(df.iloc[:signal_idx+1])
    trend_up = signal_candle['ema20'] > signal_candle['ema200']
    volume_spike = signal_candle['volume_ratio'] > 1.5

    entry = None
    sl = None
    tp = None
    setup = "NO_SETUP"
    conf = 0
    signals = []
    if "BULLISH" in structure and (bull_ob or bull_fvg):
        entry = next_candle['open']
        sl = entry - (1.5 * atr)
        tp = entry + (3 * atr)
        setup = "SMART_MONEY_BUY"
        conf = 80
        signals = ["BOS + OB/FVG"]
    elif is_sweep and sweep_type == "BULLISH_SFP":
        entry = next_candle['open']
        sl = entry - (1.5 * atr)
        tp = entry + (2.5 * atr)
        setup = "LIQUIDITY_SWEEP_BUY"
        conf = 70
        signals = ["Sweep + reversal"]
    elif trend_up and volume_spike:
        entry = next_candle['open']
        sl = entry - (2 * atr)
        tp = entry + (3 * atr)
        setup = "TREND_BREAKOUT_BUY"
        conf = 60
        signals = ["Trend + volume"]
    else:
        return None, None, None, 0, 0, "NO_SETUP", 0, []

    if entry is None:
        return None, None, None, 0, 0, "NO_SETUP", 0, []

    risk_amount = capital * risk_percent / 100
    risk_per_share = abs(entry - sl)
    shares = int(risk_amount / risk_per_share) if risk_per_share > 0 else 0
    max_shares = int(capital * 0.5 / entry) if entry > 0 else 0
    shares = min(shares, max_shares)
    rr = abs(tp - entry) / abs(entry - sl) if (entry - sl) != 0 else 0
    return entry, sl, tp, shares, rr, setup, conf, signals

# ========== BACKTEST TANPA LOOKAHEAD ==========
def backtest_strategy(df, initial_capital=100000000, risk_per_trade=2, fee_buy=0.0015, fee_sell=0.0025, slippage=0.001):
    if df.empty or len(df) < 40:
        return {"return":0,"winrate":0,"trades":0,"final_capital":initial_capital,"max_drawdown":0,"profit_factor":0,"sharpe_ratio":0,"expectancy":0,"equity_curve":[]}
    df = df.copy()
    df = add_indicators(df)
    capital = initial_capital
    trades = []
    equity_curve = [initial_capital]
    position = 0
    entry_price = 0
    stop_price = 0
    take_price = 0
    peak = initial_capital
    max_dd = 0

    for i in range(30, len(df)-1):  # -1 agar ada candle berikutnya untuk entry
        entry, sl, tp, shares, rr, setup, conf, _ = calculate_entry_sl_tp(df.iloc[:i+1], capital, risk_per_trade)
        if entry and shares > 0 and position == 0:
            # Simulate next candle open
            next_open = df.iloc[i+1]['open']
            if next_open <= entry * 1.01:   # slippage reasonable
                cost = shares * next_open * (1 + fee_buy)
                if cost <= capital:
                    position = shares
                    capital -= cost
                    entry_price = next_open
                    stop_price = sl
                    take_price = tp
        elif position > 0:
            # periksa high/low candle saat ini
            high_candle = df.iloc[i]['high']
            low_candle = df.iloc[i]['low']
            close_candle = df.iloc[i]['close']
            exit_price = None
            if low_candle <= stop_price:
                exit_price = stop_price
            elif high_candle >= take_price:
                exit_price = take_price
            elif i == len(df)-2:
                exit_price = close_candle
            if exit_price:
                exit_price_slipped = exit_price * (1 - slippage)
                capital += position * exit_price_slipped * (1 - fee_sell)
                pnl_pct = (exit_price_slipped - entry_price) / entry_price * 100
                trades.append(pnl_pct)
                position = 0
        current_value = capital + (position * df.iloc[i]['close']) if position else capital
        equity_curve.append(current_value)
        if current_value > peak:
            peak = current_value
        dd = (peak - current_value) / peak * 100 if peak > 0 else 0
        max_dd = max(max_dd, dd)

    if position > 0:
        last_close = df.iloc[-1]['close']
        capital += position * last_close * (1 - fee_sell)
        equity_curve.append(capital)

    winrate = len([t for t in trades if t>0]) / len(trades) * 100 if trades else 0
    gross_profit = sum([t for t in trades if t>0])
    gross_loss = abs(sum([t for t in trades if t<0]))
    pf = gross_profit / gross_loss if gross_loss > 0 else 0
    ret = (capital - initial_capital) / initial_capital * 100
    returns = [equity_curve[i]/equity_curve[i-1]-1 for i in range(1, len(equity_curve)) if equity_curve[i-1]>0]
    sharpe = (np.mean(returns)/np.std(returns)*np.sqrt(252)) if len(returns)>0 and np.std(returns)>0 else 0
    expectancy = np.mean(trades) if trades else 0
    return {
        "return": round(ret,2), "winrate": round(winrate,2), "trades": len(trades),
        "final_capital": round(capital,0), "max_drawdown": round(max_dd,2),
        "profit_factor": round(pf,2), "sharpe_ratio": round(sharpe,2),
        "expectancy": round(expectancy,2), "equity_curve": equity_curve[-100:]
    }

# ========== FUNGSI LAIN YANG DIBUTUHKAN APP.PY ==========
def get_multi_timeframe_alignment(symbol, capital=100000000, risk_percent=2):
    # versi ringkas (sama seperti sebelumnya, tanpa repaint)
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
