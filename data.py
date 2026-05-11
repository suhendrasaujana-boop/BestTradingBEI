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
        interval_map = {"5m": "5m", "15m": "15m", "30m": "30m", "60m": "60m", "1d": "1d", "4h": "1h"}
        
        if timeframe in ["5m", "15m", "30m", "60m"]:
            period = "5d"
        else:
            period = "2mo"
        
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period, interval=interval_map.get(timeframe, "1d"))
        
        if df.empty and not symbol.endswith('.JK'):
            ticker = yf.Ticker(f"{symbol}.JK")
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
    
    # EMA
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
    
    # Bollinger Bands
    df['bb_middle'] = df['close'].rolling(window=20).mean()
    bb_std = df['close'].rolling(window=20).std()
    df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
    df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
    
    # Support & Resistance
    df['support'] = df['low'].rolling(window=20).min()
    df['resistance'] = df['high'].rolling(window=20).max()
    
    return df

# ========== IHSG FILTER ==========
def get_ihsg_trend():
    try:
        ihsg = get_data("^JKSE", "1d")
        if ihsg.empty or len(ihsg) < 20:
            return "NEUTRAL", 50, "Data IHSG tidak cukup"
        
        ihsg = add_indicators(ihsg)
        last = ihsg.iloc[-1]
        prev = ihsg.iloc[-2]
        
        ema20 = last.get('ema20', 0)
        ema50 = last.get('ema50', 0)
        close = last.get('close', 0)
        
        if close > ema20 > ema50:
            trend = "BULLISH"
            score = 70
        elif close < ema20 < ema50:
            trend = "BEARISH"
            score = 30
        else:
            trend = "SIDEWAYS"
            score = 50
        
        daily_change = ((close - prev['close']) / prev['close']) * 100
        return trend, score, f"IHSG {trend} ({daily_change:+.1f}%)"
    except:
        return "NEUTRAL", 50, "IHSG data unavailable"

# ========== BOTTOM DETECTION ==========
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
        signals.append(f"Oversold + Volume")
        confidence += 25
    
    body = abs(last['close'] - last['open'])
    lower_wick = min(last['open'], last['close']) - last['low']
    is_hammer = (lower_wick > body * 2) and (len(str(body)) > 0)
    if is_hammer:
        signals.append("Hammer Candle")
        confidence += 20
    
    support = prev_20['low'].min()
    if abs(last['close'] - support) / support < 0.01 and last['close'] > support:
        signals.append("Support Bounce")
        confidence += 20
    
    is_bottom = confidence >= 40
    desc = " | ".join(signals) if signals else "Tidak ada sinyal"
    return is_bottom, min(100, confidence), desc

# ========== BREAKOUT DETECTION ==========
def detect_valid_breakout(df, lookback=20):
    if df.empty or len(df) < lookback + 5:
        return False, "NONE", 0, "Data tidak cukup"
    
    last = df.iloc[-1]
    prev = df.iloc[-2]
    prev_highs = df.iloc[-lookback:-1]['high']
    resistance = prev_highs.max()
    
    signals = []
    confidence = 0
    
    if last['close'] > resistance:
        signals.append(f"Break resistance")
        confidence += 30
    else:
        return False, "NONE", 0, "No breakout"
    
    if last['volume_ratio'] >= 2.0:
        signals.append(f"High volume")
        confidence += 30
    elif last['volume_ratio'] >= 1.5:
        signals.append(f"Good volume")
        confidence += 20
    else:
        return False, "FAKE_BREAKOUT", 10, "Volume rendah - fake breakout!"
    
    if last['macd_histogram'] > 0 and last['macd_histogram'] > prev['macd_histogram']:
        signals.append("MACD bullish")
        confidence += 15
    
    if last['rsi'] < 75:
        confidence += 10
    
    if last['adx'] >= 25:
        signals.append(f"Strong trend")
        confidence += 15
    
    if confidence >= 70:
        breakout_type = "STRONG_BREAKOUT"
    elif confidence >= 50:
        breakout_type = "VALID_BREAKOUT"
    else:
        breakout_type = "WEAK_BREAKOUT"
    
    desc = " | ".join(signals)
    return confidence >= 50, breakout_type, min(100, confidence), desc

# ========== REVERSAL DETECTION ==========
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

# ========== HIGH QUALITY SETUP ==========
def detect_high_quality_setup(df):
    if df.empty or len(df) < 50:
        return "NO_SETUP", 0, "Data tidak cukup"
    
    is_bottom, bottom_conf, _ = detect_bottom_pattern(df)
    is_breakout, breakout_type, breakout_conf, breakout_desc = detect_valid_breakout(df)
    reversal_type, reversal_conf, _ = detect_reversal(df)
    
    last = df.iloc[-1]
    trend_up = last['ema20'] > last['ema50']
    trend_down = last['ema20'] < last['ema50']
    ema_aligned = last['close'] > last['ema20'] > last['ema50'] if trend_up else last['close'] < last['ema20'] < last['ema50']
    momentum_bullish = last['macd_histogram'] > 0 and last['rsi'] > 50
    momentum_bearish = last['macd_histogram'] < 0 and last['rsi'] < 50
    volume_spike = last['volume_ratio'] >= 1.5
    strong_trend = last['adx'] >= 25
    
    no_trade = (last['adx'] < 20) or (45 < last['rsi'] < 55 and not volume_spike)
    if no_trade:
        return "NO_TRADE_ZONE", 0, "Sideways - hindari trading"
    
    if is_breakout and breakout_type == "STRONG_BREAKOUT" and momentum_bullish:
        return "STRONG_BREAKOUT_BUY", 90, breakout_desc
    if is_breakout and breakout_type == "VALID_BREAKOUT":
        return "BREAKOUT_BUY", 75, breakout_desc
    if is_bottom and reversal_type == "BULLISH" and bottom_conf >= 50:
        return "BOTTOM_REVERSAL_BUY", 85, "Bottom + Reversal detected"
    if reversal_type == "BULLISH" and reversal_conf >= 60:
        return "REVERSAL_BUY", 80, "Bullish reversal"
    if trend_up and ema_aligned and strong_trend and momentum_bullish and volume_spike:
        return "TREND_BUY", 70, "Strong uptrend"
    if trend_down and ema_aligned and strong_trend and momentum_bearish:
        return "TREND_SELL", 70, "Strong downtrend"
    if reversal_type == "BEARISH" and reversal_conf >= 60:
        return "REVERSAL_SELL", 75, "Bearish reversal"
    
    return "NO_SETUP", 0, "Tidak ada setup berkualitas"

# ========== RISK REWARD ==========
def calculate_risk_reward(entry_price, stop_loss, take_profit):
    risk = abs(entry_price - stop_loss)
    reward = abs(take_profit - entry_price)
    if risk == 0:
        return {"ratio": 0, "grade": "BAD", "recommendation": "Invalid"}
    rr = reward / risk
    if rr >= 2:
        grade = "EXCELLENT"
        rec = "Eksekusi"
    elif rr >= 1.5:
        grade = "GOOD"
        rec = "Bisa eksekusi"
    elif rr >= 1:
        grade = "FAIR"
        rec = "Hanya jika setup bagus"
    else:
        grade = "POOR"
        rec = "SKIP! Risk > Reward"
    return {"ratio": round(rr, 2), "grade": grade, "recommendation": rec}

# ========== POSITION SIZING ==========
def calculate_smart_position_size(capital, entry_price, stop_loss, risk_percent=2):
    if entry_price <= stop_loss:
        return 0, 0
    risk_amount = capital * (risk_percent / 100)
    risk_per_share = entry_price - stop_loss
    shares = int(risk_amount / risk_per_share) if risk_per_share > 0 else 0
    max_shares = int((capital * 0.5) / entry_price)
    shares = min(shares, max_shares)
    actual_risk = (shares * risk_per_share / capital) * 100 if capital > 0 else 0
    return shares, actual_risk

# ========== CONFIDENCE SCORE ==========
def calculate_confidence_score(df, ihsg_score=50):
    if df.empty or len(df) < 30:
        return 50, [], "NORMAL"
    
    last = df.iloc[-1]
    factors = []
    total = 50
    
    if last['ema20'] > last['ema50']:
        total += 12
        factors.append(("Trend", 12, "Bullish"))
    else:
        total -= 12
        factors.append(("Trend", -12, "Bearish"))
    
    if last['adx'] >= 25:
        total += 15
        factors.append(("ADX", 15, f"Strong"))
    elif last['adx'] >= 20:
        total += 5
        factors.append(("ADX", 5, f"Weak"))
    else:
        total -= 10
        factors.append(("ADX", -10, f"Sideways"))
    
    if last['volume_ratio'] >= 1.5:
        total += 15
        factors.append(("Volume", 15, f"Spike"))
    elif last['volume_ratio'] < 0.6:
        total -= 10
        factors.append(("Volume", -10, f"Sepi"))
    
    macd_bullish = last['macd_histogram'] > 0
    rsi_bullish = last['rsi'] > 50
    if macd_bullish and rsi_bullish:
        total += 15
        factors.append(("Momentum", 15, "Bullish"))
    elif not macd_bullish and not rsi_bullish:
        total -= 15
        factors.append(("Momentum", -15, "Bearish"))
    
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

# ========== MULTI TIMEFRAME ==========
def multi_timeframe_analysis(symbol):
    timeframes = ["1h", "1d"]
    scores = {}
    for tf in timeframes:
        df = get_data(symbol, tf)
        if not df.empty and len(df) > 20:
            df = add_indicators(df)
            score, _, _ = calculate_confidence_score(df)
            scores[tf] = score
        else:
            scores[tf] = 50
        time.sleep(0.5)
    weights = {"1h": 0.3, "1d": 0.7}
    weighted = sum(scores[tf] * weights.get(tf, 0.5) for tf in timeframes)
    return {**scores, "weighted": weighted}

# ========== SCANNER ==========
def scan_saham():
    stocks = ["BBCA.JK", "BBRI.JK", "BMRI.JK", "TLKM.JK", "ASII.JK"]
    results = []
    for stock in stocks:
        try:
            df = get_data(stock, "1d")
            if not df.empty and len(df) > 30:
                df = add_indicators(df)
                setup, quality, msg = detect_high_quality_setup(df)
                if quality >= 65:
                    if "BUY" in setup:
                        signal = "🔥 BUY"
                    elif "SELL" in setup:
                        signal = "🔴 SELL"
                    else:
                        signal = "⏸️ HOLD"
                    results.append({"Kode": stock, "Score": f"{quality:.0f}", "Setup": msg[:25], "Sinyal": signal})
            time.sleep(0.5)
        except:
            continue
    results.sort(key=lambda x: int(x['Score']), reverse=True)
    return results[:5]

def get_trading_recommendation(df):
    setup, quality, msg = detect_high_quality_setup(df)
    if quality >= 85:
        return f"🎯 {msg} - Sniper eksekusi"
    elif quality >= 70:
        return f"📈 {msg} - Setup bagus"
    elif quality >= 55:
        return f"⏸️ {msg} - Tunggu konfirmasi"
    return f"⛔ {msg} - No trade zone"

def backtest_strategy(df, initial_capital=100000000, risk_per_trade=2, fee_buy=0.0015, fee_sell=0.0025, slippage=0.001):
    if df.empty or len(df) < 30:
        return {"return": 0, "winrate": 0, "trades": 0, "final_capital": initial_capital, "max_drawdown": 0, "profit_factor": 0}
    
    df_test = df.copy()
    df_test = add_indicators(df_test)
    capital = initial_capital
    position = 0
    trades = []
    peak = initial_capital
    max_dd = 0
    
    for i in range(20, len(df_test)):
        setup, quality, _ = detect_high_quality_setup(df_test.iloc[:i+1])
        close = df_test.iloc[i]['close']
        atr = df_test.iloc[i].get('atr', close * 0.02)
        
        if (setup in ["STRONG_BREAKOUT_BUY", "BREAKOUT_BUY", "BOTTOM_REVERSAL_BUY", "REVERSAL_BUY", "TREND_BUY"]) and position == 0:
            stop_loss = close - (2 * atr)
            risk_amount = capital * (risk_per_trade / 100)
            risk_per_share = close - stop_loss
            shares = int(risk_amount / risk_per_share) if risk_per_share > 0 else 0
            if shares > 0:
                entry = close * (1 + slippage)
                cost = shares * entry * (1 + fee_buy)
                if cost <= capital:
                    position = shares
                    capital -= cost
                    entry_price = entry
                    stop_price = stop_loss
        
        elif position > 0:
            take_profit = entry_price + (3 * atr)
            if close <= stop_price:
                exit_p = close * (1 - slippage)
                capital += position * exit_p * (1 - fee_sell)
                trades.append((exit_p - entry_price) / entry_price * 100)
                position = 0
            elif close >= take_profit:
                exit_p = close * (1 - slippage)
                capital += position * exit_p * (1 - fee_sell)
                trades.append((exit_p - entry_price) / entry_price * 100)
                position = 0
        
        current = capital + (position * close) if position else capital
        if current > peak:
            peak = current
        dd = (peak - current) / peak * 100
        max_dd = max(max_dd, dd)
    
    if position > 0:
        exit_p = df_test.iloc[-1]['close'] * (1 - slippage)
        capital += position * exit_p * (1 - fee_sell)
    
    winrate = len([t for t in trades if t > 0]) / len(trades) * 100 if trades else 0
    gross_profit = sum([t for t in trades if t > 0])
    gross_loss = abs(sum([t for t in trades if t < 0]))
    pf = gross_profit / gross_loss if gross_loss > 0 else 0
    ret = ((capital - initial_capital) / initial_capital) * 100
    
    return {"return": round(ret, 2), "winrate": round(winrate, 2), "trades": len(trades), "final_capital": round(capital, 0), "max_drawdown": round(max_dd, 2), "profit_factor": round(pf, 2)}
