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
        interval_map = {"5m": "5m", "15m": "15m", "30m": "30m", "60m": "60m", "1d": "1d"}
        
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
    
    # ========== TREND INDICATORS ==========
    df['ema10'] = df['close'].ewm(span=10, adjust=False).mean()
    df['ema20'] = df['close'].ewm(span=20, adjust=False).mean()
    df['ema50'] = df['close'].ewm(span=50, adjust=False).mean()
    
    # ========== MOMENTUM ==========
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
    
    # ========== VOLUME ==========
    df['volume_ma20'] = df['volume'].rolling(window=20).mean()
    df['volume_ratio'] = df['volume'] / df['volume_ma20']
    
    # ========== VOLATILITY ==========
    high_low = df['high'] - df['low']
    high_close = abs(df['high'] - df['close'].shift())
    low_close = abs(df['low'] - df['close'].shift())
    df['tr'] = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr'] = df['tr'].rolling(window=14).mean()
    
    # ========== ADX ==========
    plus_dm = df['high'].diff()
    minus_dm = df['low'].diff()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm > 0] = 0
    
    plus_di = 100 * (plus_dm.ewm(alpha=1/14).mean() / df['atr'])
    minus_di = 100 * (abs(minus_dm).ewm(alpha=1/14).mean() / df['atr'])
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    df['adx'] = dx.rolling(window=14).mean()
    df['plus_di'] = plus_di
    df['minus_di'] = minus_di
    
    # ========== BOLLINGER BANDS ==========
    df['bb_middle'] = df['close'].rolling(window=20).mean()
    bb_std = df['close'].rolling(window=20).std()
    df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
    df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
    df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
    
    # ========== SUPPORT & RESISTANCE ==========
    df['support'] = df['low'].rolling(window=20).min()
    df['resistance'] = df['high'].rolling(window=20).max()
    
    return df

# ========== MARKET FILTER (IHSG) ==========
def get_ihsg_trend():
    """Detect IHSG trend - KRUSIAL untuk market Indonesia"""
    try:
        ihsg = get_data("^JKSE", "1d")
        if ihsg.empty or len(ihsg) < 20:
            return "NEUTRAL", 0, "Data IHSG tidak cukup"
        
        ihsg = add_indicators(ihsg)
        last = ihsg.iloc[-1]
        prev = ihsg.iloc[-2]
        
        # IHSG trend direction
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
        
        # Daily change
        daily_change = ((close - prev['close']) / prev['close']) * 100
        
        return trend, score, f"IHSG {trend} ({daily_change:+.1f}%)"
    
    except:
        return "NEUTRAL", 50, "IHSG data unavailable"

def get_ihsg_filter_penalty():
    """Return penalty/bonus based on IHSG condition"""
    trend, score, msg = get_ihsg_trend()
    
    if trend == "BEARISH":
        return -20, f"⚠️ IHSG Bearish! Turunkan ekspektasi ({msg})"
    elif trend == "SIDEWAYS":
        return -10, f"📊 IHSG Sideways, selektif ({msg})"
    else:
        return 0, f"✅ IHSG Bullish mendukung ({msg})"

# ========== HIGH QUALITY SETUP DETECTOR ==========
def detect_high_quality_setup(df):
    """
    HANYA trading setup berkualitas tinggi
    Menggunakan CONFLUENCE (banyak konfirmasi)
    """
    if df.empty or len(df) < 30:
        return "NO_SETUP", 0, "Data tidak cukup"
    
    last = df.iloc[-1]
    
    # ========== TREND FILTER ==========
    trend_up = last['ema20'] > last['ema50']
    trend_down = last['ema20'] < last['ema50']
    ema_aligned = last['close'] > last['ema20'] > last['ema50'] if trend_up else last['close'] < last['ema20'] < last['ema50']
    
    # ADX untuk kekuatan trend (hindari sideways)
    adx = last.get('adx', 0)
    strong_trend = adx >= 25
    weak_trend = adx < 20
    
    # ========== MOMENTUM FILTER ==========
    momentum_bullish = last['macd_histogram'] > 0 and last['rsi'] > 55
    momentum_bearish = last['macd_histogram'] < 0 and last['rsi'] < 45
    rsi_extreme = last['rsi'] < 30 or last['rsi'] > 70
    
    # ========== VOLUME FILTER ==========
    volume_spike = last['volume_ratio'] >= 1.5
    volume_normal = 0.8 <= last['volume_ratio'] <= 1.5
    volume_silent = last['volume_ratio'] < 0.6
    
    # ========== NO TRADE ZONE (PENTING!) ==========
    no_trade_sideways = (adx < 20) or (45 < last['rsi'] < 55 and not volume_spike)
    no_trade_silent = volume_silent
    no_trade_ema_flat = abs(last['ema20'] - last['ema50']) / last['ema50'] < 0.01
    
    if no_trade_sideways or no_trade_silent or no_trade_ema_flat:
        return "NO_TRADE_ZONE", 0, "Kondisi pasar tidak ideal untuk trading"
    
    # ========== HIGH QUALITY BUY SETUP ==========
    high_quality_buy = (
        trend_up and ema_aligned and strong_trend and
        momentum_bullish and volume_spike
    )
    
    # ========== HIGH QUALITY SELL SETUP ==========
    high_quality_sell = (
        trend_down and ema_aligned and strong_trend and
        momentum_bearish and volume_spike
    )
    
    # ========== SNIPER SETUP (Extra Quality) ==========
    sniper_buy = high_quality_buy and rsi_extreme and last['rsi'] < 30
    sniper_sell = high_quality_sell and rsi_extreme and last['rsi'] > 70
    
    if sniper_buy:
        return "SNIPER_BUY", 95, "Setup langka berkualitas sangat tinggi!"
    elif sniper_sell:
        return "SNIPER_SELL", 95, "Setup langka berkualitas sangat tinggi!"
    elif high_quality_buy:
        return "HIGH_QUALITY_BUY", 85, "Setup berkualitas tinggi - eksekusi"
    elif high_quality_sell:
        return "HIGH_QUALITY_SELL", 85, "Setup berkualitas tinggi - eksekusi"
    elif trend_up and momentum_bullish:
        return "NORMAL_BUY", 65, "Setup bagus tapi tunggu konfirmasi"
    elif trend_down and momentum_bearish:
        return "NORMAL_SELL", 65, "Setup bagus tapi tunggu konfirmasi"
    else:
        return "NO_SETUP", 0, "Tidak ada setup berkualitas"

# ========== RISK REWARD ENGINE ==========
def calculate_risk_reward(entry_price, stop_loss, take_profit):
    """Hitung Risk Reward Ratio - KRUSIAL untuk profit konsisten"""
    risk = abs(entry_price - stop_loss)
    reward = abs(take_profit - entry_price)
    
    if risk == 0:
        return 0
    
    rr_ratio = reward / risk
    
    if rr_ratio >= 2:
        grade = "EXCELLENT"
        recommendation = "Eksekusi dengan RR 1:2 atau lebih"
    elif rr_ratio >= 1.5:
        grade = "GOOD"
        recommendation = "Bisa dieksekusi"
    elif rr_ratio >= 1:
        grade = "FAIR"
        recommendation = "Hanya jika setup sangat bagus"
    else:
        grade = "POOR"
        recommendation = "SKIP! Risk lebih besar dari reward"
    
    return {
        "ratio": round(rr_ratio, 2),
        "risk_pct": round((risk / entry_price) * 100, 2),
        "reward_pct": round((reward / entry_price) * 100, 2),
        "grade": grade,
        "recommendation": recommendation
    }

# ========== CONFIDENCE SCORE ==========
def calculate_confidence_score(df, ihsg_score=50):
    """Confidence score 0-100 untuk kualitas setup"""
    if df.empty or len(df) < 30:
        return 50, []
    
    last = df.iloc[-1]
    factors = []
    total_score = 50
    
    # Trend factor (30% weight)
    if last['ema20'] > last['ema50']:
        total_score += 12
        factors.append(("Trend", 12, "Bullish"))
    else:
        total_score -= 12
        factors.append(("Trend", -12, "Bearish"))
    
    # ADX strength (20% weight)
    adx = last.get('adx', 0)
    if adx >= 25:
        total_score += 15
        factors.append(("ADX", 15, f"Strong ({adx:.0f})"))
    elif adx >= 20:
        total_score += 5
        factors.append(("ADX", 5, f"Weak ({adx:.0f})"))
    else:
        total_score -= 10
        factors.append(("ADX", -10, f"Sideways ({adx:.0f})"))
    
    # Volume confirmation (20% weight)
    if last['volume_ratio'] >= 1.5:
        total_score += 15
        factors.append(("Volume", 15, f"Spike ({last['volume_ratio']:.1f}x)"))
    elif last['volume_ratio'] >= 1.2:
        total_score += 8
        factors.append(("Volume", 8, f"Good ({last['volume_ratio']:.1f}x)"))
    elif last['volume_ratio'] < 0.6:
        total_score -= 10
        factors.append(("Volume", -10, f"Silent ({last['volume_ratio']:.1f}x)"))
    
    # Momentum alignment (20% weight)
    macd_bullish = last['macd_histogram'] > 0
    rsi_bullish = last['rsi'] > 50
    
    if macd_bullish and rsi_bullish:
        total_score += 15
        factors.append(("Momentum", 15, "Bullish aligned"))
    elif not macd_bullish and not rsi_bullish:
        total_score -= 15
        factors.append(("Momentum", -15, "Bearish aligned"))
    
    # IHSG market filter (10% weight)
    total_score += (ihsg_score - 50) * 0.2
    factors.append(("IHSG", (ihsg_score - 50) * 0.2, f"Score {ihsg_score:.0f}"))
    
    final_score = max(0, min(100, total_score))
    
    if final_score >= 80:
        grade = "SNIPER"
    elif final_score >= 65:
        grade = "HIGH"
    elif final_score >= 50:
        grade = "NORMAL"
    elif final_score >= 35:
        grade = "LOW"
    else:
        grade = "AVOID"
    
    return final_score, factors, grade

# ========== SMART POSITION SIZING ==========
def calculate_smart_position_size(capital, entry_price, stop_loss, risk_percent=2):
    """Position sizing berdasarkan risk management"""
    if entry_price <= stop_loss:
        return 0, 0
    
    risk_amount = capital * (risk_percent / 100)
    risk_per_share = entry_price - stop_loss
    shares = int(risk_amount / risk_per_share) if risk_per_share > 0 else 0
    
    # Maksimal 50% modal untuk satu posisi
    max_shares = int((capital * 0.5) / entry_price)
    shares = min(shares, max_shares)
    
    actual_risk = (shares * risk_per_share / capital) * 100
    
    return shares, actual_risk

# ========== PROFIT FACTOR CALCULATION ==========
def calculate_profit_factor(trades):
    """Hitung Profit Factor = Gross Profit / Gross Loss"""
    if not trades:
        return 0
    
    gross_profit = sum([t for t in trades if t > 0])
    gross_loss = abs(sum([t for t in trades if t < 0]))
    
    if gross_loss == 0:
        return gross_profit if gross_profit > 0 else 0
    
    return gross_profit / gross_loss

# ========== BACKTEST REALISTIS ==========
def backtest_strategy(df, initial_capital=100000000, risk_per_trade=2, 
                       fee_buy=0.0015, fee_sell=0.0025, slippage=0.001):
    """Backtest profesional dengan fee, slippage, dan position sizing"""
    if df.empty or len(df) < 30:
        return {
            "return": 0, "winrate": 0, "trades": 0, 
            "final_capital": initial_capital, "max_drawdown": 0,
            "profit_factor": 0, "sharpe_ratio": 0
        }
    
    df_test = df.copy()
    df_test = add_indicators(df_test)
    
    capital = initial_capital
    position = 0
    trades = []
    equity_curve = [initial_capital]
    peak_capital = initial_capital
    max_drawdown = 0
    
    for i in range(20, len(df_test)):
        # Deteksi setup berkualitas
        setup, quality, _ = detect_high_quality_setup(df_test.iloc[:i+1])
        close = df_test.iloc[i]['close']
        atr = df_test.iloc[i].get('atr', close * 0.02)
        
        # Dynamic stop loss berdasarkan ATR
        stop_loss = close - (2 * atr) if setup in ["HIGH_QUALITY_BUY", "SNIPER_BUY"] else 0
        
        if (setup in ["HIGH_QUALITY_BUY", "SNIPER_BUY"]) and position == 0 and stop_loss > 0:
            # Position sizing
            risk_amount = capital * (risk_per_trade / 100)
            risk_per_share = close - stop_loss
            shares = int(risk_amount / risk_per_share) if risk_per_share > 0 else 0
            
            if shares > 0:
                # With slippage & fee
                entry_price = close * (1 + slippage)
                cost = shares * entry_price * (1 + fee_buy)
                
                if cost <= capital:
                    position = shares
                    capital -= cost
                    entry_price_used = entry_price
                    stop_loss_used = stop_loss
                    position_size_used = shares
        
        elif position > 0:
            # Check stop loss or take profit
            take_profit = entry_price_used + (3 * atr)
            
            if close <= stop_loss_used:
                # Stop loss triggered
                exit_price = close * (1 - slippage)
                proceeds = position * exit_price * (1 - fee_sell)
                capital += proceeds
                
                pnl_pct = ((exit_price - entry_price_used) / entry_price_used) * 100
                trades.append(pnl_pct)
                position = 0
                
            elif close >= take_profit:
                # Take profit
                exit_price = close * (1 - slippage)
                proceeds = position * exit_price * (1 - fee_sell)
                capital += proceeds
                
                pnl_pct = ((exit_price - entry_price_used) / entry_price_used) * 100
                trades.append(pnl_pct)
                position = 0
        
        # Track equity
        current_value = capital + (position * close) if position > 0 else capital
        equity_curve.append(current_value)
        
        # Update drawdown
        if current_value > peak_capital:
            peak_capital = current_value
        drawdown = (peak_capital - current_value) / peak_capital * 100
        max_drawdown = max(max_drawdown, drawdown)
    
    # Close open position
    if position > 0:
        exit_price = df_test.iloc[-1]['close'] * (1 - slippage)
        proceeds = position * exit_price * (1 - fee_sell)
        capital += proceeds
    
    # Calculate metrics
    winrate = len([t for t in trades if t > 0]) / len(trades) * 100 if trades else 0
    total_return = ((capital - initial_capital) / initial_capital) * 100
    profit_factor = calculate_profit_factor(trades)
    
    # Simple Sharpe Ratio approximation
    returns = [equity_curve[i] / equity_curve[i-1] - 1 for i in range(1, len(equity_curve))]
    sharpe = (np.mean(returns) / np.std(returns) * np.sqrt(252)) if np.std(returns) > 0 else 0
    
    return {
        "return": round(total_return, 2),
        "winrate": round(winrate, 2),
        "trades": len(trades),
        "final_capital": round(capital, 0),
        "max_drawdown": round(max_drawdown, 2),
        "profit_factor": round(profit_factor, 2),
        "sharpe_ratio": round(sharpe, 2)
    }

# ========== SCAN SAHAM BERKUALITAS ==========
def scan_saham():
    """Scan hanya saham berkualitas (blue chips)"""
    stocks = ["BBCA.JK", "BBRI.JK", "BMRI.JK", "BBNI.JK", "TLKM.JK", "ASII.JK", "UNVR.JK", "INDF.JK", "ICBP.JK"]
    results = []
    
    for stock in stocks:
        try:
            df = get_data(stock, "1d")
            if not df.empty and len(df) > 30:
                df = add_indicators(df)
                setup, quality, setup_msg = detect_high_quality_setup(df)
                
                # Hanya tampilkan jika ada setup
                if quality >= 65:
                    score = quality
                    if "BUY" in setup:
                        signal = "🔥 BUY"
                    elif "SELL" in setup:
                        signal = "🔴 SELL"
                    else:
                        signal = "⏸️ HOLD"
                    
                    results.append({
                        "Kode": stock,
                        "Score": f"{score:.0f}",
                        "Setup": setup_msg[:30],
                        "Sinyal": signal,
                        "Harga": f"Rp{df.iloc[-1]['close']:,.0f}"
                    })
            time.sleep(0.5)
        except Exception as e:
            print(f"Error scan {stock}: {e}")
            continue
    
    results.sort(key=lambda x: int(x['Score']), reverse=True)
    return results[:5]  # Hanya top 5

def multi_timeframe_analysis(symbol):
    timeframes = ["1h", "4h", "1d"]  # Fokus ke timeframe yang lebih besar
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
    
    weights = {"1h": 0.2, "4h": 0.3, "1d": 0.5}
    weighted = sum(scores[tf] * weights.get(tf, 0.33) for tf in timeframes if tf in scores)
    
    return {**scores, "weighted": weighted, "filtered": False}

def get_trading_recommendation(df):
    """Rekomendasi berdasarkan setup berkualitas"""
    setup, quality, setup_msg = detect_high_quality_setup(df)
    
    if quality >= 85:
        return f"🎯 {setup_msg} - Sniper eksekusi, RR minimal 1:2"
    elif quality >= 70:
        return f"📈 {setup_msg} - Setup bagus, pastikan konfirmasi"
    elif quality >= 55:
        return f"⏸️ {setup_msg} - Menunggu konfirmasi lebih lanjut"
    else:
        return f"⛔ {setup_msg} - NO TRADE ZONE, hindari entry"
