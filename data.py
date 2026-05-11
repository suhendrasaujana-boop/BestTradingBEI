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
            period = "1mo"
        
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period, interval=interval_map.get(timeframe, "1d"))
        
        if df.empty:
            # Coba dengan .JK jika belum ada
            if not symbol.endswith('.JK'):
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
    """Lengkap dengan semua indikator profesional"""
    if df.empty:
        return df
    df = df.copy()
    
    # ========== 1. TREND INDICATORS ==========
    # EMA
    df['ema10'] = df['close'].ewm(span=10, adjust=False).mean()
    df['ema20'] = df['close'].ewm(span=20, adjust=False).mean()
    df['ema50'] = df['close'].ewm(span=50, adjust=False).mean()
    
    # Supertrend
    atr_period = 10
    multiplier = 3
    high_low = df['high'] - df['low']
    high_close = abs(df['high'] - df['close'].shift())
    low_close = abs(df['low'] - df['close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr_supertrend'] = tr.rolling(window=atr_period).mean()
    
    hl_avg = (df['high'] + df['low']) / 2
    df['upper_band'] = hl_avg + (multiplier * df['atr_supertrend'])
    df['lower_band'] = hl_avg - (multiplier * df['atr_supertrend'])
    
    df['supertrend'] = 0
    df['supertrend_direction'] = 1  # 1=up, -1=down
    
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
    
    # VWAP (Volume Weighted Average Price)
    df['vwap'] = (df['volume'] * (df['high'] + df['low'] + df['close']) / 3).cumsum() / df['volume'].cumsum()
    
    # ========== 2. MOMENTUM INDICATORS ==========
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
    
    # Stochastic RSI
    rsi_min = df['rsi'].rolling(window=14).min()
    rsi_max = df['rsi'].rolling(window=14).max()
    df['stoch_rsi_k'] = 100 * (df['rsi'] - rsi_min) / (rsi_max - rsi_min)
    df['stoch_rsi_d'] = df['stoch_rsi_k'].rolling(window=3).mean()
    
    # ROC (Rate of Change)
    df['roc'] = df['close'].pct_change(periods=10) * 100
    
    # ========== 3. VOLUME INDICATORS ==========
    df['volume_ma20'] = df['volume'].rolling(window=20).mean()
    df['volume_ratio'] = df['volume'] / df['volume_ma20']
    
    # OBV (On Balance Volume)
    obv = [0]
    for i in range(1, len(df)):
        if df['close'].iloc[i] > df['close'].iloc[i-1]:
            obv.append(obv[-1] + df['volume'].iloc[i])
        elif df['close'].iloc[i] < df['close'].iloc[i-1]:
            obv.append(obv[-1] - df['volume'].iloc[i])
        else:
            obv.append(obv[-1])
    df['obv'] = obv
    df['obv_ma'] = df['obv'].rolling(window=20).mean()
    
    # ========== 4. VOLATILITY INDICATORS ==========
    # ATR (Average True Range)
    df['tr'] = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr'] = df['tr'].rolling(window=14).mean()
    
    # Bollinger Bands
    df['bb_middle'] = df['close'].rolling(window=20).mean()
    bb_std = df['close'].rolling(window=20).std()
    df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
    df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
    df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
    
    # ========== 5. TREND STRENGTH ==========
    # ADX
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
    
    # ========== 6. SUPPORT & RESISTANCE ==========
    df['support'] = df['low'].rolling(window=20).min()
    df['resistance'] = df['high'].rolling(window=20).max()
    
    return df

def detect_trend_regime(df):
    """Deteksi regime pasar: UPTREND, DOWNTREND, SIDEWAYS"""
    if df.empty or len(df) < 20:
        return "UNKNOWN", 0
    
    last = df.iloc[-1]
    adx = last.get('adx', 0)
    if pd.isna(adx):
        adx = 0
    
    ema20 = last.get('ema20', 0)
    ema50 = last.get('ema50', 0)
    close = last.get('close', 0)
    
    # Trend direction
    if close > ema20 > ema50:
        trend_direction = "UPTREND"
        trend_score = 70
    elif close < ema20 < ema50:
        trend_direction = "DOWNTREND"
        trend_score = 30
    else:
        trend_direction = "SIDEWAYS"
        trend_score = 50
    
    # Trend strength
    if adx >= 25:
        strength = "STRONG"
    elif adx >= 20:
        strength = "WEAK"
    else:
        strength = "NO_TREND"
        trend_direction = "SIDEWAYS"
    
    return f"{strength} {trend_direction}".strip(), adx

def calculate_smart_score(df):
    """Sistem scoring profesional dengan bobot"""
    if df.empty or len(df) < 30:
        return 50, {}
    
    last = df.iloc[-1]
    scores = {}
    
    # ========== TREND SCORE (35%) ==========
    trend_score = 50
    ema20 = last.get('ema20', 0)
    ema50 = last.get('ema50', 0)
    close = last.get('close', 0)
    
    if close > ema20 > ema50:
        trend_score = 80
    elif close > ema20:
        trend_score = 65
    elif close < ema20 < ema50:
        trend_score = 20
    elif close < ema20:
        trend_score = 35
    
    # Supertrend confirmation
    if 'supertrend_direction' in last:
        if last['supertrend_direction'] == 1 and trend_score > 50:
            trend_score += 10
        elif last['supertrend_direction'] == -1 and trend_score < 50:
            trend_score -= 10
    
    scores['trend'] = trend_score
    
    # ========== MOMENTUM SCORE (25%) ==========
    momentum_score = 50
    
    # RSI
    rsi = last.get('rsi', 50)
    if rsi < 30:
        momentum_score += 20
    elif rsi < 40:
        momentum_score += 10
    elif rsi > 70:
        momentum_score -= 20
    elif rsi > 60:
        momentum_score -= 10
    
    # MACD
    macd_hist = last.get('macd_histogram', 0)
    if macd_hist > 0:
        momentum_score += 10
    else:
        momentum_score -= 10
    
    # Stochastic RSI
    stoch = last.get('stoch_rsi_k', 50)
    if stoch < 20:
        momentum_score += 10
    elif stoch > 80:
        momentum_score -= 10
    
    scores['momentum'] = max(0, min(100, momentum_score))
    
    # ========== VOLUME SCORE (20%) ==========
    volume_score = 50
    
    volume_ratio = last.get('volume_ratio', 1)
    if volume_ratio > 1.5:
        volume_score = 80
    elif volume_ratio > 1.2:
        volume_score = 65
    elif volume_ratio < 0.5:
        volume_score = 30
    elif volume_ratio < 0.8:
        volume_score = 40
    
    # OBV confirmation
    obv = last.get('obv', 0)
    obv_ma = last.get('obv_ma', 0)
    if obv > obv_ma and volume_score > 50:
        volume_score += 10
    elif obv < obv_ma and volume_score < 50:
        volume_score -= 10
    
    scores['volume'] = max(0, min(100, volume_score))
    
    # ========== VOLATILITY SCORE (10%) ==========
    volatility_score = 50
    atr = last.get('atr', 0)
    close_price = last.get('close', 1)
    atr_pct = (atr / close_price) * 100 if close_price > 0 else 0
    
    if 1.5 <= atr_pct <= 3:
        volatility_score = 70  # Ideal volatility
    elif atr_pct < 1:
        volatility_score = 40  # Too quiet
    elif atr_pct > 5:
        volatility_score = 30  # Too volatile
    
    scores['volatility'] = volatility_score
    
    # ========== MARKET SCORE (10%) ==========
    market_score = 50
    
    # Bollinger Band position
    bb_pos = last.get('bb_position', 0.5)
    if bb_pos < 0.2:
        market_score = 80  # Oversold
    elif bb_pos > 0.8:
        market_score = 20  # Overbought
    
    scores['market'] = market_score
    
    # ========== FINAL WEIGHTED SCORE ==========
    final_score = (
        trend_score * 0.35 +
        momentum_score * 0.25 +
        volume_score * 0.20 +
        volatility_score * 0.10 +
        market_score * 0.10
    )
    
    return max(0, min(100, final_score)), scores

def get_signal_label(score):
    if score >= 75: return ("STRONG BUY", "green", "🔥")
    elif score >= 60: return ("BUY", "lightgreen", "📈")
    elif score >= 45: return ("NEUTRAL", "gray", "⏸️")
    elif score >= 30: return ("SELL", "orange", "📉")
    else: return ("STRONG SELL", "red", "🔴")

def get_confidence_level(score):
    if score >= 75: return ("Sangat Tinggi", "green")
    elif score >= 60: return ("Tinggi", "lightgreen")
    elif score >= 45: return ("Sedang", "yellow")
    elif score >= 30: return ("Rendah", "orange")
    else: return ("Sangat Rendah", "red")

def get_market_regime(df):
    """Market regime dengan ATR dan ADX"""
    if df.empty or len(df) < 20:
        return {
            "regime": "UNKNOWN",
            "adx": 0,
            "atr": 0,
            "atr_pct": 0,
            "color": "gray",
            "description": "Data tidak cukup",
            "trading_allowed": True
        }
    
    last = df.iloc[-1]
    adx = last.get('adx', 0)
    if pd.isna(adx):
        adx = 0
    
    atr = last.get('atr', 0)
    close = last.get('close', 1)
    atr_pct = (atr / close) * 100
    
    # Determine regime
    if adx >= 25:
        if atr_pct > 4:
            regime = "TRENDING + HIGH VOLATILITY"
            color = "orange"
            desc = "⚠️ Tren kuat tapi volatilitas tinggi"
            trading_allowed = True
        elif atr_pct > 2:
            regime = "TRENDING + MODERATE VOL"
            color = "#87CEEB"
            desc = "✅ Kondisi ideal untuk trading"
            trading_allowed = True
        else:
            regime = "STRONG TRENDING"
            color = "green"
            desc = "✅ Tren kuat, sinyal valid"
            trading_allowed = True
    elif adx >= 20:
        regime = "WEAK TREND"
        color = "yellow"
        desc = "⚠️ Tren mulai terbentuk"
        trading_allowed = True
    elif adx >= 15:
        regime = "RANGING (SIDEWAYS)"
        color = "orange"
        desc = "⚠️ Pasar sideways, hati-hati sinyal palsu"
        trading_allowed = False
    else:
        regime = "STRONG RANGING"
        color = "red"
        desc = "🔴 Hindari trading, tunggu breakout"
        trading_allowed = False
    
    # Check extreme volatility
    if atr_pct > 6 and adx < 25:
        regime = "EXTREME VOLATILITY"
        color = "red"
        desc = "🔴 Volatilitas ekstrim! Jangan trading!"
        trading_allowed = False
    
    return {
        "regime": regime,
        "adx": round(adx, 1),
        "atr": round(atr, 0),
        "atr_pct": round(atr_pct, 2),
        "color": color,
        "description": desc,
        "trading_allowed": trading_allowed
    }

def get_smart_entry_signal(df):
    """Deteksi breakout, pullback, dan fake breakout"""
    if df.empty or len(df) < 30:
        return "NO_SIGNAL", ""
    
    last = df.iloc[-1]
    prev = df.iloc[-2]
    
    support = last.get('support', 0)
    resistance = last.get('resistance', float('inf'))
    volume_ratio = last.get('volume_ratio', 1)
    
    # Breakout valid
    if last['close'] > resistance and volume_ratio > 2:
        return "BREAKOUT_BUY", f"Breakout resistance dengan volume {volume_ratio:.1f}x normal"
    
    # Breakdown valid  
    if last['close'] < support and volume_ratio > 2:
        return "BREAKOUT_SELL", f"Breakdown support dengan volume {volume_ratio:.1f}x normal"
    
    # Fake breakout detection
    if (prev['close'] > resistance and last['close'] < resistance):
        return "FAKE_BREAKOUT", "Fake breakout detected! Hati-hati"
    
    # Pullback to EMA
    ema20 = last.get('ema20', 0)
    if abs(last['close'] - ema20) / ema20 < 0.02:  # Within 2% of EMA20
        if last['rsi'] > 40 and last['rsi'] < 60:
            return "PULLBACK", "Pullback ke EMA20, opportunity entry"
    
    return "NO_SIGNAL", ""

def calculate_position_size(capital, entry_price, stop_loss_price, risk_percent=2):
    """Hitung position sizing berdasarkan risk management"""
    if entry_price <= stop_loss_price:
        return 0
    
    risk_amount = capital * (risk_percent / 100)
    risk_per_share = entry_price - stop_loss_price
    shares = risk_amount / risk_per_share
    
    return int(shares)

def calculate_ihsg_filter():
    """Filter berdasarkan IHSG (market breadth)"""
    try:
        ihsg = get_data("^JKSE", "1d")
        if not ihsg.empty:
            ihsg = add_indicators(ihsg)
            ihsg_score, _ = calculate_smart_score(ihsg)
            
            if ihsg_score >= 60:
                return 0, "IHSG positif (+0%)"
            elif ihsg_score >= 45:
                return -5, "IHSG netral (-5%)"
            else:
                return -15, "IHSG negatif (-15%)"
    except:
        return 0, "IHSG tidak tersedia"
    
    return 0, ""

def multi_timeframe_analysis(symbol):
    timeframes = ["5m", "15m", "30m", "60m", "1d"]
    scores = {}
    
    for tf in timeframes:
        df = get_data(symbol, tf)
        if not df.empty and len(df) > 10:
            df = add_indicators(df)
            score, _ = calculate_smart_score(df)
            scores[tf] = score
        else:
            scores[tf] = 50
        time.sleep(0.5)
    
    weights = {"5m": 0.05, "15m": 0.10, "30m": 0.15, "60m": 0.20, "1d": 0.50}
    weighted = sum(scores[tf] * weights.get(tf, 0.2) for tf in timeframes if tf in scores)
    
    return {**scores, "weighted": weighted, "filtered": False}

def scan_saham():
    stocks = ["BBCA.JK", "BBRI.JK", "BMRI.JK", "BBNI.JK", "TLKM.JK", "ASII.JK", "UNVR.JK", "ADRO.JK", "ICBP.JK"]
    results = []
    
    for stock in stocks:
        try:
            df = get_data(stock, "1d")
            if not df.empty and len(df) > 20:
                df = add_indicators(df)
                score, _ = calculate_smart_score(df)
                signal, _, emoji = get_signal_label(score)
                results.append({
                    "Kode": stock,
                    "Score": f"{score:.0f}",
                    "Sinyal": f"{emoji} {signal}",
                    "Harga": f"Rp{df.iloc[-1]['close']:,.0f}"
                })
            time.sleep(1)
        except Exception as e:
            print(f"Error scan {stock}: {e}")
            continue
    
    results.sort(key=lambda x: int(x['Score']), reverse=True)
    return results

def get_trading_recommendation(score, df):
    regime = get_market_regime(df)
    
    if not regime["trading_allowed"]:
        return f"⛔ HINDARI TRADING - {regime['description']}"
    
    if score >= 75:
        return "🔥 BELI AGGRESIF - Semua indikator konfirmasi uptrend kuat"
    elif score >= 60:
        return "📈 BELI - Momentum positif, gunakan stop loss ATR"
    elif score <= 25:
        return "🔴 JUAL AGGRESIF - Tren turun kuat, cut loss"
    elif score <= 40:
        return "📉 JUAL - Tekanan bearish, pertimbangkan cut"
    else:
        return "⏸️ HOLD/TUNGGU - Kondisi sideways, tunggu sinyal jelas"

def backtest_strategy(df, initial_capital=100000000, risk_per_trade=2, fee=0.0015):
    """Backtest realistis dengan fee, slippage, dan position sizing"""
    if df.empty or len(df) < 30:
        return {"return": 0, "winrate": 0, "trades": 0, "final_capital": initial_capital, "max_drawdown": 0}
    
    df_test = df.copy()
    df_test = add_indicators(df_test)
    
    capital = initial_capital
    position = 0
    trades = []
    peak_capital = initial_capital
    max_drawdown = 0
    
    for i in range(20, len(df_test)):
        score, _ = calculate_smart_score(df_test.iloc[:i+1])
        
        # Calculate dynamic stop loss based on ATR
        atr = df_test.iloc[i].get('atr', 0)
        close = df_test.iloc[i]['close']
        stop_loss = close - (2 * atr) if close > atr else close * 0.95
        
        if score >= 60 and position == 0:
            # Position sizing
            risk_amount = capital * (risk_per_trade / 100)
            risk_per_share = close - stop_loss
            shares = int(risk_amount / risk_per_share) if risk_per_share > 0 else 0
            
            if shares > 0:
                # With slippage and fee
                entry_price = close * (1 + 0.0005)  # 0.05% slippage
                cost = shares * entry_price * (1 + fee)
                if cost <= capital:
                    position = shares
                    capital -= cost
                    entry_price_used = entry_price
        
        elif score <= 40 and position > 0:
            exit_price = close * (1 - 0.0005)  # slippage
            proceeds = position * exit_price * (1 - fee)
            capital += proceeds
            
            pnl_pct = ((exit_price - entry_price_used) / entry_price_used) * 100
            trades.append(pnl_pct)
            position = 0
    
    # Close any open position
    if position > 0:
        exit_price = df_test.iloc[-1]['close'] * (1 - 0.0005)
        proceeds = position * exit_price * (1 - fee)
        capital += proceeds
    
    # Calculate metrics
    winrate = len([t for t in trades if t > 0]) / len(trades) * 100 if trades else 0
    total_return = ((capital - initial_capital) / initial_capital) * 100
    
    # Calculate max drawdown
    for i in range(len(df_test)):
        current_value = capital if i == len(df_test)-1 else 0  # Simplified
        if current_value > peak_capital:
            peak_capital = current_value
        drawdown = (peak_capital - current_value) / peak_capital * 100 if peak_capital > 0 else 0
        max_drawdown = max(max_drawdown, drawdown)
    
    return {
        "return": round(total_return, 2),
        "winrate": round(winrate, 2),
        "trades": len(trades),
        "final_capital": round(capital, 0),
        "max_drawdown": round(max_drawdown, 2)
    }
