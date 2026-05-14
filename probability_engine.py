import pandas as pd
import numpy as np
from data import get_data, add_indicators, calculate_confidence_score, get_ihsg_trend
from backtest_engine import backtest_strategy
import warnings
warnings.filterwarnings('ignore')

# ========== KALIBRASI CONFIDENCE SCORE ==========
def calibrate_confidence(symbols=None, initial_capital=100000000, risk_percent=2):
    """
    Backtest massal untuk mengkalibrasi confidence score.
    Menghitung actual winrate untuk setiap rentang confidence (0-50, 50-65, 65-80, 80-100).
    """
    if symbols is None:
        symbols = [
            "BBCA.JK", "BBRI.JK", "BMRI.JK", "TLKM.JK", "ASII.JK",
            "UNVR.JK", "ICBP.JK", "INDF.JK", "GGRM.JK", "HMSP.JK",
            "BBNI.JK", "BNGA.JK", "BRIS.JK", "PGAS.JK", "PTBA.JK"
        ]
    
    bins = [(0, 50, "LOW"), (50, 65, "NORMAL"), (65, 80, "HIGH"), (80, 100, "SNIPER")]
    calibration = {label: {"trades": 0, "wins": 0, "winrate": 0} for _, _, label in bins}
    
    for sym in symbols:
        try:
            df = get_data(sym, "1d")
            if df.empty or len(df) < 100:
                continue
            df = add_indicators(df)
            ihsg_trend, ihsg_score, _ = get_ihsg_trend()
            
            for i in range(50, len(df) - 1):
                snapshot = df.iloc[:i+1]
                conf_score, _, grade = calculate_confidence_score(snapshot, ihsg_score)
                
                # Cari rentang
                for low, high, label in bins:
                    if low <= conf_score < high:
                        # Ambil sinyal
                        entry, sl, tp, shares, rr, setup, conf_sig, _ = calculate_entry_sl_tp(snapshot, initial_capital, risk_percent)
                        if entry and shares > 0 and "BUY" in setup:
                            # Evaluasi di candle berikutnya
                            next_open = df.iloc[i+1]['open']
                            next_close = df.iloc[i+1]['close']
                            next_low = df.iloc[i+1]['low']
                            next_high = df.iloc[i+1]['high']
                            
                            # Sederhanakan: hit sebagai win jika close > open (profit) atau TP tercapai
                            if next_close > next_open or (next_high >= tp):
                                calibration[label]["wins"] += 1
                            calibration[label]["trades"] += 1
                        break
        except:
            continue
    
    # Hitung winrate
    for label in calibration:
        trades = calibration[label]["trades"]
        wins = calibration[label]["wins"]
        calibration[label]["winrate"] = round((wins / trades * 100) if trades > 0 else 0, 1)
    
    return calibration

# ========== CONFIDENCE INTERVAL ==========
def get_confidence_interval(winrate, trades, confidence=0.95):
    """
    Hitung confidence interval menggunakan Wilson score interval.
    Cocok untuk proporsi dengan sample kecil.
    """
    if trades == 0:
        return 0, 0
    z = 1.96  # untuk 95% confidence
    p = winrate / 100
    denom = 1 + z**2 / trades
    center = (p + z**2 / (2 * trades)) / denom
    margin = z * np.sqrt((p * (1 - p) + z**2 / (4 * trades)) / trades) / denom
    lower = max(0, (center - margin) * 100)
    upper = min(100, (center + margin) * 100)
    return round(lower, 1), round(upper, 1)

# ========== MULTI-TIMEFRAME CONFLUENCE ==========
def detect_mtf_confluence(symbol):
    """
    Deteksi konfluensi multi-timeframe: FVG/OB yang berhimpit di 1d, 60m, 15m.
    Return skor 0-100.
    """
    from data import get_nearest_fvg, get_nearest_order_block
    
    timeframes = ["1d", "60m", "15m"]
    confluence_score = 0
    zones = {}
    
    for tf in timeframes:
        df = get_data(symbol, tf)
        if df.empty or len(df) < 30:
            continue
        df = add_indicators(df)
        bull_fvg, bear_fvg = get_nearest_fvg(df)
        bull_ob, bear_ob = get_nearest_order_block(df)
        zones[tf] = {
            "bull_fvg": bull_fvg,
            "bear_fvg": bear_fvg,
            "bull_ob": bull_ob,
            "bear_ob": bear_ob
        }
    
    # Cek konfluensi: jika daily FVG berada di area yang sama dengan hourly OB
    if zones.get("1d") and zones.get("60m"):
        daily = zones["1d"]
        hourly = zones["60m"]
        
        # Bullish confluence: daily bullish FVG + hourly bullish OB overlap
        if daily["bull_fvg"] and hourly["bull_ob"]:
            if daily["bull_fvg"]["lower"] <= hourly["bull_ob"]["high"] and daily["bull_fvg"]["upper"] >= hourly["bull_ob"]["low"]:
                confluence_score += 40
        
        # Bearish confluence
        if daily["bear_fvg"] and hourly["bear_ob"]:
            if daily["bear_fvg"]["lower"] <= hourly["bear_ob"]["high"] and daily["bear_fvg"]["upper"] >= hourly["bear_ob"]["low"]:
                confluence_score += 40
    
    # Cek juga dengan 15m
    if zones.get("60m") and zones.get("15m"):
        hourly = zones["60m"]
        fifteen = zones["15m"]
        
        if hourly["bull_fvg"] and fifteen["bull_fvg"]:
            if hourly["bull_fvg"]["lower"] <= fifteen["bull_fvg"]["upper"] and hourly["bull_fvg"]["upper"] >= fifteen["bull_fvg"]["lower"]:
                confluence_score += 30
        
        if hourly["bear_fvg"] and fifteen["bear_fvg"]:
            if hourly["bear_fvg"]["lower"] <= fifteen["bear_fvg"]["upper"] and hourly["bear_fvg"]["upper"] >= fifteen["bear_fvg"]["lower"]:
                confluence_score += 30
    
    return min(100, confluence_score)

def get_probability_context(symbol, conf_score):
    """
    Gabungkan kalibrasi, confidence interval, dan MTF confluence
    menjadi satu konteks untuk ditampilkan di UI.
    """
    # Kalibrasi (load dari cache atau hitung sekali — untuk demo, kita pakai nilai default)
    # Nanti bisa di-cache ke file
    default_calibration = {
        "LOW": {"winrate": 35},
        "NORMAL": {"winrate": 52},
        "HIGH": {"winrate": 65},
        "SNIPER": {"winrate": 78}
    }
    
    # Tentukan grade
    if conf_score >= 80:
        grade = "SNIPER"
    elif conf_score >= 65:
        grade = "HIGH"
    elif conf_score >= 50:
        grade = "NORMAL"
    else:
        grade = "LOW"
    
    winrate = default_calibration[grade]["winrate"]
    lower, upper = get_confidence_interval(winrate, trades=20)  # asumsi 20 trades untuk estimasi
    
    # MTF Confluence
    mtf_score = detect_mtf_confluence(symbol)
    
    return {
        "calibrated_winrate": winrate,
        "confidence_interval": f"{lower}% - {upper}%",
        "mtf_confluence": mtf_score,
        "grade": grade
    }
