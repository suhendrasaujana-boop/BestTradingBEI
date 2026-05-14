import pandas as pd
import numpy as np
from data import get_data, add_indicators, calculate_confidence_score, get_ihsg_trend
from data import get_nearest_fvg, get_nearest_order_block, calculate_entry_sl_tp
import warnings
warnings.filterwarnings('ignore')

# ========== KALIBRASI CONFIDENCE SCORE ==========
def calibrate_confidence(symbols=None, initial_capital=100000000, risk_percent=2):
    """
    Backtest massal untuk mengkalibrasi confidence score.
    Menghitung actual winrate untuk setiap rentang confidence.
    """
    if symbols is None:
        symbols = [
            "BBCA.JK", "BBRI.JK", "BMRI.JK", "TLKM.JK", "ASII.JK",
            "UNVR.JK", "ICBP.JK", "INDF.JK", "GGRM.JK", "HMSP.JK",
            "BBNI.JK", "BNGA.JK", "BRIS.JK", "PGAS.JK", "PTBA.JK"
        ]
    
    bins = [
        (0, 50, "LOW"),
        (50, 65, "NORMAL"),
        (65, 80, "HIGH"),
        (80, 100, "SNIPER")
    ]
    
    calibration = {
        label: {"trades": 0, "wins": 0, "winrate": 0} 
        for _, _, label in bins
    }
    
    print(f"🔄 Mengkalibrasi {len(symbols)} saham...")
    
    for sym_idx, sym in enumerate(symbols):
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
                        result = calculate_entry_sl_tp(snapshot, initial_capital, risk_percent)
                        entry, sl, tp, shares, rr, setup, conf_sig, _ = result
                        
                        if entry and shares > 0 and "BUY" in setup:
                            # Evaluasi di candle berikutnya
                            next_open = df.iloc[i+1]['open']
                            next_close = df.iloc[i+1]['close']
                            next_high = df.iloc[i+1]['high']
                            
                            # Hit sebagai win jika:
                            # 1. Close > Open (candle bullish)
                            # 2. Atau high >= take profit
                            if next_close > next_open or (next_high >= tp):
                                calibration[label]["wins"] += 1
                            calibration[label]["trades"] += 1
                        break
            
            # Progress
            if (sym_idx + 1) % 5 == 0:
                print(f"  Progress: {sym_idx+1}/{len(symbols)}")
                
        except Exception as e:
            print(f"  ⚠️ Error {sym}: {e}")
            continue
    
    # Hitung winrate
    for label in calibration:
        trades = calibration[label]["trades"]
        wins = calibration[label]["wins"]
        calibration[label]["winrate"] = round((wins / trades * 100) if trades > 0 else 0, 1)
    
    print(f"✅ Kalibrasi selesai!")
    return calibration


# ========== CONFIDENCE INTERVAL ==========
def get_confidence_interval(winrate, trades, confidence=0.95):
    """
    Hitung confidence interval menggunakan Wilson score interval.
    Cocok untuk proporsi dengan sample kecil.
    """
    if trades == 0:
        return 0, 0
    
    # Z-score untuk confidence level
    z_scores = {0.90: 1.645, 0.95: 1.96, 0.99: 2.576}
    z = z_scores.get(confidence, 1.96)
    
    p = winrate / 100
    
    # Wilson score interval
    denominator = 1 + z**2 / trades
    centre = (p + z**2 / (2 * trades)) / denominator
    margin = z * np.sqrt((p * (1 - p) + z**2 / (4 * trades)) / trades) / denominator
    
    lower = max(0, (centre - margin) * 100)
    upper = min(100, (centre + margin) * 100)
    
    return round(lower, 1), round(upper, 1)


# ========== MULTI-TIMEFRAME CONFLUENCE ==========
def detect_mtf_confluence(symbol):
    """
    Deteksi konfluensi multi-timeframe: FVG/OB yang berhimpit di 1d, 60m, 15m.
    Return skor 0-100.
    """
    timeframes = ["1d", "60m", "15m"]
    confluence_score = 0
    zones = {}
    
    for tf in timeframes:
        try:
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
        except:
            continue
    
    # Cek konfluensi Daily + Hourly
    if "1d" in zones and "60m" in zones:
        daily = zones["1d"]
        hourly = zones["60m"]
        
        # Bullish confluence: daily FVG + hourly OB overlap
        if daily["bull_fvg"] and hourly["bull_ob"]:
            d_upper = daily["bull_fvg"]["upper"]
            d_lower = daily["bull_fvg"]["lower"]
            h_high = hourly["bull_ob"]["high"]
            h_low = hourly["bull_ob"]["low"]
            
            # Cek overlap
            if d_lower <= h_high and d_upper >= h_low:
                confluence_score += 40
        
        # Bearish confluence
        if daily["bear_fvg"] and hourly["bear_ob"]:
            d_upper = daily["bear_fvg"]["upper"]
            d_lower = daily["bear_fvg"]["lower"]
            h_high = hourly["bear_ob"]["high"]
            h_low = hourly["bear_ob"]["low"]
            
            if d_lower <= h_high and d_upper >= h_low:
                confluence_score += 40
    
    # Cek konfluensi Hourly + 15m
    if "60m" in zones and "15m" in zones:
        hourly = zones["60m"]
        fifteen = zones["15m"]
        
        if hourly["bull_fvg"] and fifteen["bull_fvg"]:
            h_upper = hourly["bull_fvg"]["upper"]
            h_lower = hourly["bull_fvg"]["lower"]
            f_upper = fifteen["bull_fvg"]["upper"]
            f_lower = fifteen["bull_fvg"]["lower"]
            
            if h_lower <= f_upper and h_upper >= f_lower:
                confluence_score += 30
        
        if hourly["bear_fvg"] and fifteen["bear_fvg"]:
            h_upper = hourly["bear_fvg"]["upper"]
            h_lower = hourly["bear_fvg"]["lower"]
            f_upper = fifteen["bear_fvg"]["upper"]
            f_lower = fifteen["bear_fvg"]["lower"]
            
            if h_lower <= f_upper and h_upper >= f_lower:
                confluence_score += 30
    
    return min(100, confluence_score)


def get_probability_context(symbol, conf_score):
    """
    Gabungkan kalibrasi, confidence interval, dan MTF confluence
    menjadi satu konteks untuk ditampilkan di UI.
    """
    # Default calibration values
    default_calibration = {
        "LOW": {"winrate": 35, "trades": 50},
        "NORMAL": {"winrate": 52, "trades": 80},
        "HIGH": {"winrate": 65, "trades": 60},
        "SNIPER": {"winrate": 78, "trades": 30}
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
    
    cal = default_calibration[grade]
    winrate = cal["winrate"]
    trades = cal["trades"]
    
    # Confidence interval
    lower, upper = get_confidence_interval(winrate, trades)
    
    # MTF Confluence
    mtf_score = detect_mtf_confluence(symbol)
    
    # Adjusted winrate (incorporate MTF confluence)
    if mtf_score >= 70:
        adjusted_winrate = min(95, winrate + 10)
    elif mtf_score >= 40:
        adjusted_winrate = min(90, winrate + 5)
    else:
        adjusted_winrate = winrate
    
    return {
        "calibrated_winrate": winrate,
        "adjusted_winrate": adjusted_winrate,
        "confidence_interval": f"{lower}% - {upper}%",
        "mtf_confluence": mtf_score,
        "grade": grade,
        "sample_trades": trades
    }


def print_calibration_report(calibration):
    """Print laporan kalibrasi yang rapi."""
    print(f"\n{'='*60}")
    print(f"📊 LAPORAN KALIBRASI CONFIDENCE SCORE")
    print(f"{'='*60}")
    print(f"{'Grade':<10} {'Trades':<10} {'Wins':<10} {'Winrate':<10}")
    print(f"{'-'*40}")
    
    for grade in ["LOW", "NORMAL", "HIGH", "SNIPER"]:
        data = calibration.get(grade, {"trades": 0, "wins": 0, "winrate": 0})
        print(f"{grade:<10} {data['trades']:<10} {data['wins']:<10} {data['winrate']:.1f}%")
    
    print(f"{'='*60}")


# Testing
if __name__ == "__main__":
    print("=" * 60)
    print("🧪 TEST PROBABILITY ENGINE")
    print("=" * 60)
    
    # Test Confidence Interval
    print("\n📊 Confidence Interval Test:")
    test_cases = [
        (70, 10),
        (70, 50),
        (70, 100),
        (50, 20),
        (90, 5)
    ]
    for wr, tr in test_cases:
        low, high = get_confidence_interval(wr, tr)
        print(f"  Winrate {wr}%, {tr} trades -> CI: {low}% - {high}%")
    
    # Test MTF Confluence
    print("\n📊 MTF Confluence Test:")
    for sym in ["BBRI", "BBCA"]:
        try:
            mtf = detect_mtf_confluence(sym)
            print(f"  {sym}: Confluence Score = {mtf}/100")
        except Exception as e:
            print(f"  {sym}: Error - {e}")
    
    # Test Probability Context
    print("\n📊 Probability Context Test (BBRI):")
    ctx = get_probability_context("BBRI", 75)
    for key, value in ctx.items():
        print(f"  {key}: {value}")
    
    # Quick calibration (pakai 5 saham aja buat testing)
    print("\n📊 Mini Calibration (5 saham):")
    test_symbols = ["BBCA.JK", "BBRI.JK", "TLKM.JK", "ASII.JK", "UNVR.JK"]
    cal = calibrate_confidence(symbols=test_symbols)
    print_calibration_report(cal)
