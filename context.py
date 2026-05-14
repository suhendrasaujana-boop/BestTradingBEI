import pandas as pd
import numpy as np
from data import get_data, add_indicators

# ========== RELATIVE STRENGTH vs IHSG ==========
def relative_strength_vs_ihsg(symbol, lookback=60):
    """
    Hitung Relative Strength Rating (0-100) saham vs IHSG.
    Kombinasi: selisih return + konsistensi outperformance.
    """
    try:
        df_stock = get_data(symbol, "1d")
        df_ihsg = get_data("^JKSE", "1d")
        if df_stock.empty or df_ihsg.empty or len(df_stock) < lookback or len(df_ihsg) < lookback:
            return 50, "No data"
        
        # Return selama lookback
        stock_ret = (df_stock.iloc[-1]['close'] - df_stock.iloc[-lookback]['close']) / df_stock.iloc[-lookback]['close'] * 100
        ihsg_ret = (df_ihsg.iloc[-1]['close'] - df_ihsg.iloc[-lookback]['close']) / df_ihsg.iloc[-lookback]['close'] * 100
        diff_ret = stock_ret - ihsg_ret
        
        # Konsistensi outperformance (berapa hari di atas IHSG)
        stock_norm = df_stock['close'] / df_stock['close'].iloc[-lookback]
        ihsg_norm = df_ihsg['close'] / df_ihsg['close'].iloc[-lookback]
        outperform_days = (stock_norm > ihsg_norm).sum()
        consistency = (outperform_days / lookback) * 100
        
        # Skor gabungan
        raw_score = 50 + (diff_ret * 2) + (consistency - 50) * 0.5
        score = max(0, min(100, raw_score))
        
        if score >= 70:
            desc = "Strong Outperformer"
        elif score >= 55:
            desc = "Moderate Outperformer"
        elif score >= 45:
            desc = "In Line with IHSG"
        elif score >= 30:
            desc = "Moderate Underperformer"
        else:
            desc = "Weak Underperformer"
        
        return round(score, 1), desc
    except:
        return 50, "Error"

# ========== MARKET BREADTH ==========
def get_market_breadth(symbols=None):
    """
    Hitung persentase saham di atas EMA20 dan EMA50 dari daftar simbol.
    Jika symbols=None, gunakan daftar default saham IHSG utama.
    """
    if symbols is None:
        symbols = [
            "BBCA.JK", "BBRI.JK", "BMRI.JK", "TLKM.JK", "ASII.JK",
            "UNVR.JK", "ICBP.JK", "INDF.JK", "GGRM.JK", "HMSP.JK",
            "BBNI.JK", "BNGA.JK", "BRIS.JK", "PGAS.JK", "PTBA.JK",
            "ANTM.JK", "INCO.JK", "EXCL.JK", "MNCN.JK", "SMMA.JK"
        ]
    
    above_ema20 = 0
    above_ema50 = 0
    valid = 0
    
    for sym in symbols:
        try:
            df = get_data(sym, "1d")
            if df.empty or len(df) < 50:
                continue
            df = add_indicators(df)
            last = df.iloc[-1]
            if pd.notna(last['ema20']) and pd.notna(last['ema50']):
                if last['close'] > last['ema20']:
                    above_ema20 += 1
                if last['close'] > last['ema50']:
                    above_ema50 += 1
                valid += 1
        except:
            continue
    
    if valid == 0:
        return 50, 50, "No data"
    
    pct_20 = (above_ema20 / valid) * 100
    pct_50 = (above_ema50 / valid) * 100
    
    return round(pct_20, 1), round(pct_50, 1), f"{above_ema20}/{valid} saham di atas EMA20, {above_ema50}/{valid} di atas EMA50"

# ========== SECTOR CLASSIFICATION ==========
sector_mapping = {
    "BBCA.JK": "Perbankan", "BBRI.JK": "Perbankan", "BMRI.JK": "Perbankan",
    "BBNI.JK": "Perbankan", "BNGA.JK": "Perbankan", "BRIS.JK": "Perbankan",
    "TLKM.JK": "Telekomunikasi", "EXCL.JK": "Telekomunikasi",
    "ASII.JK": "Otomotif",
    "UNVR.JK": "Konsumer", "ICBP.JK": "Konsumer", "INDF.JK": "Konsumer",
    "GGRM.JK": "Rokok", "HMSP.JK": "Rokok",
    "PGAS.JK": "Energi", "PTBA.JK": "Energi",
    "ANTM.JK": "Tambang", "INCO.JK": "Tambang",
    "MNCN.JK": "Media", "SMMA.JK": "Media"
}

def get_sector_performance():
    """
    Hitung rata-rata return 1 bulan per sektor, urutkan dari yang terkuat.
    """
    sector_returns = {}
    for sym, sector in sector_mapping.items():
        try:
            df = get_data(sym, "1d")
            if df.empty or len(df) < 20:
                continue
            ret = (df.iloc[-1]['close'] - df.iloc[-20]['close']) / df.iloc[-20]['close'] * 100
            if sector not in sector_returns:
                sector_returns[sector] = []
            sector_returns[sector].append(ret)
        except:
            continue
    
    result = {}
    for sector, returns in sector_returns.items():
        if returns:
            avg_ret = np.mean(returns)
            result[sector] = round(avg_ret, 2)
    
    # Urutkan dari return tertinggi
    return dict(sorted(result.items(), key=lambda x: x[1], reverse=True))

def get_stock_sector(symbol):
    """Dapatkan sektor dari suatu saham."""
    return sector_mapping.get(symbol.upper(), "Unknown")

# ========== FOREIGN FLOW PROXY ==========
def foreign_flow_proxy(symbol):
    """
    Deteksi akumulasi asing menggunakan proksi volume tidak wajar.
    Volume spike + harga naik + relative strength kuat = kemungkinan inflow asing.
    """
    try:
        df = get_data(symbol, "1d")
        if df.empty or len(df) < 20:
            return "NEUTRAL", 0, "No data"
        df = add_indicators(df)
        last = df.iloc[-1]
        prev_5 = df.iloc[-6:-1]
        
        vol_spike = last['volume_ratio'] > 2.0 if pd.notna(last['volume_ratio']) else False
        price_up = last['close'] > prev_5['close'].mean()
        rs, _ = relative_strength_vs_ihsg(symbol)
        
        if vol_spike and price_up and rs > 55:
            return "INFLOW", 70, "Ciri-ciri akumulasi asing"
        elif vol_spike and not price_up:
            return "OUTFLOW", 70, "Ciri-ciri distribusi asing"
        else:
            return "NEUTRAL", 30, "Tidak ada sinyal asing"
    except:
        return "NEUTRAL", 0, "Error"

# ========== KOMPILASI KONTEKS LENGKAP ==========
def get_full_market_context(symbol):
    """
    Mengumpulkan semua konteks pasar untuk suatu saham dalam satu panggilan.
    Return dictionary siap pakai untuk UI.
    """
    context = {}
    
    # Relative Strength
    rs_score, rs_desc = relative_strength_vs_ihsg(symbol)
    context['rs_score'] = rs_score
    context['rs_desc'] = rs_desc
    
    # Sektor
    sector = get_stock_sector(symbol)
    context['sector'] = sector
    
    # Market Breadth
    breadth_20, breadth_50, breadth_desc = get_market_breadth()
    context['breadth_20'] = breadth_20
    context['breadth_50'] = breadth_50
    context['breadth_desc'] = breadth_desc
    
    # Foreign Flow
    flow, flow_conf, flow_desc = foreign_flow_proxy(symbol)
    context['flow'] = flow
    context['flow_conf'] = flow_conf
    context['flow_desc'] = flow_desc
    
    # Sektor performance
    sector_perf = get_sector_performance()
    context['sector_performance'] = sector_perf
    context['sector_rank'] = list(sector_perf.keys()).index(sector) + 1 if sector in sector_perf else 0
    context['sector_count'] = len(sector_perf)
    
    return context
