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
        
        if df_stock.empty or df_ihsg.empty:
            return 50, "Data tidak tersedia"
        
        if len(df_stock) < lookback or len(df_ihsg) < lookback:
            return 50, "Data tidak cukup"
        
        # Return selama lookback
        stock_start = df_stock.iloc[-lookback]['close']
        stock_end = df_stock.iloc[-1]['close']
        ihsg_start = df_ihsg.iloc[-lookback]['close']
        ihsg_end = df_ihsg.iloc[-1]['close']
        
        if stock_start == 0 or ihsg_start == 0:
            return 50, "Data invalid"
        
        stock_ret = (stock_end - stock_start) / stock_start * 100
        ihsg_ret = (ihsg_end - ihsg_start) / ihsg_start * 100
        diff_ret = stock_ret - ihsg_ret
        
        # Konsistensi outperformance
        stock_norm = df_stock['close'] / df_stock['close'].iloc[-lookback]
        ihsg_norm = df_ihsg['close'] / df_ihsg['close'].iloc[-lookback]
        
        # Pastikan panjang sama
        min_len = min(len(stock_norm), len(ihsg_norm))
        stock_norm = stock_norm.iloc[-min_len:]
        ihsg_norm = ihsg_norm.iloc[-min_len:]
        
        outperform_days = (stock_norm.values > ihsg_norm.values).sum()
        consistency = (outperform_days / min_len) * 100
        
        # Skor gabungan
        raw_score = 50 + (diff_ret * 2) + (consistency - 50) * 0.5
        score = max(0, min(100, raw_score))
        
        # Deskripsi
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
        
    except Exception as e:
        print(f"⚠️ Error RS {symbol}: {e}")
        return 50, "Error"


# ========== MARKET BREADTH ==========
def get_market_breadth(symbols=None):
    """
    Hitung persentase saham di atas EMA20 dan EMA50.
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
            
            if pd.notna(last.get('ema20')) and pd.notna(last.get('ema50')):
                if last['close'] > last['ema20']:
                    above_ema20 += 1
                if last['close'] > last['ema50']:
                    above_ema50 += 1
                valid += 1
        except Exception as e:
            continue
    
    if valid == 0:
        return 50, 50, "No data"
    
    pct_20 = round((above_ema20 / valid) * 100, 1)
    pct_50 = round((above_ema50 / valid) * 100, 1)
    desc = f"{above_ema20}/{valid} di atas EMA20, {above_ema50}/{valid} di atas EMA50"
    
    return pct_20, pct_50, desc


# ========== SECTOR CLASSIFICATION ==========
sector_mapping = {
    "BBCA.JK": "Perbankan", "BBRI.JK": "Perbankan", "BMRI.JK": "Perbankan",
    "BBNI.JK": "Perbankan", "BNGA.JK": "Perbankan", "BRIS.JK": "Perbankan",
    "TLKM.JK": "Telekomunikasi", "EXCL.JK": "Telekomunikasi", "ISAT.JK": "Telekomunikasi",
    "ASII.JK": "Otomotif", "UNTR.JK": "Otomotif",
    "UNVR.JK": "Konsumer", "ICBP.JK": "Konsumer", "INDF.JK": "Konsumer",
    "KLBF.JK": "Konsumer", "MAPI.JK": "Konsumer",
    "GGRM.JK": "Rokok", "HMSP.JK": "Rokok",
    "PGAS.JK": "Energi", "PTBA.JK": "Energi", "MEDC.JK": "Energi",
    "ANTM.JK": "Tambang", "INCO.JK": "Tambang", "ADRO.JK": "Tambang",
    "ITMG.JK": "Tambang", "MDKA.JK": "Tambang", "BYAN.JK": "Tambang",
    "MNCN.JK": "Media", "SMMA.JK": "Media",
    "SMGR.JK": "Semen", "CTRA.JK": "Properti", "SMRA.JK": "Properti",
    "CPIN.JK": "Pakan", "JPFA.JK": "Pakan",
    "TOWR.JK": "Infrastruktur", "GOTO.JK": "Teknologi",
    "AKRA.JK": "Distribusi", "AMMN.JK": "Tambang",
    "TPIA.JK": "Petrokimia", "DSSA.JK": "Energi",
    "CUAN.JK": "Keuangan", "ADMR.JK": "Keuangan",
    "AADI.JK": "Otomotif", "PGEO.JK": "Energi",
    "BRPT.JK": "Petrokimia", "ESSA.JK": "Petrokimia"
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
            
            start_price = df.iloc[-20]['close']
            end_price = df.iloc[-1]['close']
            
            if start_price == 0:
                continue
                
            ret = (end_price - start_price) / start_price * 100
            
            if sector not in sector_returns:
                sector_returns[sector] = []
            sector_returns[sector].append(ret)
        except:
            continue
    
    result = {}
    for sector, returns in sector_returns.items():
        if returns:
            avg_ret = np.mean(returns)
            result[sector] = {
                'return': round(avg_ret, 2),
                'count': len(returns)
            }
    
    # Urutkan dari return tertinggi
    return dict(sorted(result.items(), key=lambda x: x[1]['return'], reverse=True))


def get_stock_sector(symbol):
    """Dapatkan sektor dari suatu saham."""
    return sector_mapping.get(symbol.upper().replace('.JK', '') + '.JK', "Unknown")


# ========== FOREIGN FLOW PROXY ==========
def foreign_flow_proxy(symbol):
    """
    Deteksi akumulasi asing menggunakan proksi volume tidak wajar.
    """
    try:
        df = get_data(symbol, "1d")
        if df.empty or len(df) < 20:
            return "NEUTRAL", 0, "Data tidak cukup"
        
        df = add_indicators(df)
        last = df.iloc[-1]
        prev_5 = df.iloc[-6:-1]
        
        vol_spike = last.get('volume_ratio', 0) > 2.0 if pd.notna(last.get('volume_ratio')) else False
        price_up = last['close'] > prev_5['close'].mean()
        rs_score, _ = relative_strength_vs_ihsg(symbol)
        
        if vol_spike and price_up and rs_score > 55:
            return "INFLOW", 70, "Ciri-ciri akumulasi asing"
        elif vol_spike and not price_up:
            return "OUTFLOW", 70, "Ciri-ciri distribusi asing"
        else:
            return "NEUTRAL", 30, "Tidak ada sinyal asing"
            
    except Exception as e:
        return "NEUTRAL", 0, f"Error: {e}"


# ========== KOMPILASI KONTEKS LENGKAP ==========
def get_full_market_context(symbol):
    """
    Mengumpulkan semua konteks pasar untuk suatu saham.
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
    try:
        breadth_20, breadth_50, breadth_desc = get_market_breadth()
        context['breadth_20'] = breadth_20
        context['breadth_50'] = breadth_50
        context['breadth_desc'] = breadth_desc
    except:
        context['breadth_20'] = 50
        context['breadth_50'] = 50
        context['breadth_desc'] = "Gagal ambil data"
    
    # Foreign Flow
    flow, flow_conf, flow_desc = foreign_flow_proxy(symbol)
    context['flow'] = flow
    context['flow_conf'] = flow_conf
    context['flow_desc'] = flow_desc
    
    # Sektor performance
    try:
        sector_perf = get_sector_performance()
        context['sector_performance'] = sector_perf
        
        # Ranking sektor
        sectors_list = list(sector_perf.keys())
        if sector in sectors_list:
            context['sector_rank'] = sectors_list.index(sector) + 1
        else:
            context['sector_rank'] = len(sectors_list) + 1
        
        context['sector_count'] = len(sector_perf)
        
        # Top 3 sektor
        context['top_sectors'] = list(sector_perf.items())[:3]
    except:
        context['sector_performance'] = {}
        context['sector_rank'] = 0
        context['sector_count'] = 0
        context['top_sectors'] = []
    
    return context


# Testing
if __name__ == "__main__":
    print("=" * 60)
    print("🧪 TEST MARKET CONTEXT")
    print("=" * 60)
    
    # Test Relative Strength
    print("\n📊 Relative Strength vs IHSG:")
    for sym in ["BBRI", "TLKM", "GOTO"]:
        score, desc = relative_strength_vs_ihsg(sym)
        print(f"  {sym}: {score}/100 - {desc}")
    
    # Test Market Breadth
    print("\n📊 Market Breadth:")
    pct20, pct50, desc = get_market_breadth()
    print(f"  {desc}")
    
    # Test Sektor
    print("\n📊 Sektor Performance (1 bulan):")
    sector_perf = get_sector_performance()
    for i, (sector, data) in enumerate(sector_perf.items(), 1):
        print(f"  {i:2d}. {sector:15s}: {data['return']:+.2f}% ({data['count']} saham)")
    
    # Test Full Context
    print("\n📊 Full Context untuk BBRI:")
    ctx = get_full_market_context("BBRI")
    for key, value in ctx.items():
        if key != 'sector_performance':
            print(f"  {key}: {value}")
