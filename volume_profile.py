import pandas as pd
import numpy as np

def calculate_volume_profile(df, bins=40, lookback=60):
    """
    Hitung Volume Profile dari data OHLC.
    
    Parameters:
    - df: DataFrame dengan kolom high, low, close, volume
    - bins: jumlah bin harga
    - lookback: berapa candle terakhir yang dianalisis
    
    Returns:
    - dict dengan keys:
      - 'profile': DataFrame dengan kolom price, volume
      - 'poc': Point of Control (harga dengan volume tertinggi)
      - 'value_area_high': batas atas value area (70%)
      - 'value_area_low': batas bawah value area (70%)
    """
    if df.empty or len(df) < lookback:
        return None
    
    # Ambil data sebanyak lookback
    df_subset = df.iloc[-lookback:].copy()
    
    # Tentukan range harga
    price_high = df_subset['high'].max()
    price_low = df_subset['low'].min()
    
    if price_high <= price_low:
        return None
    
    # Buat bin harga
    bin_size = (price_high - price_low) / bins
    price_bins = np.linspace(price_low, price_high, bins + 1)
    
    # Array untuk menyimpan volume per bin
    volume_profile = np.zeros(bins)
    
    for _, row in df_subset.iterrows():
        candle_low = row['low']
        candle_high = row['high']
        volume = row['volume']
        
        if volume <= 0:
            continue
        
        # Distribusikan volume ke bin yang dilewati candle
        for j in range(bins):
            bin_low = price_bins[j]
            bin_high = price_bins[j + 1]
            
            # Cek apakah candle overlap dengan bin ini
            overlap_low = max(candle_low, bin_low)
            overlap_high = min(candle_high, bin_high)
            
            if overlap_high > overlap_low:
                # Hitung proporsi volume
                candle_range = candle_high - candle_low
                if candle_range > 0:
                    proportion = (overlap_high - overlap_low) / candle_range
                else:
                    # Doji: volume full ke bin tempat close berada
                    if bin_low <= row['close'] <= bin_high:
                        proportion = 1.0
                    else:
                        proportion = 0.0
                
                volume_profile[j] += volume * proportion
    
    # Buat DataFrame hasil
    profile_df = pd.DataFrame({
        'price': price_bins[:-1] + bin_size / 2,  # titik tengah bin
        'volume': volume_profile
    })
    
    # Filter bin dengan volume > 0
    profile_df = profile_df[profile_df['volume'] > 0].reset_index(drop=True)
    
    if profile_df.empty:
        return None
    
    # Point of Control (POC): harga dengan volume tertinggi
    poc_idx = profile_df['volume'].idxmax()
    poc = profile_df.loc[poc_idx, 'price']
    
    # Value Area (70% dari total volume)
    total_volume = profile_df['volume'].sum()
    target_volume = total_volume * 0.70
    
    # Mulai dari POC, ekspansi ke atas dan bawah sampai capai 70% volume
    sorted_by_vol = profile_df.sort_values('volume', ascending=False)
    cumulative_vol = 0
    value_prices = []
    
    for _, row in sorted_by_vol.iterrows():
        cumulative_vol += row['volume']
        value_prices.append(row['price'])
        if cumulative_vol >= target_volume:
            break
    
    value_area_high = max(value_prices) if value_prices else price_high
    value_area_low = min(value_prices) if value_prices else price_low
    
    return {
        'profile': profile_df,
        'poc': poc,
        'value_area_high': value_area_high,
        'value_area_low': value_area_low,
        'total_volume': total_volume
    }


def get_volume_profile_summary(df, lookback=60):
    """
    Dapatkan ringkasan Volume Profile untuk analisis.
    
    Returns dict dengan:
    - poc: Point of Control
    - va_high, va_low: Value Area boundaries
    - poc_position: posisi harga sekarang relatif terhadap POC
    - volume_cluster: apakah ada cluster volume signifikan
    """
    vp = calculate_volume_profile(df, bins=40, lookback=lookback)
    
    if vp is None:
        return {
            'poc': None,
            'va_high': None,
            'va_low': None,
            'poc_position': 'No data',
            'volume_cluster': False
        }
    
    last_close = df.iloc[-1]['close']
    poc = vp['poc']
    
    # Posisi harga vs POC
    if last_close > vp['value_area_high']:
        poc_position = "Above Value Area"
    elif last_close < vp['value_area_low']:
        poc_position = "Below Value Area"
    elif last_close > poc:
        poc_position = "Above POC (in Value)"
    elif last_close < poc:
        poc_position = "Below POC (in Value)"
    else:
        poc_position = "At POC"
    
    # Deteksi cluster volume
    profile = vp['profile']
    mean_vol = profile['volume'].mean()
    max_vol = profile['volume'].max()
    volume_cluster = max_vol > mean_vol * 3  # Ada bin dengan volume 3x rata-rata
    
    return {
        'poc': round(poc, 0),
        'va_high': round(vp['value_area_high'], 0),
        'va_low': round(vp['value_area_low'], 0),
        'poc_position': poc_position,
        'volume_cluster': volume_cluster
    }


# ========== VISUALISASI VOLUME PROFILE (UNTUK PLOTLY) ==========
def get_volume_profile_shapes(df, bins=40, lookback=60, chart_width=100):
    """
    Generate shapes untuk ditambahkan ke Plotly chart.
    Return list of dict shapes dan list of annotations.
    """
    vp = calculate_volume_profile(df, bins=bins, lookback=lookback)
    
    if vp is None:
        return [], [], None
    
    profile = vp['profile']
    max_vol = profile['volume'].max()
    
    if max_vol <= 0:
        return [], [], None
    
    # Normalisasi volume
    profile['volume_norm'] = profile['volume'] / max_vol * (chart_width * 0.15)
    
    shapes = []
    bin_height = (df['high'].max() - df['low'].min()) / bins
    
    for _, row in profile.iterrows():
        if row['volume'] > 0:
            shapes.append({
                'type': 'rect',
                'x0': chart_width - 1,
                'x1': chart_width - 1 + row['volume_norm'],
                'y0': row['price'] - bin_height / 2,
                'y1': row['price'] + bin_height / 2,
                ' fillcolor': 'rgba(0, 150, 255, 0.4)',
                'line': {'width': 0},
                'layer': 'below'
            })
    
    # Anotasi POC
    annotations = [{
        'x': chart_width - 1,
        'y': vp['poc'],
        'text': f"POC: {vp['poc']:.0f}",
        'showarrow': False,
        'xanchor': 'right',
        'font': {'color': 'cyan', 'size': 10}
    }]
    
    return shapes, annotations, vp['poc']


# Testing
if __name__ == "__main__":
    from data import get_data, add_indicators
    
    print("=" * 60)
    print("🧪 TEST VOLUME PROFILE")
    print("=" * 60)
    
    # Test dengan BBRI
    df = get_data("BBRI", "1d")
    if not df.empty:
        df = add_indicators(df)
        
        # Hitung Volume Profile
        vp = calculate_volume_profile(df, bins=30, lookback=60)
        
        if vp:
            print(f"\n📊 Volume Profile BBRI (60 hari):")
            print(f"  POC: Rp{vp['poc']:,.0f}")
            print(f"  Value Area High: Rp{vp['value_area_high']:,.0f}")
            print(f"  Value Area Low: Rp{vp['value_area_low']:,.0f}")
            print(f"  Total Volume: {vp['total_volume']:,.0f}")
            print(f"\n  Top 5 Volume Nodes:")
            top5 = vp['profile'].nlargest(5, 'volume')
            for _, row in top5.iterrows():
                print(f"    Rp{row['price']:,.0f}: {row['volume']:,.0f}")
        
        # Ringkasan
        summary = get_volume_profile_summary(df, lookback=60)
        print(f"\n📋 Ringkasan:")
        print(f"  Posisi: {summary['poc_position']}")
        print(f"  Volume Cluster: {'Ya' if summary['volume_cluster'] else 'Tidak'}")
    else:
        print("❌ Data BBRI tidak tersedia")
