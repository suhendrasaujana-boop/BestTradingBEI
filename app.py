import streamlit as st
import pandas as pd
from datetime import datetime
import time
from functools import lru_cache

from data import (
    get_data,
    add_indicators,
    calculate_score,
    get_signal_label,
    get_confidence_level,
    multi_timeframe_analysis,
    scan_saham,
    get_trading_recommendation,
    backtest_strategy
)

st.set_page_config(
    page_title="Robot Saham Indonesia",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Cache untuk data
@st.cache_data(ttl=30, show_spinner=False)
def get_cached_data(symbol, timeframe):
    """Ambil data dengan cache 30 detik"""
    try:
        df = get_data(symbol, timeframe)
        return df
    except Exception as e:
        st.error(f"Error ambil data: {str(e)}")
        return pd.DataFrame()

@st.cache_data(ttl=60, show_spinner=False)
def get_cached_scan():
    """Cache untuk scan saham"""
    try:
        return scan_saham()
    except Exception as e:
        st.error(f"Error scan: {str(e)}")
        return []

# SIDEBAR
with st.sidebar:
    st.title("📈 Robot Saham")
    st.markdown("---")
    
    symbol = st.text_input("Kode Saham", "BBCA.JK", key="symbol_input").upper()
    timeframe = st.selectbox("Timeframe", ["5m", "15m", "30m", "60m", "1d"], key="timeframe_select")
    
    st.markdown("---")
    auto_refresh = st.checkbox("Auto Refresh (30 detik)", value=False)
    
    st.markdown("---")
    st.caption(f"Update: {datetime.now().strftime('%H:%M:%S')}")

# Auto refresh placeholder
refresh_placeholder = st.empty()

# MAIN CONTENT
st.title(f"📊 {symbol}")

# Ambil data dengan cache
df = get_cached_data(symbol, timeframe)

if df.empty:
    st.warning(f"Data {symbol} kosong. Cek kode (contoh: BBCA.JK)")
    st.stop()

try:
    df = add_indicators(df)
    score = calculate_score(df)
    signal_label, signal_color, signal_emoji = get_signal_label(score)
    confidence_label, confidence_color = get_confidence_level(score)
    
    # Validasi df tidak kosong
    if len(df) == 0:
        st.error("Data frame kosong setelah processing")
        st.stop()
    
    last = df.iloc[-1]
    
    # ========== HARGA TERTINGGI, TERENDAH, SAAT INI ==========
    st.subheader("💰 Harga")
    col_high, col_low, col_close = st.columns(3)
    
    with col_high:
        high_val = last['high'] if not pd.isna(last['high']) else 0
        st.metric("📈 Tertinggi (High)", f"Rp{high_val:,.0f}")
    
    with col_low:
        low_val = last['low'] if not pd.isna(last['low']) else 0
        st.metric("📉 Terendah (Low)", f"Rp{low_val:,.0f}")
    
    with col_close:
        if len(df) > 1:
            close_val = last['close'] if not pd.isna(last['close']) else 0
            prev_close = df.iloc[-2]['close'] if not pd.isna(df.iloc[-2]['close']) else close_val
            change = close_val - prev_close
            change_pct = (change / prev_close) * 100 if prev_close != 0 else 0
            st.metric("💰 Saat Ini (Close)", f"Rp{close_val:,.0f}", 
                      delta=f"{change_pct:+.2f}%", delta_color="normal")
        else:
            close_val = last['close'] if not pd.isna(last['close']) else 0
            st.metric("💰 Saat Ini (Close)", f"Rp{close_val:,.0f}")
    
    st.markdown("---")
    
    # ========== LAYOUT 2 KOLOM ==========
    col_left, col_right = st.columns([2, 1.2])
    
    with col_left:
        # CHART
        st.subheader("📈 Harga & EMA")
        chart_data = df[['close', 'ema20', 'ema50']].dropna()
        if not chart_data.empty:
            st.line_chart(chart_data, height=300)
        else:
            st.info("Data chart tidak tersedia")
        
        # DETAIL INDIKATOR
        st.subheader("📊 Indikator")
        col_rsi, col_macd, col_vol = st.columns(3)
        
        with col_rsi:
            rsi_val = last.get('rsi', 50)
            if not pd.isna(rsi_val):
                st.metric("RSI", f"{rsi_val:.1f}")
                if rsi_val < 30:
                    st.info("🟢 Oversold (peluang beli)")
                elif rsi_val > 70:
                    st.warning("🔴 Overbought")
            else:
                st.metric("RSI", "N/A")
            
            if 'stoch_rsi_k' in last and not pd.isna(last['stoch_rsi_k']):
                st.caption(f"Stoch RSI: {last['stoch_rsi_k']:.1f}")
        
        with col_macd:
            macd_val = last.get('macd', 0)
            macd_signal = last.get('macd_signal', 0)
            macd_hist = last.get('macd_histogram', 0)
            st.metric("MACD", f"{macd_val:.2f}" if not pd.isna(macd_val) else "N/A")
            st.metric("Signal", f"{macd_signal:.2f}" if not pd.isna(macd_signal) else "N/A", 
                     delta=f"{macd_hist:.2f}" if not pd.isna(macd_hist) else "N/A")
        
        with col_vol:
            vol_val = last.get('volume', 0)
            vol_ma20 = last.get('volume_ma20', 0)
            st.metric("Volume", f"{vol_val:,.0f}" if not pd.isna(vol_val) else "N/A")
            st.metric("MA20", f"{vol_ma20:,.0f}" if not pd.isna(vol_ma20) else "N/A")
            if not pd.isna(vol_val) and not pd.isna(vol_ma20) and vol_val < vol_ma20 * 0.8:
                st.warning("⚠️ Volume rendah")
    
    with col_right:
        st.subheader("🎯 Sinyal")
        
        if "BUY" in signal_label:
            bg_color = "#90EE90"
        elif "SELL" in signal_label:
            bg_color = "#FFCCCC"
        else:
            bg_color = "#FFE4B5"
            
        st.markdown(f"""
        <div style="background-color:{bg_color}; padding:10px; border-radius:10px; text-align:center">
            <h2 style="margin:0; color:black">{signal_emoji} {signal_label}</h2>
            <h1 style="margin:0; color:black">{score:.0f}<span style="font-size:20px">/100</span></h1>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div style="background-color:#f0f2f6; padding:5px; border-radius:10px; text-align:center; margin-top:5px">
            <p style="margin:0; color:black">Keyakinan: <strong>{confidence_label}</strong></p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.subheader("📝 Rekomendasi")
        rekomendasi = get_trading_recommendation(score, df)
        st.markdown(rekomendasi)
        
        st.markdown("---")
        st.subheader("📊 Support & Resistance")
        col_sup, col_res = st.columns(2)
        
        support_val = last.get('support', 0)
        resistance_val = last.get('resistance', 0)
        
        with col_sup:
            st.metric("🛡️ Support", f"Rp{support_val:,.0f}" if not pd.isna(support_val) else "N/A")
        with col_res:
            st.metric("🚧 Resistance", f"Rp{resistance_val:,.0f}" if not pd.isna(resistance_val) else "N/A")
        
        # ADX check dengan validasi
        if 'adx' in last and not pd.isna(last['adx']) and last['adx'] > 0:
            st.markdown("---")
            st.subheader("📊 Trend Strength")
            adx_value = last['adx']
            if adx_value >= 25:
                st.success(f"ADX: {adx_value:.1f} (Tren Kuat ✅)")
            elif adx_value >= 20:
                st.info(f"ADX: {adx_value:.1f} (Tren Mulai)")
            else:
                st.warning(f"ADX: {adx_value:.1f} (Tren Lemah ⚠️)")
    
    # ========== MULTI TIMEFRAME ==========
    st.markdown("---")
    st.subheader("⏰ Multi Timeframe")
    
    try:
        mtf = multi_timeframe_analysis(symbol)
        avg_score = mtf.get('weighted', 0) if isinstance(mtf, dict) else 0
        final_label, final_color, final_emoji = get_signal_label(avg_score)
        
        if isinstance(mtf, dict) and mtf.get('filtered', False):
            st.warning(mtf.get('filter_message', ''))
        
        col5m, col15m, col30m, col1h, col1d = st.columns(5)
        
        # Default values jika key tidak ada
        with col5m:
            s = mtf.get("5m", 0) if isinstance(mtf, dict) else 0
            st.metric("5m", f"{s:.0f}")
        with col15m:
            s = mtf.get("15m", 0) if isinstance(mtf, dict) else 0
            st.metric("15m", f"{s:.0f}")
        with col30m:
            s = mtf.get("30m", 0) if isinstance(mtf, dict) else 0
            st.metric("30m", f"{s:.0f}")
        with col1h:
            s = mtf.get("1h", 0) if isinstance(mtf, dict) else 0
            st.metric("1h", f"{s:.0f}")
        with col1d:
            s = mtf.get("1d", 0) if isinstance(mtf, dict) else 0
            st.metric("1d", f"{s:.0f}")
        
        if "BUY" in final_label:
            final_bg = "#90EE90"
        elif "SELL" in final_label:
            final_bg = "#FFCCCC"
        else:
            final_bg = "#FFE4B5"
        
        st.markdown(f"""
        <div style="background-color:{final_bg}; padding:10px; border-radius:10px; margin-top:10px; text-align:center">
            <h3 style="margin:0; color:black">🎯 Final Signal: {final_emoji} {final_label}</h3>
            <p style="margin:0; color:black">Score: {avg_score:.1f}/100</p>
        </div>
        """, unsafe_allow_html=True)
        
    except Exception as e:
        st.error(f"Error multi timeframe: {str(e)}")
        st.info("Multi timeframe analysis tidak tersedia")
    
    # ========== BACKTEST ==========
    st.markdown("---")
    with st.expander("📊 Backtest Strategy (Threshold BUY>=60, SELL<=40)"):
        if st.button("Jalankan Backtest", key="backtest_btn"):
            try:
                with st.spinner("Menghitung performa..."):
                    result = backtest_strategy(df)
                    
                    if isinstance(result, dict):
                        col_b1, col_b2, col_b3 = st.columns(3)
                        with col_b1:
                            st.metric("Return (%)", f"{result.get('return', 0)}%")
                        with col_b2:
                            st.metric("Winrate (%)", f"{result.get('winrate', 0)}%")
                        with col_b3:
                            st.metric("Jumlah Trades", result.get('trades', 0))
                        
                        final_capital = result.get('final_capital', 100000000)
                        st.caption(f"Modal awal: Rp100.000.000 → Akhir: Rp{final_capital:,.0f}")
                        
                        trades = result.get('trades', 0)
                        winrate = result.get('winrate', 0)
                        return_pct = result.get('return', 0)
                        
                        if trades > 0:
                            if winrate >= 55 and return_pct > 20:
                                st.success(f"✅ Performa bagus: Winrate {winrate}% dengan return {return_pct}% dari {trades} trade.")
                                st.balloons()
                            else:
                                st.info(f"Hasil backtest: Winrate {winrate}%, Return {return_pct}%, Trades {trades}.")
                        else:
                            st.warning("Tidak ada sinyal trade dalam periode ini.")
                    else:
                        st.warning("Data backtest tidak valid")
            except Exception as e:
                st.error(f"Error backtest: {str(e)}")
        else:
            st.info("Klik tombol di atas untuk melihat performa strategi berdasarkan data historis.")
    
    # ========== SCANNER ==========
    st.markdown("---")
    st.subheader("🔍 Scanner Saham")
    
    if st.button("🚀 Scan Market", use_container_width=True):
        try:
            with st.spinner("Scanning market..."):
                results = get_cached_scan()
                
                if results and len(results) > 0:
                    df_scan = pd.DataFrame(results)
                    st.dataframe(df_scan, use_container_width=True, hide_index=True)
                    
                    top3 = [r.get('Kode', r.get('symbol', 'N/A')) for r in results[:3]]
                    st.success(f"🏆 Top 3: {', '.join(top3)}")
                else:
                    st.warning("Tidak ada data saham ditemukan")
        except Exception as e:
            st.error(f"Error scanner: {str(e)}")
            st.info("Pastikan fungsi scan_saham() tersedia di data.py")
    
    st.markdown("---")
    st.caption("⚠️ Disclaimer: Alat bantu analisis, bukan rekomendasi investasi.")
    
    # ========== AUTO REFRESH (DIPERBAIKI) ==========
    if auto_refresh:
        with refresh_placeholder.container():
            st.info("🔄 Auto refresh aktif, halaman akan refresh dalam 30 detik...")
            time.sleep(30)
            st.rerun()

except Exception as e:
    st.error(f"Error utama: {str(e)}")
    st.info("Silakan cek koneksi internet dan pastikan kode saham benar")
