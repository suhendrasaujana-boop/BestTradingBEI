import streamlit as st
import pandas as pd
from datetime import datetime
import time

from data import (
    get_data,
    add_indicators,
    calculate_score,
    get_signal_label,
    multi_timeframe_analysis,
    scan_saham,
    get_trading_recommendation
)

# Konfigurasi halaman
st.set_page_config(
    page_title="Robot Saham Indonesia",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .stMetric {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 10px;
    }
    div[data-testid="stHorizontalBlock"] {
        gap: 1rem;
    }
    .stButton button {
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.title("📈 Robot Saham Indonesia")
    st.markdown("---")
    
    symbol = st.text_input("📊 Kode Saham", "BBCA.JK", key="symbol_input").upper()
    
    timeframe = st.selectbox(
        "⏱️ Timeframe",
        ["5m", "15m", "30m", "60m", "1d"],
        key="timeframe_select"
    )
    
    st.markdown("---")
    
    auto_refresh = st.checkbox("🔄 Auto Refresh (setiap 30 detik)", value=False)
    if auto_refresh:
        time.sleep(30)
        st.rerun()
    
    st.markdown("---")
    st.caption(f"Last update: {datetime.now().strftime('%H:%M:%S')}")

# Main content
st.title(f"📊 Analisis {symbol}")

col1, col2 = st.columns([2, 1])

df = get_data(symbol, timeframe)

if df.empty:
    st.warning(f"⚠️ Data untuk {symbol} kosong. Cek kode saham (contoh: BBCA.JK)")
    st.stop()

df = add_indicators(df)
score = calculate_score(df)
signal_label, signal_color, signal_emoji = get_signal_label(score)

with col1:
    st.subheader("📈 Harga & Indikator")
    st.line_chart(df[['close', 'ema20', 'ema50']])
    
    with st.expander("📊 Detail Indikator"):
        last = df.iloc[-1]
        col_rsi, col_macd, col_vol = st.columns(3)
        with col_rsi:
            st.metric("RSI", f"{last['rsi']:.1f}")
            if last['rsi'] < 30:
                st.caption("🟢 Oversold (peluang beli)")
            elif last['rsi'] > 70:
                st.caption("🔴 Overbought")
        with col_macd:
            st.metric("MACD", f"{last['macd']:.4f}")
            st.metric("Signal", f"{last['macd_signal']:.4f}")
        with col_vol:
            st.metric("Volume", f"{last['volume']:,.0f}")
            st.metric("Volume MA20", f"{last['volume_ma20']:,.0f}")

with col2:
    st.subheader("🎯 Sinyal Trading")
    
    st.markdown(f"""
    <div style="background-color:{signal_color if signal_color != 'green' else '#90EE90'}; 
                padding:15px; border-radius:10px; text-align:center">
        <h2 style="margin:0">{signal_emoji} {signal_label}</h2>
        <h1 style="margin:0">{score:.0f}</h1>
        <p style="margin:0">/ 100</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.subheader("📝 Rekomendasi")
    rekomendasi = get_trading_recommendation(score, df)
    st.markdown(rekomendasi)
    
    st.code(rekomendasi, language="text")
    st.caption("📋 Klik kanan pada teks di atas → Copy")

# Multi Timeframe Analysis
st.markdown("---")
st.subheader("⏰ Multi Timeframe Analysis")

mtf = multi_timeframe_analysis(symbol)
avg_score = mtf.get('weighted', 0)
final_label, final_color, final_emoji = get_signal_label(avg_score)

cols = st.columns(5)
timeframe_names = ["5m", "15m", "30m", "1h", "1d"]
for idx, (name, col) in enumerate(zip(timeframe_names, cols)):
    if name in mtf:
        with col:
            s = mtf[name]
            if s >= 60:
                color = "🟢"
            elif s >= 40:
                color = "🟡"
            else:
                color = "🔴"
            st.metric(name, f"{s:.0f}", color)

st.markdown(f"""
<div style="background-color:#f0f2f6; padding:15px; border-radius:10px; margin-top:10px">
    <h4 style="margin:0">🎯 Final Signal: {final_emoji} {final_label}</h4>
    <p style="margin:0">Score: {avg_score:.1f}/100</p>
</div>
""", unsafe_allow_html=True)

# Scanner Saham - DIPERBAIKI
st.markdown("---")
st.subheader("🔍 Scanner Saham Terbaik")

if st.button("🚀 Scan Market Sekarang", use_container_width=True):
    with st.spinner("Scanning market (mohon tunggu 30-60 detik)..."):
        try:
            results = scan_saham()
            if results:
                df_scan = pd.DataFrame(results)
                
                # PERBAIKAN: applymap diganti dengan map
                def color_signal(val):
                    if "BUY" in str(val):
                        return "background-color: #90EE90"
                    elif "SELL" in str(val):
                        return "background-color: #FFCCCC"
                    return ""
                
                styled_df = df_scan.style.map(color_signal, subset=['Sinyal'])
                st.dataframe(styled_df, use_container_width=True, hide_index=True)
                st.success(f"🏆 Top 3: {', '.join([r['Kode'] for r in results[:3]])}")
            else:
                st.warning("Tidak ada data yang ditemukan")
        except Exception as e:
            st.error(f"Error saat scan: {str(e)}")
            st.info("Coba lagi nanti")

# Footer
st.markdown("---")
st.caption("⚠️ Disclaimer: Ini hanya alat bantu analisis. Bukan rekomendasi investasi. Trading ada risiko.")
