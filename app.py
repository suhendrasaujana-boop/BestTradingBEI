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

st.set_page_config(
    page_title="Robot Saham Indonesia",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# SIDEBAR
with st.sidebar:
    st.title("📈 Robot Saham")
    st.markdown("---")
    symbol = st.text_input("Kode Saham", "BBCA.JK", key="symbol_input").upper()
    timeframe = st.selectbox("Timeframe", ["5m", "15m", "30m", "60m", "1d"], key="timeframe_select")
    st.markdown("---")
    auto_refresh = st.checkbox("Auto Refresh (30 detik)", value=False)
    if auto_refresh:
        time.sleep(30)
        st.rerun()
    st.markdown("---")
    st.caption(f"Update: {datetime.now().strftime('%H:%M:%S')}")

# MAIN CONTENT
st.title(f"📊 {symbol}")

# Ambil data
df = get_data(symbol, timeframe)

if df.empty:
    st.warning(f"Data {symbol} kosong. Cek kode (contoh: BBCA.JK)")
    st.stop()

df = add_indicators(df)
score = calculate_score(df)
signal_label, signal_color, signal_emoji = get_signal_label(score)
last = df.iloc[-1]

# ========== HARGA TERTINGGI, TERENDAH, SAAT INI ==========
st.subheader("💰 Harga")
col_high, col_low, col_close = st.columns(3)

with col_high:
    st.metric("📈 Tertinggi (High)", f"Rp{last['high']:,.0f}")
with col_low:
    st.metric("📉 Terendah (Low)", f"Rp{last['low']:,.0f}")
with col_close:
    # Hitung perubahan harga
    if len(df) > 1:
        change = last['close'] - df.iloc[-2]['close']
        change_pct = (change / df.iloc[-2]['close']) * 100
        st.metric("💰 Saat Ini (Close)", f"Rp{last['close']:,.0f}", 
                  delta=f"{change_pct:+.2f}%", delta_color="normal")
    else:
        st.metric("💰 Saat Ini (Close)", f"Rp{last['close']:,.0f}")

st.markdown("---")

# ========== LAYOUT 2 KOLOM ==========
col_left, col_right = st.columns([2, 1.2])

with col_left:
    # CHART
    st.subheader("📈 Harga & EMA")
    st.line_chart(df[['close', 'ema20', 'ema50']], height=300)
    
    # DETAIL INDIKATOR
    st.subheader("📊 Indikator")
    col_rsi, col_macd, col_vol = st.columns(3)
    with col_rsi:
        st.metric("RSI", f"{last['rsi']:.1f}")
        if last['rsi'] < 30:
            st.info("🟢 Oversold (peluang beli)")
        elif last['rsi'] > 70:
            st.warning("🔴 Overbought")
    with col_macd:
        st.metric("MACD", f"{last['macd']:.2f}")
        st.metric("Signal", f"{last['macd_signal']:.2f}", delta=f"{last['macd_histogram']:.2f}")
    with col_vol:
        st.metric("Volume", f"{last['volume']:,.0f}")
        st.metric("MA20", f"{last['volume_ma20']:,.0f}")

with col_right:
    # SIGNAL CARD
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
    
    st.markdown("---")
    
    # REKOMENDASI
    st.subheader("📝 Rekomendasi")
    rekomendasi = get_trading_recommendation(score, df)
    st.markdown(rekomendasi)
    
    st.markdown("---")
    
    # LEVEL SUPPORT & RESISTANCE
    st.subheader("📊 Support & Resistance")
    col_sup, col_res = st.columns(2)
    with col_sup:
        st.metric("🛡️ Support", f"Rp{last['support']:,.0f}")
    with col_res:
        st.metric("🚧 Resistance", f"Rp{last['resistance']:,.0f}")

# ========== MULTI TIMEFRAME ==========
st.markdown("---")
st.subheader("⏰ Multi Timeframe")

mtf = multi_timeframe_analysis(symbol)
avg_score = mtf.get('weighted', 0)
final_label, final_color, final_emoji = get_signal_label(avg_score)

# Tampilkan MTF dengan 5 kolom
col5m, col15m, col30m, col1h, col1d = st.columns(5)

with col5m:
    s = mtf.get("5m", 0)
    st.metric("5m", f"{s:.0f}")
with col15m:
    s = mtf.get("15m", 0)
    st.metric("15m", f"{s:.0f}")
with col30m:
    s = mtf.get("30m", 0)
    st.metric("30m", f"{s:.0f}")
with col1h:
    s = mtf.get("1h", 0)
    st.metric("1h", f"{s:.0f}")
with col1d:
    s = mtf.get("1d", 0)
    st.metric("1d", f"{s:.0f}")

# Final Signal
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

# ========== SCANNER ==========
st.markdown("---")
st.subheader("🔍 Scanner Saham")

if st.button("🚀 Scan Market", use_container_width=True):
    with st.spinner("Scanning..."):
        try:
            results = scan_saham()
            if results:
                df_scan = pd.DataFrame(results)
                st.dataframe(df_scan, use_container_width=True, hide_index=True)
                st.success(f"🏆 Top 3: {', '.join([r['Kode'] for r in results[:3]])}")
            else:
                st.warning("Tidak ada data")
        except Exception as e:
            st.error(f"Error: {str(e)}")

# FOOTER
st.markdown("---")
st.caption("⚠️ Disclaimer: Alat bantu analisis, bukan rekomendasi investasi.")
