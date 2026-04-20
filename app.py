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

# Custom CSS untuk tampilan lebih rapi
st.markdown("""
<style>
    .stMetric { background-color: #f0f2f6; border-radius: 10px; padding: 10px; }
    div[data-testid="stHorizontalBlock"] { gap: 0.5rem; }
    .stButton button { width: 100%; }
    .rekomendasi-card {
        background-color: #1e1e1e;
        padding: 10px;
        border-radius: 10px;
        margin: 5px 0;
    }
    .small-font {
        font-size: 12px;
    }
</style>
""", unsafe_allow_html=True)

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

# ========== LAYOUT 2 KOLOM UTAMA ==========
col_left, col_right = st.columns([2, 1.2])

with col_left:
    # CHART
    st.subheader("📈 Harga & EMA")
    st.line_chart(df[['close', 'ema20', 'ema50']], height=300)
    
    # DETAIL INDIKATOR dalam 3 baris kecil
    st.subheader("📊 Indikator")
    last = df.iloc[-1]
    
    col_rsi, col_macd, col_vol = st.columns(3)
    with col_rsi:
        st.metric("RSI", f"{last['rsi']:.1f}")
        if last['rsi'] < 30:
            st.caption("🟢 Oversold")
        elif last['rsi'] > 70:
            st.caption("🔴 Overbought")
    with col_macd:
        st.metric("MACD", f"{last['macd']:.2f}")
        st.metric("Signal", f"{last['macd_signal']:.2f}", delta=f"{last['macd_histogram']:.2f}")
    with col_vol:
        st.metric("Volume", f"{last['volume']:,.0f}")
        st.metric("MA20", f"{last['volume_ma20']:,.0f}")

with col_right:
    # SIGNAL CARD
    st.subheader("🎯 Sinyal")
    st.markdown(f"""
    <div style="background-color:{signal_color if signal_color != 'green' else '#90EE90'}; 
                padding:10px; border-radius:10px; text-align:center">
        <h2 style="margin:0">{signal_emoji} {signal_label}</h2>
        <h1 style="margin:0">{score:.0f}<span style="font-size:20px">/100</span></h1>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # REKOMENDASI + SUPPORT RESISTANCE (GABUNG)
    st.subheader("📝 Rekomendasi")
    
    rekomendasi = get_trading_recommendation(score, df)
    st.markdown(rekomendasi)
    
    # Tampilkan Support Resistance tambahan jika perlu
    if not df.empty:
        last = df.iloc[-1]
        st.markdown("---")
        st.subheader("📊 Level Harga")
        col_sup, col_res, col_now = st.columns(3)
        with col_sup:
            st.metric("🛡️ Support", f"Rp{last['support']:,.0f}")
        with col_res:
            st.metric("🚧 Resistance", f"Rp{last['resistance']:,.0f}")
        with col_now:
            st.metric("💰 Harga", f"Rp{last['close']:,.0f}")

# ========== MULTI TIMEFRAME (LEBIH KOMPAK) ==========
st.markdown("---")
st.subheader("⏰ Multi Timeframe")

mtf = multi_timeframe_analysis(symbol)
avg_score = mtf.get('weighted', 0)
final_label, final_color, final_emoji = get_signal_label(avg_score)

# Tampilkan MTF dalam 5 card kecil
cols = st.columns(5)
tf_list = ["5m", "15m", "30m", "1h", "1d"]
for idx, (tf_name, col) in enumerate(zip(tf_list, cols)):
    if tf_name in mtf:
        s = mtf[tf_name]
        if s >= 60:
            bg = "#2e7d32"
        elif s >= 40:
            bg = "#ed6c02"
        else:
            bg = "#d32f2f"
        with col:
            st.markdown(f"""
            <div style="background-color:{bg}; padding:10px; border-radius:10px; text-align:center">
                <h4 style="margin:0">{tf_name}</h4>
                <h2 style="margin:0">{s:.0f}</h2>
            </div>
            """, unsafe_allow_html=True)

# Final Signal
st.markdown(f"""
<div style="background-color:#2d2d2d; padding:10px; border-radius:10px; margin-top:10px; text-align:center">
    <h3 style="margin:0">🎯 Final Signal: {final_emoji} {final_label}</h3>
    <p style="margin:0">Score: {avg_score:.1f}/100</p>
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
                def color_signal(val):
                    if "BUY" in str(val):
                        return "background-color: #2e7d32; color: white"
                    elif "SELL" in str(val):
                        return "background-color: #d32f2f; color: white"
                    return ""
                styled_df = df_scan.style.map(color_signal, subset=['Sinyal'])
                st.dataframe(styled_df, use_container_width=True, hide_index=True)
                st.success(f"🏆 Top 3: {', '.join([r['Kode'] for r in results[:3]])}")
            else:
                st.warning("Tidak ada data")
        except Exception as e:
            st.error(f"Error: {str(e)}")

# FOOTER
st.markdown("---")
st.caption("⚠️ Disclaimer: Alat bantu analisis, bukan rekomendasi investasi.")
