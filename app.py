import streamlit as st
import pandas as pd
from datetime import datetime
import time

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

# SIDEBAR
with st.sidebar:
    st.title("📈 Robot Saham")
    st.markdown("---")
    
    symbol = st.text_input("Kode Saham", "BBCA.JK").upper()
    timeframe = st.selectbox("Timeframe", ["5m", "15m", "30m", "60m", "1d"])
    
    st.markdown("---")
    st.subheader("⚙️ SETTING TRADING")
    
    entry_price = st.number_input("💰 Harga Entry (Rp)", min_value=0, value=0, step=100)
    cutloss_percent = st.slider("✂️ Cut Loss (%)", min_value=1, max_value=20, value=5)
    takeprofit_percent = st.slider("🎯 Take Profit (%)", min_value=1, max_value=50, value=15)
    
    st.markdown("---")
    auto_refresh = st.checkbox("Auto Refresh (30 detik)", value=False)
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
confidence_label, confidence_color = get_confidence_level(score)
last = df.iloc[-1]
current_price = last['close']

# ========== BARIS 1: HARGA ==========
st.subheader("💰 Harga")
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("📈 Tertinggi", f"Rp{last['high']:,.0f}")
with col2:
    st.metric("📉 Terendah", f"Rp{last['low']:,.0f}")
with col3:
    if len(df) > 1:
        change = last['close'] - df.iloc[-2]['close']
        change_pct = (change / df.iloc[-2]['close']) * 100
        st.metric("💰 Saat Ini", f"Rp{last['close']:,.0f}", delta=f"{change_pct:+.2f}%")
    else:
        st.metric("💰 Saat Ini", f"Rp{last['close']:,.0f}")

st.markdown("---")

# ========== BARIS 2: CHART + SINYAL ==========
col_chart, col_signal = st.columns([2, 1])

with col_chart:
    st.subheader("📈 Chart Harga & EMA")
    st.line_chart(df[['close', 'ema20', 'ema50']], height=300)
    
    st.subheader("📊 Indikator Teknikal")
    col_rsi, col_macd, col_vol = st.columns(3)
    
    with col_rsi:
        rsi_val = last['rsi']
        st.metric("RSI (14)", f"{rsi_val:.1f}")
        if rsi_val < 30:
            st.success("🟢 Oversold - Peluang Beli")
        elif rsi_val > 70:
            st.warning("🔴 Overbought - Waspada")
    
    with col_macd:
        st.metric("MACD", f"{last['macd']:.2f}")
        st.metric("Signal Line", f"{last['macd_signal']:.2f}", delta=f"{last['macd_histogram']:.2f}")
    
    with col_vol:
        st.metric("Volume", f"{last['volume']:,.0f}")
        st.metric("MA Volume", f"{last['volume_ma20']:,.0f}")

with col_signal:
    # Box Sinyal
    if "BUY" in signal_label:
        bg_color = "#90EE90"
    elif "SELL" in signal_label:
        bg_color = "#FFCCCC"
    else:
        bg_color = "#FFE4B5"
    
    st.markdown(f"""
    <div style="background-color:{bg_color}; padding:15px; border-radius:10px; text-align:center">
        <h2 style="margin:0">{signal_emoji} {signal_label}</h2>
        <h1 style="margin:0">{score:.0f}<span style="font-size:18px">/100</span></h1>
        <p style="margin:5px 0 0 0">Keyakinan: <strong>{confidence_label}</strong></p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.subheader("📝 Rekomendasi")
    st.info(get_trading_recommendation(score, df))
    
    st.markdown("---")
    st.subheader("📊 Support & Resistance")
    col_sup, col_res = st.columns(2)
    with col_sup:
        st.metric("🛡️ Support", f"Rp{last['support']:,.0f}")
    with col_res:
        st.metric("🚧 Resistance", f"Rp{last['resistance']:,.0f}")

st.markdown("---")

# ========== BARIS 3: RISK MANAGEMENT ==========
st.subheader("🛡️ RISK MANAGEMENT")

if entry_price > 0:
    cutloss_price = entry_price * (1 - cutloss_percent/100)
    takeprofit_price = entry_price * (1 + takeprofit_percent/100)
    profit_loss_pct = ((current_price - entry_price) / entry_price) * 100
    profit_loss_rp = current_price - entry_price
    
    col_entry, col_cl, col_tp, col_pl = st.columns(4)
    
    with col_entry:
        st.metric("💰 Entry Price", f"Rp{entry_price:,.0f}")
    
    with col_cl:
        st.metric("✂️ Cut Loss", f"Rp{cutloss_price:,.0f} ({cutloss_percent}%)")
        if current_price <= cutloss_price:
            st.error("🚨 CUT LOSS TRIGGERED! Segera Jual!")
    
    with col_tp:
        st.metric("🎯 Take Profit", f"Rp{takeprofit_price:,.0f} ({takeprofit_percent}%)")
        if current_price >= takeprofit_price:
            st.success("🎉 TAKE PROFIT TRIGGERED! Ambil Untung!")
    
    with col_pl:
        if profit_loss_pct >= 0:
            st.metric("📈 Profit/Loss", f"+{profit_loss_pct:.2f}%", delta=f"+Rp{profit_loss_rp:,.0f}")
        else:
            st.metric("📉 Profit/Loss", f"{profit_loss_pct:.2f}%", delta=f"Rp{profit_loss_rp:,.0f}")
else:
    st.info("💡 Masukkan Harga Entry di sidebar untuk mengaktifkan Risk Management")

st.markdown("---")

# ========== BARIS 4: MULTI TIMEFRAME ==========
st.subheader("⏰ Multi Timeframe Analysis")

mtf = multi_timeframe_analysis(symbol)
avg_score = mtf.get('weighted', 50)

col_tf1, col_tf2, col_tf3, col_tf4, col_tf5 = st.columns(5)

with col_tf1:
    st.metric("5 Menit", f"{mtf.get('5m', 50):.0f}")
with col_tf2:
    st.metric("15 Menit", f"{mtf.get('15m', 50):.0f}")
with col_tf3:
    st.metric("30 Menit", f"{mtf.get('30m', 50):.0f}")
with col_tf4:
    st.metric("1 Jam", f"{mtf.get('1h', 50):.0f}")
with col_tf5:
    st.metric("1 Hari", f"{mtf.get('1d', 50):.0f}")

final_label, _, final_emoji = get_signal_label(avg_score)
st.markdown(f"### 🎯 Final Signal: {final_emoji} {final_label} (Score: {avg_score:.1f}/100)")

st.markdown("---")

# ========== BARIS 5: SCANNER ==========
st.subheader("🔍 Scanner Saham")

if st.button("🚀 SCAN MARKET SEKARANG"):
    with st.spinner("Sedang scanning market..."):
        try:
            results = scan_saham()
            if results:
                df_scan = pd.DataFrame(results)
                st.dataframe(df_scan, use_container_width=True, hide_index=True)
                st.success(f"🏆 Top Pick: {results[0]['Kode']} dengan Score {results[0]['Score']}")
            else:
                st.warning("Tidak ada data saham ditemukan")
        except Exception as e:
            st.error(f"Error: {str(e)}")

st.markdown("---")
st.caption("⚠️ DISCLAIMER: Ini adalah alat bantu analisis teknikal, bukan rekomendasi investasi. Selakukan riset sendiri sebelum trading.")

# Auto Refresh
if auto_refresh:
    time.sleep(30)
    st.rerun()
