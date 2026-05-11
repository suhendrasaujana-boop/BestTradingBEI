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

with st.sidebar:
    st.title("📈 Robot Saham")
    st.markdown("---")
    symbol = st.text_input("Kode Saham", "BBCA.JK", key="symbol_input").upper()
    timeframe = st.selectbox("Timeframe", ["5m", "15m", "30m", "60m", "1d"], key="timeframe_select")
    
    st.markdown("---")
    st.subheader("⚙️ Setting Trading")
    entry_price = st.number_input("💰 Harga Entry (Rp)", min_value=0, value=0, step=100, key="entry_price")
    cutloss_percent = st.slider("✂️ Cut Loss (%)", min_value=1, max_value=20, value=5, key="cutloss")
    takeprofit_percent = st.slider("🎯 Take Profit (%)", min_value=1, max_value=50, value=15, key="takeprofit")
    
    st.markdown("---")
    auto_refresh = st.checkbox("Auto Refresh (30 detik)", value=False)
    st.markdown("---")
    st.caption(f"Update: {datetime.now().strftime('%H:%M:%S')}")

st.title(f"📊 {symbol}")

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

st.subheader("💰 Harga")
col_high, col_low, col_close = st.columns(3)

with col_high:
    st.metric("📈 Tertinggi (High)", f"Rp{last['high']:,.0f}")
with col_low:
    st.metric("📉 Terendah (Low)", f"Rp{last['low']:,.0f}")
with col_close:
    if len(df) > 1:
        change = last['close'] - df.iloc[-2]['close']
        change_pct = (change / df.iloc[-2]['close']) * 100
        st.metric("💰 Saat Ini (Close)", f"Rp{last['close']:,.0f}", 
                  delta=f"{change_pct:+.2f}%", delta_color="normal")
    else:
        st.metric("💰 Saat Ini (Close)", f"Rp{last['close']:,.0f}")

st.markdown("---")

col_left, col_right = st.columns([2, 1.2])

with col_left:
    st.subheader("📈 Harga & EMA")
    st.line_chart(df[['close', 'ema20', 'ema50']], height=300)
    
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
    st.subheader("🛡️ Risk Management")
    
    col_entry, col_cl, col_tp = st.columns(3)
    
    with col_entry:
        if entry_price > 0:
            st.metric("💰 Entry Price", f"Rp{entry_price:,.0f}")
            profit_loss = ((current_price - entry_price) / entry_price) * 100
            if profit_loss > 0:
                st.success(f"Profit/Loss: +{profit_loss:.2f}%")
            else:
                st.error(f"Profit/Loss: {profit_loss:.2f}%")
        else:
            st.info("Set entry price di sidebar")
    
    with col_cl:
        if entry_price > 0:
            cutloss_price = entry_price * (1 - cutloss_percent/100)
            st.metric("✂️ Cut Loss", f"Rp{cutloss_price:,.0f} ({cutloss_percent}%)")
            if current_price <= cutloss_price:
                st.error("⚠️ CUT LOSS TRIGGERED!")
        else:
            st.info("Set cut loss % di sidebar")
    
    with col_tp:
        if entry_price > 0:
            tp_price = entry_price * (1 + takeprofit_percent/100)
            st.metric("🎯 Take Profit", f"Rp{tp_price:,.0f} ({takeprofit_percent}%)")
            if current_price >= tp_price:
                st.success("🎉 TAKE PROFIT TRIGGERED!")
        else:
            st.info("Set take profit % di sidebar")
    
    st.markdown("---")
    st.subheader("📊 Support & Resistance")
    col_sup, col_res = st.columns(2)
    with col_sup:
        st.metric("🛡️ Support", f"Rp{last['support']:,.0f}")
    with col_res:
        st.metric("🚧 Resistance", f"Rp{last['resistance']:,.0f}")

st.markdown("---")
st.subheader("⏰ Multi Timeframe")

mtf = multi_timeframe_analysis(symbol)
avg_score = mtf.get('weighted', 0)
final_label, final_color, final_emoji = get_signal_label(avg_score)

col5m, col15m, col30m, col1h, col1d = st.columns(5)

with col5m: st.metric("5m", f"{mtf.get('5m', 0):.0f}")
with col15m: st.metric("15m", f"{mtf.get('15m', 0):.0f}")
with col30m: st.metric("30m", f"{mtf.get('30m', 0):.0f}")
with col1h: st.metric("1h", f"{mtf.get('1h', 0):.0f}")
with col1d: st.metric("1d", f"{mtf.get('1d', 0):.0f}")

st.markdown("---")
st.subheader("🔍 Scanner Saham")

if st.button("🚀 Scan Market", width="stretch"):
    with st.spinner("Scanning..."):
        try:
            results = scan_saham()
            if results:
                df_scan = pd.DataFrame(results)
                st.dataframe(df_scan, use_container_width=True, hide_index=True)
            else:
                st.warning("Tidak ada data")
        except Exception as e:
            st.error(f"Error: {str(e)}")

st.markdown("---")
st.caption("⚠️ Disclaimer: Alat bantu analisis, bukan rekomendasi investasi.")

if auto_refresh:
    time.sleep(30)
    st.rerun()
