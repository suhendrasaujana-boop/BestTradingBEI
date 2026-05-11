import streamlit as st
import pandas as pd
from datetime import datetime
import time

from data import (
    get_data,
    add_indicators,
    get_ihsg_trend,
    get_ihsg_filter_penalty,
    detect_bottom_pattern,
    detect_valid_breakout,
    detect_reversal,
    detect_high_quality_setup,
    calculate_confidence_score,
    calculate_entry_sl_tp,
    get_trading_recommendation,
    backtest_strategy,
    multi_timeframe_analysis,
    scan_saham
)

st.set_page_config(
    page_title="Robot Saham Pro",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# SIDEBAR
with st.sidebar:
    st.title("Robot Saham Pro")
    st.caption("Entry, SL, TP dari INDIKATOR")
    st.markdown("---")
    
    symbol = st.text_input("Kode Saham", "^JKSE").upper()
    timeframe = st.selectbox("Timeframe", ["5m", "15m", "30m", "60m", "1d"])
    
    st.markdown("---")
    st.subheader("Risk Management")
    
    capital = st.number_input("Modal (Rp)", min_value=1000000, value=100000000, step=1000000)
    risk_percent = st.slider("Risk per Trade (%)", min_value=0.5, max_value=3.0, value=2.0, step=0.5)
    
    st.markdown("---")
    st.info("Sistem hitung entry dari indikator, SL dari ATR, TP dari ATR")
    st.markdown("---")
    auto_refresh = st.checkbox("Auto Refresh (30 detik)", value=False)
    st.caption(f"Update: {datetime.now().strftime('%H:%M:%S')}")

# MAIN CONTENT
st.title(f"{symbol}")

# Load data
df = get_data(symbol, timeframe)

if df.empty:
    st.warning(f"Data {symbol} kosong. Coba: BBCA.JK, BBRI.JK, ^JKSE")
    st.stop()

df = add_indicators(df)
current_price = df.iloc[-1]['close']

# ========== MARKET FILTER ==========
st.markdown("### Market Filter (IHSG)")
ihsg_trend, ihsg_score, ihsg_msg = get_ihsg_trend()
if ihsg_trend == "BULLISH":
    st.success(f"✅ {ihsg_msg}")
elif ihsg_trend == "BEARISH":
    st.error(f"⚠️ {ihsg_msg}")
else:
    st.warning(f"📊 {ihsg_msg}")

st.markdown("---")

# ========== SIGNAL & LEVEL ==========
st.markdown("### Signal & Level dari Indikator")

entry_price, stop_loss, take_profit, shares, rr_ratio, setup_name, conf = calculate_entry_sl_tp(df, capital, risk_percent)

if entry_price and setup_name != "NO_SETUP":
    if "BUY" in setup_name:
        st.success(f"### SIGNAL: {setup_name}")
    else:
        st.info(f"### SIGNAL: {setup_name}")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("ENTRY PRICE", f"Rp{entry_price:,.0f}")
    with col2:
        st.metric("STOP LOSS", f"Rp{stop_loss:,.0f}")
    with col3:
        st.metric("TAKE PROFIT", f"Rp{take_profit:,.0f}")
    
    col4, col5, col6 = st.columns(3)
    
    with col4:
        st.metric("Position Size", f"{shares:,} saham")
    with col5:
        st.metric("Risk/Reward", f"1:{rr_ratio:.1f}")
    with col6:
        st.metric("Confidence", f"{conf:.0f}/100")
    
    if conf >= 70 and rr_ratio >= 1.5:
        st.success("✅ REKOMENDASI: EKSEKUSI")
    elif conf >= 60:
        st.info("⏸️ REKOMENDASI: TUNGGU KONFIRMASI")
    else:
        st.warning("⛔ REKOMENDASI: SKIP")

else:
    st.warning("Tidak ada setup berkualitas saat ini")

st.markdown("---")

# ========== CONFIDENCE SCORE ==========
st.markdown("### Confidence Score")
confidence, factors, grade = calculate_confidence_score(df, ihsg_score)
st.metric("Score", f"{confidence:.0f}", delta=grade)

# ========== HARGA ==========
st.markdown("### Harga")
last = df.iloc[-1]
col_h1, col_h2, col_h3 = st.columns(3)
with col_h1:
    st.metric("High", f"Rp{last['high']:,.0f}")
with col_h2:
    st.metric("Low", f"Rp{last['low']:,.0f}")
with col_h3:
    if len(df) > 1:
        change = last['close'] - df.iloc[-2]['close']
        change_pct = (change / df.iloc[-2]['close']) * 100
        st.metric("Current", f"Rp{last['close']:,.0f}", delta=f"{change_pct:+.2f}%")

# ========== CHART ==========
st.markdown("### Chart")
st.line_chart(df[['close', 'ema20', 'ema50']], height=300)

# ========== INDIKATOR ==========
st.markdown("### Indicators")
col_i1, col_i2, col_i3 = st.columns(3)
with col_i1:
    st.metric("RSI", f"{last['rsi']:.1f}")
    st.metric("ADX", f"{last['adx']:.1f}")
with col_i2:
    st.metric("MACD", f"{last['macd']:.2f}")
    st.metric("Volume Ratio", f"{last['volume_ratio']:.2f}x")
with col_i3:
    st.metric("ATR", f"Rp{last['atr']:,.0f}")

st.markdown("---")

# ========== MULTI TIMEFRAME ==========
st.markdown("### Multi Timeframe Analysis")

mtf_results, final_score = multi_timeframe_analysis(symbol, capital, risk_percent)

tf_data = []
for tf in ["5m", "15m", "30m", "60m", "1d"]:
    res = mtf_results.get(tf, {})
    tf_data.append({
        "TF": tf,
        "Score": f"{res.get('score', 50):.0f}",
        "Entry": f"Rp{res.get('entry', 0):,.0f}" if res.get('entry') else "-",
        "SL": f"Rp{res.get('stop_loss', 0):,.0f}" if res.get('stop_loss') else "-",
        "TP": f"Rp{res.get('take_profit', 0):,.0f}" if res.get('take_profit') else "-",
    })

st.dataframe(pd.DataFrame(tf_data), use_container_width=True, hide_index=True)

if final_score >= 70:
    st.success(f"✅ FINAL SIGNAL: BULLISH ({final_score:.0f}/100)")
elif final_score >= 50:
    st.warning(f"⏸️ FINAL SIGNAL: NEUTRAL ({final_score:.0f}/100)")
else:
    st.error(f"❌ FINAL SIGNAL: BEARISH ({final_score:.0f}/100)")

st.markdown("---")

# ========== REKOMENDASI ==========
st.markdown("### Rekomendasi")
st.info(get_trading_recommendation(df))

# ========== BACKTEST ==========
with st.expander("Backtest Realistis"):
    if st.button("Jalankan Backtest"):
        with st.spinner("Menghitung..."):
            result = backtest_strategy(df)
            col_b1, col_b2, col_b3 = st.columns(3)
            with col_b1:
                st.metric("Return", f"{result['return']}%")
                st.metric("Winrate", f"{result['winrate']}%")
            with col_b2:
                st.metric("Max DD", f"{result['max_drawdown']}%")
                st.metric("Profit Factor", f"{result['profit_factor']}")
            with col_b3:
                st.metric("Trades", result['trades'])
            st.caption(f"Modal Rp100jt → Rp{result['final_capital']:,.0f}")

# ========== SCANNER ==========
st.markdown("---")
st.markdown("### Scanner Saham")
if st.button("Scan Market"):
    with st.spinner("Scanning..."):
        results = scan_saham()
        if results:
            st.dataframe(pd.DataFrame(results), use_container_width=True, hide_index=True)
        else:
            st.warning("Tidak ada setup berkualitas")

st.markdown("---")
st.caption("Disclaimer: Entry, SL, TP dari indikator. Bukan rekomendasi investasi.")

if auto_refresh:
    time.sleep(30)
    st.rerun()
