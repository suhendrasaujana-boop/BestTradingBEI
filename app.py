import streamlit as st
import pandas as pd
from datetime import datetime
import time

from data import (
    get_data,
    add_indicators,
    get_ihsg_trend,
    detect_bottom_pattern,
    detect_valid_breakout,
    detect_reversal,
    detect_high_quality_setup,
    calculate_risk_reward,
    calculate_smart_position_size,
    calculate_confidence_score,
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
    st.title("🎯 Robot Saham Pro")
    st.caption("Fokus ke Setup Berkualitas")
    st.markdown("---")
    
    symbol = st.text_input("Kode Saham", "^JKSE", key="symbol_input").upper()
    timeframe = st.selectbox("Timeframe", ["1d", "1h"], key="timeframe_select")
    
    st.markdown("---")
    st.subheader("⚙️ RISK MANAGEMENT")
    
    entry_price = st.number_input("💰 Harga Entry (Rp)", min_value=0, value=0, step=100)
    capital = st.number_input("💼 Modal (Rp)", min_value=1000000, value=100000000, step=1000000)
    risk_percent = st.slider("🎲 Risk per Trade (%)", min_value=0.5, max_value=3.0, value=2.0, step=0.5)
    
    st.markdown("---")
    st.info("💡 **Prinsip:**\n- 1-3 setup berkualitas per minggu\n- Risk Reward minimal 1:2\n- Hindari overtrading")
    st.markdown("---")
    auto_refresh = st.checkbox("Auto Refresh (30 detik)", value=False)
    st.caption(f"Update: {datetime.now().strftime('%H:%M:%S')}")

# MAIN CONTENT
st.title(f"🎯 {symbol}")

# Load data
df = get_data(symbol, timeframe)

if df.empty:
    st.warning(f"Data {symbol} kosong")
    st.stop()

df = add_indicators(df)
current_price = df.iloc[-1]['close']

# ========== MARKET FILTER ==========
st.markdown("### 📊 MARKET FILTER (IHSG)")
ihsg_trend, ihsg_score, ihsg_msg = get_ihsg_trend()
if ihsg_trend == "BULLISH":
    st.success(f"✅ {ihsg_msg}")
elif ihsg_trend == "BEARISH":
    st.error(f"⚠️ {ihsg_msg}")
else:
    st.warning(f"📊 {ihsg_msg}")

st.markdown("---")

# ========== BOTTOM, BREAKOUT, REVERSAL ==========
st.markdown("### 🔍 BOTTOM, BREAKOUT & REVERSAL")

col_b1, col_b2, col_b3 = st.columns(3)

is_bottom, bottom_conf, bottom_desc = detect_bottom_pattern(df)
with col_b1:
    if is_bottom:
        st.success(f"📍 **BOTTOM**")
        st.progress(bottom_conf/100, text=f"Keyakinan: {bottom_conf:.0f}%")
        st.caption(bottom_desc[:35])
    else:
        st.info("📍 **BOTTOM**")
        st.caption("Tidak ada sinyal bottom")

is_breakout, breakout_type, breakout_conf, breakout_desc = detect_valid_breakout(df)
with col_b2:
    if breakout_type == "STRONG_BREAKOUT":
        st.success(f"🚀 **STRONG BREAKOUT**")
        st.progress(breakout_conf/100, text=f"Keyakinan: {breakout_conf:.0f}%")
        st.caption(breakout_desc[:35])
    elif breakout_type == "VALID_BREAKOUT":
        st.info(f"📈 **VALID BREAKOUT**")
        st.progress(breakout_conf/100, text=f"Keyakinan: {breakout_conf:.0f}%")
        st.caption(breakout_desc[:35])
    elif breakout_type == "FAKE_BREAKOUT":
        st.error(f"⚠️ **FAKE BREAKOUT!**")
        st.caption(breakout_desc[:35])
    else:
        st.info("🚀 **BREAKOUT**")
        st.caption("Tidak ada sinyal breakout")

reversal_type, reversal_conf, reversal_desc = detect_reversal(df)
with col_b3:
    if reversal_type == "BULLISH":
        st.success(f"🔄 **BULLISH REVERSAL**")
        st.progress(reversal_conf/100, text=f"Keyakinan: {reversal_conf:.0f}%")
        st.caption(reversal_desc[:35])
    elif reversal_type == "BEARISH":
        st.error(f"🔄 **BEARISH REVERSAL**")
        st.progress(reversal_conf/100, text=f"Keyakinan: {reversal_conf:.0f}%")
        st.caption(reversal_desc[:35])
    else:
        st.info("🔄 **REVERSAL**")
        st.caption("Tidak ada sinyal reversal")

st.markdown("---")

# ========== HIGH QUALITY SETUP ==========
st.markdown("### 🎯 HIGH QUALITY SETUP")
setup, quality, setup_msg = detect_high_quality_setup(df)

if quality >= 85:
    st.success(f"### 🔥 {setup_msg}")
    st.balloons()
elif quality >= 70:
    st.success(f"### 📈 {setup_msg}")
elif quality >= 55:
    st.info(f"### ⏸️ {setup_msg}")
else:
    st.warning(f"### ⛔ {setup_msg}")

# Confidence Score
confidence, factors, grade = calculate_confidence_score(df, ihsg_score)

col_c1, col_c2 = st.columns(2)
with col_c1:
    st.metric("Confidence Score", f"{confidence:.0f}", delta=grade)
with col_c2:
    for name, score, desc in factors[:3]:
        if score > 0:
            st.caption(f"✅ {name}: +{score:.0f}")
        else:
            st.caption(f"❌ {name}: {score:.0f}")

st.markdown("---")

# ========== PRICE ==========
st.markdown("### 💰 Harga")
last = df.iloc[-1]
col_p1, col_p2, col_p3 = st.columns(3)
with col_p1:
    st.metric("📈 High", f"Rp{last['high']:,.0f}")
with col_p2:
    st.metric("📉 Low", f"Rp{last['low']:,.0f}")
with col_p3:
    if len(df) > 1:
        change = last['close'] - df.iloc[-2]['close']
        change_pct = (change / df.iloc[-2]['close']) * 100
        st.metric("💰 Current", f"Rp{last['close']:,.0f}", delta=f"{change_pct:+.2f}%")

# ========== CHART ==========
st.markdown("### 📈 Chart")
st.line_chart(df[['close', 'ema20', 'ema50']], height=300)

# ========== INDICATORS ==========
st.markdown("### 📊 Indicators")
col_i1, col_i2, col_i3 = st.columns(3)
with col_i1:
    st.metric("RSI", f"{last['rsi']:.1f}")
    st.metric("ADX", f"{last['adx']:.1f}")
with col_i2:
    st.metric("MACD", f"{last['macd']:.2f}")
    st.metric("Volume Ratio", f"{last['volume_ratio']:.2f}x")
with col_i3:
    st.metric("ATR", f"Rp{last['atr']:,.0f}")
    st.metric("Support", f"Rp{last['support']:,.0f}")

st.markdown("---")

# ========== POSITION MANAGEMENT ==========
st.markdown("### 📋 POSITION MANAGEMENT")

if entry_price > 0 and capital > 0:
    atr = last.get('atr', last['close'] * 0.02)
    stop_loss = entry_price - (2 * atr)
    take_profit = entry_price + (3 * atr)
    
    col_ps1, col_ps2, col_ps3, col_ps4 = st.columns(4)
    with col_ps1:
        st.metric("Entry", f"Rp{entry_price:,.0f}")
    with col_ps2:
        st.metric("Stop Loss", f"Rp{stop_loss:,.0f}")
        if current_price <= stop_loss:
            st.error("🚨 STOP LOSS!")
    with col_ps3:
        st.metric("Take Profit", f"Rp{take_profit:,.0f}")
        if current_price >= take_profit:
            st.success("🎉 TAKE PROFIT!")
    
    rr = calculate_risk_reward(entry_price, stop_loss, take_profit)
    with col_ps4:
        st.metric("Risk/Reward", f"1:{rr['ratio']:.1f}")
    
    shares, actual_risk = calculate_smart_position_size(capital, entry_price, stop_loss, risk_percent)
    st.metric("Position Size", f"{shares:,} saham")
    st.caption(f"Risk: {actual_risk:.2f}% dari modal")
    
    if rr['ratio'] >= 2 and quality >= 70:
        st.success("✅ Setup memenuhi kriteria - Eksekusi")
    else:
        st.warning("⚠️ Setup belum memenuhi kriteria - Skip")
else:
    st.info("💡 Masukkan Entry Price dan Modal di sidebar")

# ========== RECOMMENDATION ==========
st.markdown("---")
st.markdown("### 📝 REKOMENDASI")
st.info(get_trading_recommendation(df))

# ========== MULTI TIMEFRAME ==========
st.markdown("---")
st.markdown("### ⏰ Multi Timeframe")
mtf = multi_timeframe_analysis(symbol)
col_tf = st.columns(2)
for i, tf in enumerate(["1h", "1d"]):
    with col_tf[i]:
        score = mtf.get(tf, 50)
        if score >= 70:
            st.success(f"**{tf}**: {score:.0f}")
        elif score >= 50:
            st.warning(f"**{tf}**: {score:.0f}")
        else:
            st.error(f"**{tf}**: {score:.0f}")

final = mtf.get('weighted', 50)
if final >= 70:
    st.success(f"🎯 Final Signal: BUY ({final:.0f})")
elif final >= 50:
    st.warning(f"⏸️ Final Signal: NEUTRAL ({final:.0f})")
else:
    st.error(f"🔴 Final Signal: SELL ({final:.0f})")

# ========== BACKTEST ==========
st.markdown("---")
with st.expander("📊 Backtest Realistis"):
    if st.button("🚀 Jalankan Backtest", width="stretch"):
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
st.markdown("### 🔍 Scanner Saham")
if st.button("🚀 SCAN MARKET", width="stretch"):
    with st.spinner("Scanning..."):
        results = scan_saham()
        if results:
            st.dataframe(pd.DataFrame(results), use_container_width=True, hide_index=True)
        else:
            st.warning("Tidak ada setup berkualitas")

st.markdown("---")
st.caption("⚠️ Disclaimer: Alat bantu analisis, bukan rekomendasi investasi. Fokus ke setup berkualitas, minimal Risk Reward 1:2.")

if auto_refresh:
    time.sleep(30)
    st.rerun()
