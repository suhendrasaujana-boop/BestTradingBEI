import streamlit as st
import pandas as pd
from datetime import datetime
import time

from data import (
    get_data,
    add_indicators,
    get_ihsg_trend,
    get_ihsg_filter_penalty,
    detect_high_quality_setup,
    calculate_risk_reward,
    calculate_confidence_score,
    calculate_smart_position_size,
    get_trading_recommendation,
    backtest_strategy,
    multi_timeframe_analysis,
    scan_saham
)

st.set_page_config(
    page_title="Robot Saham Pro - Quality Setup",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# SIDEBAR
with st.sidebar:
    st.title("🎯 Robot Saham Pro")
    st.caption("Fokus ke Setup Berkualitas, Bukan Banyak Sinyal")
    st.markdown("---")
    
    symbol = st.text_input("Kode Saham", "^JKSE", key="symbol_input").upper()
    timeframe = st.selectbox("Timeframe", ["1d", "4h", "1h"], key="timeframe_select")
    
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

# ========== MARKET FILTER (IHSG) - KRUSIAL ==========
st.markdown("### 📊 MARKET FILTER (IHSG)")
ihsg_trend, ihsg_score, ihsg_msg = get_ihsg_trend()
penalty, penalty_msg = get_ihsg_filter_penalty()

col_m1, col_m2 = st.columns(2)
with col_m1:
    if ihsg_trend == "BULLISH":
        st.success(f"✅ {ihsg_msg}")
    elif ihsg_trend == "BEARISH":
        st.error(f"⚠️ {ihsg_msg}")
    else:
        st.warning(f"📊 {ihsg_msg}")

with col_m2:
    if penalty < 0:
        st.warning(penalty_msg)
    else:
        st.info(penalty_msg)

# ========== HIGH QUALITY SETUP ==========
st.markdown("---")
st.markdown("### 🎯 HIGH QUALITY SETUP DETECTOR")

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
    st.info("💡 **No Trade Zone** - Kondisi pasar tidak ideal, lebih baik tidak trading")

# Confidence Score
confidence, factors, grade = calculate_confidence_score(df, ihsg_score)

col_c1, col_c2 = st.columns(2)
with col_c1:
    if grade == "SNIPER":
        st.metric("Confidence Score", f"{confidence:.0f}", delta="SNIPER SETUP")
    elif grade == "HIGH":
        st.metric("Confidence Score", f"{confidence:.0f}", delta="High Quality")
    elif grade == "NORMAL":
        st.metric("Confidence Score", f"{confidence:.0f}", delta="Normal")
    elif grade == "LOW":
        st.metric("Confidence Score", f"{confidence:.0f}", delta="Low - Skip")
    else:
        st.metric("Confidence Score", f"{confidence:.0f}", delta="AVOID")

with col_c2:
    st.caption("Faktor penilaian:")
    for name, score, desc in factors[:3]:
        if score > 0:
            st.caption(f"✅ {name}: +{score:.0f} ({desc})")
        else:
            st.caption(f"❌ {name}: {score:.0f} ({desc})")

# ========== PRICE DISPLAY ==========
st.markdown("---")
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
    else:
        st.metric("💰 Current", f"Rp{last['close']:,.0f}")

# ========== CHART ==========
st.markdown("### 📈 Chart")
st.line_chart(df[['close', 'ema20', 'ema50']], height=300)

# ========== TECHNICAL INDICATORS ==========
st.markdown("### 📊 Technical Indicators")
col_i1, col_i2, col_i3 = st.columns(3)

with col_i1:
    st.metric("RSI (14)", f"{last['rsi']:.1f}")
    if last['rsi'] < 30:
        st.caption("🟢 Oversold - Perhatikan reversal")
    elif last['rsi'] > 70:
        st.caption("🔴 Overbought - Hati-hati")
    
    st.metric("ADX", f"{last['adx']:.1f}")
    if last['adx'] >= 25:
        st.caption("✅ Tren Kuat")
    elif last['adx'] >= 20:
        st.caption("⚠️ Tren Mulai")
    else:
        st.caption("🔴 Sideways - Hindari")

with col_i2:
    st.metric("MACD", f"{last['macd']:.2f}")
    st.metric("Signal", f"{last['macd_signal']:.2f}", delta=f"{last['macd_histogram']:.2f}")
    if last['macd_histogram'] > 0:
        st.caption("✅ Bullish momentum")
    else:
        st.caption("❌ Bearish momentum")

with col_i3:
    st.metric("Volume Ratio", f"{last['volume_ratio']:.2f}x")
    if last['volume_ratio'] >= 1.5:
        st.success("✅ Volume Spike - Konfirmasi")
    elif last['volume_ratio'] < 0.6:
        st.warning("⚠️ Volume Sepi - Hati-hati")
    
    st.metric("ATR", f"Rp{last['atr']:,.0f}")
    st.caption(f"Volatility: {(last['atr']/last['close']*100):.2f}%")

# ========== POSITION MANAGEMENT ==========
st.markdown("---")
st.markdown("### 📋 POSITION MANAGEMENT")

if entry_price > 0 and capital > 0:
    atr = last.get('atr', last['close'] * 0.02)
    
    # ATR-based stop loss (2-3 ATR untuk hindari noise)
    stop_loss = entry_price - (2 * atr)
    take_profit = entry_price + (3 * atr)
    
    col_ps1, col_ps2, col_ps3, col_ps4 = st.columns(4)
    
    with col_ps1:
        st.metric("🎯 Entry", f"Rp{entry_price:,.0f}")
    
    with col_ps2:
        st.metric("✂️ Stop Loss", f"Rp{stop_loss:,.0f}")
        if current_price <= stop_loss:
            st.error("🚨 STOP LOSS!")
    
    with col_ps3:
        st.metric("🎯 Take Profit", f"Rp{take_profit:,.0f}")
        if current_price >= take_profit:
            st.success("🎉 TAKE PROFIT!")
    
    # Risk Reward
    rr = calculate_risk_reward(entry_price, stop_loss, take_profit)
    
    with col_ps4:
        st.metric("Risk/Reward", f"1:{rr['ratio']:.1f}")
        if rr['ratio'] >= 2:
            st.success(f"✅ {rr['grade']}")
        elif rr['ratio'] >= 1.5:
            st.info(f"📊 {rr['grade']}")
        else:
            st.error(f"❌ {rr['grade']}")
    
    # Position sizing
    shares, actual_risk = calculate_smart_position_size(capital, entry_price, stop_loss, risk_percent)
    
    col_sz1, col_sz2 = st.columns(2)
    with col_sz1:
        st.metric("📊 Position Size", f"{shares:,} saham")
        st.caption(f"Risk: {actual_risk:.2f}% dari modal")
    
    with col_sz2:
        if rr['ratio'] >= 2 and quality >= 70:
            st.success("✅ Setup memenuhi kriteria - Eksekusi")
        else:
            st.warning("⚠️ Setup tidak memenuhi kriteria - Skip")
    
    # RR Recommendation
    st.info(f"💡 {rr['recommendation']}")

else:
    st.info("💡 Masukkan Entry Price dan Modal di sidebar")

# ========== RECOMMENDATION ==========
st.markdown("---")
st.markdown("### 📝 REKOMENDASI")
st.info(get_trading_recommendation(df))

# ========== MULTI TIMEFRAME ==========
st.markdown("---")
st.markdown("### ⏰ Multi Timeframe Confirmation")

mtf = multi_timeframe_analysis(symbol)
col_tf = st.columns(3)
timeframes = ["1h", "4h", "1d"]

for i, tf in enumerate(timeframes):
    with col_tf[i]:
        score = mtf.get(tf, 50)
        if score >= 70:
            st.success(f"**{tf}**\n{score:.0f}")
        elif score >= 50:
            st.warning(f"**{tf}**\n{score:.0f}")
        else:
            st.error(f"**{tf}**\n{score:.0f}")

final_score = mtf.get('weighted', 50)
if final_score >= 70:
    st.success(f"🎯 Final Signal: BUY ({final_score:.0f}/100)")
elif final_score >= 50:
    st.warning(f"⏸️ Final Signal: NEUTRAL ({final_score:.0f}/100)")
else:
    st.error(f"🔴 Final Signal: SELL ({final_score:.0f}/100)")

# ========== BACKTEST ==========
st.markdown("---")
with st.expander("📊 Backtest Realistis (Dengan Fee, Slippage, Position Sizing)"):
    if st.button("🚀 Jalankan Backtest", width="stretch"):
        with st.spinner("Menghitung performa..."):
            result = backtest_strategy(df)
            
            col_b1, col_b2, col_b3 = st.columns(3)
            with col_b1:
                st.metric("📈 Return", f"{result['return']}%")
                st.metric("✅ Winrate", f"{result['winrate']}%")
            with col_b2:
                st.metric("📉 Max Drawdown", f"{result['max_drawdown']}%")
                st.metric("📊 Profit Factor", f"{result['profit_factor']}")
            with col_b3:
                st.metric("🔄 Trades", result['trades'])
                st.metric("📐 Sharpe Ratio", f"{result['sharpe_ratio']}")
            
            st.caption(f"💰 Modal: Rp100jt → Rp{result['final_capital']:,.0f}")
            
            # Evaluasi
            if result['profit_factor'] >= 1.5 and result['max_drawdown'] < 20:
                st.success("✅ Strategi robust! Profit factor bagus, drawdown terkendali")
            elif result['profit_factor'] >= 1.2:
                st.info("📊 Strategi cukup baik, masih bisa dioptimalkan")
            else:
                st.warning("⚠️ Strategi perlu perbaikan, profit factor < 1.2")

# ========== SCANNER ==========
st.markdown("---")
st.markdown("### 🔍 Scanner Saham (Hanya Top 5 Setup Terbaik)")

if st.button("🚀 SCAN MARKET", width="stretch"):
    with st.spinner("Scanning saham berkualitas..."):
        results = scan_saham()
        if results:
            df_scan = pd.DataFrame(results)
            st.dataframe(df_scan, use_container_width=True, hide_index=True)
            st.caption("💡 Hanya menampilkan setup dengan score >= 65")
        else:
            st.warning("Tidak ada setup berkualitas saat ini. Cek lagi nanti.")

st.markdown("---")
st.caption("""
**⚠️ DISCLAIMER & PRINSIP TRADING:**
1. Fokus ke **setup berkualitas**, bukan banyak sinyal
2. Minimal **Risk Reward 1:2** sebelum entry
3. **No Trade Zone** saat market sideways atau volume sepi
4. Maksimal **2% risk per trade** dari modal
5. **1-3 setup per minggu** sudah cukup untuk profit konsisten
""")

if auto_refresh:
    time.sleep(30)
    st.rerun()
