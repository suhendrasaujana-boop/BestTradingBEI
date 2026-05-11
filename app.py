import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import time

from data import (
    get_data,
    add_indicators,
    get_ihsg_trend,
    detect_market_structure,
    detect_smart_money_volume,
    detect_liquidity_sweep,
    detect_candlestick_pattern,
    detect_market_regime,
    get_pivot_sr,
    calculate_entry_sl_tp,
    calculate_confidence_score,
    detect_high_quality_setup,
    get_trading_recommendation,
    backtest_strategy,
    scan_saham,
    get_multi_timeframe_alignment
)

st.set_page_config(
    page_title="Smart Money Trading System",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# SIDEBAR
with st.sidebar:
    st.title("🧠 Smart Money Trading")
    st.caption("Market Structure | Smart Money | Liquidity Sweep")
    st.markdown("---")
    
    symbol = st.text_input("Kode Saham", "BBCA.JK").upper()
    timeframe = st.selectbox("Timeframe", ["1d", "60m", "30m", "15m", "5m"])
    
    st.markdown("---")
    st.subheader("💰 Risk Management")
    
    capital = st.number_input("Modal (Rp)", min_value=1000000, value=100000000, step=1000000)
    risk_percent = st.slider("Risk per Trade (%)", min_value=0.5, max_value=3.0, value=2.0, step=0.5)
    
    st.markdown("---")
    st.info("""
    **Fitur:**
    - Market Structure (BOS)
    - Smart Money Volume
    - Liquidity Sweep / SFP
    - Candlestick Patterns
    - Pivot Support/Resistance
    - Multi Timeframe Alignment
    """)
    st.markdown("---")
    auto_refresh = st.checkbox("Auto Refresh (30 detik)", value=False)
    st.caption(f"Update: {datetime.now().strftime('%H:%M:%S')}")

# MAIN CONTENT
st.title(f"🧠 {symbol} - Smart Money Analysis")

df = get_data(symbol, timeframe)

if df.empty:
    st.warning(f"Data {symbol} kosong. Coba: BBCA.JK, BBRI.JK, BMRI.JK, ASII.JK")
    st.stop()

df = add_indicators(df)
last = df.iloc[-1]
current_price = last['close']

# ========== MARKET FILTER IHSG ==========
st.markdown("### 📊 Market Filter")
ihsg_trend, ihsg_score, ihsg_msg = get_ihsg_trend()
if ihsg_trend == "BULLISH":
    st.success(f"✅ {ihsg_msg}")
elif ihsg_trend == "BEARISH":
    st.error(f"⚠️ {ihsg_msg}")
else:
    st.warning(f"📊 {ihsg_msg}")

st.markdown("---")

# ========== 4 KOLOM UTAMA ==========
st.markdown("### 🔍 Smart Money Detection")

col1, col2, col3, col4 = st.columns(4)

# Market Structure
structure, struct_conf, struct_desc = detect_market_structure(df)
with col1:
    if "BULLISH" in structure:
        st.success(f"**Market Structure**\n{structure}")
    elif "BEARISH" in structure:
        st.error(f"**Market Structure**\n{structure}")
    else:
        st.info(f"**Market Structure**\n{structure}")
    st.caption(f"Keyakinan: {struct_conf:.0f}%")
    st.caption(struct_desc[:40])

# Smart Money Volume
sm, sm_conf, sm_desc = detect_smart_money_volume(df)
with col2:
    if sm == "ACCUMULATION":
        st.success(f"**Smart Money**\n{sm}")
    elif sm == "DISTRIBUTION":
        st.error(f"**Smart Money**\n{sm}")
    else:
        st.info(f"**Smart Money**\n{sm}")
    st.caption(f"Keyakinan: {sm_conf:.0f}%")
    st.caption(sm_desc[:40])

# Liquidity Sweep
is_sweep, sweep_conf, sweep_type, sweep_desc = detect_liquidity_sweep(df)
with col3:
    if sweep_type == "BULLISH_SFP":
        st.success(f"**Liquidity Sweep**\n{sweep_type}")
    elif sweep_type == "BEARISH_SFP":
        st.error(f"**Liquidity Sweep**\n{sweep_type}")
    elif sweep_type == "FAKE_BREAKOUT":
        st.warning(f"**Liquidity Sweep**\n{sweep_type}")
    else:
        st.info(f"**Liquidity Sweep**\nTidak ada")
    st.caption(f"Keyakinan: {sweep_conf:.0f}%")
    st.caption(sweep_desc[:40])

# Candlestick Pattern
pattern, pattern_conf, pattern_desc = detect_candlestick_pattern(df)
with col4:
    if "BULLISH" in pattern:
        st.success(f"**Candlestick**\n{pattern}")
    elif "BEARISH" in pattern:
        st.error(f"**Candlestick**\n{pattern}")
    else:
        st.info(f"**Candlestick**\n{pattern}")
    st.caption(pattern_desc[:40])

st.markdown("---")

# ========== MARKET REGIME ==========
st.markdown("### 📈 Market Regime")
regime, regime_conf, regime_desc = detect_market_regime(df)

col_r1, col_r2, col_r3 = st.columns(3)
with col_r1:
    st.metric("Regime", regime, delta=f"Confidence {regime_conf:.0f}%")
with col_r2:
    st.metric("ADX", f"{last['adx']:.1f}")
with col_r3:
    atr_pct = (last['atr'] / last['close'] * 100) if last['close'] > 0 else 0
    st.metric("ATR %", f"{atr_pct:.2f}%")
st.caption(regime_desc)

st.markdown("---")

# ========== PIVOT SUPPORT & RESISTANCE ==========
st.markdown("### 📊 Pivot Support & Resistance")
support, resistance, pivot, r1, r2, s1, s2, fib_382, fib_618 = get_pivot_sr(df)

col_sr1, col_sr2, col_sr3, col_sr4 = st.columns(4)
with col_sr1:
    st.metric("Support", f"Rp{support:,.0f}")
    st.metric("S1", f"Rp{s1:,.0f}")
with col_sr2:
    st.metric("Resistance", f"Rp{resistance:,.0f}")
    st.metric("R1", f"Rp{r1:,.0f}")
with col_sr3:
    st.metric("Pivot", f"Rp{pivot:,.0f}")
    st.metric("R2", f"Rp{r2:,.0f}")
with col_sr4:
    st.metric("Fib 38.2%", f"Rp{fib_382:,.0f}")
    st.metric("Fib 61.8%", f"Rp{fib_618:,.0f}")

st.markdown("---")

# ========== ENTRY, SL, TP DARI INDIKATOR ==========
st.markdown("### 🎯 Entry - Stop Loss - Take Profit (Dari Indikator)")

entry, sl, tp, shares, rr, setup, conf, signals = calculate_entry_sl_tp(df, capital, risk_percent)

if entry:
    if "BUY" in setup:
        st.success(f"### 🔥 {setup}")
    else:
        st.info(f"### 📊 {setup}")
    
    col_e1, col_e2, col_e3 = st.columns(3)
    with col_e1:
        st.metric("🎯 ENTRY", f"Rp{entry:,.0f}")
    with col_e2:
        st.metric("✂️ STOP LOSS", f"Rp{sl:,.0f}")
    with col_e3:
        st.metric("🏁 TAKE PROFIT", f"Rp{tp:,.0f}")
    
    col_ps1, col_ps2, col_ps3 = st.columns(3)
    with col_ps1:
        st.metric("Position Size", f"{shares:,} saham")
    with col_ps2:
        st.metric("Risk/Reward", f"1:{rr:.1f}")
    with col_ps3:
        st.metric("Confidence", f"{conf:.0f}/100")
    
    st.caption(" | ".join(signals))
    
    if conf >= 70 and rr >= 1.5:
        st.success("✅ REKOMENDASI: EKSEKUSI")
    elif conf >= 60:
        st.info("⏸️ REKOMENDASI: TUNGGU KONFIRMASI")
    else:
        st.warning("⛔ REKOMENDASI: SKIP")
else:
    st.warning("⛔ Tidak ada setup trading berkualitas saat ini")

st.markdown("---")

# ========== CONFIDENCE SCORE ==========
st.markdown("### 📊 Confidence Score")
confidence, factors, grade = calculate_confidence_score(df, ihsg_score)

col_conf1, col_conf2 = st.columns([1, 2])
with col_conf1:
    st.metric("Total Score", f"{confidence:.0f}", delta=grade)
with col_conf2:
    for name, score, desc in factors[:5]:
        if score > 0:
            st.caption(f"✅ {name}: +{score:.0f} ({desc})")
        else:
            st.caption(f"❌ {name}: {score:.0f} ({desc})")

st.markdown("---")

# ========== MULTI TIMEFRAME ALIGNMENT ==========
st.markdown("### ⏰ Multi Timeframe Alignment")
st.caption("Daily (trend utama) → 1 Hour (setup) → 15m (entry timing)")

with st.spinner("Menganalisis multi timeframe..."):
    mtf_results, alignment, alignment_score, mtf_signals = get_multi_timeframe_alignment(symbol, capital, risk_percent)

col_mtf1, col_mtf2, col_mtf3 = st.columns(3)

for i, (tf_name, tf_data) in enumerate(mtf_results.items()):
    with [col_mtf1, col_mtf2, col_mtf3][i]:
        st.subheader(tf_name.upper())
        st.metric("Direction", tf_data['direction'])
        st.metric("Regime", tf_data['regime'][:10])
        if tf_data['entry']:
            st.caption(f"Entry: Rp{tf_data['entry']:,.0f}")
            st.caption(f"SL: Rp{tf_data['stop_loss']:,.0f}")
            st.caption(f"TP: Rp{tf_data['take_profit']:,.0f}")

st.markdown("---")
if alignment_score >= 80:
    st.success(f"### ✅ {alignment} - Semua timeframe searah! (Score: {alignment_score})")
elif alignment_score >= 60:
    st.info(f"### 📊 {alignment} - Sebagian searah (Score: {alignment_score})")
else:
    st.warning(f"### ⚠️ {alignment} - Timeframe kontradiksi (Score: {alignment_score})")

st.caption(" | ".join(mtf_signals))

st.markdown("---")

# ========== HARGA & CHART ==========
st.markdown("### 💰 Harga Real-time")
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
    else:
        st.metric("Current", f"Rp{last['close']:,.0f}")

st.markdown("### 📈 Chart")
st.line_chart(df[['close', 'ema20', 'ema50']], height=300)

# ========== REKOMENDASI ==========
st.markdown("---")
st.markdown("### 📝 Final Recommendation")
st.info(get_trading_recommendation(df))

# ========== BACKTEST ==========
st.markdown("---")
with st.expander("📊 Advanced Backtest (Sharpe, Expectancy, Equity Curve)"):
    if st.button("🚀 Jalankan Backtest", width="stretch"):
        with st.spinner("Menghitung performa..."):
            result = backtest_strategy(df, capital, risk_percent)
            
            col_b1, col_b2, col_b3 = st.columns(3)
            with col_b1:
                st.metric("Return", f"{result['return']}%")
                st.metric("Winrate", f"{result['winrate']}%")
            with col_b2:
                st.metric("Max DD", f"{result['max_drawdown']}%")
                st.metric("Profit Factor", f"{result['profit_factor']}")
            with col_b3:
                st.metric("Sharpe Ratio", f"{result['sharpe_ratio']}")
                st.metric("Expectancy", f"Rp{result['expectancy']:,.0f}")
            
            st.metric("Total Trades", result['trades'])
            st.caption(f"Modal Rp100jt → Rp{result['final_capital']:,.0f}")
            
            if result['equity_curve'] and len(result['equity_curve']) > 1:
                st.subheader("Equity Curve")
                fig, ax = plt.subplots(figsize=(10, 4))
                ax.plot(result['equity_curve'])
                ax.set_title("Equity Curve")
                ax.set_ylabel("Modal (Rp)")
                ax.set_xlabel("Trade")
                st.pyplot(fig)

# ========== SCANNER ==========
st.markdown("---")
st.markdown("### 🔍 Scanner Saham (Smart Money Detection)")
if st.button("🚀 SCAN MARKET", width="stretch"):
    with st.spinner("Scanning market untuk smart money setup..."):
        results = scan_saham()
        if results:
            df_scan = pd.DataFrame(results)
            st.dataframe(df_scan, use_container_width=True, hide_index=True)
            st.success(f"🏆 Top Pick: {results[0]['Kode']} (Score: {results[0]['Score']})")
        else:
            st.warning("Tidak ada setup smart money saat ini")

st.markdown("---")
st.caption("⚠️ DISCLAIMER: Sistem berbasis Market Structure & Smart Money. Entry, SL, TP dihitung dari indikator. Bukan rekomendasi investasi.")

if auto_refresh:
    time.sleep(30)
    st.rerun()
