import streamlit as st
import pandas as pd
from datetime import datetime
import time

from data import (
    get_data,
    add_indicators,
    calculate_smart_score,
    get_signal_label,
    get_confidence_level,
    multi_timeframe_analysis,
    scan_saham,
    get_trading_recommendation,
    backtest_strategy,
    get_market_regime,
    detect_trend_regime,
    get_smart_entry_signal,
    calculate_position_size,
    calculate_ihsg_filter
)

st.set_page_config(
    page_title="Robot Saham Indonesia Pro",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# SIDEBAR
with st.sidebar:
    st.title("📈 Robot Saham Pro")
    st.markdown("---")
    
    symbol = st.text_input("Kode Saham", "BBCA.JK", key="symbol_input").upper()
    timeframe = st.selectbox("Timeframe", ["5m", "15m", "30m", "60m", "1d"], key="timeframe_select")
    
    st.markdown("---")
    st.subheader("⚙️ RISK MANAGEMENT")
    
    entry_price = st.number_input("💰 Harga Entry (Rp)", min_value=0, value=0, step=100)
    risk_percent = st.slider("🎲 Risk per Trade (%)", min_value=0.5, max_value=5.0, value=2.0, step=0.5)
    
    col_sl, col_tp = st.columns(2)
    with col_sl:
        stop_loss_atr = st.slider("✂️ Stop Loss (ATR)", min_value=1, max_value=5, value=2)
    with col_tp:
        take_profit_atr = st.slider("🎯 Take Profit (ATR)", min_value=1, max_value=10, value=3)
    
    st.markdown("---")
    auto_refresh = st.checkbox("Auto Refresh (30 detik)", value=False)
    st.caption(f"Update: {datetime.now().strftime('%H:%M:%S')}")

# MAIN CONTENT
st.title(f"📊 {symbol}")

# Load data
df = get_data(symbol, timeframe)

if df.empty:
    st.warning(f"Data {symbol} kosong")
    st.stop()

df = add_indicators(df)
score, detailed_scores = calculate_smart_score(df)
signal_label, signal_color, signal_emoji = get_signal_label(score)
confidence_label, confidence_color = get_confidence_level(score)
last = df.iloc[-1]
current_price = last['close']

# ========== MARKET REGIME ==========
st.subheader("📊 MARKET REGIME")
regime = get_market_regime(df)
trend_regime, adx = detect_trend_regime(df)

col_r1, col_r2, col_r3, col_r4 = st.columns(4)
with col_r1:
    st.metric("Market Regime", regime['regime'])
with col_r2:
    st.metric("ADX", f"{regime['adx']}")
with col_r3:
    st.metric("ATR %", f"{regime['atr_pct']}%")
with col_r4:
    st.metric("Trend", trend_regime)

if not regime['trading_allowed']:
    st.error(f"⚠️ {regime['description']}")

# ========== IHSG FILTER ==========
st.subheader("📊 MARKET BREADTH (IHSG)")
ihsg_penalty, ihsg_msg = calculate_ihsg_filter()
if ihsg_penalty < 0:
    st.warning(f"IHSG Filter: {ihsg_msg}")
else:
    st.success(f"IHSG Filter: {ihsg_msg}")

# ========== PRICE DISPLAY ==========
st.subheader("💰 Harga")
col_p1, col_p2, col_p3 = st.columns(3)
with col_p1:
    st.metric("High", f"Rp{last['high']:,.0f}")
with col_p2:
    st.metric("Low", f"Rp{last['low']:,.0f}")
with col_p3:
    if len(df) > 1:
        change = last['close'] - df.iloc[-2]['close']
        change_pct = (change / df.iloc[-2]['close']) * 100
        st.metric("Current", f"Rp{last['close']:,.0f}", delta=f"{change_pct:+.2f}%")
    else:
        st.metric("Current", f"Rp{last['close']:,.0f}")

# ========== SMART ENTRY SIGNAL ==========
smart_entry, entry_desc = get_smart_entry_signal(df)
if smart_entry != "NO_SIGNAL":
    if "BUY" in smart_entry:
        st.success(f"🎯 SMART ENTRY: {smart_entry} - {entry_desc}")
    elif "SELL" in smart_entry:
        st.error(f"⚠️ SMART ENTRY: {smart_entry} - {entry_desc}")
    else:
        st.info(f"ℹ️ {smart_entry} - {entry_desc}")

st.markdown("---")

# ========== 2 COLUMN LAYOUT ==========
col_chart, col_signal = st.columns([2, 1])

with col_chart:
    st.subheader("📈 Chart & Indicators")
    st.line_chart(df[['close', 'ema20', 'ema50']], height=300)
    
    # Indicators
    st.subheader("📊 Technical Indicators")
    col_i1, col_i2, col_i3 = st.columns(3)
    
    with col_i1:
        st.metric("RSI", f"{last['rsi']:.1f}")
        if last['rsi'] < 30:
            st.info("🟢 Oversold")
        elif last['rsi'] > 70:
            st.warning("🔴 Overbought")
        
        st.metric("Stoch RSI", f"{last['stoch_rsi_k']:.1f}")
    
    with col_i2:
        st.metric("MACD", f"{last['macd']:.2f}")
        st.metric("Signal", f"{last['macd_signal']:.2f}", delta=f"{last['macd_histogram']:.2f}")
        st.metric("Volume Ratio", f"{last['volume_ratio']:.2f}x")
    
    with col_i3:
        st.metric("ATR", f"Rp{last['atr']:,.0f}")
        st.metric("BB Width", f"{last['bb_width']:.2f}")
        st.metric("Support/Res", f"Rp{last['support']:,.0f} / Rp{last['resistance']:,.0f}")

with col_signal:
    # Signal box
    if "BUY" in signal_label:
        bg = "#90EE90"
    elif "SELL" in signal_label:
        bg = "#FFCCCC"
    else:
        bg = "#FFE4B5"
    
    st.markdown(f"""
    <div style="background-color:{bg}; padding:15px; border-radius:10px; text-align:center">
        <h1>{signal_emoji} {signal_label}</h1>
        <h2>{score:.0f}<span style="font-size:16px">/100</span></h2>
        <p>Keyakinan: <strong>{confidence_label}</strong></p>
    </div>
    """, unsafe_allow_html=True)
    
    # Detailed scores
    with st.expander("📊 Detail Score"):
        for cat, s in detailed_scores.items():
            st.progress(s/100, text=f"{cat.upper()}: {s:.0f}")
    
    st.markdown("---")
    st.subheader("📝 Rekomendasi")
    st.info(get_trading_recommendation(score, df))

# ========== POSITION MANAGEMENT ==========
st.markdown("---")
st.subheader("📋 POSITION MANAGEMENT")

if entry_price > 0:
    atr = last.get('atr', 0)
    stop_loss = entry_price - (stop_loss_atr * atr)
    take_profit = entry_price + (take_profit_atr * atr)
    
    col_ps1, col_ps2, col_ps3, col_ps4 = st.columns(4)
    
    with col_ps1:
        st.metric("Entry Price", f"Rp{entry_price:,.0f}")
    
    with col_ps2:
        st.metric("Stop Loss", f"Rp{stop_loss:,.0f}", delta=f"-{stop_loss_atr} ATR")
        if current_price <= stop_loss:
            st.error("🚨 STOP LOSS TRIGGERED!")
    
    with col_ps3:
        st.metric("Take Profit", f"Rp{take_profit:,.0f}", delta=f"+{take_profit_atr} ATR")
        if current_price >= take_profit:
            st.success("🎉 TAKE PROFIT!")
    
    with col_ps4:
        pnl = ((current_price - entry_price) / entry_price) * 100
        if pnl >= 0:
            st.metric("P&L", f"+{pnl:.2f}%", delta_color="normal")
        else:
            st.metric("P&L", f"{pnl:.2f}%", delta_color="inverse")
    
    # Position sizing
    capital = st.number_input("💰 Modal (Rp)", min_value=1000000, value=100000000, step=1000000)
    position_size = calculate_position_size(capital, entry_price, stop_loss, risk_percent)
    
    if position_size > 0:
        st.info(f"📊 Position Size: {position_size} saham (Risk: Rp{(entry_price - stop_loss) * position_size:,.0f})")
    else:
        st.warning("⚠️ Stop loss terlalu dekat atau harga invalid")

else:
    st.info("💡 Masukkan Harga Entry di sidebar untuk aktifkan Position Management")

# ========== MULTI TIMEFRAME ==========
st.markdown("---")
st.subheader("⏰ Multi Timeframe")
mtf = multi_timeframe_analysis(symbol)
col_tf = st.columns(5)
for i, (tf, val) in enumerate([("5m", "5m"), ("15m", "15m"), ("30m", "30m"), ("1h", "1h"), ("1d", "1d")]):
    with col_tf[i]:
        st.metric(tf, f"{mtf.get(val, 50):.0f}")

# ========== BACKTEST ==========
st.markdown("---")
with st.expander("📊 Backtest Strategy (Realistic)"):
    if st.button("Jalankan Backtest", use_container_width=True):
        with st.spinner("Backtesting..."):
            result = backtest_strategy(df)
            col_b1, col_b2, col_b3, col_b4 = st.columns(4)
            with col_b1:
                st.metric("Return", f"{result['return']}%")
            with col_b2:
                st.metric("Winrate", f"{result['winrate']}%")
            with col_b3:
                st.metric("Trades", result['trades'])
            with col_b4:
                st.metric("Max DD", f"{result['max_drawdown']}%")
            st.caption(f"Modal: Rp100jt → Rp{result['final_capital']:,.0f}")

# ========== SCANNER ==========
st.markdown("---")
st.subheader("🔍 Scanner Saham")
if st.button("🚀 Scan Market", width="stretch"):
    with st.spinner("Scanning..."):
        results = scan_saham()
        if results:
            st.dataframe(pd.DataFrame(results), use_container_width=True, hide_index=True)

st.markdown("---")
st.caption("⚠️ Disclaimer: Alat bantu analisis, bukan rekomendasi investasi.")

if auto_refresh:
    time.sleep(30)
    st.rerun()
