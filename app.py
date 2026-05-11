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
    
    # DEFAULT SAHAM DIUBAH KE IHSG (^JKSE)
    symbol = st.text_input("Kode Saham", "^JKSE", key="symbol_input").upper()
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
    st.warning(f"Data {symbol} kosong. Cek kode saham (contoh: BBCA.JK, ^JKSE untuk IHSG)")
    st.stop()

df = add_indicators(df)
score, detailed_scores = calculate_smart_score(df)
signal_label, signal_color, signal_emoji = get_signal_label(score)
confidence_label, confidence_color = get_confidence_level(score)
last = df.iloc[-1]
current_price = last['close']

# ========== MARKET REGIME (TIDAK TERPOTONG) ==========
st.markdown("### 📊 MARKET REGIME")
regime = get_market_regime(df)
trend_regime, adx = detect_trend_regime(df)

# Pakai container biar tidak terpotong
col_r1, col_r2, col_r3, col_r4 = st.columns(4)
with col_r1:
    st.metric("Market Regime", regime['regime'], help=regime['description'])
with col_r2:
    st.metric("ADX", f"{regime['adx']}")
with col_r3:
    st.metric("ATR %", f"{regime['atr_pct']}%")
with col_r4:
    st.metric("Trend", trend_regime)

if not regime['trading_allowed']:
    st.warning(f"⚠️ {regime['description']}")

# ========== IHSG FILTER ==========
st.markdown("### 📊 MARKET BREADTH (IHSG)")
ihsg_penalty, ihsg_msg = calculate_ihsg_filter()
if ihsg_penalty < 0:
    st.warning(f"IHSG Filter: {ihsg_msg}")
else:
    st.success(f"IHSG Filter: {ihsg_msg}")

# ========== PRICE DISPLAY ==========
st.markdown("### 💰 Harga")
col_p1, col_p2, col_p3 = st.columns(3)
with col_p1:
    st.metric("📈 Tertinggi", f"Rp{last['high']:,.0f}")
with col_p2:
    st.metric("📉 Terendah", f"Rp{last['low']:,.0f}")
with col_p3:
    if len(df) > 1:
        change = last['close'] - df.iloc[-2]['close']
        change_pct = (change / df.iloc[-2]['close']) * 100
        st.metric("💰 Saat Ini", f"Rp{last['close']:,.0f}", delta=f"{change_pct:+.2f}%")
    else:
        st.metric("💰 Saat Ini", f"Rp{last['close']:,.0f}")

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
    st.markdown("### 📈 Chart & Indicators")
    st.line_chart(df[['close', 'ema20', 'ema50']], height=300)
    
    st.markdown("### 📊 Technical Indicators")
    
    # Baris 1 - RSI & Stoch
    col_i1, col_i2, col_i3 = st.columns(3)
    
    with col_i1:
        st.metric("RSI (14)", f"{last['rsi']:.1f}")
        if last['rsi'] < 30:
            st.caption("🟢 Oversold - Peluang Beli")
        elif last['rsi'] > 70:
            st.caption("🔴 Overbought - Waspada")
        st.metric("Stoch RSI", f"{last['stoch_rsi_k']:.1f}")
    
    with col_i2:
        st.metric("MACD", f"{last['macd']:.2f}")
        st.metric("Signal Line", f"{last['macd_signal']:.2f}")
        st.metric("Histogram", f"{last['macd_histogram']:.2f}")
    
    with col_i3:
        st.metric("Volume Ratio", f"{last['volume_ratio']:.2f}x")
        st.metric("ATR", f"Rp{last['atr']:,.0f}")
        st.metric("BB Width", f"{last['bb_width']:.2f}")
    
    # Baris 2 - Support Resistance
    col_sup, col_res = st.columns(2)
    with col_sup:
        st.metric("🛡️ Support", f"Rp{last['support']:,.0f}")
    with col_res:
        st.metric("🚧 Resistance", f"Rp{last['resistance']:,.0f}")

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
        <h2 style="margin:0">{signal_emoji} {signal_label}</h2>
        <h1 style="margin:0">{score:.0f}<span style="font-size:16px">/100</span></h1>
        <p style="margin:5px 0 0 0">Keyakinan: <strong>{confidence_label}</strong></p>
    </div>
    """, unsafe_allow_html=True)
    
    # Weight distribution
    st.markdown("---")
    st.markdown("### 📊 Score Breakdown")
    for cat, s in detailed_scores.items():
        st.progress(int(s), text=f"{cat.upper()}: {s:.0f}")
    
    st.markdown("---")
    st.markdown("### 📝 Rekomendasi")
    st.info(get_trading_recommendation(score, df))

# ========== POSITION MANAGEMENT ==========
st.markdown("---")
st.markdown("### 📋 POSITION MANAGEMENT")

if entry_price > 0:
    atr = last.get('atr', 0)
    stop_loss = entry_price - (stop_loss_atr * atr)
    take_profit = entry_price + (take_profit_atr * atr)
    
    col_ps1, col_ps2, col_ps3, col_ps4 = st.columns(4)
    
    with col_ps1:
        st.metric("💰 Entry", f"Rp{entry_price:,.0f}")
    
    with col_ps2:
        st.metric("✂️ Stop Loss", f"Rp{stop_loss:,.0f}", delta=f"-{stop_loss_atr} ATR")
        if current_price <= stop_loss:
            st.error("🚨 STOP LOSS TRIGGERED!")
    
    with col_ps3:
        st.metric("🎯 Take Profit", f"Rp{take_profit:,.0f}", delta=f"+{take_profit_atr} ATR")
        if current_price >= take_profit:
            st.success("🎉 TAKE PROFIT!")
    
    with col_ps4:
        pnl = ((current_price - entry_price) / entry_price) * 100
        if pnl >= 0:
            st.metric("📈 Profit/Loss", f"+{pnl:.2f}%")
        else:
            st.metric("📉 Profit/Loss", f"{pnl:.2f}%")
    
    # Position sizing
    st.markdown("---")
    col_cap, col_size = st.columns(2)
    with col_cap:
        capital = st.number_input("💰 Modal (Rp)", min_value=1000000, value=100000000, step=1000000)
    with col_size:
        position_size = calculate_position_size(capital, entry_price, stop_loss, risk_percent)
        if position_size > 0:
            st.metric("📊 Position Size", f"{position_size:,} saham")
            risk_amount = (entry_price - stop_loss) * position_size
            st.caption(f"Risk: Rp{risk_amount:,.0f} ({risk_percent}% dari modal)")
        else:
            st.warning("⚠️ Stop loss terlalu dekat")

else:
    st.info("💡 Masukkan Harga Entry di sidebar untuk aktifkan Position Management")

# ========== MULTI TIMEFRAME ==========
st.markdown("---")
st.markdown("### ⏰ Multi Timeframe Analysis")
mtf = multi_timeframe_analysis(symbol)

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

# Final signal
final_score = mtf.get('weighted', 50)
final_label, _, final_emoji = get_signal_label(final_score)

if "BUY" in final_label:
    final_bg = "#90EE90"
elif "SELL" in final_label:
    final_bg = "#FFCCCC"
else:
    final_bg = "#FFE4B5"

st.markdown(f"""
<div style="background-color:{final_bg}; padding:10px; border-radius:10px; margin-top:10px; text-align:center">
    <h3 style="margin:0">🎯 Final Signal: {final_emoji} {final_label}</h3>
    <p style="margin:0">Score: {final_score:.1f}/100</p>
</div>
""", unsafe_allow_html=True)

# ========== BACKTEST ==========
st.markdown("---")
with st.expander("📊 Backtest Strategy (Realistic)"):
    if st.button("🚀 Jalankan Backtest", width="stretch"):
        with st.spinner("Menghitung performa..."):
            result = backtest_strategy(df)
            col_b1, col_b2, col_b3, col_b4 = st.columns(4)
            with col_b1:
                st.metric("📈 Return", f"{result['return']}%")
            with col_b2:
                st.metric("✅ Winrate", f"{result['winrate']}%")
            with col_b3:
                st.metric("🔄 Trades", result['trades'])
            with col_b4:
                st.metric("📉 Max DD", f"{result['max_drawdown']}%")
            st.caption(f"💰 Modal awal: Rp100.000.000 → Akhir: Rp{result['final_capital']:,.0f}")

# ========== SCANNER ==========
st.markdown("---")
st.markdown("### 🔍 Scanner Saham")

if st.button("🚀 SCAN MARKET SEKARANG", width="stretch"):
    with st.spinner("Scanning market..."):
        try:
            results = scan_saham()
            if results:
                df_scan = pd.DataFrame(results)
                st.dataframe(df_scan, use_container_width=True, hide_index=True)
                st.success(f"🏆 Top Pick: {results[0]['Kode']} (Score: {results[0]['Score']})")
            else:
                st.warning("Tidak ada data saham")
        except Exception as e:
            st.error(f"Error: {str(e)}")

st.markdown("---")
st.caption("⚠️ DISCLAIMER: Alat bantu analisis teknikal, bukan rekomendasi investasi. Selalu lakukan riset mandiri sebelum trading.")

# Auto Refresh
if auto_refresh:
    time.sleep(30)
    st.rerun()
