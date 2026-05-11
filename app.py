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

# SIDEBAR - Hanya untuk Risk Management (bukan entry manual!)
with st.sidebar:
    st.title("🎯 Robot Saham Pro")
    st.caption("Entry, SL, TP dari INDIKATOR - Bukan manual!")
    st.markdown("---")
    
    symbol = st.text_input("Kode Saham", "^JKSE", key="symbol_input").upper()
    timeframe = st.selectbox("Timeframe", ["5m", "15m", "30m", "60m", "1d"], key="timeframe_select")
    
    st.markdown("---")
    st.subheader("⚙️ RISK MANAGEMENT")
    
    capital = st.number_input("💼 Modal (Rp)", min_value=1000000, value=100000000, step=1000000)
    risk_percent = st.slider("🎲 Risk per Trade (%)", min_value=0.5, max_value=3.0, value=2.0, step=0.5)
    
    st.markdown("---")
    st.info("💡 **Sistem akan hitung:**\n- Entry dari indikator\n- Stop Loss dari ATR (2x)\n- Take Profit dari ATR (3x)\n- Position sizing otomatis")
    st.markdown("---")
    auto_refresh = st.checkbox("Auto Refresh (30 detik)", value=False)
    st.caption(f"Update: {datetime.now().strftime('%H:%M:%S')}")

# MAIN CONTENT
st.title(f"🎯 {symbol}")

# Load data sesuai timeframe yang dipilih
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

# ========== HITUNG ENTRY, SL, TP DARI INDIKATOR ==========
st.markdown("### 🎯 SIGNAL & LEVEL DARI INDIKATOR")

entry_price, stop_loss, take_profit, shares, rr_ratio, setup_name, conf = calculate_entry_sl_tp(df, capital, risk_percent)

if entry_price and setup_name != "NO_SETUP":
    # Tampilkan dengan warna berdasarkan jenis sinyal
    if "BUY" in setup_name:
        bg_color = "#90EE90"
        st.success(f"### 🔥 SIGNAL: {setup_name}")
    elif "SELL" in setup_name:
        bg_color = "#FFCCCC"
        st.error(f"### 🔴 SIGNAL: {setup_name}")
    else:
        bg_color = "#FFE4B5"
        st.info(f"### ⏸️ SIGNAL: {setup_name}")
    
    # Tampilkan Entry, SL, TP
    col_e1, col_e2, col_e3 = st.columns(3)
    
    with col_e1:
        st.metric("🎯 ENTRY PRICE", f"Rp{entry_price:,.0f}", delta="Dari Indikator")
        st.caption(f"Berdasarkan: {setup_name}")
    
    with col_e2:
        st.metric("✂️ STOP LOSS", f"Rp{stop_loss:,.0f}", delta=f"{(stop_loss/entry_price-1)*100:+.1f}%")
        st.caption("2 × ATR (Volatilitas)")
    
    with col_e3:
        st.metric("🎯 TAKE PROFIT", f"Rp{take_profit:,.0f}", delta=f"{(take_profit/entry_price-1)*100:+.1f}%")
        st.caption("3 × ATR (Risk Reward 1:1.5)")
    
    st.markdown("---")
    
    # Position Sizing
    col_p1, col_p2, col_p3 = st.columns(3)
    
    with col_p1:
        st.metric("📊 POSITION SIZE", f"{shares:,} saham")
        st.caption(f"Berdasarkan risk {risk_percent}%")
    
    with col_p2:
        st.metric("📈 RISK/REWARD", f"1:{rr_ratio:.1f}")
        if rr_ratio >= 2:
            st.success("✅ EXCELLENT - Eksekusi")
        elif rr_ratio >= 1.5:
            st.info("📊 GOOD - Bisa eksekusi")
        else:
            st.warning("⚠️ POOR - Pertimbangkan skip")
    
    with col_p3:
        st.metric("🎲 CONFIDENCE", f"{conf:.0f}/100")
        if conf >= 70:
            st.success("✅ High confidence")
        elif conf >= 50:
            st.info("📊 Medium confidence")
        else:
            st.warning("⚠️ Low confidence")
    
    # Rekomendasi akhir
    st.markdown("---")
    if conf >= 70 and rr_ratio >= 1.5 and ihsg_trend != "BEARISH":
        st.success("### ✅ REKOMENDASI: EKSEKUSI")
        st.caption("Semua kriteria terpenuhi: Confidence tinggi, RR bagus, IHSG mendukung")
    elif conf >= 60 and rr_ratio >= 1.2:
        st.info("### ⏸️ REKOMENDASI: TUNGGU KONFIRMASI")
        st.caption("Masih kurang 1-2 kriteria, tunggu candle berikutnya")
    else:
        st.warning("### ⛔ REKOMENDASI: SKIP / HOLD")
        st.caption("Kriteria belum terpenuhi, jangan dipaksakan")

else:
    st.warning("### ⛔ TIDAK ADA SETUP BERKUALITAS")
    st.caption("Sistem tidak mendeteksi setup entry yang baik. Cek timeframe lain atau tunggu sinyal berikutnya.")

st.markdown("---")

# ========== CONFIDENCE SCORE ==========
st.markdown("### 📊 CONFIDENCE SCORE")
confidence, factors, grade = calculate_confidence_score(df, ihsg_score)

col_c1, col_c2 = st.columns([1, 2])
with col_c1:
    st.metric("Score", f"{confidence:.0f}", delta=grade)
with col_c2:
    for name, score, desc in factors[:4]:
        if score > 0:
            st.caption(f"✅ {name}: +{score:.0f} ({desc})")
        else:
            st.caption(f"❌ {name}: {score:.0f} ({desc})")

st.markdown("---")

# ========== HARGA SAAT INI ==========
st.markdown("### 💰 Harga Real-time")
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

st.markdown("---")

# ========== CHART ==========
st.markdown("### 📈 Chart")
st.line_chart(df[['close', 'ema20', 'ema50']], height=300)

# ========== INDIKATOR TEKNIKAL ==========
st.markdown("### 📊 Technical Indicators")
col_i1, col_i2, col_i3 = st.columns(3)
with col_i1:
    st.metric("RSI", f"{last['rsi']:.1f}")
    st.metric("ADX", f"{last['adx']:.1f}")
with col_i2:
    st.metric("MACD", f"{last['macd']:.2f}")
    st.metric("Volume Ratio", f"{last['volume_ratio']:.2f}x")
with col_i3:
    st.metric("ATR", f"Rp{last['atr']:,.0f}")
    st.metric("Support/Res", f"Rp{last['support']:,.0f} / Rp{last['resistance']:,.0f}")

st.markdown("---")

# ========== MULTI TIMEFRAME (LIHAT PERBANDINGAN) ==========
st.markdown("### ⏰ MULTI TIMEFRAME ANALYSIS")
st.caption("Lihat perbedaan Entry, SL, TP di setiap timeframe")

mtf_results, final_score = multi_timeframe_analysis(symbol, capital, risk_percent)

# Tampilkan tabel per timeframe
tf_data = []
for tf in ["5m", "15m", "30m", "60m", "1d"]:
    res = mtf_results.get(tf, {})
    tf_data.append({
        "Timeframe": tf,
        "Score": f"{res.get('score', 50):.0f}",
        "Setup": res.get('setup', 'N/A')[:15],
        "Entry": f"Rp{res.get('entry', 0):,.0f}" if res.get('entry') else "-",
        "Stop Loss": f"Rp{res.get('stop_loss', 0):,.0f}" if res.get('stop_loss') else "-",
        "Take Profit": f"Rp{res.get('take_profit', 0):,.0f}" if res.get('take_profit') else "-",
        "RR": f"1:{res.get('rr', 0):.1f}" if res.get('rr', 0) > 0 else "-"
    })

df_tf = pd.DataFrame(tf_data)
st.dataframe(df_tf, use_container_width=True, hide_index=True)

# Final signal
st.markdown("---")
st.markdown("### 🎯 FINAL MULTI TIMEFRAME SIGNAL")

if final_score >= 70:
    st.success(f"### ✅ BULLISH - Konfirmasi dari mayoritas timeframe")
    st.progress(final_score/100, text=f"Score: {final_score:.0f}/100")
elif final_score >= 50:
    st.warning(f"### ⏸️ NEUTRAL - Timeframe mixed, pilih timeframe yang align")
    st.progress(final_score/100, text=f"Score: {final_score:.0f}/100")
else:
    st.error(f"### ❌ BEARISH - Mayoritas timeframe downtrend")
    st.progress(final_score/100, text=f"Score: {final_score:.0f}/100")

st.markdown("---")

# ========== REKOMENDASI ==========
st.markdown("### 📝 REKOMENDASI")
st.info(get_trading_recommendation(df))

# ========== BACKTEST ==========
st.markdown("---")
with st.expander("📊 Backtest Realistis (Fee, Slippage, Drawdown)"):
    if st.button("🚀 Jalankan Backtest", width="stretch"):
        with st.spinner("Menghitung performa..."):
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
st.caption("⚠️ Disclaimer: Entry, Stop Loss, Take Profit dihitung dari INDIKATOR (bukan manual). Gunakan sebagai alat bantu, tetap lakukan riset mandiri.")

if auto_refresh:
    time.sleep(30)
    st.rerun()
