import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import time

from data import (
    get_data,
    add_indicators,
    get_ihsg_trend,
    detect_bottom_pattern,
    detect_valid_breakout,
    detect_reversal,
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
    get_multi_timeframe_alignment,
    get_nearest_fvg,
    get_nearest_order_block
)

st.set_page_config(
    page_title="Smart Money Trading System",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ========== SIDEBAR ==========
with st.sidebar:
    st.title("🧠 Smart Money Trading")
    st.caption("Market Structure | FVG | Order Block | Liquidity Sweep")
    st.markdown("---")
    
    symbol = st.text_input("Kode Saham", "BBCA.JK").upper()
    timeframe = st.selectbox("Timeframe", ["1d", "60m", "30m", "15m", "5m"])
    
    st.markdown("---")
    st.subheader("💰 Risk Management")
    capital = st.number_input("Modal (Rp)", min_value=1000000, value=100000000, step=1000000)
    risk_percent = st.slider("Risk per Trade (%)", min_value=0.5, max_value=3.0, value=2.0, step=0.5)
    
    st.markdown("---")
    st.subheader("📋 Status Posisi (Opsional)")
    has_position = st.checkbox("Saya sudah punya posisi di saham ini")
    entry_price_manual = 0
    shares_manual = 0
    if has_position:
        entry_price_manual = st.number_input("Harga Entry (Rp)", min_value=0, value=0, step=100)
        shares_manual = st.number_input("Jumlah Saham", min_value=0, value=0, step=100)
    
    st.markdown("---")
    st.info("""
    **Fitur Smart Money:**
    - Swing Structure (BOS/CHOCH)
    - Fair Value Gap (FVG)
    - Order Block
    - Liquidity Sweep
    - Multi Timeframe Alignment
    - Rekomendasi Aksi & Filter Prioritas
    """)
    st.markdown("---")
    auto_refresh = st.checkbox("Auto Refresh (30 detik)", value=False)
    st.caption(f"Update: {datetime.now().strftime('%H:%M:%S')}")

# ========== MAIN CONTENT ==========
st.title(f"🧠 {symbol} - Smart Money Analysis")

df = get_data(symbol, timeframe)

if df.empty:
    st.warning(f"Data {symbol} kosong. Coba: BBCA.JK, BBRI.JK, BMRI.JK, ASII.JK")
    st.stop()

df = add_indicators(df)
last = df.iloc[-1]
current_price = last['close']

# === MARKET FILTER IHSG ===
ihsg_trend, ihsg_score, ihsg_msg = get_ihsg_trend()

# === AMBIL DATA DARI INDIKATOR ===
entry, sl, tp, shares, rr, setup, conf, signals = calculate_entry_sl_tp(df, capital, risk_percent)
support, resistance, pivot, r1, r2, s1, s2, fib_382, fib_618 = get_pivot_sr(df)
atr = last.get('atr', last['close'] * 0.02) if last['close'] > 0 else 0

# === PRIORITY FILTER (OTOMATIS) ===
skip_reason = None
action = "SKIP"
action_color = "red"
recommendation_text = ""

# 1. IHSG BEARISH
if ihsg_trend == "BEARISH":
    skip_reason = f"IHSG BEARISH ({ihsg_msg.split('(')[-1].replace(')','')}) → Market tidak mendukung"
# 2. ADX < 20 (sideways)
elif last['adx'] < 20:
    skip_reason = f"ADX {last['adx']:.1f} (<20) → Pasar sideways, sinyal palsu tinggi"
# 3. Jika ada sinyal entry
elif entry and conf >= 70 and rr >= 1.5:
    action = "EKSEKUSI BUY"
    action_color = "green"
    recommendation_text = f"✅ Setup: {setup} | Confidence {conf:.0f} | RR 1:{rr:.1f}"
elif entry and conf >= 60:
    action = "TUNGGU KONFIRMASI"
    action_color = "orange"
    recommendation_text = f"⏸️ Setup: {setup} | Confidence {conf:.0f} | Perlu konfirmasi tambahan"
elif entry:
    action = "SKIP (KUALITAS RENDAH)"
    action_color = "red"
    recommendation_text = f"❌ Setup {setup} tidak memenuhi kriteria"
else:
    action = "SKIP (TIDAK ADA SETUP)"
    action_color = "red"
    recommendation_text = "Tidak ada setup berkualitas"

# Jika skip_reason ada, override action menjadi SKIP
if skip_reason:
    action = "SKIP"
    action_color = "red"
    recommendation_text = skip_reason

# === TAMPILAN MARKET FILTER ===
st.markdown("### 📊 Market Filter")
if ihsg_trend == "BULLISH":
    st.success(f"✅ {ihsg_msg}")
elif ihsg_trend == "BEARISH":
    st.error(f"⚠️ {ihsg_msg}")
else:
    st.warning(f"📊 {ihsg_msg}")

st.markdown("---")

# === SMART MONEY DETECTION (5 KOLOM) ===
st.markdown("### 🔍 Smart Money Detection")
col1, col2, col3, col4, col5 = st.columns(5)

structure, struct_conf, struct_desc = detect_market_structure(df)
with col1:
    if "BULLISH" in structure:
        st.success(f"**Structure**\n{structure}")
    elif "BEARISH" in structure:
        st.error(f"**Structure**\n{structure}")
    else:
        st.info(f"**Structure**\n{structure}")
    st.caption(f"Conf: {struct_conf:.0f}%")

sm, sm_conf, sm_desc = detect_smart_money_volume(df)
with col2:
    if sm == "ACCUMULATION":
        st.success(f"**Smart Money**\n{sm}")
    elif sm == "DISTRIBUTION":
        st.error(f"**Smart Money**\n{sm}")
    else:
        st.info(f"**Smart Money**\n{sm}")
    st.caption(f"Conf: {sm_conf:.0f}%")

is_sweep, sweep_conf, sweep_type, sweep_desc = detect_liquidity_sweep(df)
with col3:
    if sweep_type == "BULLISH_SFP":
        st.success(f"**Liquidity**\n{sweep_type}")
    elif sweep_type == "BEARISH_SFP":
        st.error(f"**Liquidity**\n{sweep_type}")
    else:
        st.info(f"**Liquidity**\nTidak ada")
    st.caption(f"Conf: {sweep_conf:.0f}%")

nearest_bullish_fvg, nearest_bearish_fvg = get_nearest_fvg(df)
with col4:
    if nearest_bullish_fvg:
        st.success(f"**FVG**\nBullish Gap")
        st.caption(f"At: Rp{nearest_bullish_fvg['upper']:,.0f}")
    elif nearest_bearish_fvg:
        st.error(f"**FVG**\nBearish Gap")
        st.caption(f"At: Rp{nearest_bearish_fvg['lower']:,.0f}")
    else:
        st.info(f"**FVG**\nNo gap")
        st.caption("-")

nearest_bullish_ob, nearest_bearish_ob = get_nearest_order_block(df)
with col5:
    if nearest_bullish_ob:
        st.success(f"**Order Block**\nBullish OB")
        st.caption(f"At: Rp{nearest_bullish_ob['high']:,.0f}")
    elif nearest_bearish_ob:
        st.error(f"**Order Block**\nBearish OB")
        st.caption(f"At: Rp{nearest_bearish_ob['low']:,.0f}")
    else:
        st.info(f"**Order Block**\nNo OB")
        st.caption("-")

st.markdown("---")

# === MARKET REGIME ===
st.markdown("### 📈 Market Regime")
regime, regime_conf, regime_desc = detect_market_regime(df)
col_r1, col_r2, col_r3 = st.columns(3)
with col_r1:
    st.metric("Regime", regime)
with col_r2:
    st.metric("ADX", f"{last['adx']:.1f}")
with col_r3:
    atr_pct = (last['atr'] / last['close'] * 100) if last['close'] > 0 else 0
    st.metric("ATR %", f"{atr_pct:.2f}%")
st.caption(regime_desc)

st.markdown("---")

# === PIVOT SUPPORT RESISTANCE ===
st.markdown("### 📊 Pivot Support & Resistance")
col_sr1, col_sr2, col_sr3, col_sr4 = st.columns(4)
with col_sr1:
    st.metric("Support", f"Rp{support:,.0f}")
with col_sr2:
    st.metric("Resistance", f"Rp{resistance:,.0f}")
with col_sr3:
    st.metric("Pivot", f"Rp{pivot:,.0f}")
with col_sr4:
    st.metric("Fib 61.8%", f"Rp{fib_618:,.0f}")

st.markdown("---")

# === ENTRY, SL, TP DARI INDIKATOR ===
st.markdown("### 🎯 Entry - Stop Loss - Take Profit (Dari Indikator)")
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
    
    if signals:
        st.caption(" | ".join(signals[:3]))
else:
    st.warning("⛔ Tidak ada setup trading berkualitas saat ini")

st.markdown("---")

# === TARGET HARGA ALTERNATIF (3 LEVEL) ===
if entry and atr > 0:
    st.markdown("### 🎯 Target Harga Alternatif")
    target1 = entry + (1.5 * atr)   # RR 1:1.5
    target2 = entry + (2 * atr)     # RR 1:2
    target3 = entry + (3 * atr)     # RR 1:3 (sudah ada di TP utama)
    col_t1, col_t2, col_t3 = st.columns(3)
    with col_t1:
        st.metric("Target 1 (Konservatif)", f"Rp{target1:,.0f}", delta="RR 1:1.5")
    with col_t2:
        st.metric("Target 2 (Moderat)", f"Rp{target2:,.0f}", delta="RR 1:2")
    with col_t3:
        st.metric("Target 3 (Agresif)", f"Rp{target3:,.0f}", delta="RR 1:3")

# === STATUS POSISI (jika sudah punya) ===
if has_position and entry_price_manual > 0 and shares_manual > 0:
    st.markdown("---")
    st.markdown("### 📋 Status Posisi Anda")
    pnl_percent = ((current_price - entry_price_manual) / entry_price_manual) * 100
    pnl_nominal = (current_price - entry_price_manual) * shares_manual
    col_p1, col_p2, col_p3 = st.columns(3)
    with col_p1:
        st.metric("Entry Price", f"Rp{entry_price_manual:,.0f}")
    with col_p2:
        st.metric("Current Price", f"Rp{current_price:,.0f}")
    with col_p3:
        if pnl_percent >= 0:
            st.metric("Profit/Loss", f"+{pnl_percent:.2f}%", delta=f"+Rp{pnl_nominal:,.0f}")
        else:
            st.metric("Profit/Loss", f"{pnl_percent:.2f}%", delta=f"-Rp{abs(pnl_nominal):,.0f}")
    
    # Rekomendasi posisi
    if pnl_percent > 5:
        st.success("✅ Posisi sudah aman, pertimbangkan trailing stop")
    elif pnl_percent > 0:
        st.info("📈 Posisi positif, tahan dengan stop loss di entry")
    elif pnl_percent > -5:
        st.warning("⚠️ Posisi sedikit rugi, pantau support terdekat")
    else:
        st.error("🔴 Posisi rugi besar, pertimbangkan cut loss")

# === PERINGATAN RISIKO ===
st.markdown("---")
st.markdown("### ⚠️ Peringatan Risiko")
if entry and sl:
    risk_amount = (entry - sl) * shares
    st.caption(f"• Stop Loss: Rp{sl:,.0f} ({(entry-sl)/entry*100:.2f}% dari entry)")
    st.caption(f"• Risk per trade: {risk_percent}% dari modal → Rp{capital * risk_percent / 100:,.0f}")
    if risk_amount > 0:
        st.caption(f"• Maksimal kerugian jika kena SL: Rp{risk_amount:,.0f}")
    if ihsg_trend == "BEARISH":
        st.error("⚠️ IHSG BEARISH → Risiko lebih tinggi! Sebaiknya hindari entry baru.")
    elif last['adx'] < 20:
        st.warning("⚠️ ADX rendah (<20) → Pasar sideways, stop loss rawan tersapu.")
else:
    st.info("Tidak ada setup aktif, risiko rendah.")

# === CONFIDENCE SCORE ===
st.markdown("---")
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

# === REKOMENDASI AKHIR (ACTION & KESIMPULAN) ===
st.markdown("### 🎯 REKOMENDASI AKHIR")

# Warna latar berdasarkan action
if action == "EKSEKUSI BUY":
    st.success(f"## ✅ {action}")
elif action == "TUNGGU KONFIRMASI":
    st.info(f"## ⏸️ {action}")
else:
    st.error(f"## ⛔ {action}")

if recommendation_text:
    st.markdown(f"**{recommendation_text}**")

# Kesimpulan 1 kalimat
if skip_reason:
    st.warning(f"📌 **Kesimpulan:** {skip_reason}. Sebaiknya hindari trading {symbol} hari ini.")
elif action == "EKSEKUSI BUY":
    st.success(f"📌 **Kesimpulan:** Semua filter terpenuhi. Anda bisa mempertimbangkan untuk membeli {symbol} di sekitar Rp{entry:,.0f} dengan stop loss Rp{sl:,.0f} dan target Rp{tp:,.0f}.")
elif action == "TUNGGU KONFIRMASI":
    st.info(f"📌 **Kesimpulan:** Kondisi masih kurang ideal. Tunggu konfirmasi tambahan sebelum entry.")
else:
    st.warning(f"📌 **Kesimpulan:** Tidak ada setup berkualitas. Tetap hold cash.")

st.markdown("---")

# === MULTI TIMEFRAME ALIGNMENT ===
st.markdown("### ⏰ Multi Timeframe Alignment")
with st.spinner("Menganalisis multi timeframe..."):
    mtf_results, alignment, alignment_score, mtf_signals = get_multi_timeframe_alignment(symbol, capital, risk_percent)

col_mtf1, col_mtf2, col_mtf3 = st.columns(3)
for i, (tf_name, tf_data) in enumerate(mtf_results.items()):
    with [col_mtf1, col_mtf2, col_mtf3][i]:
        st.subheader(tf_name.upper())
        st.metric("Direction", tf_data['direction'])
        if tf_data['entry']:
            st.caption(f"Entry: Rp{tf_data['entry']:,.0f}")
            st.caption(f"SL: Rp{tf_data['stop_loss']:,.0f}")

if alignment_score >= 80:
    st.success(f"### ✅ {alignment} - Semua timeframe searah! (Score: {alignment_score})")
elif alignment_score >= 60:
    st.info(f"### 📊 {alignment} - Sebagian searah (Score: {alignment_score})")
else:
    st.warning(f"### ⚠️ {alignment} - Timeframe kontradiksi (Score: {alignment_score})")

st.markdown("---")

# === SCANNER ===
st.markdown("### 🔍 Scanner Saham")
if st.button("🚀 SCAN MARKET", width="stretch"):
    with st.spinner("Scanning market..."):
        results = scan_saham()
        if results:
            df_scan = pd.DataFrame(results)
            st.dataframe(df_scan, use_container_width=True, hide_index=True)
        else:
            st.warning("Tidak ada setup berkualitas")

st.markdown("---")
st.caption("⚠️ DISCLAIMER: Sistem berbasis Smart Money (BOS, FVG, Order Block). Rekomendasi hanya alat bantu, bukan keputusan investasi.")

if auto_refresh:
    time.sleep(30)
    st.rerun()
