import streamlit as st
import pandas as pd
from datetime import datetime
import time

# Import dari data.py
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
    scan_saham
)

st.set_page_config(
    page_title="Robot Saham Indonesia",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ========== MULTI TIMEFRAME FUNCTION (DI DALAM APP.PY AGAR LEBIH MUDAH) ==========
def get_multi_timeframe_data(symbol):
    """Ambil data multi timeframe dengan rate limit handling"""
    timeframes = ["5m", "15m", "30m", "60m", "1d"]
    tf_labels = ["5 Menit", "15 Menit", "30 Menit", "1 Jam", "1 Hari"]
    results = []
    
    for i, tf in enumerate(timeframes):
        try:
            df = get_data(symbol, tf)
            if not df.empty and len(df) > 10:
                df = add_indicators(df)
                last = df.iloc[-1]
                
                # Hitung score sederhana
                score = 50
                if last['ema20'] > last['ema50']:
                    score += 15
                if last['rsi'] > 50:
                    score += 10
                if last['macd_histogram'] > 0:
                    score += 15
                if last['volume_ratio'] > 1.2:
                    score += 10
                
                results.append({
                    "Timeframe": tf_labels[i],
                    "Kode": tf,
                    "Score": min(100, score),
                    "Harga": f"Rp{last['close']:,.0f}",
                    "RSI": f"{last['rsi']:.0f}",
                    "Trend": "🔼" if last['ema20'] > last['ema50'] else "🔽"
                })
            else:
                results.append({
                    "Timeframe": tf_labels[i],
                    "Kode": tf,
                    "Score": 50,
                    "Harga": "N/A",
                    "RSI": "N/A",
                    "Trend": "⏸️"
                })
            time.sleep(0.3)  # Delay biar tidak kena rate limit
        except Exception as e:
            results.append({
                "Timeframe": tf_labels[i],
                "Kode": tf,
                "Score": 50,
                "Harga": "Error",
                "RSI": "Error",
                "Trend": "❌"
            })
    
    return results

# SIDEBAR
with st.sidebar:
    st.title("📈 Robot Saham")
    st.markdown("---")
    
    symbol = st.text_input("Kode Saham", "BBCA.JK", key="symbol_input").upper()
    timeframe = st.selectbox("Timeframe Utama", ["1d", "1h", "30m", "15m", "5m"], key="timeframe_select")
    
    st.markdown("---")
    st.subheader("⚙️ RISK MANAGEMENT")
    
    capital = st.number_input("💰 Modal (Rp)", min_value=1000000, value=100000000, step=1000000)
    risk_percent = st.slider("🎲 Risk per Trade (%)", min_value=0.5, max_value=3.0, value=2.0, step=0.5)
    
    st.markdown("---")
    st.caption(f"Update: {datetime.now().strftime('%H:%M:%S')}")
    auto_refresh = st.checkbox("Auto Refresh (30 detik)", value=False)

# MAIN CONTENT
st.title(f"📊 {symbol}")

# Load data utama
df = get_data(symbol, timeframe)

if df.empty:
    st.warning(f"⚠️ Data {symbol} kosong. Coba kode: BBCA.JK, BBRI.JK, BMRI.JK, ASII.JK, atau ^JKSE untuk IHSG")
    st.stop()

df = add_indicators(df)
current_price = df.iloc[-1]['close']

# ========== MARKET FILTER (IHSG) ==========
st.markdown("### 📊 MARKET FILTER")
ihsg_trend, ihsg_score, ihsg_msg = get_ihsg_trend()

if ihsg_trend == "BULLISH":
    st.success(f"✅ {ihsg_msg}")
elif ihsg_trend == "BEARISH":
    st.error(f"⚠️ {ihsg_msg} - Hati-hati, market sedang turun!")
else:
    st.warning(f"📊 {ihsg_msg}")

st.markdown("---")

# ========== SIGNAL DARI INDIKATOR ==========
st.markdown("### 🎯 SIGNAL & LEVEL")

entry_price, stop_loss, take_profit, shares, rr_ratio, setup_name, conf = calculate_entry_sl_tp(df, capital, risk_percent)

if entry_price and setup_name != "NO_SETUP":
    if "BUY" in setup_name:
        st.success(f"### 🔥 SIGNAL: {setup_name}")
    else:
        st.info(f"### 📈 SIGNAL: {setup_name}")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("🎯 ENTRY", f"Rp{entry_price:,.0f}")
    with col2:
        st.metric("✂️ STOP LOSS", f"Rp{stop_loss:,.0f}")
    with col3:
        st.metric("🏁 TAKE PROFIT", f"Rp{take_profit:,.0f}")
    
    col4, col5, col6 = st.columns(3)
    with col4:
        st.metric("📊 POSITION SIZE", f"{shares:,} saham")
    with col5:
        st.metric("📈 RISK/REWARD", f"1:{rr_ratio:.1f}")
    with col6:
        st.metric("🎲 CONFIDENCE", f"{conf:.0f}/100")
    
    if conf >= 70 and rr_ratio >= 1.5 and ihsg_trend != "BEARISH":
        st.success("### ✅ REKOMENDASI: EKSEKUSI")
    elif conf >= 60:
        st.info("### ⏸️ REKOMENDASI: TUNGGU KONFIRMASI")
    else:
        st.warning("### ⛔ REKOMENDASI: SKIP")
else:
    st.warning("### ⛔ TIDAK ADA SETUP BERKUALITAS")
    st.caption("Sistem tidak mendeteksi setup entry yang baik. Cek saham lain atau tunggu sinyal berikutnya.")

st.markdown("---")

# ========== CONFIDENCE SCORE ==========
st.markdown("### 📊 CONFIDENCE SCORE")
confidence, factors, grade = calculate_confidence_score(df, ihsg_score)

col_c1, col_c2 = st.columns([1, 2])
with col_c1:
    st.metric("SCORE", f"{confidence:.0f}", delta=grade)
with col_c2:
    for name, score, desc in factors[:3]:
        if score > 0:
            st.caption(f"✅ {name}: +{score:.0f}")
        else:
            st.caption(f"❌ {name}: {score:.0f}")

st.markdown("---")

# ========== HARGA SAAT INI ==========
st.markdown("### 💰 HARGA")
last = df.iloc[-1]

col_h1, col_h2, col_h3 = st.columns(3)
with col_h1:
    st.metric("📈 TERTINGGI", f"Rp{last['high']:,.0f}")
with col_h2:
    st.metric("📉 TERENDAH", f"Rp{last['low']:,.0f}")
with col_h3:
    if len(df) > 1:
        change = last['close'] - df.iloc[-2]['close']
        change_pct = (change / df.iloc[-2]['close']) * 100
        st.metric("💰 SAAT INI", f"Rp{last['close']:,.0f}", delta=f"{change_pct:+.2f}%")
    else:
        st.metric("💰 SAAT INI", f"Rp{last['close']:,.0f}")

# ========== CHART ==========
st.markdown("### 📈 CHART")
st.line_chart(df[['close', 'ema20', 'ema50']], height=300)

# ========== INDIKATOR TEKNIKAL ==========
st.markdown("### 📊 INDIKATOR")
col_i1, col_i2, col_i3 = st.columns(3)

with col_i1:
    st.metric("RSI (14)", f"{last['rsi']:.1f}")
    if last['rsi'] < 30:
        st.caption("🟢 Oversold - Potensi beli")
    elif last['rsi'] > 70:
        st.caption("🔴 Overbought - Waspada")
    else:
        st.caption("⚪ Netral")
    
    st.metric("ADX", f"{last['adx']:.1f}")
    if last['adx'] >= 25:
        st.caption("✅ Tren Kuat")
    elif last['adx'] >= 20:
        st.caption("⚠️ Tren Mulai")
    else:
        st.caption("🔴 Sideways")

with col_i2:
    st.metric("MACD", f"{last['macd']:.2f}")
    st.metric("SIGNAL", f"{last['macd_signal']:.2f}", delta=f"{last['macd_histogram']:.2f}")
    if last['macd_histogram'] > 0:
        st.caption("✅ Bullish")
    else:
        st.caption("❌ Bearish")

with col_i3:
    st.metric("VOLUME", f"{last['volume']:,.0f}")
    st.metric("VOLUME RATIO", f"{last['volume_ratio']:.2f}x")
    if last['volume_ratio'] >= 1.5:
        st.caption("✅ Volume Spike - Konfirmasi")
    elif last['volume_ratio'] < 0.6:
        st.caption("⚠️ Volume Sepi - Hati-hati")
    else:
        st.caption("⚪ Normal")
    
    st.metric("ATR", f"Rp{last['atr']:,.0f}")

st.markdown("---")

# ========== MULTI TIMEFRAME ANALYSIS ==========
st.markdown("### ⏰ MULTI TIMEFRAME ANALYSIS")
st.caption("Analisis di berbagai timeframe untuk konfirmasi sinyal")

with st.spinner("Mengambil data multi timeframe..."):
    mtf_results = get_multi_timeframe_data(symbol)

# Tampilkan dalam bentuk dataframe
df_mtf = pd.DataFrame(mtf_results)
st.dataframe(df_mtf, use_container_width=True, hide_index=True)

# Hitung rata-rata score
avg_score = sum([r["Score"] for r in mtf_results if isinstance(r["Score"], (int, float))]) / len(mtf_results) if mtf_results else 50

st.markdown("---")
st.markdown("### 🎯 FINAL MULTI TIMEFRAME SIGNAL")

if avg_score >= 70:
    st.success(f"### ✅ BULLISH - Rata-rata score {avg_score:.0f}/100")
    st.progress(avg_score/100)
elif avg_score >= 50:
    st.warning(f"### ⏸️ NEUTRAL - Rata-rata score {avg_score:.0f}/100")
    st.progress(avg_score/100)
else:
    st.error(f"### ❌ BEARISH - Rata-rata score {avg_score:.0f}/100")
    st.progress(avg_score/100)

st.markdown("---")

# ========== REKOMENDASI ==========
st.markdown("### 📝 REKOMENDASI AKHIR")
st.info(get_trading_recommendation(df))

# ========== SCANNER ==========
st.markdown("---")
st.markdown("### 🔍 SCANNER SAHAM")

if st.button("🚀 SCAN MARKET SEKARANG", width="stretch"):
    with st.spinner("Scanning market..."):
        results = scan_saham()
        if results:
            df_scan = pd.DataFrame(results)
            st.dataframe(df_scan, use_container_width=True, hide_index=True)
            st.success(f"🏆 Top Pick: {results[0]['Kode']} (Score: {results[0]['Score']})")
        else:
            st.warning("Tidak ada setup berkualitas saat ini")

st.markdown("---")
st.caption("⚠️ DISCLAIMER: Alat bantu analisis teknikal. Entry, SL, TP dihitung dari indikator. Bukan rekomendasi investasi.")

# Auto Refresh
if auto_refresh:
    time.sleep(30)
    st.rerun()
