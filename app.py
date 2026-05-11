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

# ========== SIDEBAR DENGAN FITUR ENTRY, SL, TP ==========
with st.sidebar:
    st.title("📈 Robot Saham")
    st.markdown("---")
    
    symbol = st.text_input("Kode Saham", "BBCA.JK", key="symbol_input").upper()
    timeframe = st.selectbox("Timeframe", ["5m", "15m", "30m", "60m", "1d"], key="timeframe_select")
    
    st.markdown("---")
    st.subheader("⚙️ SMART ENTRY & RISK MANAGEMENT")
    
    # ========== FITUR ENTRY, STOP LOSS, TAKE PROFIT ==========
    entry_price = st.number_input("💰 HARGA ENTRY (Rp)", min_value=0, value=0, step=100, 
                                   help="Masukkan harga beli Anda")
    
    col_sl, col_tp = st.columns(2)
    with col_sl:
        stop_loss_percent = st.slider("✂️ STOP LOSS (%)", min_value=1, max_value=20, value=5,
                                       help="Harga jual jika rugi X%")
    with col_tp:
        take_profit_percent = st.slider("🎯 TAKE PROFIT (%)", min_value=1, max_value=50, value=15,
                                         help="Harga jual jika untung X%")
    
    st.markdown("---")
    auto_refresh = st.checkbox("Auto Refresh (30 detik)", value=False)
    st.markdown("---")
    st.caption(f"Update: {datetime.now().strftime('%H:%M:%S')}")

# MAIN CONTENT
st.title(f"📊 {symbol}")

# Ambil data
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

# ========== HARGA TERTINGGI, TERENDAH, SAAT INI ==========
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

# ========== LAYOUT 2 KOLOM ==========
col_left, col_right = st.columns([2, 1.2])

with col_left:
    # CHART
    st.subheader("📈 Harga & EMA")
    st.line_chart(df[['close', 'ema20', 'ema50']], height=300)
    
    # DETAIL INDIKATOR
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
        if last['volume'] < last['volume_ma20'] * 0.8:
            st.warning("⚠️ Volume rendah")

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
    st.subheader("📊 Support & Resistance")
    col_sup, col_res = st.columns(2)
    with col_sup:
        st.metric("🛡️ Support", f"Rp{last['support']:,.0f}")
    with col_res:
        st.metric("🚧 Resistance", f"Rp{last['resistance']:,.0f}")

# ========== POSITION MANAGEMENT (ENTRY, SL, TP) ==========
st.markdown("---")
st.subheader("📋 POSITION MANAGEMENT")

if entry_price > 0:
    # Hitung harga SL dan TP
    stop_loss_price = entry_price * (1 - stop_loss_percent/100)
    take_profit_price = entry_price * (1 + take_profit_percent/100)
    
    # Hitung profit/loss
    pnl_percent = ((current_price - entry_price) / entry_price) * 100
    pnl_nominal = current_price - entry_price
    
    # Tampilkan status posisi
    col_entry_info, col_sl_info, col_tp_info, col_pnl_info = st.columns(4)
    
    with col_entry_info:
        st.metric("💰 ENTRY PRICE", f"Rp{entry_price:,.0f}")
        st.caption(f"Harga saat ini: Rp{current_price:,.0f}")
    
    with col_sl_info:
        st.metric("✂️ STOP LOSS", f"Rp{stop_loss_price:,.0f}", delta=f"-{stop_loss_percent}%")
        if current_price <= stop_loss_price:
            st.error("🚨 STOP LOSS TRIGGERED! Segera tutup posisi!")
    
    with col_tp_info:
        st.metric("🎯 TAKE PROFIT", f"Rp{take_profit_price:,.0f}", delta=f"+{take_profit_percent}%")
        if current_price >= take_profit_price:
            st.success("🎉 TAKE PROFIT TRIGGERED! Ambil keuntungan!")
    
    with col_pnl_info:
        if pnl_percent >= 0:
            st.metric("📈 PROFIT/LOSS", f"+{pnl_percent:.2f}%", delta=f"+Rp{pnl_nominal:,.0f}", delta_color="normal")
        else:
            st.metric("📉 PROFIT/LOSS", f"{pnl_percent:.2f}%", delta=f"-Rp{abs(pnl_nominal):,.0f}", delta_color="inverse")
    
    # Rekomendasi Aksi
    st.markdown("---")
    st.subheader("📌 REKOMENDASI AKSI")
    
    if current_price <= stop_loss_price:
        st.error(f"🔴 **CUT LOSS!** Harga turun {stop_loss_percent}% dari entry ({Rp{stop_loss_price:,.0f}}). Segera jual!")
    elif current_price >= take_profit_price:
        st.success(f"🟢 **TAKE PROFIT!** Harga naik {take_profit_percent}% dari entry ({Rp{take_profit_price:,.0f}}). Ambil untung!")
    elif score >= 70:
        st.success(f"🔥 **HOLD & TAMBAH POSISI** - Sinyal STRONG BUY, tren sangat kuat. Cut loss di {Rp{stop_loss_price:,.0f}}")
    elif score >= 60:
        st.success(f"📈 **HOLD** - Sinyal BUY, masih aman. Cut loss di {Rp{stop_loss_price:,.0f}}")
    elif score <= 40:
        st.warning(f"⚠️ **PERTIMBANGKAN CUT** - Sinyal bearish, harga bisa turun lebih jauh")
    elif pnl_percent > 0:
        st.info(f"✅ **HOLD DULU** - Masih profit {pnl_percent:.2f}%, pantau support di {Rp{last['support']:,.0f}")
    else:
        st.info(f"⏸️ **TUNGGU** - Belum ada sinyal jelas. Stop loss di {Rp{stop_loss_price:,.0f}}")

else:
    st.info("💡 **Masukkan Harga Entry di sidebar kiri** untuk mengaktifkan Position Management (Stop Loss & Take Profit)")

# ========== MULTI TIMEFRAME ==========
st.markdown("---")
st.subheader("⏰ Multi Timeframe")

mtf = multi_timeframe_analysis(symbol)
avg_score = mtf.get('weighted', 0)
final_label, final_color, final_emoji = get_signal_label(avg_score)

col5m, col15m, col30m, col1h, col1d = st.columns(5)

with col5m:
    st.metric("5m", f"{mtf.get('5m', 0):.0f}")
with col15m:
    st.metric("15m", f"{mtf.get('15m', 0):.0f}")
with col30m:
    st.metric("30m", f"{mtf.get('30m', 0):.0f}")
with col1h:
    st.metric("1h", f"{mtf.get('1h', 0):.0f}")
with col1d:
    st.metric("1d", f"{mtf.get('1d', 0):.0f}")

if "BUY" in final_label:
    final_bg = "#90EE90"
elif "SELL" in final_label:
    final_bg = "#FFCCCC"
else:
    final_bg = "#FFE4B5"

st.markdown(f"""
<div style="background-color:{final_bg}; padding:10px; border-radius:10px; margin-top:10px; text-align:center">
    <h3 style="margin:0; color:black">🎯 Final Signal: {final_emoji} {final_label}</h3>
    <p style="margin:0; color:black">Score: {avg_score:.1f}/100</p>
</div>
""", unsafe_allow_html=True)

# ========== BACKTEST ==========
st.markdown("---")
with st.expander("📊 Backtest Strategy (Threshold BUY>=60, SELL<=40)"):
    if st.button("Jalankan Backtest", key="backtest_btn"):
        with st.spinner("Menghitung performa..."):
            result = backtest_strategy(df)
            col_b1, col_b2, col_b3 = st.columns(3)
            with col_b1:
                st.metric("Return (%)", f"{result.get('return', 0)}%")
            with col_b2:
                st.metric("Winrate (%)", f"{result.get('winrate', 0)}%")
            with col_b3:
                st.metric("Jumlah Trades", result.get('trades', 0))
            st.caption(f"Modal awal: Rp100.000.000 → Akhir: Rp{result.get('final_capital', 0):,.0f}")
    else:
        st.info("Klik tombol di atas untuk melihat performa strategi")

# ========== SCANNER ==========
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
st.caption("⚠️ Disclaimer: Alat bantu analisis, bukan rekomendasi investasi. Risk Management ada di sidebar kiri.")

# Auto Refresh
if auto_refresh:
    time.sleep(30)
    st.rerun()
