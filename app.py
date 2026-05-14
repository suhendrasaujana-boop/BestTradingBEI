import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os
from datetime import datetime

# ===================== WAJIB PALING ATAS =====================
st.set_page_config(
    page_title="Smart Money Trading Engine",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===================== PASSWORD & SECURITY =====================
def check_password():
    """Autentikasi sederhana, gunakan Streamlit secrets untuk production."""
    correct_password = os.getenv("APP_PASSWORD", st.secrets.get("APP_PASSWORD", None))
    if not correct_password:
        correct_password = "dev123"
        st.sidebar.warning("⚠️  Gunakan environment variable APP_PASSWORD atau Streamlit secrets!")

    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False

    if not st.session_state.authenticated:
        with st.form("login_form"):
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Login")
            if submitted:
                if password == correct_password:
                    st.session_state.authenticated = True
                    st.rerun()
                else:
                    st.error("Password salah")
            st.stop()

# Panggil autentikasi
check_password()

# ===================== IMPORTS DATA =====================
from data import (
    get_data,
    add_indicators,
    get_ihsg_trend,
    detect_market_structure,
    detect_market_regime,
    detect_liquidity_sweep,
    detect_candlestick_pattern,
    detect_smart_money_volume,
    get_pivot_sr,
    get_nearest_fvg,
    get_nearest_order_block,
    detect_bos_choch,
    calculate_confidence_score,
    calculate_entry_sl_tp,
    get_trading_recommendation,
    detect_high_quality_setup,
    get_multi_timeframe_alignment,
    scan_saham,
    backtest_strategy
)

# Tidak import di sini, akan dipanggil saat dibutuhkan saja

# ===================== SIDEBAR =====================
with st.sidebar:
    st.title("⚙️ Konfigurasi")
    symbol_input = st.text_input("Kode Saham", value="BBCA").upper()
    timeframe = st.selectbox("Timeframe", ["1d", "60m", "30m", "15m", "5m"], index=0)
    modal = st.number_input("Modal (Rp)", value=100_000_000, step=10_000_000, format="%d")
    risk_pct = st.slider("Risiko per trade (%)", 0.5, 5.0, 2.0, 0.5)
    st.divider()
    st.caption("Smart Money Trading Engine v3.0")
    st.caption("Data dari Yahoo Finance (delay 15m)")

# ===================== MAIN TABS =====================
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["📊 Overview", "🧠 Smart Money", "⚖️ Risk & Entry", "🔍 Scanner", "📈 Backtest"]
)

# ===================== FUNGSI CHART =====================
@st.cache_data(ttl=120, show_spinner=False)
def load_data(symbol, tf):
    try:
        df = get_data(symbol, tf)
        if df.empty:
            return None, "Data tidak tersedia"
        df = add_indicators(df)
        return df, None
    except Exception as e:
        return None, str(e)

def plot_smart_money_chart(df):
    """Candlestick interaktif dengan FVG, OB, dan swing points."""
    if df.empty or len(df) < 30:
        return None
    df_plot = df.iloc[-100:].copy()
    fig = go.Figure()
    
    fig.add_trace(go.Candlestick(
        x=df_plot.index if 'datetime' not in df_plot.columns else df_plot['datetime'],
        open=df_plot['open'],
        high=df_plot['high'],
        low=df_plot['low'],
        close=df_plot['close'],
        name="OHLC",
        increasing_line_color='#26a69a', decreasing_line_color='#ef5350'
    ))
    
    nearest_bull, nearest_bear = get_nearest_fvg(df)
    if nearest_bull:
        fig.add_hrect(y0=nearest_bull['lower'], y1=nearest_bull['upper'],
                      fillcolor="rgba(0,255,0,0.2)", line_width=0,
                      annotation_text="Bullish FVG", annotation_position="top left")
    if nearest_bear:
        fig.add_hrect(y0=nearest_bear['lower'], y1=nearest_bear['upper'],
                      fillcolor="rgba(255,0,0,0.2)", line_width=0,
                      annotation_text="Bearish FVG", annotation_position="top left")
    
    nearest_ob_bull, nearest_ob_bear = get_nearest_order_block(df)
    if nearest_ob_bull:
        fig.add_hrect(y0=nearest_ob_bull['low'], y1=nearest_ob_bull['high'],
                      fillcolor="rgba(0,200,200,0.2)", line_width=1, line_dash="dash",
                      annotation_text="Bullish OB", annotation_position="bottom left")
    if nearest_ob_bear:
        fig.add_hrect(y0=nearest_ob_bear['low'], y1=nearest_ob_bear['high'],
                      fillcolor="rgba(200,0,200,0.2)", line_width=1, line_dash="dash",
                      annotation_text="Bearish OB", annotation_position="bottom left")
    
    from data import detect_swing_points
    swing_highs, swing_lows = detect_swing_points(df, lookback=5, confirmation=1)
    if swing_highs:
        h_idx = [x[0] for x in swing_highs if x[0] in df_plot.index]
        h_val = [x[1] for x in swing_highs if x[0] in df_plot.index]
        fig.add_trace(go.Scatter(x=h_idx, y=h_val, mode='markers',
                                 marker=dict(symbol='triangle-down', size=8, color='red'),
                                 name='Swing High'))
    if swing_lows:
        l_idx = [x[0] for x in swing_lows if x[0] in df_plot.index]
        l_val = [x[1] for x in swing_lows if x[0] in df_plot.index]
        fig.add_trace(go.Scatter(x=l_idx, y=l_val, mode='markers',
                                 marker=dict(symbol='triangle-up', size=8, color='green'),
                                 name='Swing Low'))
    
    fig.update_layout(
        height=500,
        margin=dict(l=0, r=0, t=20, b=0),
        template="plotly_dark",
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig

# ===================== TAB 1: OVERVIEW =====================
with tab1:
    st.header("📊 Market Overview")
    col1, col2 = st.columns([2, 1])
    
    with col1:
        ihsg_trend, ihsg_score, ihsg_msg = get_ihsg_trend()
        st.metric("IHSG", ihsg_trend, delta=ihsg_msg)
    with col2:
        st.write("**Saham Dipilih:**", symbol_input)
        st.write("**Timeframe:**", timeframe)
    
    # ========== MARKET CONTEXT PANEL (BARU) ==========
    st.subheader("🌐 Market Context")
    try:
           from context import get_full_market_context
        context = get_full_market_context(symbol_input)
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        with col_m1:
            st.metric("RS Rating", f"{context['rs_score']:.0f}", delta=context['rs_desc'])
        with col_m2:
            st.metric("Breadth EMA20", f"{context['breadth_20']}%")
        with col_m3:
            st.metric("Sektor", context['sector'])
        with col_m4:
            st.metric("Foreign Flow", context['flow'], delta=context['flow_desc'])
        
        st.progress(context['breadth_20'] / 100, text=f"Breadth EMA20: {context['breadth_desc']}")
        
        with st.expander("📊 Performa Sektor (1 Bulan)"):
            sector_df = pd.DataFrame(
                list(context['sector_performance'].items()),
                columns=["Sektor", "Return (%)"]
            )
            st.dataframe(sector_df, width='stretch')
    except Exception as e:
        st.warning(f"Market context gagal dimuat: {e}")
    # ========== END MARKET CONTEXT ==========

    df, error = load_data(symbol_input, timeframe)
    if error:
        st.error(f"❌ Gagal memuat data: {error}")
        st.stop()
    if df is None or len(df) < 30:
        st.warning("Data tidak cukup untuk analisis (minimal 30 candle).")
        st.stop()
    
    last = df.iloc[-1]
    colA, colB, colC, colD = st.columns(4)
    colA.metric("Close", f"Rp{last['close']:,.0f}")
    colB.metric("RSI", f"{last['rsi']:.1f}")
    colC.metric("ADX", f"{last['adx']:.1f}" if pd.notna(last['adx']) else "N/A")
    colD.metric("Volume Ratio", f"{last['volume_ratio']:.2f}x")
    
    regime, reg_conf, reg_desc = detect_market_regime(df)
    st.info(f"**Market Regime:** {regime} (confidence: {reg_conf}) – {reg_desc}")
    # ========== VOLUME PROFILE INFO ==========
    try:
        from volume_profile import volume_profile_analysis
        vp_analysis = volume_profile_analysis(df, symbol_input)
        if vp_analysis:
            st.subheader("📊 Volume Profile")
            col_v1, col_v2, col_v3 = st.columns(3)
            with col_v1:
                st.metric("Point of Control", f"Rp{vp_analysis['poc']:,.0f}")
            with col_v2:
                st.metric("Posisi vs POC", f"{vp_analysis['poc_distance_pct']:+.2f}%")
            with col_v3:
                st.caption(vp_analysis['poc_position'])
    except Exception as e:
        pass
    # ========== END VOLUME PROFILE INFO ==========
# ===================== TAB 2: SMART MONEY =====================
with tab2:
    st.header("🧠 Smart Money Analysis")
    if df is None:
        st.warning("Data belum dimuat. Silakan kembali ke tab Overview.")
    else:
        st.subheader("Price Action & Smart Money Zones")
        fig = plot_smart_money_chart(df)
        if fig:
            st.plotly_chart(fig, width='stretch')
        else:
            st.info("Chart tidak dapat ditampilkan karena data kurang.")
        
        col1, col2 = st.columns(2)
        with col1:
            structure, struct_conf, struct_desc = detect_market_structure(df)
            bos_cho, bos_conf, bos_desc = detect_bos_choch(df)
            st.metric("Market Structure", structure, delta=f"Conf: {struct_conf}")
            st.caption(f"BOS/CHOCH: {bos_cho} ({bos_desc})")
            
            nearest_bull_fvg, nearest_bear_fvg = get_nearest_fvg(df)
            if nearest_bull_fvg:
                st.success(f"✅ Bullish FVG terdekat di {nearest_bull_fvg['upper']:.2f} - {nearest_bull_fvg['lower']:.2f}")
            elif nearest_bear_fvg:
                st.error(f"❌ Bearish FVG terdekat di {nearest_bear_fvg['upper']:.2f} - {nearest_bear_fvg['lower']:.2f}")
            else:
                st.info("Tidak ada FVG valid")
        
        with col2:
            regime, reg_conf, reg_desc = detect_market_regime(df)
            st.metric("Market Regime", regime, delta=f"Conf: {reg_conf}")
            
            nearest_bull_ob, nearest_bear_ob = get_nearest_order_block(df)
            if nearest_bull_ob:
                st.success(f"✅ Bullish OB di {nearest_bull_ob['high']:.2f} - {nearest_bull_ob['low']:.2f}")
            elif nearest_bear_ob:
                st.error(f"❌ Bearish OB di {nearest_bear_ob['high']:.2f} - {nearest_bear_ob['low']:.2f}")
            else:
                st.info("Tidak ada Order Block")
        
        is_sweep, sweep_conf, sweep_type, sweep_desc = detect_liquidity_sweep(df)
        if is_sweep:
            st.warning(f"⚡ Liquidity Sweep terdeteksi: {sweep_type} ({sweep_desc})")

# ===================== TAB 3: RISK & ENTRY =====================
with tab3:
    st.header("⚖️ Risk Management & Setup")
    if df is None:
        st.warning("Data belum dimuat.")
    else:
        entry, sl, tp, shares, rr, setup_name, conf, signals = calculate_entry_sl_tp(
            df, capital=modal, risk_percent=risk_pct
        )
        if entry:
            st.success(f"**Setup:** {setup_name} (Confidence: {conf})")
            st.metric("Entry Price", f"Rp{entry:,.0f}")
            col1, col2, col3 = st.columns(3)
            col1.metric("Stop Loss", f"Rp{sl:,.0f}", delta=f"{- (entry-sl)/entry*100:.2f}%")
            col2.metric("Take Profit", f"Rp{tp:,.0f}", delta=f"+{(tp-entry)/entry*100:.2f}%")
            col3.metric("Risk:Reward", f"1:{rr:.2f}")
            st.write(f"**Jumlah Lot:** {shares} lembar (Rp{shares*entry:,.0f})")
            st.caption(" | ".join(signals))
        else:
            st.warning("Tidak ada setup valid saat ini.")
        
        support, resistance, pivot, r1, r2, s1, s2, fib382, fib618 = get_pivot_sr(df)
        st.subheader("Support & Resistance")
        cols = st.columns(4)
        cols[0].metric("Support", f"Rp{support:,.0f}")
        cols[1].metric("Resistance", f"Rp{resistance:,.0f}")
        cols[2].metric("Pivot", f"Rp{pivot:,.0f}")
        cols[3].metric("Fib 38.2%", f"Rp{fib382:,.0f}")
        
        conf_score, factors, grade = calculate_confidence_score(df, ihsg_score)
        st.subheader(f"Confidence Score: {conf_score:.0f} ({grade})")
        st.progress(conf_score/100)
        st.write("**Breakdown:**")
        for name, score, note in factors:
            st.write(f"- {name}: {score:+.1f} ({note})")
        # ========== PROBABILITY CONTEXT (FASE 3) ==========
        st.subheader("🎯 Probability Calibration")
        try:
            from probability_engine import get_probability_context
            prob_ctx = get_probability_context(symbol_input, conf_score)
            
            col_p1, col_p2, col_p3 = st.columns(3)
            with col_p1:
                st.metric("Kalibrasi Winrate", f"{prob_ctx['calibrated_winrate']}%", 
                         delta=f"Grade: {prob_ctx['grade']}")
            with col_p2:
                st.metric("Confidence Interval (95%)", prob_ctx['confidence_interval'])
            with col_p3:
                st.metric("MTF Confluence", f"{prob_ctx['mtf_confluence']}/100")
            
            st.caption("💡 Winrate dikalibrasi dari backtest historis. Confidence interval menunjukkan rentang ketidakpastian.")
        except Exception as e:
            st.warning(f"Probability engine gagal: {e}")
        # ========== END PROBABILITY CONTEXT ==========
# ===================== TAB 4: SCANNER =====================
with tab4:
    st.header("🔍 Market Scanner")
    if st.button("Scan Sekarang"):
        with st.spinner("Scanning saham-saham... (mohon tunggu)"):
            results = scan_saham()
        if results:
            st.dataframe(pd.DataFrame(results), width='stretch')
        else:
            st.info("Tidak ada sinyal kuat saat ini.")
# Di dalam with tab4:
    st.divider()
    st.subheader("🔔 Tes Notifikasi Telegram")
    if st.button("Tes Kirim Notifikasi"):
        try:
            from notification import send_telegram_message
            import asyncio
            test_msg = f"✅ Tes notifikasi Smart Money Engine berhasil!\nWaktu: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            asyncio.run(send_telegram_message(test_msg))
            st.success("Pesan tes terkirim. Cek Telegram Anda.")
        except Exception as e:
            st.error(f"Gagal mengirim: {e}")
# ===================== TAB 5: BACKTEST (UPGRADED) =====================
with tab5:
    st.header("📈 Backtest Profesional")
    if df is None:
        st.warning("Data belum dimuat.")
    else:
        bt_mode = st.radio("Mode Backtest", ["Standar", "Walk-Forward", "Monte Carlo"], horizontal=True)
        
        if st.button("Jalankan Backtest"):
            with st.spinner("Memproses..."):
                if bt_mode == "Standar":
                    bt = backtest_strategy(df, modal, risk_pct)
                    # Tambahkan metrik lanjutan dari backtest_engine
                    from backtest_engine import calculate_advanced_metrics
                    advanced = calculate_advanced_metrics(bt['equity_curve'], [], modal)
                    bt.update(advanced)
                    
                    st.success(f"Backtest selesai. Total trades: {bt['trades']}")
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Return", f"{bt['return']}%", delta=f"Rp{bt['final_capital']-modal:,.0f}")
                    col2.metric("Win Rate", f"{bt['winrate']}%")
                    col3.metric("Max Drawdown", f"{bt['max_drawdown']}%")
                    
                    col4, col5, col6 = st.columns(3)
                    col4.metric("Profit Factor", f"{bt.get('profit_factor', 0):.2f}")
                    col5.metric("Sharpe Ratio", f"{bt.get('sharpe_ratio', 0):.2f}")
                    col6.metric("Expectancy", f"{bt.get('expectancy', 0):.2f}%")
                    
                    if bt['equity_curve']:
                        eq_df = pd.DataFrame({'Equity': bt['equity_curve']})
                        st.line_chart(eq_df)
                
                elif bt_mode == "Walk-Forward":
                    wf_result, wf_error = walk_forward_backtest(symbol_input, 1, 3, modal, risk_pct)
                    if wf_error:
                        st.error(wf_error)
                    else:
                        st.success("Walk-Forward Analysis selesai")
                        st.metric("Rata-rata Return per Periode", f"{wf_result['avg_return']}%")
                        st.metric("Rata-rata Win Rate", f"{wf_result['avg_winrate']}%")
                        st.write("**Detail per Periode:**")
                        st.dataframe(pd.DataFrame(wf_result['periods']), width='stretch')
                
                else:  # Monte Carlo
                    # Ambil trades dari backtest standar dulu
                    bt_std = backtest_strategy(df, modal, risk_pct)
                    # Untuk Monte Carlo kita butuh daftar trade % return
                    # Sayangnya backtest_strategy tidak mengembalikan trades_list
                    # Kita bisa generate dari equity curve
                    if len(bt_std['equity_curve']) > 1:
                        eq = bt_std['equity_curve']
                        trade_returns = list(np.diff(eq) / eq[:-1] * 100)
                        mc_result = monte_carlo_simulation(trade_returns, 1000, modal)
                        if mc_result:
                            st.success("Monte Carlo Simulation (1000 simulasi)")
                            col1, col2, col3 = st.columns(3)
                            col1.metric("Median Return", f"{mc_result['median_return']}%")
                            col2.metric("Worst Return", f"{mc_result['worst_return']}%")
                            col3.metric("Best Return", f"{mc_result['best_return']}%")
                            col4, col5 = st.columns(2)
                            col4.metric("Median Drawdown", f"{mc_result['median_drawdown']}%")
                            col5.metric("Worst Drawdown", f"{mc_result['worst_drawdown']}%")
                            st.caption(f"Value at Risk (95% confidence): Rp{mc_result['var_95']:,.0f}")
                        else:
                            st.warning("Data trade tidak cukup untuk Monte Carlo.")
                    else:
                        st.warning("Data equity curve tidak cukup.")
import threading
import schedule
import time
from notification import scan_and_notify

def run_scheduler():
    """Menjalankan scheduler dalam thread terpisah."""
    # Nilai default untuk scanning notifikasi (bisa disesuaikan)
    modal_default = 100_000_000   # Rp 100 juta
    risk_default = 2.0            # 2%
    
    # Jadwalkan scan setiap 2 jam
    schedule.every(2).hours.do(
        scan_and_notify,
        capital=modal_default,
        risk_percent=risk_default
    )
    
    while True:
        schedule.run_pending()
        time.sleep(1)

# Jalankan scheduler jika token tersedia
if os.getenv("TELEGRAM_BOT_TOKEN") and os.getenv("TELEGRAM_CHAT_ID"):
    scheduler_thread = threading.Thread(target=run_scheduler, daemon=True)
    scheduler_thread.start()
