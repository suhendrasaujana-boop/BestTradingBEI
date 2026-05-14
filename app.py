import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import fungsi dari modul
from data import (
    get_data, 
    add_indicators, 
    calculate_entry_sl_tp,
    calculate_confidence_score,
    get_trading_recommendation,
    get_ihsg_trend,
    backtest_strategy
)
from scanner_engine import scan_saham_fast
from context import get_full_market_context
from probability_engine import get_probability_context
from database import init_db

# ========== PAGE CONFIG ==========
st.set_page_config(
    page_title="BestTradingBEI",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ========== CSS CUSTOM ==========
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sniper-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        margin: 10px 0;
    }
    .metric-card {
        background: #f0f2f6;
        padding: 15px;
        border-radius: 8px;
        text-align: center;
    }
    .buy-signal {
        color: #00c853;
        font-weight: bold;
    }
    .hold-signal {
        color: #ff9800;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# ========== INIT ==========
init_db()

# ========== SIDEBAR ==========
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/stock-share.png", width=60)
    st.markdown("## 📊 BestTradingBEI")
    st.markdown("---")
    
    menu = st.radio(
        "Menu",
        ["🏠 Dashboard", "🔍 Scanner", "📊 Detail Saham", "⚙️ Backtest", "ℹ️ Tentang"]
    )
    
    st.markdown("---")
    st.caption(f"⏰ {datetime.now().strftime('%Y-%m-%d %H:%M')}")

# ========== CACHE DATA ==========
@st.cache_data(ttl=300)
def load_ihsg():
    return get_ihsg_trend()

@st.cache_data(ttl=300)
def load_scan():
    return scan_saham_fast()

@st.cache_data(ttl=300)
def load_stock_data(symbol):
    df = get_data(symbol, "1d")
    if not df.empty:
        df = add_indicators(df)
    return df

# ========== DASHBOARD ==========
if menu == "🏠 Dashboard":
    st.markdown('<p class="main-header">📈 Smart Money Trading System</p>', unsafe_allow_html=True)
    
    # IHSG
    ihsg_trend, ihsg_score, ihsg_desc = load_ihsg()
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("IHSG Trend", ihsg_trend)
    with col2:
        st.metric("IHSG Score", f"{ihsg_score}/100")
    with col3:
        st.metric("Market", ihsg_desc.split("(")[-1].replace(")", "") if "(" in ihsg_desc else "N/A")
    with col4:
        st.metric("Last Update", datetime.now().strftime("%H:%M"))
    
    st.markdown("---")
    
    # Top Picks
    st.subheader("🔥 Top Picks Hari Ini")
    results = load_scan()
    
    if results:
        cols = st.columns(min(len(results), 3))
        for i, row in enumerate(results[:6]):
            with cols[i % 3]:
                score = int(row['Score'])
                if score >= 80:
                    st.markdown(f"""
                    <div class="sniper-box">
                        <h3>{row['Kode']}</h3>
                        <h2>{score}/100</h2>
                        <p>{row['Setup'][:50]}...</p>
                        <h4>{row['Harga']}</h4>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    with st.container():
                        st.markdown(f"**{row['Kode']}** | Score: {score} | {row['Sinyal']}")
                        st.caption(f"{row['Setup'][:60]}")
    else:
        st.info("Tidak ada sinyal kuat hari ini.")
    
    st.markdown("---")
    
    # Market Breadth
    st.subheader("📊 Market Context")
    try:
        ctx = get_full_market_context("BBRI")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Market Breadth EMA20", f"{ctx.get('breadth_20', 50)}%")
        with c2:
            st.metric("Market Breadth EMA50", f"{ctx.get('breadth_50', 50)}%")
        with c3:
            top_sectors = ctx.get('top_sectors', [])
            if top_sectors:
                st.metric("Top Sector", f"{top_sectors[0][0]}")
    except:
        pass

# ========== SCANNER ==========
elif menu == "🔍 Scanner":
    st.markdown('<p class="main-header">🔍 Stock Scanner</p>', unsafe_allow_html=True)
    
    if st.button("🔄 Refresh Scan", use_container_width=True):
        st.cache_data.clear()
        st.rerun()
    
    results = load_scan()
    
    if results:
        # Convert ke DataFrame
        df_results = pd.DataFrame(results)
        df_results['Score'] = df_results['Score'].astype(int)
        
        # Filter
        min_score = st.slider("Minimum Score", 0, 100, 60)
        df_filtered = df_results[df_results['Score'] >= min_score]
        
        st.dataframe(
            df_filtered,
            column_config={
                "Kode": "Kode",
                "Score": st.column_config.NumberColumn("Score", format="%d"),
                "Sinyal": "Sinyal",
                "Setup": "Setup",
                "Harga": "Harga"
            },
            hide_index=True,
            use_container_width=True
        )
    else:
        st.info("Loading...")

# ========== DETAIL SAHAM ==========
elif menu == "📊 Detail Saham":
    st.markdown('<p class="main-header">📊 Detail Analisis Saham</p>', unsafe_allow_html=True)
    
    symbol = st.text_input("Masukkan kode saham (contoh: BBRI)", value="BBRI").upper()
    
    if symbol:
        with st.spinner(f"Menganalisis {symbol}..."):
            df = load_stock_data(symbol)
        
        if df.empty:
            st.error(f"Data {symbol} tidak tersedia")
        else:
            last = df.iloc[-1]
            
            # Info dasar
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Harga", f"Rp{last['close']:,.0f}")
            with col2:
                st.metric("Volume", f"{last['volume']:,.0f}")
            with col3:
                st.metric("RSI", f"{last.get('rsi', 0):.0f}")
            with col4:
                st.metric("ADX", f"{last.get('adx', 0):.0f}")
            
            st.markdown("---")
            
            # Setup Trading
            entry, sl, tp, shares, rr, setup, conf, signals = calculate_entry_sl_tp(df)
            
            if entry:
                st.success(f"🎯 Setup: **{setup}** (Confidence: {conf:.0f}%)")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Entry", f"Rp{entry:,.0f}")
                with col2:
                    st.metric("Stop Loss", f"Rp{sl:,.0f}", delta=f"-{abs(entry-sl)/entry*100:.1f}%")
                with col3:
                    st.metric("Take Profit", f"Rp{tp:,.0f}", delta=f"+{abs(tp-entry)/entry*100:.1f}%")
                
                st.info(f"Risk/Reward: 1:{rr:.1f} | Lot: {shares} lembar")
            else:
                st.warning("Tidak ada setup trading saat ini")
            
            st.markdown("---")
            
            # Confidence Score
            col1, col2 = st.columns(2)
            with col1:
                score, factors, grade = calculate_confidence_score(df, load_ihsg()[1])
                color = "green" if grade in ["SNIPER", "HIGH"] else "orange" if grade == "NORMAL" else "red"
                st.markdown(f"### Confidence: <span style='color:{color}'>{score:.0f}/100 ({grade})</span>", unsafe_allow_html=True)
                
                for name, value, desc in factors:
                    st.caption(f"{name}: {value:+.1f} - {desc}")
            
            with col2:
                try:
                    prob = get_probability_context(symbol, score)
                    st.markdown(f"### Winrate: {prob.get('adjusted_winrate', 0):.0f}%")
                    st.caption(f"CI: {prob.get('confidence_interval', 'N/A')}")
                    st.caption(f"MTF Confluence: {prob.get('mtf_confluence', 0)}/100")
                except:
                    pass
            
            # Rekomendasi
            st.markdown("---")
            rec = get_trading_recommendation(df)
            if "SNIPER" in rec or "🎯" in rec:
                st.markdown(f'<div class="sniper-box">{rec}</div>', unsafe_allow_html=True)
            else:
                st.info(rec)

# ========== BACKTEST ==========
elif menu == "⚙️ Backtest":
    st.markdown('<p class="main-header">⚙️ Backtest Strategi</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        symbol_bt = st.text_input("Kode Saham", "BBRI").upper()
    with col2:
        capital = st.number_input("Modal Awal (Rp)", 10_000_000, 1_000_000_000, 100_000_000, 10_000_000)
    
    if st.button("🚀 Jalankan Backtest", use_container_width=True):
        with st.spinner("Menjalankan backtest..."):
            df = load_stock_data(symbol_bt)
            
            if df.empty:
                st.error("Data tidak tersedia")
            else:
                bt = backtest_strategy(df, initial_capital=capital)
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Return", f"{bt['return']}%")
                with col2:
                    st.metric("Winrate", f"{bt['winrate']}%")
                with col3:
                    st.metric("Total Trades", bt['trades'])
                with col4:
                    st.metric("Max DD", f"{bt['max_drawdown']}%")
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Profit Factor", f"{bt['profit_factor']:.2f}")
                with col2:
                    st.metric("Sharpe", f"{bt['sharpe_ratio']:.2f}")
                with col3:
                    st.metric("Final Capital", f"Rp{bt['final_capital']:,.0f}")
                with col4:
                    st.metric("Expectancy", f"{bt['expectancy']:.2f}%")
                
                # Equity curve chart
                if bt.get('equity_curve'):
                    equity_df = pd.DataFrame({
                        'Trade': range(len(bt['equity_curve'])),
                        'Equity': bt['equity_curve']
                    })
                    st.line_chart(equity_df.set_index('Trade'))

# ========== TENTANG ==========
elif menu == "ℹ️ Tentang":
    st.markdown('<p class="main-header">ℹ️ Tentang BestTradingBEI</p>', unsafe_allow_html=True)
    
    st.markdown("""
    ### 📈 Smart Money Trading System
    
    Sistem analisis saham berbasis **Smart Money Concept** untuk pasar Indonesia (BEI).
    
    #### Fitur:
    - 🔍 **Scanner**: Scanning 40+ saham unggulan
    - 📊 **Analisis Teknikal**: EMA, RSI, MACD, ADX, Volume Profile
    - 🧠 **Smart Money**: BOS/CHOCH, FVG, Order Block, Liquidity Sweep
    - 🎯 **Entry/SL/TP**: Kalkulasi otomatis dengan Risk Management
    - ⚙️ **Backtest**: Uji strategi dengan data historis
    - 📱 **Notifikasi**: Alert Telegram untuk sinyal SNIPER
    
    #### Disclaimer:
    Sistem ini hanya alat bantu analisis. Keputusan trading sepenuhnya tanggung jawab Anda.
    """)
    
    st.markdown("---")
    st.caption(f"© 2024 BestTradingBEI | v1.0 | {datetime.now().year}")
