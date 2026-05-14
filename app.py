import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from data import (
    get_data, 
    add_indicators, 
    calculate_entry_sl_tp,
    calculate_confidence_score,
    get_trading_recommendation,
    detect_high_quality_setup,
    get_multi_timeframe_alignment,
    get_ihsg_trend,
    backtest_strategy
)
from scanner_engine import scan_saham_fast
from context import get_full_market_context
from probability_engine import get_probability_context
from database import init_db

def main():
    print("=" * 60)
    print("📊 SMART MONEY TRADING SYSTEM")
    print("=" * 60)
    
    # Init database
    init_db()
    
    # 1. Cek IHSG
    print("\n📈 Mengecek IHSG...")
    ihsg_trend, ihsg_score, ihsg_desc = get_ihsg_trend()
    print(f"IHSG: {ihsg_desc}")
    
    # 2. Scan saham
    print("\n🔍 Scanning saham...")
    results = scan_saham_fast()
    
    if results:
        print(f"\n{'='*60}")
        print(f"TOP {len(results)} PICKS:")
        print(f"{'='*60}")
        for i, row in enumerate(results, 1):
            print(f"{i:2d}. {row['Kode']:<10} Score: {row['Score']:<5} {row['Sinyal']:<10} {row['Setup']}")
    else:
        print("❌ Tidak ada sinyal kuat hari ini")
    
    # 3. Analisis detail
    symbol = input("\n🔍 Masukkan kode saham (contoh: BBRI) untuk analisis detail [enter untuk skip]: ").strip().upper()
    
    if symbol:
        print(f"\n⏳ Menganalisis {symbol}...")
        df = get_data(symbol, "1d")
        
        if df.empty:
            print(f"❌ Data {symbol} tidak tersedia")
            return
        
        df = add_indicators(df)
        last = df.iloc[-1]
        
        # Basic info
        print(f"\n{'='*60}")
        print(f"ANALISIS {symbol}")
        print(f"{'='*60}")
        print(f"Harga: Rp{last['close']:,.0f}")
        print(f"Volume: {last['volume']:,.0f}")
        
        # Setup trading
        entry, sl, tp, shares, rr, setup, conf, signals = calculate_entry_sl_tp(df)
        
        if entry:
            print(f"\n🎯 RENCANA TRADING:")
            print(f"Setup: {setup}")
            print(f"Confidence: {conf:.0f}%")
            print(f"Entry: Rp{entry:,.0f}")
            print(f"Stop Loss: Rp{sl:,.0f}")
            print(f"Take Profit: Rp{tp:,.0f}")
            print(f"Risk/Reward: 1:{rr:.1f}")
            print(f"Lot: {shares} lembar")
        
        # Confidence score
        score, factors, grade = calculate_confidence_score(df, ihsg_score)
        print(f"\n📊 Confidence Score: {score:.0f}/100 ({grade})")
        
        for name, value, desc in factors:
            sign = "+" if value > 0 else ""
            print(f"  {name:15s}: {sign}{value:.1f} - {desc}")
        
        # Rekomendasi
        rec = get_trading_recommendation(df)
        print(f"\n💡 Rekomendasi: {rec}")
        
        # Multi timeframe
        print("\n🕐 Multi Timeframe Analysis...")
        mtf, alignment, mtf_score, mtf_signals = get_multi_timeframe_alignment(symbol)
        print(f"Alignment: {alignment} (Score: {mtf_score})")
        for sig in mtf_signals:
            print(f"  - {sig}")
        
        # Market context
        print("\n🌍 Market Context...")
        try:
            ctx = get_full_market_context(symbol)
            print(f"Sektor: {ctx.get('sector', 'Unknown')}")
            print(f"RS vs IHSG: {ctx.get('rs_score', 50)}/100 - {ctx.get('rs_desc', '')}")
            print(f"Market Breadth: {ctx.get('breadth_desc', '')}")
            print(f"Foreign Flow: {ctx.get('flow_desc', '')}")
        except Exception as e:
            print(f"Context error: {e}")
        
        # Probability
        print("\n🎲 Probability Context...")
        prob = get_probability_context(symbol, score)
        print(f"Calibrated Winrate: {prob.get('calibrated_winrate', 0)}%")
        print(f"Confidence Interval: {prob.get('confidence_interval', 'N/A')}")
        print(f"MTF Confluence: {prob.get('mtf_confluence', 0)}/100")
        
        # Backtest
        print("\n📈 Backtest (100 hari)...")
        bt = backtest_strategy(df)
        print(f"Return: {bt['return']}%")
        print(f"Winrate: {bt['winrate']}%")
        print(f"Trades: {bt['trades']}")
        print(f"Max DD: {bt['max_drawdown']}%")
        print(f"Sharpe: {bt['sharpe_ratio']}")

if __name__ == "__main__":
    main()
