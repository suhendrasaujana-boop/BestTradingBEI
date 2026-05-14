import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime
from data import get_data, add_indicators, backtest_strategy

DB_PATH = "backtest_results.db"

def init_db():
    """Inisialisasi database SQLite untuk hasil backtest."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS backtests (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        symbol TEXT,
        start_date TEXT,
        end_date TEXT,
        initial_capital REAL,
        final_capital REAL,
        return_pct REAL,
        winrate REAL,
        trades INTEGER,
        max_drawdown REAL,
        profit_factor REAL,
        sharpe_ratio REAL,
        expectancy REAL,
        calmar_ratio REAL,
        ulcer_index REAL,
        avg_win REAL,
        avg_loss REAL,
        max_consecutive_losses INTEGER,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    )''')
    conn.commit()
    conn.close()

def save_backtest_result(symbol, start_date, end_date, initial_capital, metrics):
    """Simpan hasil backtest ke database."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    try:
        c.execute('''INSERT INTO backtests (
            symbol, start_date, end_date, initial_capital, final_capital,
            return_pct, winrate, trades, max_drawdown, profit_factor,
            sharpe_ratio, expectancy, calmar_ratio, ulcer_index,
            avg_win, avg_loss, max_consecutive_losses
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''', (
            symbol,
            start_date,
            end_date,
            initial_capital,
            metrics.get('final_capital', 0),
            metrics.get('return', 0),
            metrics.get('winrate', 0),
            metrics.get('trades', 0),
            metrics.get('max_drawdown', 0),
            metrics.get('profit_factor', 0),
            metrics.get('sharpe_ratio', 0),
            metrics.get('expectancy', 0),
            metrics.get('calmar_ratio', 0),
            metrics.get('ulcer_index', 0),
            metrics.get('avg_win', 0),
            metrics.get('avg_loss', 0),
            metrics.get('max_consecutive_losses', 0)
        ))
        conn.commit()
        print(f"💾 Hasil backtest {symbol} disimpan ke database.")
    except Exception as e:
        print(f"⚠️ Gagal simpan backtest: {e}")
    finally:
        conn.close()

def calculate_advanced_metrics(equity_curve, trades, initial_capital):
    """
    Hitung metrik lanjutan dari equity curve dan daftar trade (% return).
    
    Parameters:
    - equity_curve: list nilai equity dari waktu ke waktu
    - trades: list persentase return per trade
    - initial_capital: modal awal
    
    Returns:
    - dict dengan metrik lanjutan
    """
    if not equity_curve or len(equity_curve) < 2:
        return {
            'calmar_ratio': 0,
            'ulcer_index': 0,
            'avg_win': 0,
            'avg_loss': 0,
            'max_consecutive_losses': 0
        }
    
    equity = np.array(equity_curve)
    
    # Calmar Ratio: CAGR / Max Drawdown
    total_return = (equity[-1] - initial_capital) / initial_capital
    days = len(equity)
    years = days / 252
    
    if total_return > -1 and years > 0:
        cagr = (1 + total_return) ** (1 / years) - 1
    else:
        cagr = 0
    
    # Drawdown
    peak = np.maximum.accumulate(equity)
    drawdown = (peak - equity) / np.where(peak > 0, peak, 1)
    max_dd = np.max(drawdown) * 100
    
    calmar = cagr / (max_dd / 100) if max_dd > 0 else 0
    
    # Ulcer Index
    ulcer = np.sqrt(np.mean(drawdown**2)) * 100 if len(drawdown) > 0 else 0
    
    # Average Win / Average Loss
    if trades:
        wins = [t for t in trades if t > 0]
        losses = [t for t in trades if t < 0]
        avg_win = np.mean(wins) if wins else 0
        avg_loss = np.mean(losses) if losses else 0
    else:
        avg_win = 0
        avg_loss = 0
    
    # Max Consecutive Losses
    max_cons_loss = 0
    cons_loss = 0
    for t in (trades or []):
        if t < 0:
            cons_loss += 1
            max_cons_loss = max(max_cons_loss, cons_loss)
        else:
            cons_loss = 0
    
    return {
        'calmar_ratio': round(calmar, 2),
        'ulcer_index': round(ulcer, 2),
        'avg_win': round(avg_win, 2),
        'avg_loss': round(avg_loss, 2),
        'max_consecutive_losses': max_cons_loss
    }

def walk_forward_analysis(symbol, train_months=12, test_months=3, initial_capital=100000000, risk_percent=2):
    """
    Walk-Forward Analysis: rolling training dan testing.
    
    Parameters:
    - symbol: kode saham
    - train_months: bulan data training
    - test_months: bulan data testing per window
    - initial_capital: modal awal
    - risk_percent: risiko per trade dalam %
    
    Returns:
    - dict hasil walk-forward atau None jika gagal
    """
    try:
        df = get_data(symbol, "1d")
    except:
        return None, "Gagal ambil data"
    
    if df.empty or len(df) < 252:
        return None, "Data tidak cukup (minimal 1 tahun)"
    
    df = add_indicators(df)
    
    # Konversi datetime
    if 'datetime' in df.columns:
        df['date'] = pd.to_datetime(df['datetime'])
    else:
        df['date'] = pd.to_datetime(df.index)
    
    df = df.sort_values('date').reset_index(drop=True)
    
    # Hitung jumlah candle per bulan (approx 21 trading days)
    candles_per_month = 21
    train_candles = train_months * candles_per_month
    test_candles = test_months * candles_per_month
    
    results = []
    start_idx = train_candles
    
    while start_idx + test_candles < len(df):
        # Data testing
        test_start = start_idx
        test_end = min(start_idx + test_candles, len(df))
        test_df = df.iloc[test_start:test_end]
        
        if len(test_df) < 20:
            start_idx += test_candles
            continue
        
        # Jalankan backtest pada periode testing
        bt = backtest_strategy(test_df, initial_capital, risk_percent)
        
        period_start = df.iloc[test_start]['date'].strftime('%Y-%m-%d')
        period_end = df.iloc[test_end - 1]['date'].strftime('%Y-%m-%d')
        
        results.append({
            'period_start': period_start,
            'period_end': period_end,
            'return': bt['return'],
            'winrate': bt['winrate'],
            'trades': bt['trades'],
            'max_drawdown': bt['max_drawdown']
        })
        
        start_idx += test_candles
    
    if not results:
        return None, "Tidak ada periode testing yang valid"
    
    # Rata-rata hasil
    avg_return = np.mean([r['return'] for r in results])
    avg_winrate = np.mean([r['winrate'] for r in results])
    total_trades = sum([r['trades'] for r in results])
    
    return {
        'periods': results,
        'avg_return': round(avg_return, 2),
        'avg_winrate': round(avg_winrate, 2),
        'total_trades': total_trades,
        'num_periods': len(results)
    }, None

def monte_carlo_simulation(trades, num_simulations=1000, initial_capital=100000000):
    """
    Monte Carlo simulation: acak urutan trade untuk estimasi kemungkinan drawdown.
    
    Parameters:
    - trades: list persentase return per trade
    - num_simulations: jumlah simulasi
    - initial_capital: modal awal
    
    Returns:
    - dict hasil simulasi
    """
    if not trades or len(trades) < 5:
        return None
    
    trades = np.array(trades)
    final_equities = []
    max_drawdowns = []
    
    for _ in range(num_simulations):
        # Acak urutan trade
        shuffled = np.random.permutation(trades)
        equity = initial_capital
        peak = equity
        max_dd = 0
        
        for t in shuffled:
            equity *= (1 + t / 100)
            if equity > peak:
                peak = equity
            dd = (peak - equity) / peak * 100
            if dd > max_dd:
                max_dd = dd
        
        final_equities.append(equity)
        max_drawdowns.append(max_dd)
    
    final_equities = np.array(final_equities)
    max_drawdowns = np.array(max_drawdowns)
    
    return {
        'median_return': round((np.median(final_equities) - initial_capital) / initial_capital * 100, 2),
        'worst_return': round((np.min(final_equities) - initial_capital) / initial_capital * 100, 2),
        'best_return': round((np.max(final_equities) - initial_capital) / initial_capital * 100, 2),
        'median_drawdown': round(np.median(max_drawdowns), 2),
        'worst_drawdown': round(np.max(max_drawdowns), 2),
        'var_95': round(np.percentile(final_equities, 5), 0)
    }

def run_full_backtest(symbol, initial_capital=100000000, risk_percent=2, save_to_db=True):
    """
    Jalankan backtest lengkap: standar + walk-forward + Monte Carlo.
    
    Returns:
    - dict hasil lengkap
    - error message (None jika sukses)
    """
    print(f"\n📊 Memulai backtest lengkap untuk {symbol}...")
    
    # 1. Ambil data
    try:
        df = get_data(symbol, "1d")
    except Exception as e:
        return None, f"Gagal ambil data: {e}"
    
    if df.empty:
        return None, "Data kosong"
    
    df = add_indicators(df)
    
    # 2. Backtest standar
    print("  ⚙️ Backtest standar...")
    bt = backtest_strategy(df, initial_capital, risk_percent)
    
    # 3. Advanced metrics
    print("  ⚙️ Menghitung metrik lanjutan...")
    advanced = calculate_advanced_metrics(
        bt.get('equity_curve', []),
        bt.get('trades_list', []),
        initial_capital
    )
    
    # Gabungkan
    full_metrics = {**bt, **advanced}
    
    # 4. Walk-Forward
    print("  ⚙️ Walk-Forward Analysis...")
    wf_result, wf_error = walk_forward_analysis(
        symbol, 
        train_months=12, 
        test_months=3,
        initial_capital=initial_capital,
        risk_percent=risk_percent
    )
    
    # 5. Monte Carlo (jika ada trades)
    mc_result = None
    if bt['trades'] > 5:
        print("  ⚙️ Monte Carlo Simulation...")
        mc_result = monte_carlo_simulation(
            bt.get('trades_list', []),
            num_simulations=1000,
            initial_capital=initial_capital
        )
    
    # 6. Simpan ke database
    if save_to_db and bt['trades'] > 0:
        try:
            if 'datetime' in df.columns:
                dates = df['datetime']
            else:
                dates = df.index
            
            start_date = str(dates.iloc[0])[:10] if len(dates) > 0 else "N/A"
            end_date = str(dates.iloc[-1])[:10] if len(dates) > 0 else "N/A"
            
            save_backtest_result(symbol, start_date, end_date, initial_capital, full_metrics)
        except Exception as e:
            print(f"  ⚠️ Gagal simpan ke DB: {e}")
    
    print(f"  ✅ Backtest selesai!")
    
    return {
        'standard': full_metrics,
        'walk_forward': wf_result,
        'monte_carlo': mc_result
    }, None

def print_backtest_summary(result):
    """Print ringkasan hasil backtest dengan format rapi."""
    if result is None:
        print("❌ Tidak ada hasil backtest.")
        return
    
    std = result.get('standard', {})
    
    print(f"\n{'='*60}")
    print(f"📊 HASIL BACKTEST STANDAR")
    print(f"{'='*60}")
    print(f"  Return:        {std.get('return', 0):>8}%")
    print(f"  Winrate:       {std.get('winrate', 0):>8}%")
    print(f"  Total Trades:  {std.get('trades', 0):>8}")
    print(f"  Max Drawdown:  {std.get('max_drawdown', 0):>8}%")
    print(f"  Profit Factor: {std.get('profit_factor', 0):>8}")
    print(f"  Sharpe Ratio:  {std.get('sharpe_ratio', 0):>8}")
    print(f"  Expectancy:    {std.get('expectancy', 0):>8}%")
    print(f"  Calmar Ratio:  {std.get('calmar_ratio', 0):>8}")
    print(f"  Avg Win:       {std.get('avg_win', 0):>8}%")
    print(f"  Avg Loss:      {std.get('avg_loss', 0):>8}%")
    print(f"  Max Cons Loss: {std.get('max_consecutive_losses', 0):>8}")
    
    wf = result.get('walk_forward')
    if wf:
        print(f"\n{'='*60}")
        print(f"📊 WALK-FORWARD ANALYSIS")
        print(f"{'='*60}")
        print(f"  Avg Return:    {wf.get('avg_return', 0):>8}%")
        print(f"  Avg Winrate:   {wf.get('avg_winrate', 0):>8}%")
        print(f"  Total Trades:  {wf.get('total_trades', 0):>8}")
        print(f"  Periods:       {wf.get('num_periods', 0):>8}")
    
    mc = result.get('monte_carlo')
    if mc:
        print(f"\n{'='*60}")
        print(f"📊 MONTE CARLO SIMULATION (1000 runs)")
        print(f"{'='*60}")
        print(f"  Median Return: {mc.get('median_return', 0):>8}%")
        print(f"  Best Return:   {mc.get('best_return', 0):>8}%")
        print(f"  Worst Return:  {mc.get('worst_return', 0):>8}%")
        print(f"  Median DD:     {mc.get('median_drawdown', 0):>8}%")
        print(f"  Worst DD:      {mc.get('worst_drawdown', 0):>8}%")
        print(f"  VaR 95%:       Rp{mc.get('var_95', 0):>,.0f}")


# Testing
if __name__ == "__main__":
    init_db()
    
    print("=" * 60)
    print("🧪 TEST BACKTEST ENGINE")
    print("=" * 60)
    
    # Test dengan BBRI
    result, error = run_full_backtest("BBRI", initial_capital=100000000, risk_percent=2, save_to_db=False)
    
    if error:
        print(f"❌ Error: {error}")
    else:
        print_backtest_summary(result)
