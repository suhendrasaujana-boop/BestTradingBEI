import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime
from data import get_data, add_indicators, backtest_strategy

DB_PATH = "backtest_results.db"

def init_db():
    """Inisialisasi database SQLite."""
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
    c.execute('''INSERT INTO backtests (
        symbol, start_date, end_date, initial_capital, final_capital,
        return_pct, winrate, trades, max_drawdown, profit_factor,
        sharpe_ratio, expectancy, calmar_ratio, ulcer_index,
        avg_win, avg_loss, max_consecutive_losses
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''', (
        symbol, start_date, end_date, initial_capital,
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
    conn.close()

def calculate_advanced_metrics(equity_curve, trades, initial_capital):
    """
    Hitung metrik lanjutan dari equity curve dan daftar trade (% return).
    """
    if not equity_curve or len(equity_curve) < 2:
        return {}
    
    equity = np.array(equity_curve)
    returns = np.diff(equity) / equity[:-1]
    
    # Calmar Ratio: CAGR / Max Drawdown
    total_return = (equity[-1] - initial_capital) / initial_capital
    days = len(equity)
    cagr = (1 + total_return) ** (252 / days) - 1 if days > 0 else 0
    peak = np.maximum.accumulate(equity)
    drawdown = (peak - equity) / peak
    max_dd = np.max(drawdown) * 100
    calmar = cagr / (max_dd / 100) if max_dd > 0 else 0
    
    # Ulcer Index
    ulcer = np.sqrt(np.mean(drawdown**2)) * 100 if len(drawdown) > 0 else 0
    
    # Average Win / Average Loss
    wins = [t for t in trades if t > 0]
    losses = [t for t in trades if t < 0]
    avg_win = np.mean(wins) if wins else 0
    avg_loss = np.mean(losses) if losses else 0
    
    # Max Consecutive Losses
    max_cons_loss = 0
    cons_loss = 0
    for t in trades:
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

def walk_forward_backtest(symbol, train_years=1, test_months=3, initial_capital=100000000, risk_percent=2):
    """
    Walk-Forward Analysis: latih di periode sebelumnya, uji di periode berikutnya.
    """
    df = get_data(symbol, "1d")
    if df.empty or len(df) < 252 * (train_years + 0.5):
        return None, "Data tidak cukup"
    
    df = add_indicators(df)
    df['date'] = pd.to_datetime(df['datetime'] if 'datetime' in df.columns else df.index)
    df = df.sort_values('date')
    
    results = []
    start_date = df['date'].iloc[0] + pd.DateOffset(years=train_years)
    end_date = df['date'].iloc[-1]
    
    current_train_end = start_date
    while current_train_end < end_date:
        test_start = current_train_end
        test_end = min(test_start + pd.DateOffset(months=test_months), end_date)
        
        # Ambil data training (dari awal sampai test_start, untuk "latih" = gunakan data historis)
        train_df = df[df['date'] < test_start]
        # Ambil data testing
        test_df = df[(df['date'] >= test_start) & (df['date'] < test_end)]
        
        if len(test_df) < 20:
            current_train_end = test_end
            continue
        
        # Jalankan backtest di data testing dengan strategi yang sudah "dilatih" (parameter sama)
        # Karena strategi kita tidak memiliki optimisasi parameter, kita langsung gunakan backtest_strategy
        bt = backtest_strategy(test_df, initial_capital, risk_percent)
        results.append({
            'period_start': test_start.strftime('%Y-%m-%d'),
            'period_end': test_end.strftime('%Y-%m-%d'),
            'return': bt['return'],
            'winrate': bt['winrate'],
            'trades': bt['trades']
        })
        
        current_train_end = test_end
    
    if not results:
        return None, "Tidak ada periode testing"
    
    avg_return = np.mean([r['return'] for r in results])
    avg_winrate = np.mean([r['winrate'] for r in results])
    total_trades = sum([r['trades'] for r in results])
    
    return {
        'periods': results,
        'avg_return': round(avg_return, 2),
        'avg_winrate': round(avg_winrate, 2),
        'total_trades': total_trades
    }, None

def monte_carlo_simulation(trades, num_simulations=1000, initial_capital=100000000):
    """
    Monte Carlo simulation pada daftar trade (% return).
    Mengacak urutan trade untuk melihat distribusi kemungkinan drawdown.
    """
    if not trades or len(trades) < 5:
        return None
    
    trades = np.array(trades)
    final_equities = []
    max_drawdowns = []
    
    for _ in range(num_simulations):
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
        'var_95': round(np.percentile(final_equities, 5), 0)  # 95% confidence level
    }

def run_full_backtest(symbol, initial_capital=100000000, risk_percent=2, save_to_db=True):
    """
    Jalankan backtest lengkap: standar + walk-forward + Monte Carlo.
    """
    df = get_data(symbol, "1d")
    if df.empty:
        return None, "Data tidak tersedia"
    
    # Backtest standar
    df = add_indicators(df)
    bt = backtest_strategy(df, initial_capital, risk_percent)
    
    # Hitung metrik lanjutan
    advanced = calculate_advanced_metrics(bt['equity_curve'], bt.get('trades_list', []), initial_capital)
    
    # Gabungkan semua metrik
    full_metrics = {**bt, **advanced}
    
    # Walk-Forward
    wf_result, wf_error = walk_forward_backtest(symbol, train_years=1, test_months=3, initial_capital=initial_capital, risk_percent=risk_percent)
    
    # Monte Carlo (gunakan equity curve dari backtest standar untuk menghasilkan trade %)
    # Karena kita hanya punya equity, kita hitung trade % dari equity curve jika trades_list tidak ada.
    # Lebih baik gunakan trades_list jika ada di backtest_strategy.
    # Untuk sekarang, kita bisa skip MC jika tidak ada trades_list.
    
    if save_to_db and bt['trades'] > 0:
        # Simpan hasil
        dates = df['datetime'] if 'datetime' in df.columns else df.index
        start_date = str(dates.iloc[0])
        end_date = str(dates.iloc[-1])
        save_backtest_result(symbol, start_date, end_date, initial_capital, full_metrics)
    
    return {
        'standard': full_metrics,
        'walk_forward': wf_result,
        'monte_carlo': None  # akan kita perbaiki nanti
    }, None
