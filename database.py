import sqlite3
import pandas as pd
from datetime import datetime, timedelta

DB_PATH = "market_data.db"

def get_connection():
    """Buka koneksi ke database SQLite."""
    conn = sqlite3.connect(DB_PATH)
    return conn

def init_db():
    """Buat tabel daily_prices jika belum ada."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS daily_prices (
            symbol TEXT NOT NULL,
            date TEXT NOT NULL,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            volume REAL,
            PRIMARY KEY (symbol, date)
        )
    ''')
    conn.commit()
    conn.close()
    print("✅ Database siap.")

def store_data(symbol, df):
    """
    Simpan dataframe OHLC ke database.
    df harus memiliki kolom: datetime/open/high/low/close/volume (atau setara).
    """
    if df.empty:
        return
    
    conn = get_connection()
    cursor = conn.cursor()
    
    # Pastikan kolom sesuai
    df_copy = df.copy()
    if 'datetime' in df_copy.columns:
        df_copy['date'] = pd.to_datetime(df_copy['datetime']).dt.strftime('%Y-%m-%d')
    elif 'date' in df_copy.columns:
        df_copy['date'] = pd.to_datetime(df_copy['date']).dt.strftime('%Y-%m-%d')
    else:
        df_copy['date'] = pd.to_datetime(df_copy.index).strftime('%Y-%m-%d')
    
    # Normalisasi nama kolom ke lowercase
    df_copy.columns = [c.lower() for c in df_copy.columns]
    
    inserted = 0
    for _, row in df_copy.iterrows():
        try:
            cursor.execute('''
                INSERT OR REPLACE INTO daily_prices (symbol, date, open, high, low, close, volume)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                symbol,
                row['date'],
                row.get('open', 0),
                row.get('high', 0),
                row.get('low', 0),
                row.get('close', 0),
                row.get('volume', 0)
            ))
            inserted += 1
        except Exception as e:
            print(f"Gagal insert {symbol} {row['date']}: {e}")
    
    conn.commit()
    conn.close()
    print(f"💾 {symbol}: {inserted} baris disimpan ke database.")

def load_data(symbol, start_date=None, end_date=None):
    """
    Ambil data OHLC dari database.
    Return dataframe dengan kolom: date, open, high, low, close, volume.
    """
    conn = get_connection()
    
    query = "SELECT date, open, high, low, close, volume FROM daily_prices WHERE symbol = ?"
    params = [symbol]
    
    if start_date:
        query += " AND date >= ?"
        params.append(start_date)
    if end_date:
        query += " AND date <= ?"
        params.append(end_date)
    
    query += " ORDER BY date ASC"
    
    df = pd.read_sql_query(query, conn, params=params)
    conn.close()
    
    if df.empty:
        return df
    
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date')
    return df

def get_last_date(symbol):
    """Dapatkan tanggal terakhir yang tersedia di database untuk suatu saham."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT MAX(date) FROM daily_prices WHERE symbol = ?", (symbol,))
    result = cursor.fetchone()
    conn.close()
    return result[0] if result[0] else None

def update_daily_data(symbols=None):
    """
    Update data harian dengan jeda aman (2 detik per saham).
    Hanya mengunduh data yang belum ada di database.
    """
    from data import get_data as fetch_from_yahoo
    import time as time_module
    
    if symbols is None:
        symbols = [
            "BBCA.JK", "BBRI.JK", "BMRI.JK", "BBNI.JK",
            "TLKM.JK", "ASII.JK", "UNTR.JK", "ICBP.JK",
            "INDF.JK", "KLBF.JK", "SMGR.JK", "CTRA.JK",
            "SMRA.JK", "PTBA.JK", "CPIN.JK", "GOTO.JK",
            "MDKA.JK", "ADRO.JK", "ANTM.JK", "AKRA.JK",
            "BRIS.JK", "INCO.JK", "ITMG.JK", "JPFA.JK",
            "MAPI.JK", "MEDC.JK", "PGAS.JK", "TOWR.JK",
            "EXCL.JK", "ISAT.JK", "AMMN.JK", "BYAN.JK",
            "TPIA.JK", "DSSA.JK", "CUAN.JK", "ADMR.JK",
            "AADI.JK", "PGEO.JK", "BRPT.JK", "ESSA.JK",
        ]
    
    today = datetime.now().strftime('%Y-%m-%d')
    updated = 0
    skipped = 0
    
    for i, sym in enumerate(symbols):
        try:
            last_date = get_last_date(sym)
            if last_date and last_date >= today:
                skipped += 1
                continue  # Sudah punya data hari ini
            
            print(f"[{i+1}/{len(symbols)}] Mengunduh {sym}...")
            df = fetch_from_yahoo(sym, "1d")
            if not df.empty:
                store_data(sym, df)
                updated += 1
            
            # Jeda 2 detik antar request (40 saham = 80 detik)
            if i < len(symbols) - 1:
                time_module.sleep(2)
                
        except Exception as e:
            print(f"⚠️ Gagal update {sym}: {e}")
            # Jika kena rate limit, tunggu lebih lama
            if "rate" in str(e).lower() or "limit" in str(e).lower():
                print("⏳ Rate limit terdeteksi, menunggu 60 detik...")
                time_module.sleep(60)
    
    print(f"✅ {updated} saham diperbarui, {skipped} saham sudah ada di database.")
    return updated

# Inisialisasi database saat file di-import
init_db()
