import os
import time
import pickle
import pandas as pd
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from data import get_data, add_indicators, detect_high_quality_setup

CACHE_DIR = "cache"

def _ensure_cache_dir():
    """Buat folder cache jika belum ada."""
    if not os.path.exists(CACHE_DIR):
        os.makedirs(CACHE_DIR)

def _get_cache_filename():
    """Nama file cache berdasarkan tanggal hari ini."""
    today = datetime.now().strftime("%Y-%m-%d")
    return os.path.join(CACHE_DIR, f"{today}.pkl")

def _load_cache():
    """Load cache dari file pickle."""
    cache_file = _get_cache_filename()
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'rb') as f:
                data = pickle.load(f)
                if isinstance(data, dict) and len(data) >= 30:
                    print(f"📦 Cache ditemukan: {len(data)} saham")
                    return data
        except Exception as e:
            print(f"⚠️ Gagal load cache: {e}")
    return None

def _save_cache(data):
    """Simpan data ke cache pickle."""
    _ensure_cache_dir()
    cache_file = _get_cache_filename()
    try:
        with open(cache_file, 'wb') as f:
            pickle.dump(data, f)
        print(f"💾 Cache disimpan: {len(data)} saham")
    except Exception as e:
        print(f"⚠️ Gagal simpan cache: {e}")

def _fetch_single(symbol):
    """
    Ambil data satu saham.
    Return (symbol, df) atau None jika gagal.
    """
    try:
        df = get_data(symbol, "1d")
        if df is None or df.empty or len(df) < 30:
            return None
        df = add_indicators(df)
        return (symbol, df)
    except Exception as e:
        print(f"⚠️ Gagal fetch {symbol}: {e}")
        return None

def _fetch_all_parallel(symbols, max_workers=5, timeout=15):
    """
    Ambil semua saham secara paralel.
    max_workers=5 agar tidak kena rate limit Yahoo Finance.
    timeout=15 detik per saham.
    """
    data_dict = {}
    total = len(symbols)
    completed = 0
    
    print(f"🔄 Mengambil {total} saham ({max_workers} paralel)...")
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_symbol = {
            executor.submit(_fetch_single, sym): sym 
            for sym in symbols
        }
        
        for future in as_completed(future_to_symbol):
            symbol = future_to_symbol[future]
            completed += 1
            
            try:
                result = future.result(timeout=timeout)
                if result is not None:
                    sym, df = result
                    data_dict[sym] = df
                else:
                    print(f"  [{completed}/{total}] {symbol}: Data tidak cukup")
            except Exception as e:
                print(f"  [{completed}/{total}] {symbol}: Error - {e}")
            
            # Progress setiap 10 saham
            if completed % 10 == 0:
                print(f"  Progress: {completed}/{total} ({len(data_dict)} berhasil)")
    
    return data_dict

def scan_saham_fast(symbols=None):
    """
    Scan saham dengan caching.
    Return list of dict dengan keys: Kode, Score, Setup, Sinyal, Harga.
    """
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
    
    start = time.time()
    
    # Cek cache dulu
    data_dict = _load_cache()
    
    if data_dict is None or len(data_dict) < 30:
        print("📡 Cache tidak tersedia, fetching data dari Yahoo Finance...")
        data_dict = _fetch_all_parallel(symbols, max_workers=5, timeout=15)
        
        if len(data_dict) >= 30:
            _save_cache(data_dict)
        else:
            print(f"⚠️ Hanya {len(data_dict)} saham berhasil di-fetch (minimal 30 untuk cache)")
    else:
        print("✅ Menggunakan data dari cache")
    
    # Deteksi setup
    print(f"\n🔍 Menganalisis setup untuk {len(data_dict)} saham...")
    results = []
    
    for symbol, df in data_dict.items():
        try:
            setup, quality, msg = detect_high_quality_setup(df)
            if quality >= 60:
                signal = "🔥 BUY" if "BUY" in setup else "⏸️ HOLD"
                results.append({
                    "Kode": symbol.replace(".JK", ""),
                    "Score": f"{quality:.0f}",
                    "Setup": msg[:50],
                    "Sinyal": signal,
                    "Harga": f"Rp{df.iloc[-1]['close']:,.0f}"
                })
        except Exception as e:
            print(f"⚠️ Gagal deteksi setup {symbol}: {e}")
    
    # Sort by score
    results.sort(key=lambda x: int(x['Score']), reverse=True)
    
    elapsed = time.time() - start
    print(f"\n✅ Scanner selesai dalam {elapsed:.1f} detik. {len(results)} sinyal ditemukan.")
    
    return results[:10] if len(results) >= 10 else results


# Untuk testing mandiri
if __name__ == "__main__":
    print("=" * 60)
    print("🧪 TEST SCANNER ENGINE")
    print("=" * 60)
    
    results = scan_saham_fast()
    
    if results:
        print(f"\n{'='*60}")
        print(f"TOP {len(results)} SAHAM:")
        print(f"{'='*60}")
        for i, row in enumerate(results, 1):
            print(f"{i:2d}. {row['Kode']:<8} | Score: {row['Score']:<4} | {row['Sinyal']:<10} | {row['Setup']}")
    else:
        print("\n❌ Tidak ada sinyal yang ditemukan.")
