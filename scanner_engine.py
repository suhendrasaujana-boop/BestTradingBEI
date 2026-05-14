import os
import time
import pickle
import pandas as pd
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from data import get_data, add_indicators, detect_high_quality_setup

CACHE_DIR = "cache"

def _ensure_cache_dir():
    if not os.path.exists(CACHE_DIR):
        os.makedirs(CACHE_DIR)

def _get_cache_filename():
    today = datetime.now().strftime("%Y-%m-%d")
    return os.path.join(CACHE_DIR, f"{today}.pkl")

def _load_cache():
    """Baca cache harian (pickle)."""
    cache_file = _get_cache_filename()
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'rb') as f:
                data = pickle.load(f)
                if isinstance(data, dict) and len(data) >= 30:
                    return data
        except:
            pass
    return None

def _save_cache(data):
    """Simpan cache harian (pickle)."""
    _ensure_cache_dir()
    cache_file = _get_cache_filename()
    with open(cache_file, 'wb') as f:
        pickle.dump(data, f)

def _fetch_single(symbol):
    """Ambil data satu saham + indikator, return tuple (symbol, df) atau None."""
    try:
        df = get_data(symbol, "1d")
        if df.empty or len(df) < 30:
            return None
        df = add_indicators(df)
        return (symbol, df)
    except:
        return None

def _fetch_all_parallel(symbols, max_workers=10):
    """Ambil semua saham secara paralel, return dict {symbol: df}."""
    data_dict = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_fetch_single, sym): sym for sym in symbols}
        for future in as_completed(futures):
            result = future.result()
            if result is not None:
                symbol, df = result
                data_dict[symbol] = df
    return data_dict

def scan_saham_fast():
    """Scanner cepat dengan cache + threading."""
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
    
    # Cek cache harian
    data_dict = _load_cache()
    if data_dict is None or len(data_dict) < 30:
        # Fetch paralel
        data_dict = _fetch_all_parallel(symbols, max_workers=10)
        _save_cache(data_dict)
    
    # Proses deteksi setup
    results = []
    for symbol, df in data_dict.items():
        try:
            setup, quality, msg = detect_high_quality_setup(df)
            if quality >= 60:
                signal = "🔥 BUY" if "BUY" in setup else "⏸️ HOLD"
                results.append({
                    "Kode": symbol,
                    "Score": f"{quality:.0f}",
                    "Setup": msg[:40],
                    "Sinyal": signal,
                    "Harga": f"Rp{df.iloc[-1]['close']:,.0f}"
                })
        except:
            continue
    
    results.sort(key=lambda x: int(x['Score']), reverse=True)
    return results[:10]
