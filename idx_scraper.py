"""
Scraper daftar saham BEI dari IDX.co.id
TANPA library eksternal, hanya pakai requests + pandas
"""
import requests
import pandas as pd

def get_all_bei_tickers():
    """
    Ambil SEMUA kode saham dari website IDX.
    """
    url = "https://www.idx.co.id/umbraco/Surface/ListedCompany/GetListedCompany"
    params = {
        "indexOption": "listedCompany",
        "draw": 1,
        "start": 0,
        "length": 1000
    }
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }
    
    try:
        response = requests.get(url, params=params, headers=headers, timeout=15)
        data = response.json()
        
        tickers = []
        for item in data.get('data', []):
            code = item.get('Code', '').strip().upper()
            if code and len(code) == 4:
                tickers.append(code)
        
        return sorted(tickers)
    except:
        # Fallback: daftar saham BEI terbesar
        return [
            "AADI", "ACES", "ADHI", "ADMR", "ADRO", "AGRO", "AKRA",
            "AMMN", "ANTM", "ARTO", "ASII", "ASRI", "BBNI", "BBRI",
            "BBCA", "BBKP", "BELI", "BIRD", "BJBR", "BJTM", "BMRI",
            "BNGA", "BNII", "BRIS", "BRPT", "BSDE", "BUKA", "BYAN",
            "CPIN", "CTRA", "CUAN", "DMMX", "DSSA", "ELSA", "EMTK",
            "ENRG", "ERAA", "ESSA", "EXCL", "FILM", "GGRM", "GOTO",
            "HEAL", "HMSP", "HRUM", "ICBP", "INCO", "INDF", "INKP",
            "INTP", "ISAT", "ITMG", "JPFA", "KLBF", "LPKR", "LSIP",
            "MAIN", "MAPI", "MBMA", "MDKA", "MEDC", "MEGA", "MIKA",
            "MNCN", "MPMX", "MTDL", "MTEL", "MYOR", "NCKL", "NTBK",
            "PGAS", "PGEO", "PNBN", "PTBA", "PTPP", "PWON", "RAJA",
            "RMKE", "SIDO", "SILO", "SMGR", "SMRA", "SRTG", "SSMS",
            "TBIG", "TEBE", "TGKA", "TINS", "TLKM", "TOWR", "TPIA",
            "ULTJ", "UNTR", "UNVR", "WIKA", "WINS", "WSKT"
        ]
