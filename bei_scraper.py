from beisfinder import get_stock_list

def get_all_bei_tickers():
    """Ambil semua kode saham BEI."""
    stocks = get_stock_list()
    return stocks
