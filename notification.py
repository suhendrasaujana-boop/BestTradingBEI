import os
import asyncio
from datetime import datetime
from data import scan_saham, get_data, add_indicators, calculate_entry_sl_tp

TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", None)
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", None)

if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
    print("⚠️ Token/chat_id Telegram belum disetel. Notifikasi tidak akan dikirim.")

async def send_telegram_message(message: str):
    """Mengirim pesan ke Telegram menggunakan python-telegram-bot."""
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        return
    
    try:
        from telegram import Bot
        bot = Bot(token=TELEGRAM_TOKEN)
        await bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=message)
        print(f"[{datetime.now()}] Notifikasi terkirim.")
    except Exception as e:
        print(f"Gagal mengirim notifikasi: {e}")

def generate_sniper_message(stock_code: str, entry, sl, tp, rr, setup_name, confidence):
    """Format pesan sniper dengan entry, SL, TP, RR."""
    message = f"""🔥 Sniper Alert: {stock_code}
Setup: {setup_name}
Confidence: {confidence}/100
Entry: Rp{entry:,.0f}
Stop Loss: Rp{sl:,.0f}
Take Profit: Rp{tp:,.0f}
Risk/Reward: 1:{rr:.1f}
"""
    return message

def scan_and_notify(capital=100_000_000, risk_percent=2):
    """
    Scan semua saham prioritas dan kirim notifikasi Telegram jika ada sinyal SNIPER baru.
    """
    # Ambil hasil scan
    results = scan_saham()
    if not results:
        return
    
    for row in results:
        score = int(row['Score'])
        if score >= 80 and "BUY" in row['Sinyal']:
            # Dapatkan detail entry/SL/TP
            try:
                df = get_data(row['Kode'], "1d")
                if df.empty:
                    continue
                df = add_indicators(df)
                entry, sl, tp, shares, rr, setup_name, conf, _ = calculate_entry_sl_tp(
                    df, capital=capital, risk_percent=risk_percent
                )
                if entry is None:
                    continue
                
                # Format pesan
                msg = generate_sniper_message(
                    stock_code=row['Kode'],
                    entry=entry,
                    sl=sl,
                    tp=tp,
                    rr=rr,
                    setup_name=setup_name,
                    confidence=conf
                )
                
                # Kirim pesan (async)
                asyncio.run(send_telegram_message(msg))
                print(f"Notifikasi dikirim untuk {row['Kode']}")
            except Exception as e:
                print(f"Error saat mengirim notifikasi {row['Kode']}: {e}")
