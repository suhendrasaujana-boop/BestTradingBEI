import os
import asyncio
from datetime import datetime
from data import scan_saham, get_data, add_indicators, calculate_entry_sl_tp

# Konfigurasi Telegram
TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", None)
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", None)

if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
    print("⚠️ Token/chat_id Telegram belum disetel. Notifikasi tidak akan dikirim.")
    print("   Set environment variable:")
    print("   export TELEGRAM_BOT_TOKEN='your_token'")
    print("   export TELEGRAM_CHAT_ID='your_chat_id'")


async def send_telegram_message(message: str, parse_mode: str = "HTML"):
    """Mengirim pesan ke Telegram."""
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print("❌ Token/Chat ID tidak disetel.")
        return False
    
    try:
        from telegram import Bot
        from telegram.error import TelegramError
        
        bot = Bot(token=TELEGRAM_TOKEN)
        await bot.send_message(
            chat_id=TELEGRAM_CHAT_ID, 
            text=message,
            parse_mode=parse_mode
        )
        print(f"✅ [{datetime.now().strftime('%H:%M:%S')}] Notifikasi terkirim.")
        return True
        
    except TelegramError as e:
        print(f"❌ Telegram error: {e}")
        return False
    except Exception as e:
        print(f"❌ Gagal mengirim: {e}")
        return False


def send_telegram_sync(message: str, parse_mode: str = "HTML"):
    """Wrapper sync untuk mengirim pesan Telegram."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # Dalam environment async, buat task
            asyncio.create_task(send_telegram_message(message, parse_mode))
        else:
            loop.run_until_complete(send_telegram_message(message, parse_mode))
    except RuntimeError:
        # Tidak ada event loop, buat baru
        asyncio.run(send_telegram_message(message, parse_mode))


def generate_sniper_message(stock_code: str, entry, sl, tp, rr, setup_name, confidence, signals=""):
    """Format pesan sniper dengan HTML formatting."""
    stars = "⭐" * min(5, confidence // 20 + 1)
    
    message = f"""<b>🔥 SNIPER ALERT: {stock_code}</b>
{stars}

<b>Setup:</b> {setup_name}
<b>Confidence:</b> {confidence}/100

<b>💰 Entry:</b> Rp{entry:,.0f}
<b>🛑 Stop Loss:</b> Rp{sl:,.0f}
<b>🎯 Take Profit:</b> Rp{tp:,.0f}

<b>📊 Risk/Reward:</b> 1:{rr:.1f}
<b>📉 Risk:</b> Rp{abs(entry-sl):,.0f} ({(abs(entry-sl)/entry*100):.1f}%)
<b>📈 Reward:</b> Rp{abs(tp-entry):,.0f} ({(abs(tp-entry)/entry*100):.1f}%)

{signals if signals else ''}
<b>⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</b>
"""
    return message


def generate_scan_summary(results):
    """Format ringkasan scan harian."""
    if not results:
        return "<b>📊 Daily Scan: Tidak ada sinyal kuat hari ini</b>"
    
    today = datetime.now().strftime('%Y-%m-%d')
    message = f"<b>📊 DAILY SCAN - {today}</b>\n"
    message += f"<b>Top {len(results)} Picks:</b>\n\n"
    
    for i, row in enumerate(results, 1):
        emoji = "🟢" if "BUY" in row['Sinyal'] else "🟡"
        message += f"{i}. {emoji} <b>{row['Kode']}</b> - Score: {row['Score']}\n"
        message += f"   {row['Setup'][:60]}\n"
        message += f"   Harga: {row['Harga']}\n\n"
    
    return message


def scan_and_notify(capital=100_000_000, risk_percent=2):
    """
    Scan semua saham dan kirim notifikasi Telegram.
    """
    print(f"\n🔍 [{datetime.now().strftime('%H:%M:%S')}] Memulai scan & notifikasi...")
    
    # Scan saham
    results = scan_saham()
    
    if not results:
        print("❌ Tidak ada hasil scan.")
        msg = "<b>📊 Daily Scan: Tidak ada sinyal kuat hari ini</b>"
        send_telegram_sync(msg)
        return
    
    # Kirim ringkasan
    summary = generate_scan_summary(results)
    send_telegram_sync(summary)
    
    # Kirim detail untuk sinyal SNIPER (Score >= 80)
    sniper_count = 0
    for row in results:
        score = int(row['Score'])
        if score >= 80 and "BUY" in row['Sinyal']:
            try:
                df = get_data(row['Kode'], "1d")
                if df.empty:
                    continue
                
                df = add_indicators(df)
                entry, sl, tp, shares, rr, setup_name, conf, signals = calculate_entry_sl_tp(
                    df, capital=capital, risk_percent=risk_percent
                )
                
                if entry is None:
                    continue
                
                # Format pesan sniper
                msg = generate_sniper_message(
                    stock_code=row['Kode'],
                    entry=entry,
                    sl=sl,
                    tp=tp,
                    rr=rr,
                    setup_name=setup_name,
                    confidence=conf,
                    signals=" | ".join(signals) if signals else ""
                )
                
                send_telegram_sync(msg)
                sniper_count += 1
                
            except Exception as e:
                print(f"⚠️ Error memproses {row['Kode']}: {e}")
    
    print(f"✅ Selesai: {len(results)} sinyal, {sniper_count} sniper alert terkirim.")


def send_test_message():
    """Kirim pesan test untuk verifikasi koneksi Telegram."""
    test_msg = f"""<b>🧪 TEST MESSAGE</b>
✅ Bot berfungsi dengan baik!
⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

📈 <b>BestTradingBEI</b> siap digunakan."""
    
    success = send_telegram_sync(test_msg)
    if success:
        print("✅ Test message terkirim!")
    else:
        print("❌ Gagal mengirim test message.")


def send_custom_alert(symbol, entry, sl, tp, setup_name, confidence, note=""):
    """Kirim alert kustom untuk satu saham."""
    rr = abs(tp - entry) / abs(entry - sl) if abs(entry - sl) > 0 else 0
    
    msg = generate_sniper_message(
        stock_code=symbol,
        entry=entry,
        sl=sl,
        tp=tp,
        rr=rr,
        setup_name=setup_name,
        confidence=confidence,
        signals=note
    )
    
    return send_telegram_sync(msg)


# Testing
if __name__ == "__main__":
    print("=" * 60)
    print("🧪 TEST NOTIFICATION")
    print("=" * 60)
    
    # Cek token
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print("\n⚠️ Token/Chat ID tidak disetel. Test dibatalkan.")
        print("\nUntuk testing, set environment variable:")
        print("  export TELEGRAM_BOT_TOKEN='123:abc'")
        print("  export TELEGRAM_CHAT_ID='-123456'")
    else:
        print(f"\n✅ Token: {TELEGRAM_TOKEN[:10]}...")
        print(f"✅ Chat ID: {TELEGRAM_CHAT_ID}")
        
        # Kirim test message
        print("\nMengirim test message...")
        send_test_message()
        
        # Test generate message
        print("\n📝 Contoh pesan sniper:")
        msg = generate_sniper_message(
            stock_code="BBRI",
            entry=5000,
            sl=4900,
            tp=5300,
            rr=3.0,
            setup_name="SMART_MONEY_COMBO_BUY",
            confidence=85,
            signals="Bullish FVG | Bullish OB"
        )
        print(msg)
