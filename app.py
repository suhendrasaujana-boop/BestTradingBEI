import streamlit as st
from data import get_data, add_indicators

st.title("Robot Saham Indonesia")

symbol = st.text_input("Kode Saham", "BBCA.JK", key="symbol_input")

timeframe = st.selectbox(
    "Timeframe",
    ["5m","15m","30m","60m","1d"],
    key="timeframe_select"
)

df = get_data(symbol, timeframe)

if df.empty:
    st.warning("Data kosong")
else:
    df = add_indicators(df)
    st.line_chart(df[['close','ema20','ema50']])
    st.write(df.tail())
from data import get_data, add_indicators, calculate_score
df = add_indicators(df)
score = calculate_score(df)

if score >= 80:
    signal = "STRONG BUY"
elif score >= 60:
    signal = "BUY"
elif score >= 40:
    signal = "WAIT"
else:
    signal = "SELL"

st.metric("Score", score)
st.metric("Signal", signal)
from data import multi_timeframe_analysis
st.subheader("Multi Timeframe Analysis")

mtf = multi_timeframe_analysis(symbol)

st.write(mtf)
avg_score = sum(mtf.values()) / len(mtf)

if avg_score >= 80:
    final_signal = "STRONG BUY"
elif avg_score >= 60:
    final_signal = "BUY"
elif avg_score >= 40:
    final_signal = "WAIT"
else:
    final_signal = "SELL"

st.metric("Final Score", round(avg_score,2))
st.metric("Final Signal", final_signal)
