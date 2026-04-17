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
