import streamlit as st
from data import get_data

st.title("Robot Saham Indonesia")

symbol = st.text_input("Kode Saham", "BBCA.JK")

timeframe = st.selectbox(
    "Timeframe",
    ["5m","15m","30m","60m","1d"]
)

df = get_data(symbol, timeframe)

if df.empty:
    st.warning("Data kosong")
else:
    st.line_chart(df['close'])
    st.write(df.tail())
