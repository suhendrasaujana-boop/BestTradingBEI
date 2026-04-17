import streamlit as st
from data import get_data

st.title("Robot Saham Indonesia")

symbol = st.text_input("Kode Saham", "BBCA.JK")

timeframe = st.selectbox(
    "Timeframe",
    ["5m","15m","30m","60m","1d"]
)

st.write("DEBUG: sebelum ambil data")

df = get_data(symbol, timeframe)

st.write("DEBUG: setelah ambil data")

st.write("Dataframe shape:", df.shape)

st.write("Columns:", df.columns)

if not df.empty:
    st.write(df.tail())
