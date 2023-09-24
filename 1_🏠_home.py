from collections import namedtuple
import altair as alt
import math
import pandas as pd
import streamlit as st
import yfinance as yf
from streamlit_option_menu import option_menu

with st.sidebar :
    selected = option_menu(
        menu_title = "aa",
        options = ["a", "b"])

tickedData = yf.Ticker('GOOGL')
tickedData.calendar
#tickerDf = tickedData.history(period='1d', start='10.09.2023', end='23.09.2023')
tickerDf = tickedData.history(period='1d', start='2023-09-20', end='2023-09-24')

st.line_chart(tickerDf.Close)