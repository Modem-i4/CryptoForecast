import pandas as pd
import pandas_ta as ta
import streamlit as st
import yfinance as yf
from datetime import datetime, timedelta
import plotly.graph_objects as go

st.set_page_config(
    page_title="Forecast",
    page_icon="😋"
    )

st.markdown("""
            # Welcome to Crypto Forecast app!
App works for __BTC__, __ETH__, __XRP__ on __1D__, __1H__, __15m__ and __5m__ timeframes, for any selected period
## The functionality
### Modeling
* Collecting price data and calculating _MACD_, _MACDh_, _MACDs_, _RSI_
* Analyzing, customizing simple model regressors and test-train seletion
* Scraping news on specified topic
* NLI analysis with _financial-roberta-large-sentiment_ and _bart_large_mnli_, aggregating results and removing outliers
* Advanced modeling with both _technical_ and _sentiment_ regressors, evaluating model to achieve the best result
### Application of the model
* Forecasting next candle
* Backtesting on a range
* Step-by-step simulation with applying trading strategy and positions visualuzation
* Automatic trading simulation with iterations logging and statistics collecting

##### Read the [paper.docx](https://github.com/Modem-i4/CryptoForecast/blob/master/Paper.docx) to learn more about it!
""")

st.markdown("# Price data collection")
if 'config' in st.session_state :
    config = st.session_state.config
else :
    config = {"ticker": "BTC", "interval": "1h", "date_from": datetime(2023,1,1), "date_to": datetime(2023,10,19), "query": "Crypto", "all_classes": ["Regulation", "Global Events", "Forecasts", "Technology", "Crime"], "m_classes": ["Regulation", "Global Events", "Forecasts", "Technology", "Crime"], 'all_indicators' : ['MACD', 'MACDh', 'MACDs', 'RSI', 'Volume'], "m_indicators": ['MACD', 'MACDh'], 'time_intervals': ["5m", "15m", "1h", "1d"]}
st.session_state.pop('m', None)

col_from, col_to, col_ticker = st.columns(3)
with col_from :
    config['date_from'] = st.date_input("From", value=config['date_from'], max_value=datetime.now())
with col_to :
    config['date_to'] = st.date_input("To", value=config['date_to'], max_value=datetime.now())
with col_ticker :
    config['ticker'] = st.selectbox("Select coin:", ["BTC", "ETH", "XRP"])

config['interval'] = st.selectbox("Select a time interval:", config['time_intervals'], index=config['time_intervals'].index(config['interval']))

st.session_state.config = config

macd_params = {
        'fast': 12,
        'slow': 26,
        'signal': 9
    }

tickedData = yf.Ticker(f'{config["ticker"]}-USD')
tickerDf = tickedData.history(interval=config['interval'], start=config['date_from'], end=(config['date_to']+pd.Timedelta(config['interval'])))
if len(tickerDf) == 0 :
    st.write(f"##Reduce the requested time interval if you want to have data for each {config['interval']}")
    config['date_from'] = config['date_to'] - timedelta(days=54)
    st.rerun()

fig = go.Figure(go.Scatter(x=tickerDf.index, y=tickerDf.Close, mode='lines', name='Close'))
fig.update_layout(yaxis_range=[tickerDf.Close.min(), tickerDf.Close.max()]) 
st.plotly_chart(fig)

df = tickerDf[['Close', 'High', 'Low', 'Volume']].reset_index().rename(columns={'Date': 'ds', 'Datetime': 'ds', 'Close': 'y'})

macd = ta.macd(df.y, signal=macd_params['signal'], slow=macd_params['slow'], fast=macd_params['fast'])
macd = macd.rename(columns={macd.columns[0]: 'MACD', macd.columns[1]: 'MACDh', macd.columns[2]: 'MACDs'})

df = pd.concat([df, macd], axis=1)
df['RSI'] = ta.rsi(df.y)

df.ds = df.ds.dt.tz_convert(None)

df_ordered = df.sort_index(ascending=False)
df_ordered

st.session_state.btc_price = df
