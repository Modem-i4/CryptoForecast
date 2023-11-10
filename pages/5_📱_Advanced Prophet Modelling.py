
import streamlit as st
st.set_page_config(page_icon="😋")

from prophet import Prophet
import pandas as pd
import pandas_ta as ta
import numpy as np

from datetime import datetime, timedelta

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

#loading
df = st.session_state.btc_price
config = st.session_state.config
st.session_state.pop('m', None)

st.header("Advanced modelling")
st.markdown("#__Modeling and decomposition with price and sentimental regressors__")

st.subheader("News")

data_cols = st.columns([1.2,1,1,1,3])
with data_cols[0]: st.write("Default data")
with data_cols[1]: use_own_data = st.toggle("data_touse", label_visibility='hidden')
with data_cols[2]: st.write("Own data")

if use_own_data :
    news_aggregated = pd.read_json(f'data/news_aggregated_{config["interval"]}.json')
else :
    news_aggregated = pd.read_json(f'default/news_aggregated_{config["interval"]}.json')

with data_cols[4]: 
    if st.checkbox("Apply news cummulative effect"):
        for i in range(1, 3) : news_aggregated += news_aggregated.shift(i)/(1.5**i)
        news_aggregated.fillna(0, inplace=True)

news_aggregated = news_aggregated.reset_index().rename(columns={'index': 'ds'})
##news_aggregated = news_aggregated.shift(1)
news_aggregated

st.subheader("Shifted data")
df = pd.merge(df, news_aggregated, on='ds')

regressors = config['all_classes'] + config['all_indicators']
df[regressors] = df[regressors].shift(1)
df.dropna(inplace=True)
df

config['m_indicators'] = []
st.subheader("Correlations")
correlation_matrix = df.corr()['y'].drop(['ds', 'y', 'High', 'Low'])

st.bar_chart(correlation_matrix)

st.subheader("Modelling")
st.write("Available indicators:")
ind_cols = st.columns(5)
for idx, checkbox in enumerate(config['all_indicators']):
    with ind_cols[idx]:
        if st.checkbox(checkbox, value=checkbox in ['MACDh', 'MACDs']) : config['m_indicators'].append(checkbox)


config['m_classes']  = []
st.write("Available classes:")
class_cols = st.columns(5)
for idx, checkbox in enumerate(config['all_classes']):
    with class_cols[idx]:
        if st.checkbox(checkbox, value=True) : config['m_classes'].append(checkbox)

st.session_state.config = config

train_slider = st.slider('Select train sample size', min_value=0.6, max_value=1.0, value=0.8)
train_size = int(train_slider * len(df))
train_df = df.iloc[:train_size]
test_df = df.iloc[train_size:]

n_changepoints = 0 if st.toggle("Use pure news/indicators effect(disabling changepoints)") else 25

m = Prophet(interval_width=0.95, seasonality_mode='multiplicative', n_changepoints=n_changepoints, weekly_seasonality=False) #multiplicative, additive
for reg in config['m_indicators'] + config['m_classes'] : m.add_regressor(reg) 
#m.add_country_holidays(country_name='US')
m.fit(train_df)

df_to_predict = df# df #// test_df

#df_to_predict = df[(df[all_config['m_classes'] ] != 0).any(axis=1)]

forecast = m.predict(df_to_predict)

metrics_cols = st.columns(3)
with metrics_cols[0]: st.write(f'MAPE = {mean_absolute_error(df_to_predict.y, forecast.yhat)}')
with metrics_cols[1]: st.write(f'RMSE = {np.sqrt(mean_squared_error(df_to_predict.y, forecast.yhat))}')
with metrics_cols[2]: st.write(f'R^2 = {r2_score(df_to_predict.y, forecast.yhat)}')


fig1 = m.plot(forecast)
ax = fig1.gca()
ax.scatter(test_df['ds'], test_df['y'], color='red', label='Actual', s=10)
ax.legend()
fig1

st.bar_chart(news_aggregated, x='ds')

fig2 = m.plot_components(forecast, figsize=(10,8), weekly_start=1, uncertainty=1)
fig2
