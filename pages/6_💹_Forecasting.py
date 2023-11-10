

import streamlit as st
st.set_page_config(page_icon="😋")

from prophet import Prophet
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, time

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score



#loading
df = st.session_state.btc_price
config = st.session_state.config

st.header("Forecasting")
st.markdown("#__Using your last trained model__")

data_cols = st.columns(5)
with data_cols[0]: st.markdown("**#Data**")
with data_cols[1]: st.write("Default data")
with data_cols[2]: use_own_data = st.toggle("data_touse", label_visibility='hidden')
with data_cols[3]: st.write("Own data")

mode_cols = st.columns(5)
with mode_cols[0]: st.markdown("**#Mode**")
with mode_cols[1]: st.write("Backtesting")
with mode_cols[2]: mode_forecasting = st.toggle("forecasting_mode", label_visibility='hidden')
with mode_cols[3]: st.write("Forecasting")

mode_cols = st.columns(5)
with mode_cols[0]: st.markdown("**#Forecast**")
with mode_cols[1]: st.write("Price only")
with mode_cols[2]: forecast_low_high = st.toggle("forecast_low_high", label_visibility='hidden')
with mode_cols[3]: st.write("Price, High/Low")

if use_own_data :
    news_aggregated = pd.read_json(f'data/news_aggregated_{config["interval"]}.json')
    news_evaluated= pd.read_json('data/news_evaluated.json')
else :
    news_aggregated = pd.read_json(f'default/news_aggregated_{config["interval"]}.json')
    news_evaluated= pd.read_json('default/news_evaluated.json')

news_aggregated = news_aggregated.reset_index().rename(columns={'index': 'ds'})
df = pd.merge(df, news_aggregated, on='ds')



info_cols_3 = st.columns(3)
if mode_forecasting :
    with info_cols_3[1] :
        st.subheader("Forecasting")
    selected_datetime = df.iloc[-1].ds
else :
    with info_cols_3[1] :
        st.subheader("Backtesting")
    pred_cols = st.columns(3)
    with pred_cols[0] : st.subheader('Choose date to predict from: ')
    with pred_cols[1] : selected_date = st.date_input('Select a date', value=df['ds'].iloc[len(df)-5], min_value=config['date_from'], max_value=df['ds'].iloc[len(df)-2])
    with pred_cols[2] : selected_time = st.time_input('Select a time', step=3600, value=time(0))
    selected_datetime = datetime.combine(selected_date, selected_time) 


next_candles = pd.DataFrame([{'ds': df['ds'].iloc[-1] + pd.Timedelta(config['interval'])}])
regressors = config['all_classes'] + config['all_indicators']
next_candles[regressors] = df[regressors].iloc[-1]

df[regressors] = df[regressors].shift(1)
next_candles = pd.concat([next_candles, pd.DataFrame([df.iloc[-1]])]).sort_values(by='ds').reset_index()
df.dropna(inplace=True)

if mode_forecasting :
    train_df = df
    test_df = next_candles
else:
    train_df = df#df[df['ds'] <= selected_datetime].sort_values(by='ds')
    test_df = df[df['ds'] >= selected_datetime].sort_values(by='ds').reset_index()


forecast_fields = ['Low', 'High', 'Price'] if forecast_low_high else ['Price']
m = {}
for f_field in forecast_fields :
    m[f_field] = Prophet(interval_width=0.8, seasonality_mode='multiplicative',  weekly_seasonality=False) #multiplicative, additive

    for reg in config['m_classes'] + config['m_indicators'] : m[f_field].add_regressor(reg)
    df_to_fit = train_df.rename(columns={'y': 'Price'}).rename(columns={f_field: 'y'})
    m[f_field].fit(df_to_fit)



if mode_forecasting and False : # Forecasting regressors test
    prediction_horizon = 1
    data_cols = st.columns([2,2,1,2,3])
    with data_cols[0]: st.markdown("**#Forecast regressors**")
    with data_cols[1]: st.write("No, use actual data")
    with data_cols[2]: forecast_regressors = st.toggle("forecast_regressors", label_visibility='hidden')
    with data_cols[3]: st.write("Yes, try to predict")
    if forecast_regressors:
        prediction_horizon = 5
        test_df = m['Price'].make_future_dataframe(prediction_horizon, freq=config['interval'],  include_history=False)
        for indicator in config['m_indicators']:
            indicator_m = Prophet(interval_width=0.95, seasonality_mode='multiplicative')
            indicator_df = df[['ds', indicator]].rename(columns={indicator: 'y'})
            indicator_m.fit(indicator_df)
            test_df[indicator] = indicator_m.predict(indicator_df)['yhat']
        test_df[config['all_classes']] = 0
    test_df.fillna(0, inplace=True)


if mode_forecasting :
    last_y = train_df['y'].iloc[-1]
else :
    last_y = test_df['y'].iloc[0]
df_to_predict = test_df
forecasts = pd.DataFrame(df_to_predict.ds)
for f_field in forecast_fields :
    forecast = m[f_field].predict(df_to_predict)
    forecasts[f_field] = forecast.yhat
    if f_field == 'Price': forecasts[['yhat_lower', 'yhat_upper']] = forecast[['yhat_lower', 'yhat_upper']]
if not forecast_low_high:
    forecasts['Low'] = forecasts['Price']
    forecasts['High'] = forecasts['Price']

shift_cols = ['Price', 'yhat_lower', 'yhat_upper', 'Low', 'High']
forecasts[shift_cols] = forecasts[shift_cols] - forecasts['Price'].iloc[0] + last_y

#m.plot(forecast)
if not mode_forecasting :
    metrics_cols = st.columns(3)
    with metrics_cols[0]: st.write(f'MAPE = {mean_absolute_error(df_to_predict.y, forecasts.Price)}')
    with metrics_cols[1]: st.write(f'RMSE = {np.sqrt(mean_squared_error(df_to_predict.y, forecasts.Price))}')
    with metrics_cols[2]: st.write(f'R^2 = {r2_score(df_to_predict.y, forecasts.Price)}')
st.write("#Forecasts:")
forecasts
fig = make_subplots(rows=3, cols=1, row_heights=[5,1,1], shared_xaxes=True, vertical_spacing=0.1)

fig.add_trace(go.Candlestick(x=df['ds'],
                                open=df['y'].shift(1),
                                high=df['High'],
                                low=df['Low'],
                                close=df['y'],                             
                                increasing_line_color='lightblue',
                                decreasing_line_color='lightgray',
                                name="Actual",
                                increasing_line_width=5, decreasing_line_width=5))

fig.add_trace(go.Candlestick(x=forecasts['ds'],
                                     open=forecasts['Price'].shift(1),
                                     high=forecasts['High'],
                                     low=forecasts['Low'],
                                     close=forecasts['Price'],
                                     name="Forecast"))

df_ext = pd.concat([df, next_candles[['ds'] + regressors]])
for m_class in config['m_classes'] :
    fig.add_trace(go.Scatter(x=df_ext['ds'], y=df_ext[m_class], mode='lines', name=m_class), row=2, col=1)

m_indicators_filtered = [item for item in config['m_indicators'] if item != 'Volume' and item != 'RSI']
for m_indicator in m_indicators_filtered :
    fig.add_trace(go.Scatter(x=df_ext['ds'], y=df_ext[m_indicator], mode='lines', name=m_indicator), row=3, col=1)

fig.update_layout(xaxis_rangeslider_visible=False, showlegend=False)
if mode_forecasting :
    range_x=[
        selected_datetime - pd.Timedelta(config['interval'])*15, 
        selected_datetime + pd.Timedelta(config['interval'])]
else:
    range_x=[selected_datetime - pd.Timedelta(config['interval'])*15, forecasts['ds'].iloc[-1]]
fig.update_xaxes(range=range_x)
range_y = train_df[(train_df['ds'] >= range_x[0]) & (train_df['ds'] <= range_x[1])]

fig.update_yaxes(range=[
    min(range_y['Low'].min(), forecasts['Low'].min(), test_df['Low'].min())-100,
    max(range_y['High'].max(), forecasts['High'].max(), test_df['High'].max())+100
], row=1, col=1)
fig.update_yaxes(range=[range_y[config['m_classes']].min(axis=1).min(), range_y[config['m_classes']].max(axis=1).max()], row=2, col=1)
fig.update_yaxes(range=[range_y[m_indicators_filtered].min(axis=1).min(),range_y[m_indicators_filtered].max(axis=1).max()], row=3, col=1)
st.plotly_chart(fig)

events_cols = st.columns(2)
past_events = news_evaluated[news_evaluated['date'] <= selected_datetime].set_index('date')
upcoming_events = news_evaluated[news_evaluated['date'] > selected_datetime].set_index('date').sort_index()

columns_order = ['class', 'gross_effect', 'label', 'title', 'description', 'label_score', 'class_score']

with events_cols[0] :
    st.subheader("Past events")
    past_events[:15][columns_order]
with events_cols[1] :
    st.subheader("Upcomming events")
    upcoming_events[:15][columns_order]

center_cols = st.columns([1,1.5,1])
with center_cols[1]: st.write('Fullscreening the tables might be helpful')
