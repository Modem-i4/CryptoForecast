
import streamlit as st
st.set_page_config(page_icon="😋")

from prophet import Prophet
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from prophet.plot import add_changepoints_to_plot

#add things
df = st.session_state.btc_price
config = st.session_state.config

st.header("Simple modelling")
st.markdown("#__Modeling and decomposition with price regressors only__")

st.subheader("Shifted Dataframe")
df[config['all_indicators']] = df[config['all_indicators']].shift(1)
df.dropna(inplace=True)
df

indicators = []

st.subheader("Correlations")
correlation_matrix = df.corr()['y'].drop(['ds', 'y', 'High', 'Low'])

st.bar_chart(correlation_matrix)

st.subheader("Modelling")

st.write("Available indicators :")
ind_cols = st.columns(5)
for idx, checkbox in enumerate(config['all_indicators']):
    with ind_cols[idx]:
        if st.checkbox(checkbox, value=checkbox in ['MACDh', 'MACDs']) : indicators.append(checkbox)

m = Prophet(interval_width=0.95, seasonality_mode='multiplicative') #multiplicative, additive

m.add_country_holidays(country_name='US')
for reg in indicators : m.add_regressor(reg) 

train_slider = st.slider('Select train sample size', min_value=0.6, max_value=1.0, value=0.8)
train_size = int(train_slider * len(df))
train_df = df.iloc[:train_size]
test_df = df.iloc[train_size:]

m.seasonality_strength = 0.5

m.fit(train_df)


df_to_predict = df # test_df

forecast = m.predict(df_to_predict)

metrics_cols = st.columns(3)
with metrics_cols[0]: st.write(f'MAPE = {mean_absolute_error(df_to_predict.y, forecast.yhat)}')
with metrics_cols[1]: st.write(f'RMSE = {np.sqrt(mean_squared_error(df_to_predict.y, forecast.yhat))}')
with metrics_cols[2]: st.write(f'R^2 = {r2_score(df_to_predict.y, forecast.yhat)}')

fig1 = m.plot(forecast)
ax = fig1.gca()
ax.scatter(test_df['ds'], test_df['y'], color='red', label='Actual', s=10)
ax.legend()
if st.checkbox("Show changepoints"):
    add_changepoints_to_plot(fig1.gca(), m, forecast)
fig1
fig2 = m.plot_components(forecast, figsize=(10,8), weekly_start=1, uncertainty=1)
fig2
