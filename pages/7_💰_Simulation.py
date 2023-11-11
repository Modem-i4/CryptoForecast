import streamlit as st
st.set_page_config(page_icon="😋")

from prophet import Prophet
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from datetime import datetime, timedelta, time

#loading
df = st.session_state.btc_price
config = st.session_state.config
pd_timedelta = pd.Timedelta(config['interval'].replace('m', 'T'))

def reset_stats() : 
    st.session_state.orders_book = pd.DataFrame()
    st.session_state.balance = 10000
    st.session_state.balance_history = pd.DataFrame()

st.header(f"Simulation ({config['ticker']}, {config['interval']})")
st.subheader(f"Based on your last trained model")
st.write(f"Indicator regressors: {', '.join(config['m_indicators'])}")
st.write(f"Sentiment regressors: {', '.join(config['m_classes'])}")

data_cols = st.columns(5)
with data_cols[0]: st.markdown("**#Data**")
with data_cols[1]: st.write("Default data")
with data_cols[2]: use_own_data = st.toggle("data_touse", label_visibility='hidden', on_change=lambda: st.session_state.pop('m', None))
with data_cols[3]: st.write("Own data")

if use_own_data :
    news_aggregated = pd.read_json(f'data/news_aggregated_{config["interval"]}.json')
else :
    news_aggregated = pd.read_json(f'default/news_aggregated_{config["interval"]}.json')
for i in range(1, 3) : news_aggregated += news_aggregated.shift(i)/(1.5**i)
news_aggregated.fillna(0, inplace=True)
news_aggregated = news_aggregated.reset_index().rename(columns={'index': 'ds'})
df = pd.merge(df, news_aggregated, on='ds')

regressors =  config['all_classes'] + config['all_indicators']#
df[regressors] = df[regressors].shift(1)
df.dropna(inplace=True)

pred_cols = st.columns(3)
with pred_cols[0] : st.subheader('Choose date to simulate from: ')
with pred_cols[1] : selected_date = st.date_input('Select a date', value=df['ds'].iloc[0]+timedelta(1), min_value=config['date_from'], max_value=df['ds'].iloc[len(df)-2], on_change=reset_stats)
with pred_cols[2] : selected_time = st.time_input('Select a time', step=3600, value=time(0), on_change=reset_stats)
start_datetime = datetime.combine(selected_date, selected_time) 

forecast_fields = ['Low', 'High', 'Price']
def make_models() :
    m = {}
    for f_field in forecast_fields :
        m[f_field] = Prophet(interval_width=0.8, seasonality_mode='multiplicative', weekly_seasonality=False) #multiplicative, additive

        for reg in config['m_classes'] + config['m_indicators'] : m[f_field].add_regressor(reg)
        df_to_fit = df.rename(columns={'y': 'Price'}).rename(columns={f_field: 'y'})
        #m[f_field].seasonality_prior_scale = 0.5
        m[f_field].fit(df_to_fit)
    return m

if 'm' not in st.session_state:
    m = make_models()
    st.session_state.m = m
else:
    m = st.session_state.m

#----------------------------------------------#
#----------------------------------------------#
#----------------------------------------------#
#----------------------------------------------#
#----------------------------------------------#

if 'orders_book' not in st.session_state : reset_stats()

mode_cols = st.columns(5)
with mode_cols[0]: st.markdown("**#Mode**")
with mode_cols[1]: st.write("Manual")
with mode_cols[2]: automatic_mode = st.toggle("a_m", label_visibility='hidden', on_change=reset_stats, value=True) ###
with mode_cols[3]: st.write("Automatic")

draw_iterations = False
if automatic_mode:
    with mode_cols[0]: st.markdown("**#Draw iterations**")
    with mode_cols[1]: st.write("No")
    with mode_cols[2]: draw_iterations = st.toggle("d_i", label_visibility='hidden', value=False) ###
    with mode_cols[3]: st.write("Yes")

periods_ahead = 0
periods_to_test = 1

if automatic_mode:
    periods_to_test = st.number_input("Select amount of periods to test", step=1, on_change=reset_stats)
    prog_cols = st.columns([5,1])
    with prog_cols[0] : progress = st.progress(0.0, f"0/{periods_to_test}")
    with prog_cols[1] : stop_prediction = st.button('stop', use_container_width=True)
else:
    stop_prediction = False
    periods_ahead = st.number_input("Select a period ahead to predict", step=1)
overlays_score = 0
st.subheader("#Stats")
stats_cols = st.columns([2.5,2,2,1])
with stats_cols[0]: stats_time = st.empty()
with stats_cols[1]: stats_balance = st.empty()
with stats_cols[2]: stats_change = st.empty()
with stats_cols[3]: 
    if st.button("reset", use_container_width=True) : reset_stats()
stats_table = st.table(pd.DataFrame({
            'orders': 0,'percent_change': 0,'usd_change': 0
        }, index=['earn', 'loss', 'total']))
balance_history = st.line_chart()
stats_orders = st.empty()
orders_book = st.session_state.orders_book
def percent_change(val1, val2): return ((val2 - val1) / abs(val1)) * 100

if automatic_mode : reset_stats()
for idx in range(periods_ahead, periods_ahead + periods_to_test) :
    selected_datetime = start_datetime + pd_timedelta*idx
    last_balance = st.session_state.balance
    
    test_df = pd.concat([df[df['ds'] == selected_datetime],
                        df[df['ds'] == selected_datetime + pd_timedelta]]).reset_index()
    last_y = test_df['y'].iloc[0]

    forecasts = pd.DataFrame(test_df.ds)
    for f_field in forecast_fields :
        forecast = m[f_field].predict(test_df)
        forecasts[f_field] = forecast.yhat
        if f_field == 'Price': forecasts[['yhat_lower', 'yhat_upper']] = forecast[['yhat_lower', 'yhat_upper']]

    shift_cols = ['Price', 'yhat_lower', 'yhat_upper', 'Low', 'High']
    forecasts[shift_cols] = forecasts[shift_cols] - forecasts['Price'].iloc[0] + last_y
    if draw_iterations:
        st.divider()
        fig = make_subplots(rows=2, cols=1, row_heights=[3,1], shared_xaxes=True, vertical_spacing=0.1)

        fig.add_trace(go.Candlestick(x=df['ds'],
                                        open=df['y'].shift(1),
                                        high=df['High'],
                                        low=df['Low'],
                                        close=df['y'],                             
                                        increasing_line_color='blue',
                                        decreasing_line_color='gray',
                                        name="Actual",
                                        increasing_line_width=5, decreasing_line_width=5))

        fig.add_trace(go.Candlestick(x=forecasts['ds'],
                                            open=forecasts['Price'].shift(1),
                                            high=forecasts['High'],
                                            low=forecasts['Low'],
                                            close=forecasts['Price'],
                                            name="Forecast"))
        for a_class in config['all_classes'] :
            fig.add_trace(go.Scatter(x=df['ds'], y=df[a_class], mode='lines', name=a_class), row=2, col=1)

        fig.update_layout(xaxis_rangeslider_visible=False, showlegend=False)

        fig.update_xaxes(range=[
            selected_datetime - pd_timedelta*10, 
            selected_datetime + pd_timedelta*3.5])

        scale_df = df[abs(df['ds'] - selected_datetime) < pd_timedelta*10]
        fig.update_yaxes(range=[
            min(scale_df['Low'].min(), forecasts['Low'].min(), test_df['Low'].min()),
            max(scale_df['High'].max(), forecasts['High'].max(), test_df['High'].max())
        ], row=1, col=1)

    orders = []

    entry = forecasts.rename(columns={'Price': 'Close'}).iloc[1].to_dict()
    entry['Open'] = forecasts['Price'].iloc[0]
    entry['High'] = max(entry['High'], entry['Open'], entry['Close'])
    entry['Low'] = min(entry['Low'], entry['Open'], entry['Close'])

    #dir = 'LONG' if (entry['High']+entry['Low'])/2 > entry['Open'] else 'SHORT'
    dir = 'LONG' if entry['Close'] > entry['Open'] else 'SHORT'

    last_3_entries = df[df['ds'] < selected_datetime].iloc[-4:-1]
    last_6_entries = df[df['ds'] < selected_datetime].iloc[-7:-1]
    order = {'completed': False, 'dir': dir, 'profit': '', 'entry': entry['Open'], 'close':None, 'bet': st.session_state.balance*1, 'open_dt': selected_datetime+pd_timedelta}
    if(dir == 'LONG') :
        order['sl'] = min(last_3_entries['y'].min(), entry['Low'])*0.9999
        order['sl'] = max(order['sl'], order['entry']*0.975) ##
        orders.append({**order, 'tp': entry['Close']})
        orders.append({**order, 'tp': entry['High']*0.8+entry['Close']*0.2})

        if last_6_entries['High'].max() > orders[1]['tp'] : ##
            orders.append({**order, 'tp': last_6_entries['High'].max()*0.85+entry['Close']*0.15})
    else: # SHORT
        order['sl'] = max(last_3_entries['y'].max(), entry['High'])*1.0001
        order['sl'] = min(order['sl'], order['entry']*1.025) ##
        orders.append({**order, 'tp': entry['Close']})
        orders.append({**order, 'tp': entry['Low']*0.85+entry['Close']*0.15})

        if last_6_entries['Low'].min() < orders[1]['tp'] : ##
            orders.append({**order, 'tp': last_6_entries['Low'].min()*0.85+entry['Close']*0.15})
    #for i, order in enumerate(orders): 
    if percent_change(orders[0]['entry'], orders[0]['tp']) < 0.5: 
        orders[1]['tp'] = (orders[0]['tp'] + orders[1]['tp'])/2
        del orders[0]

    orders_book = pd.concat([orders_book, pd.DataFrame(orders)], ignore_index=True).sort_index(ascending=False)

    def close_pos(index, price) :
        order = orders_book.loc[index]
        change = percent_change(order.entry, price) * (-1 if order.dir == "SHORT" else 1)
        profitable = change >= 0
        profit_str = f"{change:.2f}%, {change*order.bet/100:.2f}$"
        st.session_state.balance += change*order.bet/100
        orders_book.at[index, 'close'] = price
        orders_book.at[index, 'profit'] = profit_str
        orders_book.at[index, 'completed'] = True
        orders_book.at[index, 'change_perc'] = change

        if draw_iterations : 
            fig.add_hline(y=price, line_dash="dash", line_color=("green" if profitable else "red"), annotation_text=profit_str, annotation_font_color=("green" if profitable else "red"), row=1)
            fig.add_scatter(x=[order.open_dt], y=[order.entry], line_color=("green" if profitable else "red"), row=1, col=1)
            display = pd.DataFrame(orders_book.loc[index]).transpose()
            display


    if len(orders_book) > 0:
        #validate
        for order in orders_book[~orders_book['completed']].itertuples():
            if (order.open_dt+pd_timedelta*3 < selected_datetime and dir != order.dir) or \
                order.open_dt+pd_timedelta*5 < selected_datetime:
                if draw_iterations : st.write('Closing a position because of exparation')
                close_pos(order.Index, entry['Open'])
                continue
            high, low = entry['High'], entry['Low']
            if (not high > order.tp > low) and high > order.sl > low and percent_change(low, high) > 1 :
                if draw_iterations : st.write('Closing a position according to the new predictions')
                close_pos(order.Index, entry['Open'])
        #rebalance
        for order in orders_book[~orders_book['completed']].itertuples():
            real_entry = df[df['ds'] > selected_datetime].iloc[0]
            high, low = real_entry['High'], real_entry['Low']
            if high >= order.tp >= low : close_pos(order.Index, order.tp)
            elif high > order.sl > low : close_pos(order.Index, order.sl)

            #if high > order.tp > low and high > order.sl > low : overlays_score += 1
            
    st.session_state.orders_book = orders_book
    
    selected_datetime_end = selected_datetime + pd_timedelta
    if draw_iterations:
        extra_info_cols = st.columns(3)
        with extra_info_cols[0]: 
            selected_datetime_end
        with extra_info_cols[1]: st.write(f'balance: {st.session_state.balance:.2f}')
        with extra_info_cols[2]: st.write(f'change: {percent_change(last_balance, st.session_state.balance):.2f}%')

        st.plotly_chart(fig)
    
    stats_time.write(f"Used time: {idx*pd.Timedelta(config['interval'])}")
    stats_balance.write(f"Balance: {st.session_state.balance:.0f} (+{percent_change(10000,st.session_state.balance):.2f}%)")
    if automatic_mode: progress.progress((idx+1)/periods_to_test, f"{idx+1}/{periods_to_test}")
    st.session_state.balance_history = pd.concat([st.session_state.balance_history, 
                                                  pd.DataFrame({'balance': st.session_state.balance}, index=[selected_datetime_end])])
    balance_history.line_chart(st.session_state.balance_history)
    ob = orders_book[orders_book['completed']]
    if 'change_perc' in ob :
        earn = ob[ob['change_perc'] > 0]
        loss = ob[ob['change_perc'] < 0]
        
        stats_table.table(pd.DataFrame({
            'orders': [len(earn), len(loss), len(ob)],
            'percent_change': [earn['change_perc'].sum(), loss['change_perc'].sum(), ob['change_perc'].sum()],
            'usd_change': [(earn['change_perc']*earn['bet']).sum()/100, (loss['change_perc']*loss['bet']).sum()/100, (ob['change_perc']*ob['bet']).sum()/100]
        }, index=['earn', 'loss', 'total'])) 
        stats_change.write(f"Earn/loss: {(earn['change_perc']*earn['bet']).sum()/(loss['change_perc']*loss['bet']).sum()*-1:.2f}")#ob['change_perc'].abs().sum()*100:.2f}%")

    if stop_prediction : break

st.write(f"Overlays score: {overlays_score}")
if len(orders_book) > 0:
    stats_orders.write(orders_book)

