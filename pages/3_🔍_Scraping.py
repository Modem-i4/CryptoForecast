import urllib.request
import json
import streamlit as st
import math

st.set_page_config(page_icon="😋")

st.title("News Scrapper!")
st.markdown("#__Collecting news on specified topic from [Coindesk](https://coindesk.com)__")


def collect_data():
    results = []
    baseUrl = f'https://api.queryly.com/json.aspx?queryly_key=d0ab87fd70264c0a&query={config["query"]}&batchsize=100&daterange={config["date_from"].strftime("%m/%d/%Y")},{config["date_to"].strftime("%m/%d/%Y")}&sort=date'
    with urllib.request.urlopen(baseUrl) as data:
        data = json.load(data)
        news_number = data['metadata']['total']
        st.write(f'Total records: {news_number}')
    for page in range(0, math.ceil(news_number/100)):
        url = f'{baseUrl}&endindex={100*page}'
        with urllib.request.urlopen(url) as data:
            data = json.load(data)
            data = data['items']
            for entry in data:
                result = {'title': entry['title'], 'description':entry['description'], 'date': entry['pubdateunix'] }
                results.append(result)
            progress.progress(value=page/news_number/100)
    
    with open(f'data/news.json', 'w') as f:
        f.write(json.dumps(results))
    st.session_state.results = results
config = st.session_state.config
col_from, col_to, col_query, col_go = st.columns([1, 1, 2, 1])
with col_from :
    config['date_from'] = st.date_input("From", value=config['date_from'])
with col_to :
    config['date_to'] = st.date_input("To", value=config['date_to'])
with col_query :
    config['query'] = st.text_input("Query", value="Crypto")
with col_go : 
    st.markdown("### ")
    results = st.button('Collect!', on_click=collect_data)

st.markdown("#Collection progress:")
progress = st.progress(value=0)

show_spoiler = st.checkbox("Expand code")
if show_spoiler:
    st.code("""
def collect_data():
    results = []
    baseUrl = f'https://api.queryly.com/json.aspx?queryly_key=d0ab87fd70264c0a&query={config["query"]}&batchsize=100&daterange={config["date_from"].strftime("%m/%d/%Y")},{config["date_to"].strftime("%m/%d/%Y")}&sort=date'
    with urllib.request.urlopen(baseUrl) as data:
        data = json.load(data)
        news_number = data['metadata']['total']
        st.write(f'Total records: {news_number}')
    for page in range(0, math.ceil(news_number/100)):
        url = f'{baseUrl}&endindex={100*page}'
        with urllib.request.urlopen(url) as data:
            data = json.load(data)
            data = data['items']
            for entry in data:
                result = {'title': entry['title'], 'description':entry['description'], 'date': entry['pubdateunix'] }
                results.append(result)
            progress.progress(value=page/news_number/100)
    f = open(f'data/news.json', 'w')
    f.write(json.dumps(results))
    f.close()              
""")

if hasattr(st.session_state, 'results'):
    st.dataframe(st.session_state.results)

