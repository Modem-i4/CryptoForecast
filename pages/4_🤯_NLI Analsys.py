import streamlit as st
import pandas as pd

from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer

st.set_page_config(page_icon="😋")

st.title("NLI Analyzer!")

data_cols = st.columns([1.8, 1,1,1, 1.8])
with data_cols[1]: st.write("Default data")
with data_cols[2]: use_own_data = st.toggle("dtuse", label_visibility='hidden')
with data_cols[3]: st.write("Own data")

st.markdown("#__Analyzing, filtering and aggregating collected news data__")

if use_own_data:
    df = pd.read_json('data/news.json')
else:
    df = pd.read_json('default/news.json')
df
def save() :
    st.session_state.df = df
    with open('data/news_evaluated.json', 'w') as f:
        f.write(df.to_json())


config = st.session_state.config
if 'df' in st.session_state :
    df = st.session_state.df
    config = st.session_state.config

st.subheader("Sentiment analysys")

col_btn_1, col_progress_1 = st.columns([1, 5])
with col_btn_1:
    sentiment_analyze_btn = st.button('Analyse news.json!')
with col_progress_1:
    progress_text = st.empty().text('Analyzing progress bar')
    progress = st.progress(value=1)

if sentiment_analyze_btn :
    progress_text.text('Model loading...')
    model = AutoModelForSequenceClassification.from_pretrained("soleimanian/financial-roberta-large-sentiment")
    classifier = pipeline("sentiment-analysis",model="soleimanian/financial-roberta-large-sentiment")

    for idx, row in df.iterrows():
        result = classifier(row['title'] + ". " + row['description'])[0]
        df.loc[idx, 'label'] = result['label']
        df.loc[idx, 'label_score'] = result['score']
        progress_text.text(f'{idx+1}/{(df.index[-1]+1)} evaluated')
        progress.progress(value = (idx+1)/(df.index[-1]+1))
    df[['title', 'label', 'label_score']]
    save()

st.markdown("__#With financial-roberta-large-sentiment__")

if st.checkbox("Expand code"):
    st.code("""
def collect_data():
    model = AutoModelForSequenceClassification.from_pretrained("soleimanian/financial-roberta-large-sentiment")
    classifier = pipeline("sentiment-analysis",model="soleimanian/financial-roberta-large-sentiment")
    for idx, (key, row) in enumerate(df.iterrows()):
        result = classifier(row['title'] + ". " + row['description'])[0]
        df.loc[idx, 'label'] = result['label']
        df.loc[idx, 'label_score'] = result['score']             
""")

st.subheader("Filtering and evaluating effect of sentiment")

cols_filter_1 = st.columns(2)
with cols_filter_1[0]:
    threshold_filter_1 = st.number_input(label="Treshold", min_value=0.0, max_value=0.99, value=0.9, step=0.01)
with cols_filter_1[1]:
    st.write("")
    st.write("")
    run_filter_1 = st.button('Apply Filter And Evaluate Effect', use_container_width=True)
if run_filter_1 :
    df = df[(df['label'] != 'neutral') & (df['label_score'] >= threshold_filter_1)]
    df['label_effect'] = df.apply(lambda row: row['label_score'] if row['label'] != 'negative' else -row['label_score'], axis=1)
    df[['date', 'label_effect', 'title']]
    st.bar_chart(df, x='date', y='label_effect')
    save()

if st.checkbox("Expand code   "):
    st.code("""
    df = df[(df['label'] != 'neutral') & (df['label_score'] >= 0.9)]
    df['label_effect'] = df.apply(lambda row: row['label_score'] if row['label'] != 'negative' else -row['label_score'], axis=1)      
""")

st.subheader("Categorizing of sentiment")

classes_cols = st.columns(5)
default_classes = ['Regulation', 'Global Events', 'Forecasts', 'Technology', 'Crime']
classes = default_classes

for i in range(5) :
    with classes_cols[i] :
        classes[i] = st.text_input(label=f'Class {i+1}', value=classes[i])

config['all_classes'] = classes
st.session_state.config = config

col_btn_2, col_progress_2 = st.columns([1, 5])
with col_btn_2:
    sentiment_analyze_btn = st.button('Categorize sentiment!')
with col_progress_2:
    progress_text2 = st.empty().text('Categorizyng progress bar')
    progress2 = st.progress(value=1)

if sentiment_analyze_btn :
    progress_text2.text('Model loading...')
    nli_model = AutoModelForSequenceClassification.from_pretrained('facebook/bart-large-mnli')
    tokenizer = AutoTokenizer.from_pretrained('facebook/bart-large-mnli')
    classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

    for idx, row in df.iterrows():
        result = classifier(row['title'] + ". " + row['description'], classes)
        df.loc[idx, 'class'] = result['labels'][0]
        df.loc[idx, 'class_score'] = result['scores'][0]
        progress_text2.text(f'{idx+1}/{(df.index[-1]+1)} evaluated')
        progress2.progress(value = (idx+1)/(df.index[-1]+1))
    df[['class', 'class_score', 'title']]
    st.session_state.classes = classes
    save()

st.markdown("__#With bart-large-mnli__")
if st.checkbox("Expand code  "):
    st.code("""
    nli_model = AutoModelForSequenceClassification.from_pretrained('facebook/bart-large-mnli')
    tokenizer = AutoTokenizer.from_pretrained('facebook/bart-large-mnli')
    classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

    for idx, (key, row) in enumerate(df.iterrows()):
        result = classifier(row['title'] + ". " + row['description'], classes)
        df.loc[idx, 'class'] = result['labels'][0]
        df.loc[idx, 'class_score'] = result['scores'][0]
""")


st.subheader("Evaluating Gross Effect")
cols_filter_2 = st.columns(2)
with cols_filter_2[0]:
    threshold_filter_2 = st.number_input(label="Treshold ", min_value=0.0, max_value=0.95, value=0.35, step=0.05)
with cols_filter_2[1]:
    st.write("")
    st.write("")
    run_filter_2 = st.button('Filter and evaluate', use_container_width=True)
if run_filter_2 :
    df = df[df['class_score'] >= threshold_filter_2]
    df['gross_effect'] = df['label_effect'] * df['class_score']
    df[['date', 'gross_effect', 'label', 'label_effect', 'class', 'class_score', 'title']]
    st.bar_chart(df, x='date', y='gross_effect')
    save()

if st.checkbox("Expand code "):
    st.code("""
    df = df[df['class_score'] >= threshold_filter_2]
    df['gross_effect'] = df['label_effect'] * df['class_score']   
""")
    
st.subheader("Transformation into model parameters")


st.write("Turn on if you need saved data for this specific step")
data_cols = st.columns([2, 1,1,1, 2])
with data_cols[1]: st.write("Default data")
with data_cols[2]: use_own_data1 = st.toggle("data_to_use", label_visibility='hidden', value=True)
with data_cols[3]: st.write("Own data")

cols_transform = st.columns([2,1])
with cols_transform[0]:
    transform_interval = st.selectbox("Time interval you have selected. You may change it if necessary:", config['time_intervals'], index=config['time_intervals'].index(config['interval']))
with cols_transform[1]:
    st.write("")
    st.write("")
    run_transform = st.button('Transform!', use_container_width=True)


if run_transform :
    st.markdown("##Transforming")
    if use_own_data1:
        news = pd.read_json('data/news_evaluated.json')
    else:
        news = pd.read_json('default/news_evaluated.json')
    df_transformed = pd.DataFrame()
    df_transformed[['date'] + classes] = 0
    for idx, n in news.iterrows() :
        df_transformed.loc[idx, 'date'] = n['date']
        df_transformed.loc[idx, n['class']] = n['gross_effect']
    df_transformed
    st.bar_chart(df_transformed, x='date')

    st.markdown("##Aggregating according to the time interval")
    df_aggregated = df_transformed.copy()
    df_aggregated.set_index('date', inplace=True)
    df_aggregated = df_aggregated.resample(transform_interval.replace('m', 'T')).sum()
    df_aggregated
    st.bar_chart(df_aggregated)
    with open(f'data/news_aggregated_{transform_interval}.json', 'w') as f :
        f.write(df_aggregated.to_json())


if st.checkbox("Expand code      "):
    st.code("""
def run_transform() :
    #Transforming
    df_transformed = pd.DataFrame()
    df_transformed[['date'] + classes] = 0
    for idx, n in news.iterrows() :
        df_transformed.loc[idx, 'date'] = n['date']
        df_transformed.loc[idx, n['class']] = n['gross_effect']
    df_transformed

    #Aggregating according to the time interval
    df_aggregated = df_transformed.copy()
    df_aggregated.set_index('date', inplace=True)
    df_aggregated = df_aggregated.resample(transform_interval.replace('m', 'T')).sum()
    df_aggregated         
""")