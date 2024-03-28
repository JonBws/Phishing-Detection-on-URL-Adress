import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier

st.write('## :globe_with_meridians: Predict Web Phishing Application')

with open ('vc_model.pkl', 'rb') as file:
    model = pickle.load(file)

def input_features(url):
    url_length = len(url)
    n_dots = url.count('.')
    n_hypens = url.count('-')
    n_underline = url.count('_')
    n_slash = url.count('/')
    n_questionmark = url.count('?')
    n_equal = url.count('=')
    n_at = url.count('@')
    n_and = url.count('&')
    n_exclamation = url.count('!')
    n_space = url.count(' ')
    n_tilde = url.count('~')
    n_comma = url.count(',')
    n_plus = url.count('+')
    n_asterisk = url.count('*')
    n_hashtag = url.count('#')
    n_dollar = url.count('$')
    n_percent = url.count('%')
    
    features = {
        'url_length': [url_length],
        'n_dots': [n_dots],
        'n_hypens': [n_hypens],
        'n_underline': [n_underline],
        'n_slash': [n_slash],
        'n_questionmark': [n_questionmark],
        'n_equal': [n_equal],
        'n_at': [n_at],
        'n_and': [n_and],
        'n_exclamation': [n_exclamation],
        'n_space': [n_space],
        'n_tilde': [n_tilde],
        'n_comma': [n_comma],
        'n_plus': [n_plus],
        'n_asterisk': [n_asterisk],
        'n_hastag': [n_hashtag],
        'n_dollar': [n_dollar],
        'n_percent': [n_percent]
    }
    
    input_df = pd.DataFrame(features)
    return input_df
url = st.text_input('Masukan URL :')
predict_button = st.button("Predict")

if predict_button:  # Only execute logic when button is clicked
    if url:
        df = input_features(url)
        prediction = model.predict(df)  # Assuming model is trained for binary classification

        if prediction == 1:
            st.write("### :red[This web is phishing]")
        else:
            st.write("### :green[This web is Legitimate]")
    else:
        st.warning("Please enter a URL to make a prediction.")

