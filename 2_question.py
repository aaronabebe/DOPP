import pandas as pd 
import streamlit as st 
import numpy as np

st.markdown("# DOPP3")
st.markdown('Which characteristics are predictive for countries with large populations living in extreme poverty?')

with st.echo():
    # READ TRANSFORMED CSV FILE
    raw = pd.read_csv("transformed.csv")  
    st.write(raw.head(100))

    feature_descriptions = pd.read_csv("feature_descriptions.csv")
    st.write(feature_descriptions)

    #FEATURES WITH LESS THAN 50% MISSING VALUES
    features = feature_descriptions.where(feature_descriptions['na_percent']<=50.0).dropna(0)
    
    #ONLY DEMOGRAFIC FEATURES!
    cols = features['Unnamed: 0'].tolist()
    cols = cols[0:7]+ cols[13:18] + [cols[25]]
    #st.write(cols)

    dataset = raw[cols]
    st.write(dataset.head(100))


#st.balloons()