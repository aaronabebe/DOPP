import pandas as pd 
import streamlit as st


st.markdown("# DOPP - Exercise 3 - Question 3")
st.markdown("## Importing the dataset from data_prep.py")
st.markdown('_creating a data set to predict extreme poverty_')
st.markdown("## Transform base data")
with st.echo():
    # READ FROM DATA_PREP
    
    df_p = pd.read_csv("data/transformed.csv") 
    poor = df_p


st.write(poor)
st.write(poor.shape)

st.markdown("## Target Column")


