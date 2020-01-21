import pandas as pd 
import streamlit as st


st.markdown("# DOPP - Exercise 3 - Question 3")
st.markdown("## Importing the dataset from data_prep.py")
st.markdown('_The data set looks like')
with st.echo():
    # READ FROM DATA_PREP
    df_p = pd.read_csv("data/transformed.csv") 
    poor = df_p
    
st.write(poor)
st.write(poor.shape)


st.markdown("## Identify countries in poverty at least once")
with st.echo():
    #print (poor.columns)
    #countries poor.poverty

    p_poor = poor[poor['poverty'] == 1]
    p_countries = p_poor.LOCATION.unique()
    p_countries


st.write(poor.columns)
st.write(poor.poverty)




st.markdown("## Target Column")

st.balloons()
