import pandas as pd 
import streamlit as st

def just_demographic(df_poor):
    feature_descriptions = pd.read_csv("data/feature_descriptions.csv")
    #st.write(feature_descriptions)

    #FEATURES WITH LESS THAN 50% MISSING VALUES
    features = feature_descriptions.where(feature_descriptions['na_percent']<=50.0).dropna(0)
    
    #ONLY DEMOGRAFIC FEATURES!
    cols = features['Unnamed: 0'].tolist()
    cols = cols[0:7]+ cols[13:18] + [cols[25]]
    #st.write(cols)

    df_poor = df_poor[cols]
    return (df_poor)

def alles_good_papi():
    return("Alles good papi")

##



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

    # Poor countries
    p_poor = poor[poor['poverty'] == 1]
    c_poor = p_poor.LOCATION.unique() # poor countries

    # No Poor countries
    no_poor = poor[poor['poverty'] == 0]
    no_poor = no_poor[poor['TIME'] == 2015]
    c_no_poor = no_poor.LOCATION.unique() # no poor countries 2015


    #c_poor
    #c_no_poor
    
    e_countries = set(c_no_poor).intersection(c_poor) # Identifying emerging countries
    #print (e_countries)
    #e_countries # Emerging
    e_poor = poor[poor['LOCATION'].isin(e_countries)]
    #print(len(e_countries))
    #print(set(e_countries))
    e_poor = just_demographic(e_poor)



st.write(len(e_countries))
st.write(len(poor.LOCATION.unique()))
st.write(e_poor)
#st.write(poor.poverty)




st.markdown("## Target Column")



st.write(alles_good_papi())
st.balloons()
