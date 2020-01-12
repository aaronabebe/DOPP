import pandas as pd 
import streamlit as st

st.markdown("# Data Preparation - DOPP3")
st.markdown('_creating a data set to predict extreme poverty_')
st.markdown("## Transform base data")
with st.echo():
    # READ SOURCE CSV
    raw = pd.read_csv("unesco_poverty_dataset.csv") 
    keys = raw.DEMO_IND.unique() 

    # DEFINE BASE CSV
    base = raw[['LOCATION', 'TIME']]

    # FOR EVERY VAR JOIN ON LOCATION & TIME 
    for i in range(0,len(keys)):
        loop = raw.loc[raw.DEMO_IND == keys[i]]
        base = pd.merge(base, loop[['LOCATION', 'TIME', 'Value']],  how='left', left_on=['LOCATION','TIME'], right_on = ['LOCATION','TIME']) 
        base.columns = base.columns.str.replace('Value', keys[i])

    # DROP DUPLICATES
    base = base.drop_duplicates()
    st.write(base.head(100))
    st.write(base.shape)

st.markdown("## Features")
with st.echo():
    # GET DATA PER COLUMN
    na_percent = []
    na_total = []
    minimum = []
    maximum = []
    for col in base.columns:
        na_percent.append(round(base[col].isna().sum() / base.shape[0] * 100, 2))
        na_total.append(base[col].isna().sum())
        minimum.append(base[col].min())
        maximum.append(base[col].max())

    # GET VARIABLE DESCRIPTIONS
    descriptions = raw['Indicator'].drop_duplicates().tolist()
    descriptions.insert(0, 'LOCATION')
    descriptions.insert(1, 'TIME')

    features = pd.DataFrame(
        {'descriptions': descriptions, 
        'na_percent': na_percent, 
        'na_total': na_total,
        'minimum': minimum,
        'maximum': maximum},
        index=base.columns) 
    st.write(features)
    st.write(features.shape)
    
st.markdown("## Target Column")
st.markdown('_what is the target column supposed to be?_')
st.markdown('Lets assume the current definition: __daily GNI per person < 1.9__ ')
with st.echo():
    # calculate GNI per day (PPP)
    base['target'] = base['NY_GNP_PCAP_PP_CD'] / 365
    # check for 1.9 threshold
    poor = base[base.target < 1.9][['LOCATION', 'TIME', 'target']]
    # tada
    perc_poor_countries_ever = round(poor['LOCATION'].drop_duplicates().shape[0] / base['LOCATION'].drop_duplicates().shape[0] * 100,2)

st.write(poor)
st.write(poor.shape)
st.write('From 1970-2018, all countries considered, only ', perc_poor_countries_ever, '% lived in extreme poverty?')
st.write('this cant be right...')


# EXPORT BASE DATAFRAME
base.drop(['target'], axis=1).to_csv('transformed.csv', sep=',', na_rep="NA")
# EXPORT FEATURE DESCRIPTORS
features.to_csv('feature_descriptions.csv', sep=',', na_rep="NA")

# BALLOOOOOOONS
st.balloons()