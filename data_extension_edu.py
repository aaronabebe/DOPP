import pandas as pd 
import streamlit as st

st.markdown("# Extending the base dataset with different data")
st.markdown("## Base Data")
with st.echo():
    # LOAD BASE DATA
    base = pd.read_csv("transformed.csv", index_col="Unnamed: 0")
    st.write(base)
    st.write(base.shape)

st.markdown('## Education Data')
st.markdown('_pick the number of variables to use from the dataset (max 2992)_')
with st.echo():
    nr_vars = 50
st.write('Number of Variables used:',nr_vars)

with st.echo():
    # LOAD EDU DATA
    raw_edu = pd.read_csv("unesco_education_dataset.csv") 
    keys = raw_edu.EDULIT_IND.unique() 

    # DEFINE BASE CSV
    stem = raw_edu[['LOCATION', 'TIME']]

    # FOR EVERY VAR JOIN ON LOCATION & TIME 
    for i in range(0, nr_vars):
        print(keys[i], i, '/', nr_vars)
        loop = raw_edu.loc[raw_edu.EDULIT_IND == keys[i]]
        stem = pd.merge(stem, loop[['LOCATION', 'TIME', 'Value']],  how='left', left_on=['LOCATION','TIME'], right_on = ['LOCATION','TIME']) 

    # FIX COLUMNS
    stem.columns = ['LOCATION', 'TIME'] + [str(col) for col in keys][:nr_vars]

    # DROP DUPLICATES
    edu = stem.drop_duplicates()

    st.write(edu)
    st.write(edu.shape)

st.markdown("## Features")
st.markdown("extracted similarly to _data_prep.py_")
# GET DATA PER COLUMN
na_percent = []
na_total = []
minimum = []
maximum = []
for col in edu.columns:
    na_percent.append(round(edu[col].isna().sum() /edu.shape[0] * 100, 2))
    na_total.append(edu[col].isna().sum())
    minimum.append(edu[col].min())
    maximum.append(edu[col].max())

# GET VARIABLE DESCRIPTIONS
descriptions = raw_edu['Indicator'].drop_duplicates().tolist()[:nr_vars]
descriptions.insert(0, 'LOCATION')
descriptions.insert(1, 'TIME')

features = pd.DataFrame(
    {'descriptions': descriptions, 
    'na_percent': na_percent, 
    'na_total': na_total,
    'minimum': minimum,
    'maximum': maximum},
    index=edu.columns) 

st.write(features)
st.write(features.shape)

st.markdown('## Additional Information & Experimentation')
with st.echo():
    # Number of distinct countries in LOCATION column
    st.write('number of countries', len(edu['LOCATION'].unique()))

# SAVE
base.to_csv('edu_transformed_%d.csv' % (nr_vars), sep=',', na_rep="NA")
st.balloons()