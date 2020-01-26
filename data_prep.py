import pandas as pd 
import streamlit as st
import plots as plot


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
    st.write(base.shape)
    base = base.drop_duplicates()
    st.write(base.head(100))
    
st.markdown("## Target Column")
st.markdown('_what is the target column supposed to be?_')
st.markdown('Lets assume the current definition: __daily GNI per person < 1.9__  _(which is a generous approach, since the threshold has risen throughout the years)_ - more on that later')
st.markdown('Still there are 3 different metrics to count the GNI of a state: __LCU, Atlas, PPP__, so which is the correct one?')

st.markdown('### Calculate Poverty Line by PPP')
with st.echo():
    # calculate GNI per day (PPP)
    base['target'] = base['NY_GNP_PCAP_PP_CD'] / 365
    # check for 1.9 threshold
    poor = base[base.target < 1.9][['LOCATION', 'TIME', 'target']]
    # tada
    perc_poor_countries_ever = round(poor['LOCATION'].drop_duplicates().shape[0] / base['LOCATION'].drop_duplicates().shape[0] * 100,2)

st.write(poor)
st.write(poor.shape)

st.write('From 1970-2019, all countries considered, ', perc_poor_countries_ever, '% have lived in extreme poverty at least once.')
st.plotly_chart(plot.line_chart(base.copy(), y='target', y_name='PPP', threshold=1.9))
st.write('this can\'t be right ...')




st.markdown('### Calculate Poverty Line by Atlas')
base['target'] = base['NY_GNP_PCAP_CD'] / 365
poor = base[base.target < 1.9][['LOCATION', 'TIME', 'target']]
perc_poor_countries_ever = round(poor['LOCATION'].drop_duplicates().shape[0] / base['LOCATION'].drop_duplicates().shape[0] * 100,2)
st.write(poor)
st.write(poor.shape)

st.write('From 1970-2019, all countries considered, ', perc_poor_countries_ever, '% have lived in extreme poverty at least once.')
st.plotly_chart(plot.line_chart(base.copy(), y='target', y_name='Atlas', threshold=1.5))

st.markdown('### Calculate Poverty Line by LCU')
base['target'] = base['NY_GNP_PCAP_CN'] / 365
poor = base[base.target < 1.9][['LOCATION', 'TIME', 'target']]
perc_poor_countries_ever = round(poor['LOCATION'].drop_duplicates().shape[0] / base['LOCATION'].drop_duplicates().shape[0] * 100,2)
st.write(poor)
st.write(poor.shape)
st.write('From 1970-2019, all countries considered, ', perc_poor_countries_ever, '% have lived in extreme poverty at least once.')
st.plotly_chart(plot.line_chart(base.copy(), y='target', y_name='LCU', threshold=1))


st.plotly_chart(plot.combined_line_chart(base.copy(), 'target'))


# DROP TARGET COLUMN AGAIN
base = base.drop(['target'], axis=1)

st.markdown('### Calculate Poverty Line by Combination of LCU, PPP and Atlas')
st.markdown('Appearantly the the poverty threshold startet as 1$/day in 1996 (measure unknown)')
st.markdown('then moved on to 1.25$/day in 2005 (measure unknown, presumably Atlas)')
st.markdown('finally it went to 1.9$/day in 2015 (PPP)')
st.markdown('_the $ values being average per capita income of a person per day_')
st.markdown("[World Bank Press Release, October 2015]('https://www.worldbank.org/en/news/press-release/2015/10/04/world-bank-forecasts-global-poverty-to-fall-below-10-for-first-time-major-hurdles-remain-in-goal-to-end-poverty-by-2030')")
with st.echo():
    # SEPARATE INTO 3 SUB-TABLES: 1970-2004, 2005-2014, 2015-2019
    sub_0 = base[base['TIME'] < 2005]
    sub_1 = base[(base['TIME'] >= 2005) & (base['TIME'] < 2015)]
    sub_2 = base[base['TIME'] >= 2015]

    # WRITE TARGET VARIABLES
    sub_0['poverty'] = base['NY_GNP_PCAP_CN'].apply(lambda x: (x / 365) < 1)
    sub_1['poverty'] = base['NY_GNP_PCAP_CD'].apply(lambda x: (x / 365) < 1.25)
    sub_2['poverty'] = base['NY_GNP_PCAP_PP_CD'].apply(lambda x: (x / 365) < 1.9)

    # RE-CONCAT SUB-DATAFRAMES
    base = pd.concat([sub_0, sub_1, sub_2])

    # SHOW HOW MANY COUNTRIES WERE POOR AT LEAST ONCE
    poor = base[base['poverty'] == True]
    perc_poor_countries_ever = round(poor['LOCATION'].drop_duplicates().shape[0] / base['LOCATION'].drop_duplicates().shape[0] * 100,2)
    st.write(poor)
    st.write(poor.shape)
    st.write('From 1970-2019, all countries considered, ', perc_poor_countries_ever, '% have lived in extreme poverty at least once.')


st.markdown('what percentage of the world population lives in extreme poverty?')
st.plotly_chart(plot.question1(base))

st.markdown("## Final Data Set")
st.write(base)
st.write(base.shape)

st.plotly_chart(plot.scatter_poor_rich(base.copy(), x='SP_DYN_TFRT_IN', x_name='Fertility Rate', y='NY_GDP_PCAP_CD', y_name='GDP per capita'))
st.plotly_chart(plot.scatter(base.copy(), x='SP_DYN_TFRT_IN', x_name='Fertility Rate', y='NY_GDP_PCAP_CD', y_name='GDP per capita'))
st.plotly_chart(plot.world_map(base.copy(), y='SP_DYN_TFRT_IN', y_name='Fertility Rate'))

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
    descriptions.insert(38, 'poverty')

    features = pd.DataFrame(
        {'descriptions': descriptions, 
        'na_percent': na_percent, 
        'na_total': na_total,
        'minimum': minimum,
        'maximum': maximum},
        index=base.columns) 
    st.write(features)
    st.write(features.shape)


# EXPORT BASE DATAFRAME
base.to_csv('transformed.csv', sep=',', na_rep="NA")
# EXPORT FEATURE DESCRIPTORS
features.to_csv('feature_descriptions.csv', sep=',', na_rep="NA")
