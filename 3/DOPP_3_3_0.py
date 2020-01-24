import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy
import seaborn as sns
import sklearn as sk
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import mutual_info_classif
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.preprocessing import MinMaxScaler
import streamlit as st

#@st.cache
#np.random.seed(9103)

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

def interpolation_df_poor(df_poor):
    columns = df_poor.columns
    #c_stay = columns[:1]
    #c_inter = columns[:]
    countries = df_poor['LOCATION'].unique()
    #st.write(countries)
    #for (c in columns[1:-1]):
    for c in countries:
        #st.write(c)
        df_c = df_poor[df_poor['LOCATION']==c]
        #df_c = df_c[c_inter]
        #st.write(df_c)
        #break

        df_c = df_c.interpolate(method ='linear', 
                                    limit_direction ='forward',
                                    axis = 0)
        df_c = df_c.interpolate(method ='linear', 
                                    limit_direction ='backward',
                                    axis = 0)
        
        df_poor[df_poor['LOCATION']==c] = df_c

        #st.write(df_poor[df_poor['LOCATION']==c])
        #break
    return(df_poor)

def e_poor_selectKBest(df_e_poor, score_f = f_classif):
    e_poor = df_e_poor
    # Split dataset to train
    X = e_poor.iloc[:,2:-2] # All the columns less the last one
    y = e_poor.iloc[:,-2:-1] # Just the last column
    dfcolumns = pd.DataFrame(X.columns)

    scaler = MinMaxScaler()
    scaler.fit(X)
    X = scaler.transform(X)
    # Create the feature selector
    #perhaps a switch
    bestFeatures = SelectKBest(score_func=score_f, k='all')
    #st.write(X)
    fit = bestFeatures.fit(X,y)
    dfscores = pd.DataFrame(fit.scores_)

    # Create a data frame to see the impact of the features
    featureScores = pd.concat([dfcolumns,dfscores],axis=1)
    featureScores.columns = ['Features','Scores']
    featureScores.sort_values(by='Scores', ascending=False, inplace=True)
    
    return(featureScores)

def e_poor_feature_importance(df_e_poor):
    e_poor = df_e_poor
    # Split dataset to train
    X = e_poor.iloc[:,2:-2] # All the columns less the last one
    y = e_poor.iloc[:,-2:-1] # Just the last column
    model = ExtraTreesClassifier()
    model.fit(X,y)
    print(model.feature_importances_)
    feat_importances = pd.Series(model.feature_importances_, index=X.columns)
    feat_importances.nlargest(20).plot(kind='barh', figsize = (13, 6), fontsize=12)
    st.pyplot()
    return("All fertig pa pitura")

def plot_correlation_matrix(df_e_poor, n=20):
    data = e_poor.iloc[:,2:-2] # All features
    #data['poverty'] = data['poverty'].apply(lambda x: 1 if x==True else 0)
    # Split dataset to train
    #X = e_poor.iloc[:,2:] # All the columns less the last one
    #y = e_poor.iloc[:,-1] # Just the last column
    columns = data.columns
    plt.clf()
    correlation = data.corr()

    #columns = correlation.nlargest(n, 'poverty').index
    st.write('OK')
    #st.write(columnss)
    st.write(columns)
    st.write(data[columns].values)

    correlation_map = np.corrcoef(data[columns].values.T)
    sns.set(font_scale=1, rc={'figure.figsize':(30,30)})
    heatmap = sns.heatmap(correlation_map, cbar=True, annot=False, 
                            square=True, fmt='.2f', yticklabels=columns.values, 
                            xticklabels=columns.values)
    heatmap.set_xticklabels(heatmap.get_xticklabels(), rotation = 90, fontsize = 16)
    heatmap.set_yticklabels(heatmap.get_yticklabels(), rotation = 0, fontsize = 16)
    st.pyplot()
    return("All fertig pa pitura")



##

st.markdown("# DOPP - Exercise 3 - Question 3")
st.markdown("## Importing the dataset from data_prep.py")
st.markdown('The data set looks like')
with st.echo():
    # READ FROM DATA_PREP
    df_p = pd.read_csv("data/transformed.csv")
    poor = df_p
    poor = just_demographic(poor)
    #st.write(poor.dtypes)
    poor = interpolation_df_poor(poor)
    
st.write(poor)
st.write(poor.shape)


st.markdown("## Identifying emerging countries from poverty")
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
    poor['emerging'] = poor['LOCATION'].isin(e_countries)
    e_poor = poor[poor['LOCATION'].isin(e_countries)]
    st.write(poor)
    st.write(e_poor)

    #poor['emerging'] = poor['LOCATION'].apply(lambda x: (x.isin(e_countries)))
    #sub_0['poverty'] = base['NY_GNP_PCAP_CN'].apply(lambda x: (x / 365) < 1)
    #print(len(e_countries))
    #print(set(e_countries))
    #e_poor = just_demographic(e_poor)



st.write(len(e_countries))
st.write(len(poor.LOCATION.unique()))
st.write(e_poor)
#st.write(poor.poverty)

#st.markdown("## Filling up missing values")
#st.markdown("Starting with a  simple interpolation")
#with st.echo():
#    #e_poor.interpolation(method="Linear")
#    print("Not implemented yet")


st.markdown("## Building a Model")
st.markdown("[Feature Selection Techniques in Machine Learning with Python] ('https://towardsdatascience.com/feature-selection-techniques-in-machine-learning-with-python-f24e7da3f36e')")
with st.echo():
    st.write("Best ")
    st.write("f_classif")
    st.write(e_poor_selectKBest(e_poor))
    st.write("chi2")
    st.write(e_poor_selectKBest(e_poor,score_f=chi2))
    st.write("mutual_info_classif")
    st.write(e_poor_selectKBest(e_poor,score_f=mutual_info_classif))
    st.write("feature_importance")
    e_poor_feature_importance(e_poor)
    st.write("Correlation Matrix")
    plot_correlation_matrix(e_poor)
    
st.write(alles_good_papi())
st.balloons()
