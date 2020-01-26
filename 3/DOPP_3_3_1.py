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
from sklearn import tree
from sklearn.tree import _tree
from sklearn.tree import DecisionTreeClassifier as DTC

#from sklearn.decomposition import PCA

import streamlit as st


#@st.cache
#np.random.seed(9103)

def just_demographic(df_poor):
    feature_descriptions = pd.read_csv("data/feature_descriptions.csv")

    #FEATURES WITH LESS THAN 50% MISSING VALUES
    features = feature_descriptions.where(feature_descriptions['na_percent']<=50.0).dropna(0)
    
    #ONLY DEMOGRAFIC FEATURES!
    cols = features['Unnamed: 0'].tolist()
    cols = cols[0:7]+ cols[13:18] + [cols[25]]
    df_poor = df_poor[cols]
    return (df_poor)

def alles_good_papi():
    return("Alles good papi")

def interpolation_df_poor(df_poor):
    columns = df_poor.columns
    countries = df_poor['LOCATION'].unique()
    for c in countries:
        df_c = df_poor[df_poor['LOCATION']==c]
        df_c = df_c.interpolate(method ='linear', 
                                    limit_direction ='forward',
                                    axis = 0)
        df_c = df_c.interpolate(method ='linear', 
                                    limit_direction ='backward',
                                    axis = 0)
        df_c.fillna(poor.drop(labels='LOCATION', axis=1).mean(), inplace =True)
        df_poor[df_poor['LOCATION']==c] = df_c
    return(df_poor)

def thresholds(df_poor):
    poor = df_poor
    thresholds = {}
    columns = poor.columns[2:-2]
    st.write(columns)
    X = poor.iloc[:,2:-2]
    #st.write(len(X))
    #st.write(columns)
    y = poor.iloc[:,-2:-1]
    #st.write(len(y))
    #st.write(y)
    for c in columns:
        #st.write(c)
        #st.write(X[c])
        X_ = np.array(X[c]).reshape(-1, 1)#X.iloc[:,c]
        clf_tree = DTC(criterion="gini",
                        max_depth=1, 
                        splitter="best")
        clf_tree.fit(X_,y)
        #thresholds[c] = clf_tree.tree_.threshold[0]
        threshold = clf_tree.tree_.threshold[0]
        #st.write(X_.mean())
        #st.write(threshold)
        #break        
        thresholds[c] = threshold    
    return(thresholds)


def e_poor_selectKBest(df_e_poor, score_f = f_classif, k='all'):
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
    bestFeatures = SelectKBest(score_func=score_f, k=k)
    fit = bestFeatures.fit(X,y)
    dfscores = pd.DataFrame(fit.scores_)

    # Create a data frame to see the impact of the features
    featureScores = pd.concat([dfcolumns,dfscores],axis=1)
    featureScores.columns = ['Features','Scores']
    featureScores.sort_values(by='Scores', ascending=False, inplace=True)
    scores = featureScores.Scores
    rel_scores = np.array(scores)/np.abs(np.array(scores)).sum()
    #rel_scores.reshape(-1, 1)
    #st.write(rel_scores.shape)

    #rel_scores.names = ['Relative']
    #df = pd.DataFrame(rel_scores, columns=list('Relative'))
    #st.write(df)
    #featureScores = pd.concat([featureScores, df], axis=1)

    #pd.concat([scores,rel_scores], axis=1)

    #parameters = fit.
    st.write(scores)
    st.write(rel_scores)
    
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
    # Split dataset to train
    columns = data.columns
    plt.clf()
    correlation = data.corr()
    #columns = correlation.nlargest(n, 'poverty').index
    #st.write('OK')
    #st.write(columns)
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

st.markdown("# DOPP - Exercise 3 - Question 3")
st.markdown("## Importing the dataset from data_prep.py")
st.markdown('The data set looks like')
with st.echo():
    # READ FROM DATA_PREP
    df_p = pd.read_csv("data/transformed.csv")
    poor = df_p
    poor = just_demographic(poor)
    poor = interpolation_df_poor(poor)
    
st.write(poor)
st.write(poor.shape)


st.markdown("## Identifying emerging countries from poverty")
with st.echo():
    # Poor countries
    p_poor = poor[poor['poverty'] == 1]
    c_poor = p_poor.LOCATION.unique() # poor countries

    # No Poor countries
    no_poor = poor[(poor['poverty'] == 0) & (poor['TIME'] == 2015)]
    #no_poor = no_poor[poor['TIME'] == 2015]
    c_no_poor = no_poor.LOCATION.unique() # no poor countries 2015
    e_countries = set(c_no_poor).intersection(c_poor) # Identifying emerging countries
    #e_countries # Emerging
    poor['emerging'] = poor['LOCATION'].isin(e_countries)
    e_poor = poor[poor['LOCATION'].isin(e_countries)]
    st.write(poor)
    st.write(e_poor)
    st.write(thresholds(df_poor=poor))

st.write(len(e_countries))
st.write(len(poor.LOCATION.unique()))
st.write(e_poor)

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