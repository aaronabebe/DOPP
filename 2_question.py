import pandas as pd 
import streamlit as st 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import cross_val_score
from sklearn import linear_model
from sklearn import neighbors
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
import category_encoders as ec
from sklearn import preprocessing



st.markdown("# DOPP3")
st.markdown('## Which characteristics are predictive for countries with large populations living in extreme poverty?')

with st.echo():
    # READ TRANSFORMED CSV FILE
    raw = pd.read_csv("transformed.csv")  
    #st.write(raw.head(100))

    feature_descriptions = pd.read_csv("feature_descriptions.csv")
    #st.write(feature_descriptions)

    # FEATURES WITH LESS THAN 50% MISSING VALUES
    features = feature_descriptions.where(feature_descriptions['na_percent']<=50.0).dropna(0)
    
    # ONLY DEMOGRAFIC FEATURES!
    cols = features['Unnamed: 0'].tolist()
    #cols_to_drop = 7:13 + 18:25
    cols = cols[0:7]+ cols[13:18] + [cols[25]]
    #st.write(cols)

    dataset = raw[cols]


st.write(dataset.head(100))  
    

st.markdown('## Exploratory Data Analysis')
st.markdown('After selection of only demographic features with less than 50% missing values we are left with the following attributes: ')

# PRINTING DESCRIPTIONS OF FEATURES
for col in cols:
    desc = pd.DataFrame(features.where(features['Unnamed: 0'] == col).dropna(0)['descriptions'])
    st.write(col, ':', desc.to_string(header=False, index=False)) 
st.write('All attributes except *Location*(**categorical**) are **numeric**')

# CORRELATION MATRIX
plt.figure(figsize=(35,30))
sns.set(font_scale=3.1)
sns.heatmap(dataset.corr(),annot=True)
st.pyplot()

st.write('From the correlation matrix, we can see that all *population measures correlate* with each other (as excepted). Moreover, we can notice a strong negative correlation between *fertility rate* and *life expectancy at birth* as well as *life expectancy at birth* and *mortality rate*.')

#st.write(dataset.drop(labels=[cols[0], cols[-1]], axis=1).head(10))

# PLOT DISTRIBUTION OF THE ATTRIBUTES
fig = dataset.drop(labels=[cols[0], cols[-1]], axis=1).hist(figsize=(14,14), xlabelsize=10, ylabelsize=10, bins=20)
[x.title.set_size(20) for x in fig.ravel()]
plt.tight_layout()
st.pyplot()

st.markdown('## Missing Values')
st.markdown('The dataset contains a lot of missing values. We did **linear interpolation** for *each attribute* in *each country* separately. The values that were **not handled by the interpolation** were set to the *mean* of the column (probably because they are on the beginning/end of the column). Some attributes of the certain countries were without any values, we set those to 0. ')

with st.echo():
    by_country = dataset.groupby(by=dataset['LOCATION'])  
    dataset_full = pd.DataFrame(columns=cols)


    for name, group in by_country :
        tdf = pd.DataFrame(columns=cols) 

        tdf['TIME'] = group['TIME']
        tdf['poverty'] = group['poverty']

        # cols with all NaN values
        all_null = group.isna().all()  
        null_cols = all_null.where(all_null == 1).dropna(0).index.tolist()
        tdf[null_cols] = 0

        # cols for interpolation
        cols_to_int = all_null.where(all_null == 0).dropna(0).index.tolist()[2:]
        cols_to_int.remove('poverty')

        #st.write(group[cols_to_int].isnull().values.sum())

        tdf[cols_to_int] = group[cols_to_int].interpolate(method='linear', axis=0)
        tdf['LOCATION'] = name 

        # fill the NaN values that were not interpolated
        tdf.fillna(tdf.mean(), inplace=True)

        dataset_full = pd.concat([dataset_full,tdf])

dataset_full.sort_index(inplace=True)
st.write(dataset_full.head(100))
#st.write(X.shape)
#st.write(X.isnull().values.any())

st.markdown('## ML Models')

with st.echo():
    y = dataset_full['poverty']
    y = y.apply(lambda x: 1 if x==True else 0)
    #st.write(y.head(100))

    X = dataset_full.drop(labels=['poverty'], axis=1)
    encoder = ec.BinaryEncoder(cols=['LOCATION'])
    L_enc = encoder.fit_transform(X['LOCATION'])
    X = pd.concat([X.drop(labels=['LOCATION'], axis=1),L_enc], axis=1)
    
    #st.write(X.head(100))

    skf = KFold(n_splits=10, random_state = 30)

    def print_performance (classifier, X, y, scores= ['accuracy', 'precision', 'recall'], model=''):
        for score in scores:
            #cv1 = cross_val_score(classifier, X, y, cv=skf.split(X,y), scoring=score).mean()
            cv2 = cross_val_score(classifier, X, y, cv=10, scoring=score)
            cv2_m = cv2.mean()
            cv2_sd = cv2.std()
            st.write(model + ' ' + score +" : " + str(round(cv2_m, 5))+ ' +- '+ str(round(cv2_sd, 5)))

    def r_classifier (X, y, alpha=1.0, fit_intercept=True, normalize=True, solver='auto', max_iter=1000, tol=0.0001) :
        reg = linear_model.RidgeClassifier(alpha=alpha, fit_intercept=fit_intercept, normalize=normalize, max_iter=max_iter, tol=0.001, solver='auto', random_state=30)
        print_performance(reg, X , y, model='Ridge Calssifier', scores= ['accuracy'])
        reg.fit(X,y)
        return reg   

    # pogledaj prije koristenja
    def knn (X, y, n_neighbors=5, weights='uniform', algorithm='auto', leaf_size=30, p=2, metric='minkowski', metric_params=None, n_jobs=None) :
        reg = neighbors.KNeighborsRegressor(n_neighbors=n_neighbors, weights=weights, algorithm=algorithm, leaf_size=leaf_size, p=p, metric=metric, metric_params=metric_params, n_jobs=n_jobs)
        print_performance(reg, X , y, model='KNN', scores=["r2" ,"explained_variance", "neg_mean_squared_error", "neg_mean_absolute_error"])
        reg.fit(X,y)
        return reg 

    def find_best_parameters (classifier, parameters, X, y):
        clf = GridSearchCV(classifier, parameters, cv=10)
        clf.fit(X, y)
        return clf.best_params_

ridge = r_classifier(X,y)
parameters= {'alpha':list(np.arange(0.1, 10.0, 0.1)), 'solver':['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga']}
find_best_parameters(ridge, parameters, X, y)

R_coef = ridge.coef_
X_cols = X.columns

for i in range(0,len(X_cols)):
    st.write(str(X_cols[i]) + ' : ' + str(round(R_coef[0,i],6)))

st.balloons()