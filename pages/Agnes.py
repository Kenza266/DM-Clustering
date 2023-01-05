# import time 
import pickle
import numpy as np
import pandas as pd
import streamlit as st
from Agnes import Agnes
from utils import report

from sklearn.metrics import silhouette_score
from sklearn.metrics import adjusted_rand_score
from sklearn.cluster import AgglomerativeClustering as SK_Agnes


st.set_page_config(layout="wide")
st.title('Clustering with Agnes')

data = pd.read_csv('datset.csv') 
cat = ['OverTime', 'MaritalStatus', 'JobRole', 'Gender', 'EducationField', 'Department', 'BusinessTravel', 'Attrition']
data = data.drop(['EmployeeCount', 'Over18', 'StandardHours', 'EmployeeNumber'], axis=1)
for i in cat:
    data[i] = (data[i].astype('category').cat.codes).apply(np.int64)
X, y = data.drop(['Attrition'], axis=1), data['Attrition']

X, y = np.array(X), list(y)

data_x = data.drop(['Attrition'], axis=1)
X_standardized = np.array((data_x - data_x.mean()) / data_x.std())
X_normalized = np.array(((data_x - data_x.min()) / (data_x.max() - data_x.min())))

similarity = st.sidebar.selectbox(
    'Distance function',
    ('Manhattan', 'Hamming')) 

if similarity == 'Manhattan':
    preprocessing = st.sidebar.selectbox(
        'Preprocessing',
        ('Raw', 'Norm', 'Std')) 
    if preprocessing == 'Raw':
        input = X
        file = 'Agnes//Distances_Manhattan.pkl'
    elif preprocessing == 'Norm':
        input = X_normalized
        file = 'Agnes//Distances_Manhattan_Norm.pkl'
    elif preprocessing == 'Std':
        input = X_normalized
        file = 'Agnes//Distances_Manhattan_Std.pkl'        

elif similarity == 'Hamming':  
    input = X
    file = 'Agnes//Distances_Hamming.pkl'

linkage = st.sidebar.selectbox(
    'Linkage',
    ('Average', 'Complete', 'Single')) 

agnes = Agnes(similarity.lower()) 
with open(file, 'rb') as f:
    dist_matrix = pickle.load(f) 

if st.sidebar.button('Run'):
    agnes = SK_Agnes(2, affinity=similarity.lower(), linkage=linkage.lower())
    clustering = agnes.fit(input) 

    st.text(report(y, list(clustering.labels_)))
    st.text('Silhouette score:' + str(silhouette_score(X, list(clustering.labels_))))
    st.text('Adjusted rand score:' + str(adjusted_rand_score(y, list(clustering.labels_))))