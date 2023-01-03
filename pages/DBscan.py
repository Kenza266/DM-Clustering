import time 
import numpy as np
import pandas as pd
import streamlit as st
from DBscan import DBscan

st.set_page_config(layout="wide")
st.title('Clustering with DBscan')

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
    if similarity == 'Raw':
        input = X
        dist_matrix = np.load('DBscan//Distances_Manhattan.npy')
    elif similarity == 'Norm':
        input = X_normalized
        dist_matrix = np.load('DBscan//Distances_Manhattan_Norm.npy')
    elif similarity == 'Std':
        input = X_normalized
        dist_matrix = np.load('DBscan//Distances_Manhattan_Std.npy')         

elif similarity == 'Hamming':
    input = X
    dist_matrix = np.load('DBscan//Distances_Hamming.npy')

eps = st.sidebar.number_input('Eps', step=1, min_value=1, max_value=2500, value=100)
min_samples = st.sidebar.number_input('MinPts', step=1, min_value=1, max_value=50, value=25)

if st.sidebar.button('Run'):
    start = time.time()
    dbscan = DBscan(eps=eps, min_samples=min_samples, similarity=similarity.lower()) 
    clusters = dbscan.cluster(input, dist_matrix=dist_matrix) 
    end = time.time() - start
    st.write(end)