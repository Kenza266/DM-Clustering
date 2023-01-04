import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st

st.set_page_config(layout="wide")

st.title('Dataset')

data = pd.read_csv('datset.csv') 
cat = ['OverTime', 'MaritalStatus', 'JobRole', 'Gender', 'EducationField', 'Department', 'BusinessTravel', 'Attrition']
data = data.drop(['EmployeeCount', 'Over18', 'StandardHours', 'EmployeeNumber'], axis=1)
for i in cat:
    data[i] = (data[i].astype('category').cat.codes).apply(np.int64)

st.dataframe(data)

tsne = Image.open('tsne.png')
pca = Image.open('pca.png')

st.image(tsne, caption='Images//TSNE Distribution', width=850)
st.image(pca, caption='Images//PCA Distribution',  width=850)