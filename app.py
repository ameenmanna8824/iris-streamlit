import streamlit as st
from sklearn.datasets import load_iris
data = load_iris()
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
x = data.data
y = data.target
model.fit(x,y)
st.header("Iris Flower classification")
sl = st.number_input("Enter Sepal Length")
sw = st.number_input("Enter Sepal Width")
pl = st.number_input("Enter Petal Length")
pw = st.number_input("Enter Petal Width")
y_pred = model.predict([[sl,sw,pl,pw]])
op = data.target_names[y_pred[0]]
st.write(op)
