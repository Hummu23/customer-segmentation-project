
import streamlit as st
import numpy as np
import joblib

scaler = joblib.load("models/scaler.joblib")
clf = joblib.load("models/classifier.joblib")

st.title("Customer Segment Predictor")

R = st.number_input("Recency", 0, 1000, 30)
F = st.number_input("Frequency", 0, 500, 5)
M = st.number_input("Monetary", 0.0, 100000.0, 100.0)

if st.button("Predict"):
    X = np.log1p([[R,F,M]])
    Xs = scaler.transform(X)
    pred = clf.predict(Xs)[0]
    st.success(f"Predicted Segment: {pred}")
