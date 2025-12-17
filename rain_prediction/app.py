import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras.models import load_model

model = load_model("rain_model.h5")
scaler = joblib.load("scaler.pkl")

st.title("🌦️ Rain Prediction ANN")
st.sidebar.header("Enter Weather Details")

def user_input_features():
    MinTemp = st.sidebar.number_input("Min Temperature (°C)", -10.0, 40.0, 10.0)
    MaxTemp = st.sidebar.number_input("Max Temperature (°C)", -5.0, 45.0, 25.0)
    Rainfall = st.sidebar.number_input("Rainfall (mm)", 0.0, 500.0, 0.0)
    Evaporation = st.sidebar.number_input("Evaporation (mm)", 0.0, 30.0, 5.0)
    Sunshine = st.sidebar.number_input("Sunshine (hours)", 0.0, 15.0, 5.0)
    WindGustSpeed = st.sidebar.number_input("Wind Gust Speed (km/h)", 0, 150, 20)
    WindSpeed9am = st.sidebar.number_input("Wind Speed 9AM (km/h)", 0, 100, 15)
    WindSpeed3pm = st.sidebar.number_input("Wind Speed 3PM (km/h)", 0, 100, 20)
    Humidity9am = st.sidebar.number_input("Humidity 9AM (%)", 0, 100, 50)
    Humidity3pm = st.sidebar.number_input("Humidity 3PM (%)", 0, 100, 50)
    Pressure9am = st.sidebar.number_input("Pressure 9AM (hPa)", 980, 1050, 1010)
    Pressure3pm = st.sidebar.number_input("Pressure 3PM (hPa)", 980, 1050, 1012)
    Temp9am = st.sidebar.number_input("Temp 9AM (°C)", -5, 45, 20)
    Temp3pm = st.sidebar.number_input("Temp 3PM (°C)", -5, 45, 25)
    data = {
        "MinTemp": MinTemp,
        "MaxTemp": MaxTemp,
        "Rainfall": Rainfall,
        "Evaporation": Evaporation,
        "Sunshine": Sunshine,
        "WindGustSpeed": WindGustSpeed,
        "WindSpeed9am": WindSpeed9am,
        "WindSpeed3pm": WindSpeed3pm,
        "Humidity9am": Humidity9am,
        "Humidity3pm": Humidity3pm,
        "Pressure9am": Pressure9am,
        "Pressure3pm": Pressure3pm,
        "Temp9am": Temp9am,
        "Temp3pm": Temp3pm
    }
    return pd.DataFrame(data, index=[0])

input_df = user_input_features()
st.subheader("Input Parameters")
st.write(input_df)

input_scaled = scaler.transform(input_df)
pred_prob = model.predict(input_scaled).flatten()
pred_class = (pred_prob > 0.5).astype(int)
st.subheader("Prediction")
st.success("🌧️ Rain Tomorrow" if pred_class[0] == 1 else "☀️ No Rain Tomorrow")
st.subheader("Probability")
st.write(f"{pred_prob[0]*100:.2f}%")

st.subheader("Confusion Matrix (Test Data)")
try:
    X_test = joblib.load("X_test.pkl")
    y_test = joblib.load("y_test.pkl")
    y_pred_test = model.predict(X_test).flatten()
    y_pred_labels = (y_pred_test > 0.5).astype(int)
    cm = confusion_matrix(y_test, y_pred_labels)
    fig, ax = plt.subplots()
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['No Rain','Rain'])
    disp.plot(cmap=plt.cm.Blues, ax=ax)
    st.pyplot(fig)
except:
    st.info("Test data not available.")

st.subheader("Prediction Distribution (Train & Test)")
try:
    pred_train = joblib.load("pred_train.pkl")
    y_train = joblib.load("y_train.pkl")
    pred_test = joblib.load("pred_test.pkl")
    y_test = joblib.load("y_test.pkl")
    fig, axes = plt.subplots(1,2, figsize=(12,5))
    axes[0].hist(pred_train[y_train==0], bins=20, alpha=0.5, label='No Rain')
    axes[0].hist(pred_train[y_train==1], bins=20, alpha=0.5, label='Rain')
    axes[0].set_title('Training Data Predictions')
    axes[0].legend()
    axes[1].hist(pred_test[y_test==0], bins=20, alpha=0.5, label='No Rain')
    axes[1].hist(pred_test[y_test==1], bins=20, alpha=0.5, label='Rain')
    axes[1].set_title('Test Data Predictions')
    axes[1].legend()
    st.pyplot(fig)
except:
    st.info("Prediction distributions not available.")