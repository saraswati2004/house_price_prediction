import streamlit as st
import pickle
import numpy as np
import joblib

st.set_page_config(page_title="USA Housing Price Predictor", layout="centered")

st.title("🏡 USA Housing Price Prediction App")
st.write("Enter the house details below to get an estimated price.")

# ---------------------------
# TRY LOADING MODEL (pickle or joblib)
# ---------------------------
model = None

model_path = "housing_model1.pkl"

# Try joblib first
try:
    model = joblib.load(model_path)
except:
    try:
        with open(model_path, "rb") as f:
            model = pickle.load(f)
    except Exception as e:
        st.error("❌ Unable to load the model. It may be corrupted or not a pickle/joblib file.")
        st.stop()

# ---------------------------
# INPUT FIELDS
# ---------------------------
income = st.number_input("Average Area Income", min_value=0.0, value=60000.0)
house_age = st.number_input("Average House Age", min_value=0.0, value=5.0)
rooms = st.number_input("Average Number of Rooms", min_value=0.0, value=6.0)
bedrooms = st.number_input("Average Number of Bedrooms", min_value=0.0, value=3.0)
population = st.number_input("Area Population", min_value=0.0, value=30000.0)

# Prepare input
features = np.array([[income, house_age, rooms, bedrooms, population]])

# ---------------------------
# PREDICT BUTTON
# ---------------------------
if st.button("Predict Housing Price"):
    try:
        price = model.predict(features)[0]
        st.success(f"🏠 Estimated House Price: **${price:,.2f}**")
    except Exception as e:
        st.error(f"Prediction error: {e}")
