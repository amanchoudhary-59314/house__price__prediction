import streamlit as st
import pandas as pd
from src.pipelines.predict_pipeline import CustomData, PredictPipeline

# Title of the app
st.title("House Price Prediction")

# Sidebar for user input
st.sidebar.header("Input Parameters")

def user_input_features():
    area = st.sidebar.number_input("Area", min_value=0, value=7200)
    bedrooms = st.sidebar.number_input("Bedrooms", min_value=0, value=3)
    bathrooms = st.sidebar.number_input("Bathrooms", min_value=0, value=2)
    stories = st.sidebar.number_input("Stories", min_value=0, value=1)
    mainroad = st.sidebar.selectbox("Mainroad", ["yes", "no"])
    guestroom = st.sidebar.selectbox("Guestroom", ["yes", "no"])
    basement = st.sidebar.selectbox("Basement", ["yes", "no"])
    hotwaterheating = st.sidebar.selectbox("Hotwaterheating", ["yes", "no"])
    airconditioning = st.sidebar.selectbox("Airconditioning", ["yes", "no"])
    parking = st.sidebar.number_input("Parking", min_value=0, value=3)
    prefarea = st.sidebar.selectbox("Prefarea", ["yes", "no"])
    furnishingstatus = st.sidebar.selectbox("Furnishingstatus", ["furnished", "semi-furnished", "unfurnished"])
    
    data = CustomData(
        area=area,
        bedrooms=bedrooms,
        bathrooms=bathrooms,
        stories=stories,
        mainroad=mainroad,
        guestroom=guestroom,
        basement=basement,
        hotwaterheating=hotwaterheating,
        airconditioning=airconditioning,
        parking=parking,
        prefarea=prefarea,
        furnishingstatus=furnishingstatus
    )
    return data

data = user_input_features()
pred_df = data.get_data_as_data_frame()

# Button to trigger prediction
if st.sidebar.button("Predict"):
    predict_pipeline = PredictPipeline()
    results = predict_pipeline.predict(pred_df)
    st.write(f"Predicted House Price: {results[0]}")