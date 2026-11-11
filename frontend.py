import streamlit as st
import requests
import pandas as pd

# 1. Set up the page
st.set_page_config(page_title="California Housing Predictor", page_icon="🏠", layout="wide")

st.title("🏠 California Housing Price Predictor")
st.markdown("""
This app uses a Neural Network deployed on **Render** to predict the median house value based on location and housing features.
Enter the details below and click **Predict**!
""")

# 2. Create columns for the layout
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("📍 Location")
    longitude = st.slider("Longitude", -124.3, -114.3, -122.2)
    latitude = st.slider("Latitude", 32.5, 42.0, 37.8)
    ocean_proximity = st.selectbox(
        "Ocean Proximity", 
        ["NEAR BAY", "INLAND", "<1H OCEAN", "NEAR OCEAN", "ISLAND"]
    )
    housing_median_age = st.slider("Housing Median Age", 1, 52, 25)

with col2:
    st.subheader("🏠 House Features")
    total_rooms = st.number_input("Total Rooms", min_value=1, value=2000)
    total_bedrooms = st.number_input("Total Bedrooms", min_value=1, value=400)
    households = st.number_input("Households", min_value=1, value=500)
    population = st.number_input("Population", min_value=1, value=1000)

with col3:
    st.subheader("💰 Income")
    median_income = st.number_input("Median Income (in $10k units)", min_value=0.5, max_value=15.0, value=5.0, step=0.1)
    st.info("Note: The model was trained on an older dataset where income was measured in $10,000s.")

st.divider()

# 3. The Predict Button
if st.button("Predict House Price", type="primary", use_container_width=True):
    
    # The payload exactly matches the FastAPI HouseFeatures model
    payload = {
        "longitude": longitude,
        "latitude": latitude,
        "housing_median_age": housing_median_age,
        "total_rooms": total_rooms,
        "total_bedrooms": total_bedrooms,
        "population": population,
        "households": households,
        "median_income": median_income,
        "ocean_proximity": ocean_proximity
    }

    # YOUR LIVE RENDER URL
    api_url = "https://housing-price-api-4ovw.onrender.com/predict"
    
    with st.spinner("Asking the Neural Network..."):
        try:
            response = requests.post(api_url, json=payload)
            response.raise_for_status() # Check for HTTP errors
            result = response.json()
            
            predicted_price = result["predicted_house_value"]
            
            # Display the result beautifully
            st.success("Prediction Successful!")
            
            # Format as currency
            formatted_price = f"${predicted_price:,.2f}"
            st.metric(label="Estimated Median House Value", value=formatted_price)
            
            # Add some context
            if predicted_price > 400000:
                st.balloons()
                st.write("🎉 That's a highly expensive property!")
            else:
                st.write("This is within the typical range for the area.")

        except requests.exceptions.RequestException as e:
            st.error(f"Could not connect to the API. Is the Render server awake? Error: {e}")