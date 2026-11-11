from fastapi import FastAPI
from pydantic import BaseModel
import tensorflow as tf
import joblib
import numpy as np
import pandas as pd

app = FastAPI(title="Housing Price Predictor API")

# Load artifacts from the /app/artifacts/ folder inside Docker
model = tf.keras.models.load_model('/app/artifacts/housing_model.keras')
scaler = joblib.load('/app/artifacts/housing_scaler.joblib')
expected_columns = joblib.load('/app/artifacts/columns.joblib')

# --- Define the Input Schema ---
class HouseFeatures(BaseModel):
    longitude: float
    latitude: float
    housing_median_age: float
    total_rooms: float
    total_bedrooms: float
    population: float
    households: float
    median_income: float
    # We handle ocean_proximity via a string, and manually map it to 1/0 below
    ocean_proximity: str 

@app.get("/health")
def health_check():
    return {"status": "ok"}
@app.get("/")
def read_root():
    return {"message": "Housing API is live!"}

@app.post("/predict")
def predict_price(features: HouseFeatures):
    # 1. Convert input to a dictionary
    data = features.dict()

    # 2. Handle the categorical variable (One-Hot Encoding manually for the API)
    proximity = data.pop('ocean_proximity').upper()
    
    # We must match the categories from the training data 
    data['ocean_proximity_<1H OCEAN'] = 1 if proximity == '<1H OCEAN' else 0
    data['ocean_proximity_INLAND'] = 1 if proximity == 'INLAND' else 0
    data['ocean_proximity_ISLAND'] = 1 if proximity == 'ISLAND' else 0
    data['ocean_proximity_NEAR BAY'] = 1 if proximity == 'NEAR BAY' else 0
    data['ocean_proximity_NEAR OCEAN'] = 1 if proximity == 'NEAR OCEAN' else 0

    # 3. Create a DataFrame using the exact column order the model was trained on
    input_df = pd.DataFrame([data])
    
    # Ensure all columns exist, fill missing ones with 0 just in case
    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col] = 0
            
    input_df = input_df[expected_columns]

    # 4. Scale and Predict
    scaled_features = scaler.transform(input_df)
    
    # 5. Get the raw prediction
    predicted_value = model.predict(scaled_features, verbose=0)[0][0]

    # 6. Return the prediction
    return {
        "predicted_house_value": float(predicted_value)
    }