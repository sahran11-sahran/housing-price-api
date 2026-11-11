# 🏠 California Housing Price Predictor

An end-to-end Machine Learning deployment project. This repository contains a Neural Network regression model trained to predict the median house value in California based on location and housing features, deployed as a live REST API with a full-stack frontend.

## 🚀 Live Demo
*   **API Documentation (Swagger UI):** [Click here to test the live API](https://housing-price-api-4ovw.onrender.com/docs)
*   **Interactive Frontend:** (Run locally using the instructions below)

## 📸 Screenshot
![Housing Predictor UI](app-screenshot.png)

## 🛠️ Tech Stack
*   **Machine Learning:** TensorFlow / Keras, Scikit-Learn (StandardScaler, SimpleImputer)
*   **Backend API:** FastAPI, Uvicorn, Pydantic
*   **Frontend UI:** Streamlit
*   **Infrastructure:** Docker, GitHub (CI/CD), Render (Cloud Hosting)

## 🧠 How it works
1.  **Preprocessing:** The API receives raw JSON data. It uses a saved `StandardScaler` to scale numerical features and manually one-hot encodes the categorical `ocean_proximity` feature.
2.  **Prediction:** The processed data is fed into a Neural Network (Regression) trained on the California Housing dataset.
3.  **Serving:** The FastAPI backend is containerized using Docker and hosted on Render. The Streamlit frontend sends HTTP requests to this live API.

## 💻 How to Run Locally

**1. Run the Backend (API) via Docker:**
```bash
# Build the image
docker build -t housing-api .

# Run the container
docker run --rm -p 8000:8000 housing-api
