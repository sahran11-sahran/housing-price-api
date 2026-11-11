FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the app and artifacts folders
# (Make sure you place housing_model.keras, housing_scaler.joblib, and columns.joblib inside the artifacts folder)
COPY app/ ./app/
COPY artifacts/ ./artifacts/

EXPOSE 8000

# Start the API
CMD ["uvicorn", "app.app:app", "--host", "0.0.0.0", "--port", "8000"]