# app.py
from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np

# Initialize the FastAPI app
app = FastAPI()

# Load the pre-trained car price detection model
with open("car_price_detection.pkl", "rb") as model_file:
    model = pickle.load(model_file)

# Define the input data schema using Pydantic
class CarPriceInput(BaseModel):
    engine_size: float
    city_mpg: float

# Root endpoint for health check
@app.get("/")
def read_root():
    return {"message": "Car Price Detection API is running"}

# Prediction endpoint
@app.post("/predict/")
def predict_car_price(input_data: CarPriceInput):
    # Extract the input features as a NumPy array
    features = np.array([[input_data.engine_size, input_data.city_mpg]])
    
    # Make the prediction using the loaded model
    predicted_price = model.predict(features)[0]
    
    # Return the prediction as a JSON response
    return {"predicted_price": predicted_price}
