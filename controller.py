from fastapi import FastAPI
import numpy as np
import service
from model import ModelParams

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Hello, World!"}

@app.post("/predict")
async def predict_image(image: np.array):
    predictor = service.ImagePredictor()
    return predictor.predict_image(image)

@app.post("/train-model")
async def train_custom_model(model_params: ModelParams):
    return model_params
