from fastapi import FastAPI, File, UploadFile
import numpy as np
import service
import cv2
#from model import ModelParams

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Hello, World!"}

@app.post("/predict")
async def predict_image(image: UploadFile = File(...)):
    img_contents = await image.read()
    img_arr = np.fromstring(img_contents, np.uint8)
    img = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
    
    predictor = service.ImagePredictor()
    result = predictor.predict_image(img)
    print(result)
    return {"result": 200}

#@app.post("/train-model")
#async def train_custom_model(model_params: ModelParams):
#    return model_params
