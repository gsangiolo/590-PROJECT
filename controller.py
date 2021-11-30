from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import service
import cv2
#from model import ModelParams

app = FastAPI()

origins = ['*']

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

image_service = service.ImagePredictor()

@app.get("/")
async def root():
    return {"message": "Hello, World!"}

@app.get("/models-list")
async def get_all_models():
    return {"result": image_service.getAllModels()}

@app.post("/predict")
async def predict_image(model: str = Form(...), image: UploadFile = File(...)):
    img_contents = await image.read()
    img_arr = np.fromstring(img_contents, np.uint8)
    img = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
    print("Model: ", model)
    # predictor = service.ImagePredictor()
    result = image_service.predict_image(img, model)
    print(result)
    return {"result": result.tolist()}

#@app.post("/train-model")
#async def train_custom_model(model_params: ModelParams):
#    return model_params
