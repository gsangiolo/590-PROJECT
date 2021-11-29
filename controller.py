from fastapi import FastAPI, File, UploadFile
import numpy as np
import service
import cv2

# from fastapi.responses import FileResponse

# #from model import ModelParams

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Hello, World!"}


# # not sure if this will work with entering image ID
# @app.get("/images/{image_id}")
# async def show_image(image_id: str):
#     full_image_id = image_id + ".jpg"
#     return FileResponse(full_image_id)


# # Alternative option, have the user upload the image
# # and return predicted class
         

@app.post("/predict")
async def predict_image(image: UploadFile = File(...)):
#     imagetype = image.filename.split(".")[-1] in ("jpg", "jpeg", "png")
#     if not imagetype:
#         return "Image filetype must be .jpg, .jpeg, or .png"
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
