import numpy as np
from keras import models

class ImagePredictor:
    def predict_image(image):
        model = models.load_model('super_simple_model')
        return model.predict(image)