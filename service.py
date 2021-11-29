import numpy as np
from keras import models

class ImagePredictor:
    def predict_image(self, image):
        image_reshape = np.reshape(image, (1, np.prod(image.shape)))
        print(image_reshape.shape)
        print("\n\n")
        model = models.load_model('super_simple_model')
        return model.predict(image_reshape)
