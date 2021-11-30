import zipfile
import os
import cv2
import shutil
import numpy as np
from keras import models
from boto3 import client
from tensorflow.python.lib.io import file_io

class ImagePredictor:

    def __init__(self):
        self.conn = client('s3', aws_access_key_id='AKIAQW4FQ6E4M74BPQXK', aws_secret_access_key='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')

    def predict_image(self, image, modelName='super_simple_model'):
        image_reshape = np.reshape(image, (1, np.prod(image.shape)))
        print(image_reshape.shape)
        print("\n\n")
        model = self.loadModelFromS3(modelName)
        return model.predict(image_reshape)

    def getAllModels(self):
        return [key['Key'][:-4] for key in self.conn.list_objects(Bucket='anly590-project')['Contents'] if 'images_training_rev1' not in key['Key']]

    def loadModelFromS3(self, modelName):
        self.conn.download_file('anly590-project', modelName + '.zip', 'model.zip')
        with zipfile.ZipFile('model.zip', 'r') as zip_ref:
            zip_ref.extractall('model')
        try: # Simple fix -- some models have a directory underneath, others don't!
            model = models.load_model('model/' + modelName)
        except:
            model = models.load_model('model')
        os.remove('model.zip')
        shutil.rmtree('model')
        return model

    def getImageById(self, imageId):
        self.conn.download_file('anly590-project', 'images_training_rev1/' + imageId + '.jpg', 'image.jpg')
        image = cv2.imread('image.jpg')
        res, img_png = cv2.imencode(".png", image)
        os.remove('image.jpg')
        return img_png

    def getRandomImage(self):
        objects = self.conn.list_objects_v2('anly590-project', Prefix='images_training_rev1')
        image_keys = [obj['Key'] for obj in objects['Contents']]
        imageId = image_keys[random.randint(0, len(image_keys))]
        self.conn.download_file('anly590-project', 'images_training_rev1/' + imageId + '.jpg', 'image.jpg')
        image = cv2.imread('image.jpg')
        res, img_png = cv2.imencode(".png", image)
        os.remove('image.jpg')
        return img_png
