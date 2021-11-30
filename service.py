import zipfile
import pickle
import os
import random
import cv2
import shutil
import numpy as np
import pandas as pd
from keras import models
from boto3 import client
from tensorflow.python.lib.io import file_io
# import numpy as np

class ImagePredictor:

    def __init__(self):
        self.conn = client('s3', aws_access_key_id='AKIAQW4FQ6E4M74BPQXK', aws_secret_access_key='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')

    def predict_image(self, image, modelName='super_simple_model'):
        image = cv2.resize(image, (424, 424), interpolation = cv2.INTER_AREA)
        image_reshape = np.reshape(image, (1, 424*424*3)) #np.prod(image.shape)))
        print(image_reshape.shape)
        print("\n\n")
        model = self.loadModelFromS3(modelName)
        return model.predict(image_reshape)

    def getAllModels(self):
        return [key['Key'][7:-4] for key in self.conn.list_objects_v2(Bucket='anly590-project', Prefix='models/')['Contents'] if 'images_training_rev1' not in key['Key']]

    def loadModelFromS3(self, modelName):
        self.conn.download_file('anly590-project', 'models/' + modelName + '.zip', 'model.zip')
        with zipfile.ZipFile('model.zip', 'r') as zip_ref:
            zip_ref.extractall('model')
        try: # Simple fix -- some models have a directory underneath, others don't!
            model = models.load_model('model/' + modelName)
        except:
            model = pickle.load(open('model/' + modelName, 'rb'))
        os.remove('model.zip')
        shutil.rmtree('model')
        return model

    def getImageById(self, imageId):
        self.conn.download_file('anly590-project', 'images_training_rev1/images_training_rev1/' + imageId + '.jpg', 'image.jpg')
        image = cv2.imread('image.jpg')
        res, img_png = cv2.imencode(".png", image)
        os.remove('image.jpg')
        return img_png

    def getRandomImage(self):
        image_keys = self.getAllImageIds()
        imageId = image_keys[random.randint(0, len(image_keys))]
        self.conn.download_file('anly590-project', imageId, 'image.jpg')
        image = cv2.imread('image.jpg')
        res, img_png = cv2.imencode(".png", image)
        os.remove('image.jpg')
        return img_png

    def getAllImageIds(self):
        objects = self.conn.list_objects_v2(Bucket='anly590-project', Prefix='images_training_rev1/')
        image_keys = [obj['Key'] for obj in objects['Contents']]
        return image_keys
    
    def getRandomImageByClass(self, myClass, threshold=0.5):
        self.conn.download_file('anly590-project', 'images_training_rev1/training_solutions_rev1/training_solutions_rev1.csv', 'solutions.csv')
        solutions = pd.read_csv('solutions.csv', index_col='GalaxyID', na_values=['(NA)']).fillna(0)
        myClass = int(myClass)
        solutions_in_class = solutions[solutions[solutions.columns[myClass]] >= threshold]
        image_keys = solutions_in_class.index.tolist()
        success = False
        count = 0
        while not success:
            imageId = 'images_training_rev1/images_training_rev1/' + str(image_keys[random.randint(0, len(image_keys) - 1)]) + '.jpg'
            try:
                self.conn.download_file('anly590-project', imageId, 'image.jpg')
                success = True
            except:
                print("Could not find this image? ", imageId)
            count += 1
            if count >= 1000:
                os.remove('solutions.csv')
                return None
        image = cv2.imread('image.jpg')
        res, img_png = cv2.imencode(".png", image)
        os.remove('image.jpg')
        os.remove('solutions.csv')
        return img_png
