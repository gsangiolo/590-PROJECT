import numpy as np
import cv2 as cv
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models, Sequential, optimizers, losses, metrics, datasets, callbacks
import os
import pandas as pd
import keras_tuner as kt


global multiclass
multiclass = True


def get_all_images_from_directory(dir, max_images=1000):
    for filename in os.listdir(dir):
        sample_img = cv.imread(os.path.join(dir, filename))
        break
    shape = sample_img.shape
    images = np.zeros(shape=(max_images, shape[0], shape[1], shape[2]))
    image_names = []
    #max_images = 1000
    count = 0

    for filename in os.listdir(dir):
        if count >= max_images:
            break
        #print(os.path.join(dir, filename))
        img = cv.imread(os.path.join(dir, filename))
        images[count] = img
        image_names.append(filename[:-4])
        count += 1

    #images = np.array(images)
    return images, image_names


def get_image_labels(images, image_names, solutions_path, multiclass_threshold=0.5):
#solutions = pd.read_csv('../images_training_rev1/training_solutions_rev1/training_solutions_rev1.csv')
    solutions = pd.read_csv(solutions_path)

    y = []

    if not multiclass:
        for name in image_names:
            y.append(np.argmax(np.array(solutions[solutions['GalaxyID'] == int(name)].drop('GalaxyID', axis=1))))
    else:
        for name in image_names:
            row = solutions[solutions['GalaxyID']==int(name)].drop('GalaxyID', axis=1).values.tolist()[0]
            y.append(np.array([0 if r < multiclass_threshold else 1 for r in row]))

    classes = solutions.columns[1:].tolist()
    y = np.array(y)
    return y, classes


def generate_new_image_data(images, y, max_it=10):
    datagen = ImageDataGenerator()
    datagen.fit(images)

    print(images.shape)
    print(len(y))

    it = datagen.flow(images, y)

    count = 0
    for item in it:
        images = np.append(images, item[0], axis=0)
        print(y.shape)
        print(len(item[1]))
        y = np.append(y, item[1], axis=0)
        if count >= max_it:
            break
        if count % 100 == 0:
            print(count)
        count += 1

    # print(images.shape)
    # print(len(y))

    images = images.reshape((images.shape[0], images[0].shape[0] * images[0].shape[1] * images[0].shape[2]))
    return images, y


