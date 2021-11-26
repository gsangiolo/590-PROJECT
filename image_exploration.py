#!/usr/bin/env python
# coding: utf-8

# In[25]:


import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tensorflow.keras import layers, models, Sequential, optimizers, losses, metrics, datasets, callbacks
from tensorflow import matmul
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import keras_tuner as kt
import seaborn as sns
import os
import pandas as pd
#get_ipython().run_line_magic('matplotlib', 'inline')
sns.set()


#dir = '../images_training_rev1/images_training_rev1/'
#filename = '100008.jpg'


#img = cv.imread(dir + filename)


#plt.imshow(img)

#print(img.shape)

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

    print(images.shape)
    print(len(y))

    images = images.reshape((images.shape[0], images[0].shape[0] * images[0].shape[1] * images[0].shape[2]))
    return images, y


def split_and_encode(images, y, test_size=0.2):
    x_train, x_test, y_train, y_test = train_test_split(images, y, test_size=test_size)
    
    if not multiclass:
        lb = LabelEncoder()
        y_train = lb.fit_transform(y_train)
        y_test = lb.transform(y_test)
    return x_train, x_test, y_train, y_test


# From Tensorflow Docs:
def model_builder(hp):
    model = Sequential()

    if not multiclass:
        model.add(layers.Flatten(input_shape=(1, x_train.shape[1])))

    # Tune the number of units in the first Dense layer
    # Choose an optimal value between 32-512
    hp_units = hp.Int('units', min_value=32, max_value=512, step=32)
    model.add(layers.Dense(units=hp_units, activation='relu'))
    if not multiclass:
        model.add(layers.Dense(n_classes, activation='softmax'))
    else:
        model.add(layers.Dense(n_classes, activation='sigmoid'))

    # Tune the learning rate for the optimizer
    # Choose an optimal value from 0.01, 0.001, or 0.0001
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])

    if not multiclass:
        model.compile(optimizer=optimizers.Adam(learning_rate=hp_learning_rate),
                    loss=losses.SparseCategoricalCrossentropy(from_logits=True),
                    metrics=['accuracy'])
    else:
        model.compile(optimizer=optimizers.Adam(learning_rate=hp_learning_rate),
                loss='binary_crossentropy',
                metrics=['accuracy'])

    return model


def tune_hyperparams(x_train, x_test, y_train, y_test):
    tuner = kt.Hyperband(model_builder,
                         objective='val_accuracy',
                         max_epochs=10,
                         factor=3,
                         directory='my_dir',
                         project_name='intro_to_kt')

    stop_early = callbacks.EarlyStopping(monitor='val_loss', patience=5)

    tuner.search(x_train, y_train, epochs=50, validation_split=0.2, batch_size=128, callbacks=[stop_early])
    #tuner.search(it, epochs=50, validation_split=0.2, batch_size=128, callbacks=[stop_early])

    # Get the optimal hyperparameters
    best_hps=tuner.get_best_hyperparameters(num_trials=1)[0]

    print(f"""
    The hyperparameter search is complete. The optimal number of units in the first densely-connected
    layer is {best_hps.get('units')} and the optimal learning rate for the optimizer
    is {best_hps.get('learning_rate')}.
    """)
    return best_hps

def fit_model_from_hps(best_hps, x_train, y_train, x_test, y_test):
    # Build the model with the optimal hyperparameters and train it on the data for 50 epochs
    model = tuner.hypermodel.build(best_hps)
    history = model.fit(x_train, y_train, epochs=50, validation_split=0.2)

    val_acc_per_epoch = history.history['val_accuracy']
    best_epoch = val_acc_per_epoch.index(max(val_acc_per_epoch)) + 1
    print('Best epoch: %d' % (best_epoch,))

    hypermodel = tuner.hypermodel.build(best_hps)

    # Retrain the model
    history = hypermodel.fit(x_train, y_train, epochs=best_epoch, validation_split=0.2)


    eval_result = hypermodel.evaluate(x_test, y_test)
    print("[test loss, test accuracy]:", eval_result)
    return history, eval_result

if __name__ == '__main__':
    global multiclass
    multiclass = True
    images, image_names = get_all_images_from_directory('../images_training_rev1/images_training_rev1/')
    y, classes = get_image_labels(images, image_names, '../images_training_rev1/training_solutions_rev1/training_solutions_rev1.csv', multiclass_threshold=0.5)
    global n_classes
    n_classes = len(classes)
    images, y = generate_new_image_data(images, y, max_it=2)
    x_train, x_test, y_train, y_test = split_and_encode(images, y, test_size=0.2)
    best_hps = tune_hyperparams(x_train, x_test, y_train, y_test)
    history, eval_result = fit_model_from_hps(best_hps, x_train, y_train, x_test, y_test)

