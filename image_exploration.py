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
import keras_tuner as kt
import seaborn as sns
import os
import pandas as pd
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set()


# In[13]:


dir = '../images_training_rev1/images_training_rev1/'
filename = '100008.jpg'


# In[14]:


img = cv.imread(dir + filename)


# In[15]:


plt.imshow(img)


# In[16]:


print(img)


# In[18]:


for filename in os.listdir(dir):
    print(os.path.join(dir, filename))


# In[20]:


solutions = pd.read_csv('../images_training_rev1/training_solutions_rev1/training_solutions_rev1.csv')
solutions.head()


# In[24]:


classes = solutions.columns[1:].tolist()
n_classes = len(classes)


# In[ ]:


# From Tensorflow Docs:
def model_builder(hp):
    model = Sequential()
    model.add(layers.Flatten(input_shape=(1, x_train.shape[1])))

    # Tune the number of units in the first Dense layer
    # Choose an optimal value between 32-512
    hp_units = hp.Int('units', min_value=32, max_value=512, step=32)
    model.add(layers.Dense(units=hp_units, activation='relu'))
    model.add(layers.Dense(n_bins, activation='softmax'))

    # Tune the learning rate for the optimizer
    # Choose an optimal value from 0.01, 0.001, or 0.0001
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])

    model.compile(optimizer=optimizers.Adam(learning_rate=hp_learning_rate),
                loss=losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])

    return model

print("Shape: " + str(x_train.shape))

#x_train = x_train.reshape(1, x_train.shape[0] * x_train.shape[1])

tuner = kt.Hyperband(model_builder,
                     objective='val_accuracy',
                     max_epochs=10,
                     factor=3,
                     directory='my_dir',
                     project_name='intro_to_kt')

stop_early = callbacks.EarlyStopping(monitor='val_loss', patience=5)

tuner.search(x_train, y_train, epochs=50, validation_split=0.2, batch_size=128, callbacks=[stop_early])

# Get the optimal hyperparameters
best_hps=tuner.get_best_hyperparameters(num_trials=1)[0]

print(f"""
The hyperparameter search is complete. The optimal number of units in the first densely-connected
layer is {best_hps.get('units')} and the optimal learning rate for the optimizer
is {best_hps.get('learning_rate')}.
""")

# Build the model with the optimal hyperparameters and train it on the data for 50 epochs
model = tuner.hypermodel.build(best_hps)
history = model.fit(x_train, y_train, epochs=50, validation_split=0.2)

val_acc_per_epoch = history.history['val_accuracy']
best_epoch = val_acc_per_epoch.index(max(val_acc_per_epoch)) + 1
print('Best epoch: %d' % (best_epoch,))

hypermodel = tuner.hypermodel.build(best_hps)

# Retrain the model
hypermodel.fit(x_train, y_train, epochs=best_epoch, validation_split=0.2)


eval_result = hypermodel.evaluate(x_test, y_test)
print("[test loss, test accuracy]:", eval_result)

