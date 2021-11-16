import pandas as pd
# from skimage.transform import resize
# import cv2
# import tqdm
import datetime
import os
from preprocess import preprocess_image
import tensorflow as tf
from sklearn.model_selection import train_test_split

from tensorflow import keras
from tensorflow.keras import models, layers, regularizers, optimizers
from cnn import SimpleCNN
from tensorflow.keras import backend as K
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

ORIGINAL_SIZE = 424
IMG_SIZE = 128
BATCH_SIZE = 32

data_dir = 'galaxy-zoo-the-galaxy-challenge/images_training_rev1'
labels_df = pd.read_csv('galaxy-zoo-the-galaxy-challenge/training_solutions_rev1.csv')
labels_df.set_index('GalaxyID', inplace=True)

(X_train, X_test, y_train, y_test) = train_test_split(labels_df.index.astype(int).astype(str),
                                                        labels_df.values[:,:], test_size=0.2, random_state=42)


def load_and_preprocess_image(path):
    # img_path = os.path.join(data_dir, path + '.jpg')
    img_path = data_dir + '/' + path + '.jpg'
    image = tf.io.read_file(img_path)
    return preprocess_image(image, image_size=IMG_SIZE)

# Convert training Data to TF Data
path_ds = tf.data.Dataset.from_tensor_slices(X_train)
image_ds = path_ds.map(load_and_preprocess_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
label_ds = tf.data.Dataset.from_tensor_slices(tf.cast(y_train, tf.float32))
image_label_ds = tf.data.Dataset.zip((image_ds, label_ds))

ds = image_label_ds
ds = ds.apply(
    tf.data.experimental.shuffle_and_repeat(buffer_size=1000))
ds = ds.batch(BATCH_SIZE)

# Convert testing Data to TF Data
path_ds_test = tf.data.Dataset.from_tensor_slices(X_test)
image_ds_test = path_ds_test.map(load_and_preprocess_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
label_ds_test = tf.data.Dataset.from_tensor_slices(tf.cast(y_test, tf.float32))
image_label_ds_test = tf.data.Dataset.zip((image_ds_test, label_ds_test))

ds_test = image_label_ds_test
ds_test = ds_test.batch(BATCH_SIZE)

# base model
# TODO add if for selecting models
cnn_model = SimpleCNN(img_size=IMG_SIZE, num_classes=labels_df.shape[1])
plot_model(cnn_model, show_shapes=True, to_file=f'{cnn_model.name}.jpg')

def root_mean_square_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))

cnn_model.compile(loss='categorical_crossentropy', 
                    optimizer=optimizers.Adam(learning_rate=1e-8), 
                    metrics=['acc', root_mean_square_error])

# define early stopping callback and model checkpoint callback
checkout_best_model = 'model/Best_Model_SimpleCNN.h5'
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=150)
mc = ModelCheckpoint(checkout_best_model, monitor='val_loss', mode='min', verbose=1, save_best_only=True)

# define tensorboard callback
log_dir = 'logs/fit/' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

history = cnn_model.fit(ds, epochs=35, steps_per_epoch=X_train.shape[0] // BATCH_SIZE, 
                        use_multiprocessing=True,validation_steps=X_test.shape[0] // BATCH_SIZE, validation_data=ds_test, 
                        callbacks=[es, mc, tensorboard_callback])

cnn_model.save(f'model/{cnn_model.name}.h5')
