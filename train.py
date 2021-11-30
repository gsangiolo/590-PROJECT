import pandas as pd
# from skimage.transform import resize
# import cv2
# import tqdm
import datetime
import os
# from utils import preprocess_image
import tensorflow as tf
from sklearn.model_selection import train_test_split
from utils.util import *
import visualkeras

from tensorflow import keras
from tensorflow.keras import optimizers
from model.models import SimpleCNN, Xception_Img, Inception_Img, ResNet50_Img
from tensorflow.keras import backend as K
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

ORIGINAL_SIZE = 424
IMG_SIZE = 128
BATCH_SIZE = 32
# MODEL = 'XCEPTION'
# MODEL = 'DFF'
MODEL = 'SimpleCNN'

data_dir = 'galaxy-zoo-the-galaxy-challenge/images_training_rev1'
solution_dir = 'galaxy-zoo-the-galaxy-challenge/training_solutions_rev1.csv'

physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)


# labels_df = pd.read_csv('galaxy-zoo-the-galaxy-challenge/training_solutions_rev1.csv')
# labels_df.set_index('GalaxyID', inplace=True)

# (X_train, X_test, y_train, y_test) = train_test_split(labels_df.index.astype(int).astype(str),
#                                                         labels_df.values[:,:], test_size=0.2, random_state=42)


# def load_and_preprocess_image(path):
#     # img_path = os.path.join(data_dir, path + '.jpg')
#     img_path = data_dir + '/' + path + '.jpg'
#     image = tf.io.read_file(img_path)
#     return preprocess_image(image, image_size=IMG_SIZE)

# # Convert training Data to TF Data
# path_ds = tf.data.Dataset.from_tensor_slices(X_train)
# image_ds = path_ds.map(load_and_preprocess_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
# label_ds = tf.data.Dataset.from_tensor_slices(tf.cast(y_train, tf.float32))
# image_label_ds = tf.data.Dataset.zip((image_ds, label_ds))

# ds = image_label_ds
# ds = ds.apply(
#     tf.data.experimental.shuffle_and_repeat(buffer_size=1000))
# ds = ds.batch(BATCH_SIZE)

# # Convert testing Data to TF Data
# path_ds_test = tf.data.Dataset.from_tensor_slices(X_test)
# image_ds_test = path_ds_test.map(load_and_preprocess_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
# label_ds_test = tf.data.Dataset.from_tensor_slices(tf.cast(y_test, tf.float32))
# image_label_ds_test = tf.data.Dataset.zip((image_ds_test, label_ds_test))

# ds_test = image_label_ds_test
# ds_test = ds_test.batch(BATCH_SIZE)

# base model
# cnn_model = SimpleCNN(img_size=IMG_SIZE, num_classes=labels_df.shape[1])
# plot_model(cnn_model, show_shapes=True, to_file=f'{cnn_model.name}.jpg')

# def root_mean_square_error(y_true, y_pred):
#     return K.sqrt(K.mean(K.square(y_pred - y_true)))

# cnn_model.compile(loss='categorical_crossentropy', 
#                     optimizer=optimizers.Adam(learning_rate=1e-8), 
#                     metrics=['acc', root_mean_square_error])

# # define early stopping callback and model checkpoint callback
# checkout_best_model = 'model/Best_Model_SimpleCNN.h5'
# es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=150)
# mc = ModelCheckpoint(checkout_best_model, monitor='val_loss', mode='min', verbose=1, save_best_only=True)

# # define tensorboard callback
# log_dir = 'logs/fit/' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
# tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

# history = cnn_model.fit(ds, epochs=35, steps_per_epoch=X_train.shape[0] // BATCH_SIZE, 
#                         use_multiprocessing=True,validation_steps=X_test.shape[0] // BATCH_SIZE, validation_data=ds_test, 
#                         callbacks=[es, mc, tensorboard_callback])

# cnn_model.save(f'model/{cnn_model.name}.h5')


def split_and_encode(images, y, test_size=0.2):
    x_train, x_test, y_train, y_test = train_test_split(images, y, test_size=test_size)
    print("X_train shape:", x_train.shape)

    if not multiclass:
        lb = LabelEncoder()
        y_train = lb.fit_transform(y_train)
        y_test = lb.transform(y_test)
    print("y_train", x_train)
    print("y_test", y_train)
    return x_train, x_test, y_train, y_test


def model_builder(hp):

    print("Model Building...")
    if MODEL == 'DFF':
        print('Using DFF model')
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

        # save plot 
        plot_model(model, show_shapes=True, to_file=f'{model.name}.jpg')

        if not multiclass:
            model.compile(optimizer=optimizers.Adam(learning_rate=hp_learning_rate),
                        loss=losses.SparseCategoricalCrossentropy(from_logits=True),
                        metrics=['accuracy'])
        else:
            model.compile(optimizer=optimizers.Adam(learning_rate=hp_learning_rate),
                    loss='binary_crossentropy',
                    metrics=['accuracy'])
    
    elif MODEL == 'SimpleCNN':
        print('SimpleCNN')
        model = SimpleCNN(img_size=IMG_SIZE, num_classes=n_classes)
        hp_learning_rate = hp.Choice('learning_rate', values=[1e-5, 1e-6, 1e-7])
        # save plot 
        plot_model(model, show_shapes=True, to_file=f'{model.name}.jpg')
        visualkeras.layered_view(model, to_file='SimpleCNN_Layered.png')
        if not multiclass:
            model.compile(optimizer=optimizers.Adam(learning_rate=hp_learning_rate),
                        loss=losses.SparseCategoricalCrossentropy(from_logits=True),
                        metrics=['accuracy'])
        else:
            model.compile(optimizer=optimizers.Adam(learning_rate=hp_learning_rate),
                    loss='binary_crossentropy',
                    metrics=['accuracy'])
    
    elif MODEL == 'XCEPTION':
        print('XCEPTION')
        model = Xception_Img(img_size=IMG_SIZE, num_classes=n_classes)
        hp_learning_rate = hp.Choice('learning_rate', values=[1e-5, 1e-6, 1e-7])
        # save plot 
        plot_model(model, show_shapes=True, to_file=f'{model.name}.jpg')
        visualkeras.layered_view(model, to_file='Xception_Layered.png')
        if not multiclass:
            model.compile(optimizer=optimizers.RMSprop(learning_rate=hp_learning_rate),
                        loss=losses.SparseCategoricalCrossentropy(from_logits=True),
                        metrics=['accuracy'])
        else:
            model.compile(optimizer=optimizers.RMSprop(learning_rate=hp_learning_rate),
                    loss='binary_crossentropy',
                    metrics=['accuracy'])

    elif MODEL == 'RESNET':
        print('ResNet50')
        model = ResNet50_Img(img_size=IMG_SIZE, num_classes=n_classes)
        hp_learning_rate = hp.Choice('learning_rate', values=[1e-5, 1e-6, 1e-7])
        # save plot 
        plot_model(model, show_shapes=True, to_file=f'{model.name}.jpg')
        visualkeras.layered_view(model, to_file='ResNet50_Layered.png')
        if not multiclass:
            model.compile(optimizer=optimizers.RMSprop(learning_rate=hp_learning_rate),
                        loss=losses.SparseCategoricalCrossentropy(from_logits=True),
                        metrics=['accuracy'])
        else:
            model.compile(optimizer=optimizers.RMSprop(learning_rate=hp_learning_rate),
                    loss='binary_crossentropy',
                    metrics=['accuracy'])

    return model


def fit_model_from_hps(x_train, y_train, x_test, y_test, hyperband=True):
    if hyperband:
        print('Using Hyperband')
        tuner = kt.Hyperband(model_builder,
                            objective='val_accuracy',
                            max_epochs=10,
                            factor=3,
                            # overwrite=True,
                            # max_trail = 3
                            )
    else:
        print('Using RandomSearch')
        tuner = kt.RandomSearch(model_builder,
                                objective='val_accuracy',
                                max_trials=4)

    stop_early = callbacks.EarlyStopping(monitor='val_loss', patience=5)

    tuner.search(x_train, y_train, epochs=50, validation_split=0.2, batch_size=BATCH_SIZE, callbacks=[stop_early, keras.callbacks.TensorBoard(log_dir='logs/fit/tuner')])
    #tuner.search(it, epochs=50, validation_split=0.2, batch_size=128, callbacks=[stop_early])

    # Get the optimal hyperparameters
    best_hps=tuner.get_best_hyperparameters(num_trials=1)[0]
    print(best_hps.get('learning_rate'))

    # print(f"""
    # The hyperparameter search is complete. The optimal number of units in the first densely-connected
    # layer is {best_hps.get('units')} and the optimal learning rate for the optimizer
    # is {best_hps.get('learning_rate')}.
    # """)

    # Build the model with the optimal hyperparameters and train it on the data for 50 epochs
    model = tuner.hypermodel.build(best_hps)
    # model.name = 'hypermodel' + MODEL

    history = model.fit(x_train, y_train, epochs=40, validation_split=0.2)

    val_acc_per_epoch = history.history['val_accuracy']
    best_epoch = val_acc_per_epoch.index(max(val_acc_per_epoch)) + 1
    print('Best epoch: %d' % (best_epoch,))

    hypermodel = tuner.hypermodel.build(best_hps)
    # hypermodel.name = 'hypermodel ' + MODEL

    print('set best model')
    # bast model checkout and early stopping
    checkout_best_model = 'weights/Best_Model_' + hypermodel.name + '.h5'
    # es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=150)
    mc = ModelCheckpoint(checkout_best_model, monitor='val_loss', mode='min', verbose=1, save_best_only=True)

    # define tensorboard callback
    log_dir = 'logs/fit/' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    # Retrain the model
    history = hypermodel.fit(x_train, y_train, 
                            epochs=best_epoch, 
                            validation_split=0.2,
                            # steps_per_epoch=x_train.shape[0] // BATCH_SIZE, 
                            use_multiprocessing=True,
                            # validation_steps=X_test.shape[0] // BATCH_SIZE, 
                            # validation_data=ds_test,
                            callbacks=[
                            # es, 
                            mc, 
                            tensorboard_callback]
                            )

    hypermodel.save(f'model/{hypermodel.name}.h5')

    eval_result = hypermodel.evaluate(x_test, y_test)
    print("[test loss, test accuracy]:", eval_result)
    return history, eval_result




if __name__ == '__main__':
    # global multiclass
    multiclass = True
    images, image_names = get_all_images_from_directory(data_dir)
    global classes
    y, classes = get_image_labels(images, image_names, solution_dir, multiclass_threshold=0.5)
    global n_classes
    n_classes = len(classes)
    if MODEL == 'DFF':
        images, y = generate_new_image_data(images, y, max_it=2)
    else:
        images, y = generate_new_image_data(images, y, max_it=1, return_images=True)
    x_train, x_test, y_train, y_test = split_and_encode(images, y, test_size=0.2)
    history, eval_result = fit_model_from_hps(x_train, y_train, x_test, y_test)
    
