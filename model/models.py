from tensorflow import keras
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
from tensorflow.keras import models, layers, regularizers, optimizers

from tensorflow.keras.applications import VGG16, Xception, ResNet50, EfficientNetB0, MobileNetV2, InceptionV3
from tensorflow.python.keras.layers.pooling import GlobalAveragePooling2D

multiclass = True

def SimpleCNN(img_size, num_classes):
    model = Sequential(name='SimpleCNN')
    model.add(Conv2D(128, (3, 3), padding='same', input_shape=(img_size, img_size, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.1))

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.1))

    model.add(Conv2D(32, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.1))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.25))
    
    if multiclass:
        model.add(Dense(num_classes, kernel_initializer='he_normal'))
        model.add(Activation('sigmoid'))
    else:
        model.add(Dense(num_classes), kernel_initializer='he_normal')
        model.add(Activation('softmax'))

    return model

def Xception_Img(img_size, num_classes):
    # model = Sequential(name='Xception')
    # model.add(Xception(include_top=False, input_shape=(img_size, img_size, 3), weights='imagenet', pooling='avg'))
    # model.add(Flatten())
    # model.add(Dense(512))
    # model.add(Activation('relu'))
    # model.add(Dropout(0.1))
    base_model = Xception(include_top=False, input_shape=(img_size, img_size, 3), weights='imagenet')
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)

    if multiclass:
        predictions = Dense(num_classes, activation='sigmoid')(x)
    else:
        predictions = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)

    for layer in base_model.layers:
        layer.trainable = False
    
    model._name = "Xception"
    
    return model

def Inception(img_size, num_classes):
    base_model = InceptionV3(include_top=False, input_shape=(img_size, img_size, 3), weights='imagenet', pooling='avg')
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)

    if multiclass:
        predictions = Dense(num_classes, activation='sigmoid')(x)
    else:
        predictions = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)

    for layer in base_model.layers:
        layer.trainable = False

    return model

def MobileNet(img_size, num_classes):
    model = Sequential(name='MobileNet')
    model.add(MobileNetV2(include_top=False, input_shape=(img_size, img_size, 3), weights='imagenet', pooling='avg'))
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.1))

    if multiclass:
        model.add(Dense(num_classes, kernel_initializer='he_normal'))
        model.add(Activation('sigmoid'))
    else:
        model.add(Dense(num_classes, kernel_initializer='he_normal'))
        model.add(Activation('softmax'))

    return model