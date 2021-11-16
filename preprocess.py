from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import os, sys
import tensorflow as tf


def preprocess_image(image, image_size):
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize_with_crop_or_pad(image, image_size, image_size)
    image = tf.cast(image, tf.float32) / 255.0  # normalize to [0,1] range

    return image
