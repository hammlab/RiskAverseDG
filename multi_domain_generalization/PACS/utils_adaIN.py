
import tensorflow as tf
import numpy as np
from os.path import join
import tensorflow_datasets as tfds
import socket

HEIGHT = 224
WIDTH = 224
NCH = 3

def get_image_from_coco(coco):
    image = coco['image']
    image = tf.cast(image, tf.float32)

    image_size = tf.shape(image)[:2]
    min_length = tf.reduce_min(image_size)
    image_size = image_size * 512 // min_length
    image = tf.image.resize(image, image_size)
    image = tf.image.random_crop(image, [224, 224, NCH]) / 255.0
    return image

def get_coco_training_set():
    split = tfds.Split.TRAIN
    coco = tfds.load(name='coco/2017', split=split)
    return coco.map(get_image_from_coco)


def get_image_from_wikiart(filename):
    image = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(image, channels=3)
    image_size = tf.shape(image)[:2]
    min_length = tf.reduce_min(image_size)
    image_size = image_size * 512 // min_length
    image = tf.image.resize(image, image_size)
    image = tf.image.random_crop(image, [224, 224, NCH]) / 255.0
    return image

def get_wikiart_set():
    names = tf.data.Dataset.list_files(join("path_to_wikiart", "**/*.jpg"))
    images = names.map(get_image_from_wikiart).apply(tf.data.experimental.ignore_errors())
    return images

def get_encoder(preprocessing=True):
    vgg19 = tf.keras.applications.VGG19(
        include_top=False,
        weights="imagenet",
        input_shape=(HEIGHT, WIDTH, NCH),
    )
    vgg19.trainable = False
    mini_vgg19 = tf.keras.Model(vgg19.input, vgg19.get_layer("block4_conv1").output)

    inputs = tf.keras.layers.Input([HEIGHT, WIDTH, NCH])
    x = inputs
    if preprocessing:
        x = tf.cast(inputs, tf.float32) * 255
        x = tf.keras.applications.vgg19.preprocess_input(x)
    mini_vgg19_out = mini_vgg19(x)
    return tf.keras.Model(inputs, mini_vgg19_out, name="mini_vgg19")
       
def get_mean_std(x, epsilon=1e-5):
    axes = [1, 2]

    # Compute the mean and standard deviation of a tensor.
    mean, variance = tf.nn.moments(x, axes=axes, keepdims=True)
    standard_deviation = tf.sqrt(variance + epsilon)
    return mean, standard_deviation


def ada_in(style, content):
    """Computes the AdaIn feature map.

    Args:
        style: The style feature map.
        content: The content feature map.

    Returns:
        The AdaIN feature map.
    """
    content_mean, content_std = get_mean_std(content)
    style_mean, style_std = get_mean_std(style)
    t = style_std * (content - content_mean) / content_std + style_mean
    return t

def get_decoder():
    config = {"kernel_size": 3, "strides": 1, "padding": "same", "activation": "relu"}
    decoder = tf.keras.Sequential(
        [
            tf.keras.layers.InputLayer((None, None, 512)),
            tf.keras.layers.Conv2D(filters=512, **config),
            tf.keras.layers.UpSampling2D(),
            tf.keras.layers.Conv2D(filters=256, **config),
            tf.keras.layers.Conv2D(filters=256, **config),
            tf.keras.layers.Conv2D(filters=256, **config),
            tf.keras.layers.Conv2D(filters=256, **config),
            tf.keras.layers.UpSampling2D(),
            tf.keras.layers.Conv2D(filters=128, **config),
            tf.keras.layers.Conv2D(filters=128, **config),
            tf.keras.layers.UpSampling2D(),
            tf.keras.layers.Conv2D(filters=64, **config),
            tf.keras.layers.Conv2D(
                filters=3,
                kernel_size=3,
                strides=1,
                padding="same",
                #activation="sigmoid",
            ),
        ]
    )
    return decoder  

def deprocess(image, mode='BGR'):
    if mode == 'BGR':
        return image + np.array([103.939, 116.779, 123.68])
    else:
        return image + np.array([123.68, 116.779, 103.939])

def preprocess(image, mode='BGR'):
    if mode == 'BGR':
        return image - np.array([103.939, 116.779, 123.68])
    else:
        return image - np.array([123.68, 116.779, 103.939])