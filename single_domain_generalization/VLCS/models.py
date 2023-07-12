
import tensorflow as tf
def classificationNN(rep_dim, num_classes):
    inputs = tf.keras.layers.Input(shape=(rep_dim), name='inputs')
    
    x = inputs
    x = tf.keras.layers.Dense(num_classes)(x)
    return tf.keras.Model(inputs=inputs, outputs=x)