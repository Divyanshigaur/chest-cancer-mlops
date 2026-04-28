import tensorflow as tf
from tensorflow.keras import layers

def get_preprocessing():
    return layers.Rescaling(1./255)

def get_augmentation():
    """
    Applies light data augmentation to improve model generalization.
    Designed considering already augmented dataset to avoid overfitting.
    """

    return tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.02),
])

