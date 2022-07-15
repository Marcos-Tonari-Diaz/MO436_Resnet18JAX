import numpy as np
from PIL import Image
import glob
from sklearn.utils import shuffle
import tensorflow as tf


def load_imagenet_val(data_dir):
    batch_size = 32
    img_height = 224
    img_width = 224
    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        labels=None,
        shuffle=False,
        image_size=(img_height, img_width),
        batch_size=batch_size)
    return val_ds


if __name__ == "__main__":
    val_ds = load_imagenet_val("work/datasets/imagenet/val")
    for i, batch in enumerate(val_ds):
        print(batch.shape)
