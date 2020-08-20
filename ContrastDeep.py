from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import numpy as np
import os
import PIL
import PIL.Image
from PIL import Image
import matplotlib.pyplot as plt


NOMBRE_MODELO='contrastDeep'
NUM_CLASES=5

# Train
train_datagen = ImageDataGenerator( rescale = 1.0/255. )

train_dir='..\\imagen'

with tf.device('/gpu:0'):

  train_generator = train_datagen.flow_from_directory(train_dir,
                                                    batch_size=1,
                                                    class_mode='binary',
                                                    shuffle=True,
                                                    target_size=(150, 150))


  for item in train_generator:
    plt.imshow(item[0][0])
    plt.show()
    break