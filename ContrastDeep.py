'''
  Autores: 
  Jose Cifuentes
  Oscar Juarez
  Paul Belches

  20/08/2020
'''
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
train_datagen = ImageDataGenerator( 
	rescale = 1.0/255.,
	brightness_range=(1,2),
	rotation_range=180.,
	shear_range=0.2,
	horizontal_flip=True,
	vertical_flip=True,)

train_dir='../train'

with tf.device('/gpu:0'):
	train_generator = train_datagen.flow_from_directory(train_dir,
                                                    batch_size=1,
                                                    class_mode='binary',
                                                    seed=1,
                                                    shuffle=True,
                                                    target_size=(474,358))


  

	contador=0
	for item in train_generator:
		if(contador==5):
			break
		plt.imshow(item[0][0])
		plt.show()	
		contador+=1
		
    