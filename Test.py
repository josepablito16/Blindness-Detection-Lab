import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

NOMBRE_MODELO='prueba'
NUM_CLASES=3

#CARGAR MODELO
model = tf.keras.models.Sequential([tf.keras.layers.Flatten(input_shape=(150, 150,3)), 
                                    tf.keras.layers.Dense(128, activation=tf.nn.relu), 
                                    tf.keras.layers.Dense(NUM_CLASES, activation=tf.nn.softmax)])

model.load_weights(NOMBRE_MODELO)

# Test
test_datagen  = ImageDataGenerator( rescale = 1.0/255. )
validation_dir='/home/jose/Descargas/Tensor/Test'
validation_generator =  test_datagen.flow_from_directory(validation_dir,
                                                         batch_size=20,
                                                         class_mode  = 'binary',
                                                         target_size = (150, 150))

predictions = model.predict(validation_generator)

import numpy as np

for i in predictions:
	
	print(i)
	print(np.argmax(i))
	print(max(i))
	print()
