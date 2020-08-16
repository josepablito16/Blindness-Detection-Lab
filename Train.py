from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import pickle

NOMBRE_MODELO='prueba'
NUM_CLASES=3

# Train
train_datagen = ImageDataGenerator( rescale = 1.0/255. )

train_dir='/home/jose/Descargas/Tensor/Train'
train_generator = train_datagen.flow_from_directory(train_dir,
                                                    batch_size=20,
                                                    class_mode='binary',
                                                    target_size=(150, 150))



model = tf.keras.models.Sequential([tf.keras.layers.Flatten(input_shape=(150, 150,3)), 
                                    tf.keras.layers.Dense(128, activation=tf.nn.relu), 
                                    tf.keras.layers.Dense(NUM_CLASES, activation=tf.nn.softmax)])


model.compile(optimizer='adam',
              loss = 'sparse_categorical_crossentropy',
              metrics=['accuracy'])


history =model.fit(
	train_generator,
	epochs=15,
	verbose=2)


#GUARDAR MODELO
