from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf

NOMBRE_MODELO='grayscale'
NUM_CLASES=5

# Train
train_datagen = ImageDataGenerator( rescale = 1.0/255. )

train_dir='..\\train'

with tf.device('/gpu:0'):

  train_generator = train_datagen.flow_from_directory(train_dir,
                                                    batch_size=400,
                                                    class_mode='binary',
                                                    target_size=(150, 150),
                                                    color_mode='grayscale')

  model = tf.keras.models.Sequential([tf.keras.layers.Flatten(input_shape=(150, 150, 1)), 
                                  tf.keras.layers.Dense(128, activation=tf.nn.relu), 
                                  tf.keras.layers.Dense(NUM_CLASES, activation=tf.nn.softmax)])


  model.compile(optimizer='adam',
                loss = 'sparse_categorical_crossentropy',
                metrics=['accuracy'])


  model.fit(train_generator, epochs=15, verbose=2)


  #GUARDAR MODELO
  model.save_weights(NOMBRE_MODELO)