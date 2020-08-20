from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf

NOMBRE_MODELO='deep1'
NUM_CLASES=5

# Train
train_datagen = ImageDataGenerator( rescale = 1.0/255. )

train_dir='..\\train'

with tf.device('/gpu:0'):

  train_generator = train_datagen.flow_from_directory(train_dir,
                                                    batch_size=400,
                                                    class_mode='binary',
                                                    shuffle=True,
                                                    target_size=(150, 150))

  model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2), 
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'), 
    tf.keras.layers.MaxPooling2D(2,2),
    # Flatten the results 
    tf.keras.layers.Flatten(),
    # neuron hidden layer
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(NUM_CLASES, activation='softmax')
    ])


  model.compile(optimizer='adam',
                loss = 'sparse_categorical_crossentropy',
                metrics=['accuracy'])


  model.fit(train_generator, epochs=25, verbose=2)


  #GUARDAR MODELO
  model.save_weights(NOMBRE_MODELO)