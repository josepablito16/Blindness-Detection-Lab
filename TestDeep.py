import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

NOMBRE_MODELO='deep1'
NUM_CLASES=5

#CARGAR MODELO
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

model.load_weights(NOMBRE_MODELO)

# Test
test_datagen  = ImageDataGenerator( rescale = 1.0/255. )
validation_dir='..\\test'
validation_generator =  test_datagen.flow_from_directory(validation_dir,
                                                         batch_size=20,
                                                         class_mode  = 'binary',
                                                         target_size = (150, 150))

model.compile(optimizer='adam', loss = 'sparse_categorical_crossentropy', metrics=['accuracy'])

test_loss, test_acc = model.evaluate(validation_generator)
print(test_acc)
