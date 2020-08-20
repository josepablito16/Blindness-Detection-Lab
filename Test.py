import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

NOMBRE_MODELO='grayscale'
NUM_CLASES=5

#CARGAR MODELO
model = tf.keras.models.Sequential([tf.keras.layers.Flatten(input_shape=(150, 150,1)), 
                                    tf.keras.layers.Dense(128, activation=tf.nn.relu), 
                                    tf.keras.layers.Dense(NUM_CLASES, activation=tf.nn.softmax)])

model.load_weights(NOMBRE_MODELO)

# Test
test_datagen  = ImageDataGenerator( rescale = 1.0/255. )
validation_dir='..\\test'
validation_generator =  test_datagen.flow_from_directory(validation_dir,
                                                         batch_size=400,
                                                         class_mode  = 'binary',
                                                         target_size = (150, 150),
                                                         color_mode='grayscale')

model.compile(optimizer='adam',
            loss = 'sparse_categorical_crossentropy',
            metrics=['accuracy'])

test_loss, test_acc = model.evaluate(validation_generator)
print(test_acc)
