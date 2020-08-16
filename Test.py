import pickle

NOMBRE_MODELO='prueba'

#CARGAR MODELO

predictions = model.predict(validation_generator)

import numpy as np

for i in predictions:
	
	print(i)
	print(np.argmax(i))
	print(max(i))
	print()
