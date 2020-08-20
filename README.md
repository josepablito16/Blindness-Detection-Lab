# Blindness-Detection-Lab

Para este laboratorio se desarrolló tanto una red nueronal simple como una red neuronal de Deep learning para 
la competencia APTOS 2019 Blindness Detection [https://www.kaggle.com/c/aptos2019-blindness-detection/discussion]

## Preprocesamiento de datos 
Antes de entrenar un modelo debemos descargar la base de datos de kaggle y descomprimirla una carpeta afuera del repositorio. 
Posteriormente ejecutar el script *Prepocess.py* el cual distribuirá las imágenes de forma adecuada para ser utilizadas. 

## Entrenamiento de modelos
Posteriormente se puede ejecutar el script *Train.py* para un modelo simple o *TrainDeep.py* para un modelo de Deep learning. 
Es necesario cambiar la variable *NOMBRE_MODELO* para poder guardas el modelo y poder hacer el test independientemente. 

## Test de modelos
Para realizar el test de un modelo se puede ejecutar el script *Test.py* para testear un modelo simple o *TestDeep.py* para 
uno de Deep learning. Es necesario cambiar la variable *NOMBRE_MODELO* para poder cargar el modelo que se desee. 

