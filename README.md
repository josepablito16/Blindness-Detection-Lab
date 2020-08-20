# Blindness-Detection-Lab

Para este laboratorio se desarrolló tanto una red nueronal simple como una red neuronal de Deep learning para 
la competencia [APTOS 2019 Blindness Detection](https://www.kaggle.com/c/aptos2019-blindness-detection/discussion)

## Preprocesamiento de datos 
Antes de entrenar un modelo debemos descargar la base de datos de kaggle y descomprimirla en una carpeta afuera del repositorio. 
Posteriormente, se debe ejecutar el script *Prepocess.py* usando la *versión 3 de python* el cual distribuirá las imágenes de forma adecuada y en distintas carpetas. Estas luego están listas para ser utilizadas por el siguiente algoritmo. 

## Entrenamiento de modelos
Para entrenar un modelo, se debe ejecutar el script *Train.py* para un modelo simple o *TrainDeep.py* para un modelo de Deep learning. 
Es necesario cambiar la variable *NOMBRE_MODELO* para poder guardar el modelo entrenado y poder hacer el test independientemente. 

## Test de modelos
Para realizar el test de un modelo se puede ejecutar el script *Test.py* para testear un modelo simple o *TestDeep.py* para 
uno de Deep learning. Es necesario cambiar la variable *NOMBRE_MODELO* dentro de estos archivos para poder cargar el modelo que se desee.

## Modelos realizados por nosotros
En este repositorio no se encuentran todos los modelos hechos por nosotros, pues algunos son muy pesados como para ser subidos en github. Por ello, se pueden descargar de [este medio](https://drive.google.com/drive/folders/1aTbxeU3_qV5qujcTgDbVM07FrjNLfe5-?usp=sharing).

## Requerimientos: 
* pip install tensorflow
* pip install numpy
* pip install Pillow
