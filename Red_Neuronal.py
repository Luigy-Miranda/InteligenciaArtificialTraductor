import tensorflow
import opt_einsum
import matplotlib.pyplot as plt
import numpy as np
import tensorflow.keras.optimizers
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.preprocessing.image import ImageDataGenerator #Ayuda a pre-procesar las fotos
from tensorflow.python.keras.models import Sequential  #Permite hacer redes neuronales secuenciales
from tensorflow.python.keras.layers import Dropout, Flatten, Dense, Activation
from tensorflow.python.keras.layers import Convolution2D, MaxPooling2D  #Capas para hacer la convolucion
from tensorflow.python.keras import backend as k


#File

#Nombre del modelo y pesos con lo que sera guardado
file_name='Modelo_FORTE2.h5'
peso_name='Pesos_FORTE2.h5'
k.clear_session()
#-----------------------------------Data-----------------------------------------------
data_train = 'C:/Users/luigy/Documents/Repositorio/Entrenamiento'
data_test = 'C:/Users/luigy/Documents/Repositorio/Validacion'

#----------------------------------Parametros
ancho,alto=200,200                   #Tamano de las imagenes
interaciones=100                     #Interacciones que tendra
batch_size=1                         #Imagenes que se enviaran
pasos=600/1                          #Numero de fotogramas
pasos_validacion=600/1               #Numero de fotogramas (validacion)

#Extractor de informacion
filtro_conv1=32
filtro_conv2=64
filtro_conv3=128

tam_filtro1=(4,4)
tam_filtro2=(3,3)
tam_filtro3=(2,2)

tam_pool=(2,2)
clases=6

lr=0.0005 #Es buena para solo 5 clases  =  lr=0.0000004 o probar con 0.005

preprocesamiento_train = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.3,
    zoom_range=0.3,
    horizontal_flip=True
)
preprocesamiento_test = ImageDataGenerator(
    rescale = 1./255
)
imagen_train = preprocesamiento_train.flow_from_directory(
    data_train,
    target_size=(alto,ancho),
    batch_size=batch_size,
    class_mode='categorical'
)
imagen_test = preprocesamiento_train.flow_from_directory(
    data_test,
    target_size=(alto,ancho),
    batch_size=batch_size,
    class_mode='categorical'
)
#---------------------------------------RED NEURONAL CONVOLUCIONAL (CNN)
cnn = Sequential()
#---------------------------------------Capa 1 - Parametros ya definidos y con activacion RELU
cnn.add(Convolution2D(filtro_conv1, tam_filtro1, padding='same', input_shape=(alto,ancho,3), activation='relu'))
cnn.add(MaxPooling2D(pool_size=tam_pool)) #Extraccion de caracteristicas
#---------------------------------------Capa 2
cnn.add(Convolution2D(filtro_conv2, tam_filtro2, padding='same', input_shape=(alto,ancho,3), activation='relu'))
cnn.add(MaxPooling2D(pool_size=tam_pool)) #Extraccion de caracteristicas
#---------------------------------------Capa 3
cnn.add(Convolution2D(filtro_conv3, tam_filtro3, padding='same', input_shape=(alto,ancho,3), activation='relu'))
cnn.add(MaxPooling2D(pool_size=tam_pool)) #Extraccion de caracteristicas

cnn.add(Flatten())
#Asignamos 1536 neuronas con una activacion RELU
cnn.add(Dense(768,activation='relu'))
#Esta funcion apagara neuronas aleatoriamente para mejorar la presicion
cnn.add(Dropout(0.5))
#Agregamos una capa densa a las clases
cnn.add(Dense(clases, activation='softmax'))

#Optimizamos con el learning_rate y definido con el modelo Adam
optimizar = tensorflow.keras.optimizers.Adam(learning_rate= lr)
#Compilamos
cnn.compile(loss='categorical_crossentropy', optimizer=optimizar, metrics=['accuracy'])

print('Comenzando entrenamiento...')
historial = cnn.fit(imagen_train,steps_per_epoch=pasos,epochs=interaciones,validation_data=imagen_test,validation_steps=pasos_validacion)
print('Entrenado!')

#Generamos la grafica para visualizar las perdidas que hubo
plt.title('Resultados del entrenamiento')
plt.xlabel('Numero de Epocas')
plt.ylabel('Magnitud de Perdida')
plt.plot(historial.history['loss'])
plt.show()

#Guardamos el modelo
cnn.save(file_name)
cnn.save_weights(peso_name)

