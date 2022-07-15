import cv2
import mediapipe as mp
import os
import numpy as np
from keras_preprocessing.image import load_img,img_to_array
from tensorflow.python.keras.models import load_model
#-----------------------------------------------------------------------
#Cargamos el modelo y sus pesos
modelo = 'C:/Users/luigy/PycharmProjects/ia/Modelo_FORTE.h5'
peso = 'C:/Users/luigy/PycharmProjects/ia/Pesos_FORTE.h5'
cnn = load_model(modelo)
cnn.load_weights(peso)
#-----------------------------------------------------------------------
#Direccion donde esten los datos de validacion
direccion = 'C:/Users/luigy/Documents/Repaldo/Validacion'
dire_img= os.listdir(direccion)
#Pintamos los nombres de las clases en consola
print("Nombres: ", dire_img)
#Iniciamos la camara
cap = cv2.VideoCapture(0)
las_manos = mp.solutions.hands
manitos = las_manos.Hands()
pinta = mp.solutions.drawing_utils
while(1):
    ret,frame = cap.read()
    color = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    copia = frame.copy()
    resultado = manitos.process(color)
    position = []
    if resultado.multi_hand_landmarks:
        for mano in resultado.multi_hand_landmarks:
            for id, lm in enumerate(mano.landmark):
                alto, ancho, c = frame.shape
                corx, cory = int(lm.x*ancho), int(lm.y*alto)
                position.append([id,corx,cory])
                pinta.draw_landmarks(frame,mano,las_manos.HAND_CONNECTIONS)
            if len(position) !=0:
                pto_i1 = position[4]
                pto_i2 = position[20]
                pto_i3 = position[12]
                pto_i4 = position[8]
                pto_i5 = position[9]
                #pto_i1 = position[4]
                #pto_i2 = position[17]
                #pto_i3 = position[10]
                #pto_i4 = position[0]
                x1,y1 = (pto_i5[1]-100),(pto_i5[2]-100)
                ancho, alto = (x1+200),(y1+200)
                x2,y2 = x1 + ancho, y1 + alto
                dedos_registro = copia[y1:y2, x1:x2]
                #Reajustamos el fotograma
                dedos_data = cv2.resize(dedos_registro,(200,200),interpolation = cv2.INTER_CUBIC)
                #Convertimos en un array de imagen
                x = img_to_array(dedos_data)
                x = np.expand_dims(x, axis=0)
                vector = cnn.predict(x)
                #Generara el resultado en una Array
                resultado = vector[0]
                res = np.argmax(resultado)
                #Fuente que se usara para el PutText
                fuente = cv2.QT_FONT_NORMAL
                #Si la posicion es 0, obtendra la carpeta "Hola", y sera mostrada parametrizada con el puText
                if res == 0:
                    texto_dicho='{}'.format(dire_img[0])
                    #Mostramos el rectangulo
                    cv2.rectangle(frame,(x1,y1), (x2,y2), (0,255,0),3)
                    cv2.putText(frame,'{}'.format(dire_img[0]), (150,100), fuente, 3,(0,255,0),1,cv2.LINE_AA)
                    #cv2.putText(frame,'Hola', (x1,y1-10), 1, 1.3,(0,255,0),1,cv2.LINE_AA)
                #Si la posicion es 1, obtendra la carpeta "Lo Siento"
                elif res == 1:
                    texto_dicho='{}'.format(dire_img[1])
                    #Mostramos el rectangulo
                    cv2.rectangle(frame,(x1,y1), (x2,y2), (0,255,0),3)
                    cv2.putText(frame,'{}'.format(dire_img[1]), (150,100), fuente, 3,(0,255,0),1,cv2.LINE_AA)
                    #cv2.putText(frame,'Lo Siento', (x1,y1-5), 1, 1.3,(0,0,255),1,cv2.LINE_AA)
                #Si la posicion es 2, obtendra la carpeta "Muy Bien"
                elif res == 2:
                    texto_dicho='{}'.format(dire_img[2])
                    #Mostramos el rectangulo
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 3)
                    cv2.putText(frame, '{}'.format(dire_img[2]), (150,100), fuente, 3, (0,255,0), 1, cv2.LINE_AA)
                    #cv2.putText(frame, 'Muy Bien!', (x1, y1-5), 1, 1.3, (255, 0, 0), 1, cv2.LINE_AA)
                #Si la posicion es 3, obtendra la carpeta "No"
                elif res == 3:
                    texto_dicho='{}'.format(dire_img[3])
                    #Mostramos el rectangulo
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 3)
                    cv2.putText(frame, '{}'.format(dire_img[3]), (150,100), fuente, 3, (0,255,0), 1, cv2.LINE_AA)
                    #cv2.putText(frame, 'No', (x1, y1-5), 1, 1.3, (255, 0, 255), 1, cv2.LINE_AA)
                #Si la posicion es 4, obtendra la carpeta "Por Favor"
                elif res == 4:
                    texto_dicho='{}'.format(dire_img[4])
                    #Mostramos el rectangulo
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 3)
                    cv2.putText(frame, '{}'.format(dire_img[4]), (150,100), fuente, 3, (0,255,0), 1, cv2.LINE_AA)
                    #cv2.putText(frame, 'Por Favor', (x1, y1-5), 1, 1.3, (0, 255, 255), 1, cv2.LINE_AA)
                #Si la posicion es 5, obtendra la carpeta "Yo"
                elif res == 5:
                    texto_dicho = '{}'.format(dire_img[5])
                    #Mostramos el rectangulo
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 3)
                    cv2.putText(frame, '{}'.format(dire_img[5]),(150,100), fuente, 3, (0,255,0), 1, cv2.LINE_AA)
                    # cv2.putText(frame, 'Yo', (x1, y1-5), 1, 1.3, (0, 255, 255), 1, cv2.LINE_AA)
                else:
                    texto_dicho = 'No se ha encontrado resultados'
                    #Mostramos el rectangulo
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 3)
                print(texto_dicho)

    cv2.imshow("Video", frame)
    k = cv2.waitKey(1)
    #Para romper el buble
    if k == 27:
        break
#Cerrar_todo
cap.release()
cv2.destroyAllWindows()