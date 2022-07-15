

# Entrenamiento - GYM
import cv2
import mediapipe as np
import os

nombre = 'Yo'
direccion = 'C:/Users/luigy/Documents/Repositorio/Validacion'
contenido = direccion + '/' + nombre
if not os.path.exists(contenido):
    print('Carpeta creada: ', contenido)
    os.makedirs(contenido)
#Contador para limitar el numero de fotogramas
con = 0
#Iniciamos la camara con la libreria Cv2
cap = cv2.VideoCapture(0)
#Almacenamos la detencion y el seguimiento de las manos en las variables
las_manos = np.solutions.hands
manos = las_manos.Hands()
#Dibuja los 21 puntos criticos de la palma de la mano
pinta = np.solutions.drawing_utils
#Bucle para repetir el procedimiento de captura del fotograma
while (1):
    ret,frame = cap.read()
    #Asignamos el color que tendra el cuadro de reconcimiento
    color = cv2.cvtColor(frame, cv2.COLOR_YCrCb2BGR)
    copia = frame.copy()
    #Resultado guardara la clase manos con el color que se desea
    resultado = manos.process(color)
    #Posiciones guardara las posicion de los puntos criticos de las manos
    #En este caso solo guardaremos 5 puntos teniendo como punto central el punto 9
    posiciones = []
    if resultado.multi_hand_landmarks:
        for mano in resultado.multi_hand_landmarks:
            for id, lm in enumerate(mano.landmark):
                alto, ancho, c = frame.shape
                corx, cory = int(lm.x*ancho), int(lm.y*alto)
                posiciones.append([id,corx,cory])
                pinta.draw_landmarks(frame,mano,las_manos.HAND_CONNECTIONS)
            #Guardamos la posiciones de las manos
            if len(posiciones)!= 0:
                pto_i1 = posiciones[4]
                pto_i2 = posiciones[20]
                pto_i3 = posiciones[12]
                pto_i4 = posiciones[8]
                pto_i5 = posiciones[9]
                # pto_i1 = position[4]
                # pto_i2 = position[17]
                # pto_i3 = position[10]
                # pto_i4 = position[0]
                #Este calculo servira para armar el cuadrito verde de 200x200
                x1,y1 = (pto_i5[1]-100),(pto_i5[2]-100)
                ancho,alto = (x1+200),(y1+200)
                x2,y2 = x1 + ancho, y1 + alto
                #Guardamos en esta variable los dedos con sus posiciones
                dedos_registro = copia[y1:y2, x1:x2]
                #Genera el rectangulo con los parametros ya obtenidos
                cv2.rectangle(frame, (x1,y1) , (x2,y2), (0, 255, 0), 3)
                #Redimenzionamos las imagenes y ajustamos el fotograma
                dedos_registro = cv2.resize(dedos_registro,(200,200),interpolation = cv2.INTER_CUBIC)
                #Guardamos el fotograma en la carpeta
                cv2.imwrite(contenido + '/Dedo_{}.jpg'.format(con),dedos_registro)
                con = con + 1
    cv2.imshow("Video",frame)
    k = cv2.waitKey(1)
    #Para detener el ciclo con la cantidad de imagenes que se desean y si
    if k == 27 or con >= 200:
        break
cap.release()
cv2.destroyAllWindows()