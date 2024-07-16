import cv2
import os
import face_recognition as fr
import numpy as np
from datetime import datetime

class ReconocimientoFacial:
    def __init__(self, ruta_empleados):
        self.ruta = ruta_empleados
        self.mis_imagenes = []
        self.nombres_empleados = []
        self.lista_empleados = os.listdir(self.ruta)

        for empleado in self.lista_empleados:
            imagen_actual = cv2.imread(f"{self.ruta}/{empleado}")
            self.mis_imagenes.append(imagen_actual)
            self.nombres_empleados.append(os.path.splitext(empleado)[0])

        self.lista_empleados_codificada = self.codificar(self.mis_imagenes)

    def codificar(self, imagenes):
        lista_codificada = []

        for imagen in imagenes:
            imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
            codificado = fr.face_encodings(imagen)[0]  # Codificar la cara
            lista_codificada.append(codificado)

        return lista_codificada

    def capturar_imagen(self):
        captura = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        exito, imagen = captura.read()
        captura.release()
        if exito:
            cv2.imshow("Foto Empleado", imagen)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            return imagen
        else:
            print("No se pudo tomar la foto")
            return None

    def reconocer_empleado(self, imagen):
        if imagen is None:
            return False

        cara_captura = fr.face_locations(imagen)
        cara_captura_codificada = fr.face_encodings(imagen, known_face_locations=cara_captura)

        for caracodif, caraubic in zip(cara_captura_codificada, cara_captura):
            coincidencias = fr.compare_faces(self.lista_empleados_codificada, caracodif, 0.6)
            distancias = fr.face_distance(self.lista_empleados_codificada, caracodif)

            if True in coincidencias:
                indice_coincidencia = np.argmin(distancias)

                if distancias[indice_coincidencia] <= 0.6:
                    nombre_empleado = self.nombres_empleados[indice_coincidencia]
                    fecha_hora_actual = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                    # Mostrar la imagen del empleado reconocido
                    imagen_reconocida = self.mis_imagenes[indice_coincidencia]
                    # Detectar la cara en la imagen reconocida
                    cara_reconocida = fr.face_locations(imagen_reconocida)[0]

                    # Dibujar el rectángulo y agregar texto en la imagen reconocida
                    cv2.rectangle(imagen_reconocida,
                                  (cara_reconocida[3], cara_reconocida[0]),
                                  (cara_reconocida[1], cara_reconocida[2]),
                                  (0, 255, 0),
                                  2)

                    imagen_reconocida = cv2.putText(imagen_reconocida, f"Bienvenido {nombre_empleado}",
                                                    (cara_reconocida[3], cara_reconocida[0] - 10),
                                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    imagen_reconocida = cv2.putText(imagen_reconocida, fecha_hora_actual,
                                                    (cara_reconocida[3], cara_reconocida[0] - 30),
                                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                    # cv2.imshow("Empleado Reconocido", imagen_reconocida)
                    # cv2.waitKey(0)
                    # cv2.destroyAllWindows()

                    # calcular el centro del rostro para dibujar un rectangulo en la imagen capturada
                    cv2.circle(imagen, (int((caraubic[1] + caraubic[3]) / 2), int((caraubic[2] + caraubic[0]) / 2)), 10, (0, 255, 0), 220)

                    # Mostrar la imagen capturada con el mensaje de bienvenida
                    imagen = cv2.putText(imagen, f"Bienvenido {nombre_empleado}",
                                         (caraubic[3], caraubic[0] - 10),
                                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    imagen = cv2.putText(imagen, fecha_hora_actual,
                                         (caraubic[3], caraubic[0] - 30),
                                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                    #Combinar las dos imagenes y mostrarlas juntas
                    # Redimensionar ambas imágenes a la misma dimensión
                    height, width, _ = imagen.shape
                    imagen_reconocida = cv2.resize(imagen_reconocida, (width, height))

                    # Asegurar que ambas imágenes sean del mismo tipo
                    if imagen.dtype != imagen_reconocida.dtype:
                        imagen_reconocida = imagen_reconocida.astype(imagen.dtype)

                    # Concatenar las imágenes
                    imagenComb = cv2.hconcat([imagen, imagen_reconocida])

                    cv2.imshow("Fotos en una sola venta", imagenComb)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()

                    return True

        print("No se encontraron coincidencias")
        return False