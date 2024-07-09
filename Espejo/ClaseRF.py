import cv2
import os
import face_recognition as fr
import numpy

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
            codificado = fr.face_encodings(imagen)[0]  # donde esta la cara
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
            if coincidencias:
                indice_coincidencia = numpy.argmin(distancias)
                if distancias[indice_coincidencia] <= 0.6:
                    print(f"Bienvenido {self.nombres_empleados[indice_coincidencia]}")
                    return True
        print("No se encontraron coincidencias")
        return False
