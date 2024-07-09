# 201 crear base de dats de rostros
# 202. Encontrar Coincidencias en la Base de Dato
import cv2
import os
import face_recognition as fr
import numpy

ruta = "Empleados"  # carpeta donde se encuentran las fotos de los empleados esta en directorio de trabajo
mis_imagenes = []
nombres_empleados = []
lista_empleados = os.listdir(ruta)
# print(lista_empleados)
for empleado in lista_empleados:
    imagen_actual = cv2.imread(f"{ruta}/{empleado}")
    mis_imagenes.append(imagen_actual)
    nombres_empleados.append(os.path.splitext(empleado)[0])


# print(nombres_empleados)
# funcion codificar imagenes para obtener las codificaciones de las caras
def codificar(imagenes):
    lista_codificada = []
    for imagen in imagenes:
        imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
        codificado = fr.face_encodings(imagen)[0]  # donde esta la cara
        lista_codificada.append(codificado)
    return lista_codificada


lista_empleados_codificada = codificar(mis_imagenes)
print(len(lista_empleados_codificada))

# 202 tomar una foto de la camara
captura = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# 202 leer la foto
exito, imagen = captura.read()
# 202 mostrar la imagen
cv2.imshow("Foto Empleado", imagen)
cv2.waitKey(0)

if not exito:
    print("No se pudo tomar la foto")
else:
    # 202reconocer cara en factura
    cara_captura = fr.face_locations(imagen)
    # 202 codificar la cara
    cara_captura_codificada = fr.face_encodings(imagen, known_face_locations=cara_captura)
    print(cara_captura_codificada)
    # 202 buscar coincidencias
    """zip(cara_captura_codificada, cara_captura): La función zip toma dos (o más) 
    listas y las empareja elemento a elemento. Esto significa que en cada iteración, caracodif toma
     un valor de cara_captura_codificada y caraubic toma el valor correspondiente de cara_captura."""
    for caracodif, caraubic in zip(cara_captura_codificada, cara_captura):
        coincidencias = fr.compare_faces(lista_empleados_codificada, caracodif, 0.6)  # devuelve una lista de booleanos
        distancias = fr.face_distance(lista_empleados_codificada, caracodif)  # devuelve una lista de distancia

    print(coincidencias)
    print(distancias)

    # 202 indice de coincidencia
    indice_coincidencia = numpy.argmin(distancias)
    print(f"Indice de Coincidencia {indice_coincidencia}")

    # 202 mostrar resultados
    try:  # 202 manejar excepciones si no se encuentra coincidencias

        # 202 mostrar coincidencias si las hay
        if distancias[indice_coincidencia]:
            print(f"Bienvenido {nombres_empleados[indice_coincidencia]}")
        else:
            print("No se encontró coincidencia")
    except:
        print("No se encontraron coincidencias")