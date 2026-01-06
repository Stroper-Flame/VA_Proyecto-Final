import os                                              #Importar dependencias
import cv2

def leer_imagen(ruta_imagen):
    imagen=cv2.imread(ruta_imagen)
    if imagen is None:                                                                                 #Funcion para leer las imagenes
        raise FileNotFoundError(f"No se pudo leer la imagen en la ruta: {ruta_imagen}")     
    return cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)

def mascara_grises(ruta_imagen):
    mascara = cv2.imread(ruta_imagen, cv2.IMREAD_GRAYSCALE)
    if mascara is None:                                                                                 #Mascara en escala de grises
        raise FileNotFoundError(f"No se pudo leer la máscara: {ruta_imagen}")
    return mascara

def directorio_Imagenes(directorio, ext=".png"):
    imagenes=[]                                                                                     #Listar imagenes en un directorio
    for archivo in os.listdir(directorio):
        if archivo.lower().endswith(ext):
            imagenes.append(os.path.join(directorio, archivo))
    return sorted(imagenes)