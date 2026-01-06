import os                                             #Importar dependencias        
import cv2
from PIL import Image    #Debido a que son imagenes satelitales, usamos PIL para leerlas
import numpy as np


def leer_imagen(ruta_imagen):
    imagen=Image.opend(ruta_imagen).convert('RGB')
    if imagen is None:                                                                                 #Funcion para leer las imagenes
        raise FileNotFoundError(f"No se pudo leer la imagen en la ruta: {ruta_imagen}")     
    return np.array(imagen)

def mascara_grises(ruta_imagen):
    mascara = Image.open(ruta_imagen).convert('L')
    if mascara is None:                                                                                 #Mascara en escala de grises
        raise FileNotFoundError(f"No se pudo leer la máscara: {ruta_imagen}")
    return np.array(mascara)

def dir_imagenes(directorio, ext=".png"):
    imagenes = []
    for archivo in os.listdir(directorio):                                                               #Funcion para obtener las rutas de las imagenes en un directorio
        if archivo.lower().endswith(ext):
            ruta = os.path.join(directorio, archivo)
            imagenes.append(ruta)
    return sorted(imagenes)