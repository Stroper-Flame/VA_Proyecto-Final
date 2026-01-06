import os
import torch
import cv2
from torch.utils.data import Dataset

from lectura import leer_imagen, prep_mascara


class XView2Dataset(Dataset):
    def __init__(self, img_dir, mask_dir):
        self.img_dir = img_dir
        self.mask_dir = mask_dir

        self.imagenes = sorted([
            f for f in os.listdir(img_dir)
            if f.endswith(".png")
        ])

    def __len__(self):
        return len(self.imagenes)

    def _ruta_mascara(self, nombre_img):
        return os.path.join(
            self.mask_dir,
            nombre_img.replace(".png", "_target.png")
        )

    def __getitem__(self, idx):
        nombre_img = self.imagenes[idx]

        ruta_img = os.path.join(self.img_dir, nombre_img)
        ruta_mask = self._ruta_mascara(nombre_img)

        imagen = leer_imagen(ruta_img)
        mascara = prep_mascara(ruta_mask)
    
        #Redimensionar a 256, downscale.
        imagen = cv2.resize(imagen, (256, 256), interpolation=cv2.INTER_LINEAR)
        mascara = cv2.resize(mascara, (256, 256), interpolation=cv2.INTER_NEAREST)

        # Imagen → Tensor [3,H,W]
        imagen = torch.from_numpy(imagen).permute(2, 0, 1).float() / 255.0

        # Máscara → Tensor [1,H,W]
        mascara = torch.from_numpy(mascara).unsqueeze(0)

        return imagen, mascara
