import cv2
import numpy as np

def foo_bar(img=None):
    # On reçois une image de la partie c++ 
    if img is not None:
        # Si image reçue, on l'enregistre dans le répertoire (c'est juste pour tester)
        cv2.imwrite('../image_recue.png', img)
        # Faire quelque chose avec l'image ici ...
        
        # Je retourne un array à la partie c++
        imageSize = np.array(img.shape, dtype=np.double) 
        imageSize = imageSize.reshape(( len(img.shape),1 ))
        return imageSize
    else:
        # Pas d'image reçue
        return None
