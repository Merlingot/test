import cv2
import numpy as np

def foo_bar(img=None): 
    if img is not None:
        print('image received')
        return img
    else:
        print('no image')
        return None
