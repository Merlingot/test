import cv2
import numpy as np

def foo_bar(img=None):  
    if img is not None:
       cv2.imshow("image in python", img)

    else:
        print('allo')