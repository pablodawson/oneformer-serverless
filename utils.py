import cv2
import numpy as np

def create_shadows(img):
    # grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    dilated_img = cv2.dilate(gray, np.ones((7,7), np.uint8))

    #Median Blur
    bg_img = cv2.medianBlur(dilated_img, 21)

    #normalizar
    normalized= cv2.normalize(bg_img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    #a rgba
    rgba = cv2.cvtColor(normalized, cv2.COLOR_GRAY2RGBA)

    #canal alpha es proporcional a los negros de la imagen
    rgba[:,:,3] = 255 - rgba[:,:,0]

    return rgba