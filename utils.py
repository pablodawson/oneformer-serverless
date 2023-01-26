import cv2
import numpy as np
from PIL import Image

def create_shadows(img):
    # grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    dilated_img = cv2.dilate(gray, np.ones((7,7), np.uint8))

    #Median Blur
    bg_img = cv2.medianBlur(dilated_img, 21)

    #normalizar
    normalized= cv2.normalize(bg_img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    #a rgba
    rgba = cv2.cvtColor(normalized, cv2.COLOR_GRAY2RGBA)

    #canal alpha es proporcional a los negros de la imagen
    rgba[:,:,3] = 255 - rgba[:,:,0]

    rgba = Image.fromarray(rgba)

    return rgba

def create_overlay(img, segmentation, floor_id = 3, shadow_strength = 1.0):

    shadows = np.array(create_shadows(img))

    img = cv2.cvtColor(img, cv2.COLOR_RGB2RGBA)

    if (img.shape[0] != segmentation.shape[0]) or (img.shape[1] != segmentation.shape[1]):
        segmentation = cv2.resize(segmentation, (img.shape[1], img.shape[0]))
    
    # En la segmentacion poner sombras
    img[segmentation == floor_id] = np.uint8(shadows[segmentation == floor_id] * shadow_strength)

    img = Image.fromarray(img, 'RGBA')

    return img

# Los labels necesarios solamente, escalados de 0 a 255
def labels_only(img_array, labels=[0]):
    output_array = np.zeros((img_array.shape[0], img_array.shape[1], 4)) # RGBA

    output_array[img_array == labels[0]] = [255,255,255,255]
    #output_array[img_array == labels[1]] = 160 
    
    img = output_array.astype(np.uint8)
    
    img = Image.fromarray(img)
    
    return img

if __name__ == "__main__":
    room = cv2.imread("input/room.jpg")
    segmentation = cv2.imread("output.jpg", 0)
    

    img = create_overlay(room, segmentation)

    img.save("overlay.png", "PNG")