import cv2
import numpy as np
from PIL import Image
from vanishing_point_detection import get_vanishing_point
from skimage import io
import numpy as np
from lu_vp_detect import VPDetection

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

def create_visualizer(img_array):

    output_array = np.zeros((img_array.shape[0], img_array.shape[1], 4)) # RGBA

    output_array[img_array == 3] = [255,255,255,255]
    output_array[img_array == 0] = [160,0,0,255]
    
    img = output_array.astype(np.uint8)
    
    img = Image.fromarray(img)
    
    return img

def create_overlay(img, segmentation, floor_id = 3, shadow_strength = 1.0):

    shadows = np.array(create_shadows(img))

    img = cv2.cvtColor(img, cv2.COLOR_RGB2RGBA)

    if (img.shape[0] != segmentation.shape[0]) or (img.shape[1] != segmentation.shape[1]):
        segmentation = cv2.resize(segmentation, (img.shape[1], img.shape[0]))
    
    # En la segmentacion poner sombras
    img[segmentation == floor_id] = np.uint8(shadows[segmentation == floor_id] * shadow_strength)
    img[segmentation == 28] = np.uint8(shadows[segmentation == 28] * shadow_strength)

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

def normalized(a, axis=-1, order=2):
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2==0] = 1
    return a / np.expand_dims(l2, axis)

def get_py_from_vp(u_i, v_i, K):
    p_infinity = np.array([u_i, v_i, 1])
    K_inv = np.linalg.inv(K)
    r3 = K_inv @ p_infinity    
    r3 /= np.linalg.norm(r3)
    yaw = -np.arctan2(r3[0], r3[2])
    pitch = np.arcsin(r3[1])    
    return np.rad2deg(pitch), np.rad2deg(yaw)

def get_angle(image, method=1, angle='radians'):
    if method==1:
        _, best_hypothesis_1, best_hypothesis_2, best_hypothesis_3, _, _ = get_vanishing_point(image, threshold=4)
    else: 
        length_thresh = 15 # Minimum length of the line in pixels
        principal_point = (image.shape[1]//2, image.shape[0]//2)
        focal_length = 1500 # Specify focal length in pixels
        seed = None

        vpd = VPDetection(length_thresh, principal_point, focal_length, seed)
        vps = vpd.find_vps(image)
        best_hypothesis_1, best_hypothesis_2, best_hypothesis_3 = vpd.vps_2D
        
    hypothesis_list = [best_hypothesis_1, best_hypothesis_2, best_hypothesis_3]
    hypothesis = []

    for i, h in enumerate(hypothesis_list):
        
        if h[1]>image.shape[1] or h[1]<-image.shape[1]*5:
            print("Eliminada la hipotesis " + str(i+1))
        else:
            print("Hipotesis " + str(i+1) + " aceptada")
            hypothesis.append(h)

    if len(hypothesis) <= 0:
        print("Ningun VP lo suficientemente bueno.")
        hypothesis = [np.array([image.shape[0]/2, image.shape[1]/2,1])]
    
    width = image.shape[1]
    height = image.shape[0]

    fov_vertical = 40 # Poner el FOV vertical en grados
    fov_horizontal = fov_vertical*width/height #57.2

    fy = np.divide(height/2, np.tan(np.radians(fov_vertical/2)))
    fx = np.divide(width/2, np.tan(np.radians(fov_horizontal/2)))

    K = np.array([[fx, 0.0, width/2], [0.0, fy, height/2], [0.0, 0.0, 1.0]])

    i=0
    angleX, angleY = get_py_from_vp(hypothesis[i][0],hypothesis[i][1],K)
    print([hypothesis[i][0]/image.shape[1], hypothesis[i][1]/image.shape[0], 1])

    if angle=='radians':
        return np.deg2rad(angleX), np.deg2rad(-angleY)
    else:
        return angleX, -angleY

if __name__ == "__main__":
    pass