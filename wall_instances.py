import cv2
import numpy as np
import diplib as dip

img = cv2.imread('input/bedroom.png')
wallseg = cv2.imread('seg_output.png')

if (img.shape != wallseg.shape):
    img= cv2.resize(img, (wallseg.shape[1], wallseg.shape[0]))

wall_only = np.zeros_like(img)

wall_only[wallseg == 255] = img[wallseg == 255]

wall_only = cv2.cvtColor(wall_only, cv2.COLOR_BGR2GRAY)

#b = dip.GradientMagnitude(wall_only, sigmas=0.5)
#edges = np.array(b, dtype=np.uint8)
#canny edge
edges = cv2.Canny(wall_only, 50, 100)
cv2.imwrite('edges.jpg', edges)

#thresh = cv2.threshold(edges, 9, 255, cv2.THRESH_BINARY)[1]
#cv2.imwrite('edges.jpg', thresh)

# Detect vertical lines using Hough Line Transform
lines = cv2.HoughLines(edges, 3, np.pi/2, 120)

# Draw the detected lines on a copy of the original image
copy = np.copy(img)
for line in lines:
    rho, theta = line[0]
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a * rho
    y0 = b * rho
    x1 = int(x0 + 1000 * (-b))
    y1 = int(y0 + 1000 * (a))
    x2 = int(x0 - 1000 * (-b))
    y2 = int(y0 - 1000 * (a))
    cv2.line(copy, (x1, y1), (x2, y2), (0, 0, 255), 2)

# Display the result
cv2.imwrite('Result.jpg', copy)