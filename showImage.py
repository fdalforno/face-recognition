import numpy as np
import cv2

# Load an color image in grayscale
img = cv2.imread('../img/picture_5.jpg')

print(type(img))
cv2.imshow('image b',img[:,:,0])
cv2.imshow('image g',img[:,:,1])
cv2.imshow('image r',img[:,:,2])
cv2.waitKey(0)
cv2.destroyAllWindows()