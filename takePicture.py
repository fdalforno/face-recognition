import numpy as np
import cv2
import os
 
cap = cv2.VideoCapture(0)

image_folder = "../img/"
image_file_name = "picture_{0}.jpg"
 
while(True):
    # Cattura fotogramma per fotogramma
    ret, frame = cap.read()
 
    # Visualizza il frame risultante
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
 
# Al termine, rilasciare il tappo di acquisizione
cap.release()
cv2.destroyAllWindows()


image_number = len([f for f in os.listdir(image_folder) if f.endswith('.jpg') and os.path.isfile(os.path.join(image_folder, f))])
image_path = os.path.join(image_folder,image_file_name.format(image_number))

cv2.imwrite(image_path, frame)

