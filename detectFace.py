import argparse

import cv2
import numpy as np

from faceDetector import detectFace, detectFaceSSD,detectLandMarks,show_box_haar,show_box_sdd,show_landmarks,get_embeddings

print(cv2.__version__)


ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to input image")
args = vars(ap.parse_args())

image = cv2.imread(args["image"])

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

face = image.copy()
boxes1 = detectFace(gray)
show_box_haar(face,boxes1)

face = image.copy()
boxes,conf = detectFaceSSD(image)
show_box_sdd(face,boxes,conf)

face = image.copy()
landmarks = detectLandMarks(gray,boxes1)
show_landmarks(face,landmarks)

for face in boxes1:
    x1,y1,x2,y2 = face
    store_face = image[y1:y2,x1:x2]

    get_embeddings(store_face)

    cv2.imshow("face", store_face)
    cv2.waitKey(0)
    cv2.destroyAllWindows()