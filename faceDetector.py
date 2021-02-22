import cv2
import urllib.request as urlreq
import os
import numpy as np

haarcascade_url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_alt2.xml"
haarcascade = "haarcascade_frontalface_alt2.xml"

LBFmodel_url = "https://github.com/kurnianggoro/GSOC2017/raw/master/data/lbfmodel.yaml"
LBFmodel = "lbfmodel.yaml"

ssdmodel_url = "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20180205_fp16/res10_300x300_ssd_iter_140000_fp16.caffemodel"
ssdmodel_proto_url = "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"
ssdmodel = "res10_300x300_ssd_iter_140000_fp16.caffemodel"
ssdmodel_proto = "deploy.prototxt"

embeddings_url = "https://storage.cmusatyalab.org/openface-models/nn4.small2.v1.t7"
embeddings_model = "nn4.small2.v1.t7"

models_dir = "../modelli"

def detectFace(image_gray):
    if (haarcascade in os.listdir(models_dir)):
        print("File exists")
    else:
        #scarico il modello del detect
        urlreq.urlretrieve(haarcascade_url, haarcascade)
        print("File downloaded")

    #creo il classificatore
    haarcascade_path = os.path.join(models_dir,haarcascade)
    detector = cv2.CascadeClassifier(haarcascade_path)

    #estraggo lo facce dal classificatore
    faces = detector.detectMultiScale(image_gray)
    #mostro le coordinate
    #print("Faces:\n", faces)
    return_data = []
    for face in faces:
        print(type(face))

        x,y,w,h = face
        return_data.append((x,y,x+w,y+h))

   
    return return_data


def detectFaceSSD(image,min_confidence = 0.5):

    if (ssdmodel in os.listdir(models_dir)):
        print("File model exists")
    else:
        #scarico il modello del detect
        urlreq.urlretrieve(ssdmodel_url, ssdmodel)
        print("File model downloaded")

    if (ssdmodel_proto in os.listdir(models_dir)):
        print("File proto exists")
    else:
        #scarico il modello del detect
        urlreq.urlretrieve(ssdmodel_proto_url, ssdmodel_proto)
        print("File proto downloaded")

    (h, w) = image.shape[:2]
    model_file = os.path.join(models_dir,ssdmodel)
    proto_file = os.path.join(models_dir,ssdmodel_proto)

    #carico il modello dnn
    net = cv2.dnn.readNetFromCaffe(proto_file, model_file)
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

    #do in ingresso il modello alla rete
    net.setInput(blob)
    detections = net.forward()

    boxes = []
    confidences = []
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > min_confidence:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            boxes.append(box.astype("int"))
            confidences.append(confidence)



    #print("Faces:\n", boxes)
    return boxes,confidences

def detectLandMarks(image_face,faces):
    if (LBFmodel in os.listdir(models_dir)):
        print("File exists")
    else:
        #scarico il modello del detect
        urlreq.urlretrieve(LBFmodel_url, LBFmodel)
        print("File downloaded")

    model_file = os.path.join(models_dir,LBFmodel)
    landmark_detector  = cv2.face.createFacemarkLBF()
    landmark_detector.loadModel(model_file)
    
    data = []
    for face in faces:
        startx,starty,endx,endy = face
        data.append([startx,starty,endx-startx,endy-starty])

    data = np.array(data)

    _, landmarks = landmark_detector.fit(image_face, data)
    return landmarks

def get_embeddings(face):

    if (embeddings_model in os.listdir(models_dir)):
        print("File exists")
    else:
        #scarico il modello del detect
        urlreq.urlretrieve(embeddings_url, embeddings_model)
        print("File downloaded")

    model_file = os.path.join(models_dir,embeddings_model)
    embedder = cv2.dnn.readNetFromTorch(model_file)

    faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255,(96, 96), (0, 0, 0), swapRB=True, crop=False)
    embedder.setInput(faceBlob)
    vec = embedder.forward()

    print(vec)

def show_box_haar(image,boxes):
    for box in boxes:
        startX, startY, endX, endY = box
        image = cv2.rectangle(image,(startX, startY), (endX, endY),(255,0,0),2)

    cv2.imshow('haar_box',image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def show_box_sdd(image,boxes,confidences):
    for i,box in enumerate(boxes):
        startX, startY, endX, endY = box
        text = "{:.2f}%".format(confidences[i] * 100)
        image = cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)

        y = startY - 10 if startY - 10 > 10 else startY + 10
        image = cv2.putText(image, text, (startX, y),cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)

        cv2.imshow("ssd_box", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def show_landmarks(image_cropped,landmarks):
    for landmark in landmarks:
        for x,y in landmark[0]:
            # display landmarks on "image_cropped"
            # with white colour in BGR and thickness 1
            cv2.circle(image_cropped, (x, y), 1, (255, 255, 255), 1)
        
    cv2.imshow("landmarks", image_cropped)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



            

