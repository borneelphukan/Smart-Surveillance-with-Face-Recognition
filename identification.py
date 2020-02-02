from imutils import paths
from imutils.video import FPS
from imutils.video import VideoStream
import imutils
import pickle
import os
import numpy as np
import cv2
import argparse
import time

argp = argparse.ArgumentParser()
argp.add_argument("-d", "--detector", required=True)
argp.add_argument("-m", "--embedding-model", required=True)
argp.add_argument("-r", "--recognizer", required=True)
argp.add_argument("-l", "--labelencoder", required=True)
argp.add_argument("-c", "--confidence", type=float, default=0.5)
args = vars(argp.parse_args())

#loading the face detector
print("Loading Face Detection Algorithm (1)...")
proto_path = os.path.sep.join([args["detector"], "deploy.prototxt"])
model_path = os.path.sep.join([args["detector"], "res10_300x300_ssd_iter_140000.caffemodel"])
detector = cv2.dnn.readNetFromCaffe(proto_path, model_path)

#loading the face embeddings
print("Loading Extracted Face Embeddings (2)...")
embedder = cv2.dnn.readNetFromTorch(args["embedding_model"])

#loading the configured face recognition model with label encoder
recognizer = pickle.loads(open(args["recognizer"], "rb").read())
label_encoder = pickle.loads(open(args["labelencoder"], "rb").read())

#starting up camera
print("Firing up the WebCam...")
video = VideoStream(src=0).start()
time.sleep(2.0)

#start FPS throughput estimator
fps = FPS().start()

while True:
    frame = video.read()
    frame = imutils.resize(frame, width=600)
    (height, width) = frame.shape[:2]

    #making blob around the image
    imageBlob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0), swapRB=False, crop=False)

    #Using face detector caffe model to detect faces in images
    detector.setInput(imageBlob)
    detections = detector.forward()

    #loop over the detections
    for i in range(0, detections.shape[2]):
        confidence = detections[0,0,i,2]

        #filtering weak detection
        if confidence > args["confidence"]:
            #compute the x, y coordinate of bounding box of face
            box = detections[0,0,i,3:7] * np.array([width, height, width, height])
            (start_x, start_y, end_x, end_y) = box.astype("int")

            #extracting ROI of face
            face = frame[start_y:end_y, start_x:end_x]
            (face_height, face_width) = face.shape[:2]

            #face width and face height should be large
            if face_width < 20 or face_height < 20:
                continue

            '''
            1. construct a face blob.
            2. Pass the blob through the embedder.
            3. Use the generate 128d quantification of face
            '''

            faceBlob = cv2.dnn.blobFromImage(face, 1.0/255, (96, 96), (0,0,0), swapRB=True, crop=False)
            embedder.setInput(faceBlob)
            face_vectors = embedder.forward()

            #classification for face recognition
            predict = recognizer.predict_proba(face_vectors)[0]
            j = np.argmax(predict)
            proba = predict[j]
            name = label_encoder.classes_[j]

            #bounding box configuration
            text = "{}: {:.2f}%".format(name, proba*100)
            y = start_y - 10 if start_y else start_y + 10
            cv2.rectangle(frame, (start_x, start_y), (end_x, end_y), (0,0,255), 2)
            cv2.putText(frame, text, (start_x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,255,0), 2)

    #fps updated
    fps.update()

    #output frame display
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == 27:
        break

fps.stop()
cv2.destroyAllWindows()
video.stop()

print("\n")
print("Developed by Borneel, Anmol")
print("Neeloi, Gargi, Vishaka and Saptami")
