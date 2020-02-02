from imutils import paths
import numpy as np
import argparse
import imutils
import pickle
import cv2
import os
#Argument Parsing
'''
--data -> path of input directory of images
--embeddings -> path of face embeddings after training
--detector -> path to pre-trained Caffe deep learning model provided by OpenCV to 
                detect faces. This model detects and localizes faces in an 
                image.
--embedding-model -> path for pre-built and pre-trained FaceNet Neural Network.
--confidence -> minimum probability to filter weak detections
'''
argp = argparse.ArgumentParser()
argp.add_argument("-i", "--dataset", required=True)
argp.add_argument("-e", "--embeddings", required=True)
argp.add_argument("-d", "--detector", required=True)
argp.add_argument("-m", "--embedding-model", required=True)
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

#grabbing the path of the input images
image_path = list(paths.list_images(args["dataset"]))

#creating a list of extracted face embeddings and person name
flagged_embeddings = []
flagged_names = []

total = 0

#looping over every images
for (i, img_path) in enumerate(image_path):
    print("Extracting embeddings from images {}/{}".format(i+1, len(image_path)))
    person_name = img_path.split(os.path.sep)[-2]

    #loading image, resizing to width 600px maintaining image ratio and grabbing the dimension
    face_image = cv2.imread(img_path)
    face_image = imutils.resize(face_image, width=600)
    (height, width) = face_image.shape[:2]

    #making blob around the image
    imageBlob = cv2.dnn.blobFromImage(cv2.resize(face_image, (300, 300)), 1.0, (300,300), (104.0, 177.0, 123.0), swapRB=False, crop=False)

    #Using face detector caffe model to detect faces in images
    detector.setInput(imageBlob)
    detections = detector.forward()
    
    #(detection -> Number of faces)
    #Ensuring atleast one face is found
    if len(detections) > 0:
        #finding the bounding box with the largest probability
        #and filtering out weaker probability of embeddings
        i = np.argmax(detections[0,0,:,2])
        confidence = detections[0,0,i,2]

        if confidence > args["confidence"]:
            #Computing x, y coordinates of the bounding box
            box = detections[0,0,i, 3:7] * np.array([width, height, width, height])
            (start_x, start_y, end_x, end_y) = box.astype("int")

            #extracting face ROI from the embeddings and grabbing ROI dimension
            face = face_image[start_y:end_y, start_x:end_x]
            (face_height, face_width) = face.shape[:2]

            #face width and face height should be large
            if face_width < 20 or face_height < 20:
                continue

            '''
            1. construct a face blob.
            2. Pass the blob through the embedder.
            3. Use the generate 128d quantification of face
            '''

            faceBlob = cv2.dnn.blobFromImage(face, 1.0/255, (96,96), (0,0,0), swapRB=True, crop=False)
            embedder.setInput(faceBlob)
            face_vectors = embedder.forward()

            #adding the person name and face embeddings in a new list
            flagged_names.append(person_name)
            flagged_embeddings.append(face_vectors.flatten())
            total = total + 1
#saving the embeddings and person name
print("Saving {} face embeddings...".format(total))
print("Save Complete")

labelled_data = {"embeddings": flagged_embeddings, "names": flagged_names}
file_object = open(args["embeddings"], "wb")
file_object.write(pickle.dumps(labelled_data))
file_object.close()
