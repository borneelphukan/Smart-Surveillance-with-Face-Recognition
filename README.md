# SMART SURVEILLANCE SYSTEM USING FACIAL RECOGNITION SYSTEM (PROJECT IDEATION)

## Introduction

Security surveillance has always been a prime necessity, especially in criminal prone zones. However, the recent changing trend in the surveillance system has proved that a security camera with the added advantage of recognizing the person on the frame is more preferable to the traditional security cameras. These make the process of tracking culprits easier for the authorities. Therefore through this project we have come up with a surveillance software powered by deep learning which uses the combination of a per-trained Neural Network Model and Support Vector Classifier to detect the face of the person in frame and classify them according to the face data currently stored in the system. 

## Structure

In this project we are combining two algorithms to detect and classify faces for digital detection. A FaceNet neural network model and a Support Vector Classifier.

Our dataset is a combination of images, divided into classes (folders). Each class contains 20 images and each class represents a person.

The FaceNet model consist of two major components. Detector and Embedder.
    • The detector takes the weights included with the FaceNet model which helps in localizing faces in an image.
    • The embedder works as a feature extraction tool. It extracts the embedding weights from the pre-trained model and uses those weights in extracting features from our own face database.

Once the feature extraction and detection are done, we use the extracted features from the image data to train our Support Vector Classifier (SVC). We use a linear kernel for the same. The trained weights are then stored in the form of pickle files.

After the SVC is trained, we move on to using those pickle files for the main detection and identification of the person in frame. Here we make use of openCV and using the weights, we carry out the proper detection along with the probability of occurrence.

## Flow Chart

![Dataflow Diagram](https://github.com/borneelphukan/Face-Recognition-and-Identification/blob/borneelphukan-patch-1/data%20flow%20diagram.jpeg)

## Tools Used

   1. **OpenCV** – We are using OpenCV for the purpose of detecting the face objects through the webcam of our device.
   2. **Scikit-learn** – We are using a Support Vector Classifier from the SVM class in scikit-learn library in order to classify the different faces embedding that we have extracted from our face database and identify the different people on frame.
   3. **Pytorch** – In order to calculate the 128d embeddings from the face dataset and extract those embeddings, we need to make use of the pre-trained FaceNet neural network. This FaceNet needs Pytorch framework to function and therefore make the embedder responsive.
   4. **Caffe** – We use a pre-trained detector that is stored as a caffe model which is used to detect the area on the screen where the face is located. The detector is a part of the pre-trained FaceNet Neural Network.

## Other Details

 **Language Written In:** Python 3.6
 
 **Environment (IDE):** Anaconda IDE (Jupyter Notebook/Spyder), PyCharm CE, VSCode
