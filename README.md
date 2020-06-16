# SMART SURVEILLANCE SYSTEM USING FACIAL RECOGNITION SYSTEM (PROJECT IDEATION)

## Abstract

Surveillance System has always been around us and with the increase in global crime
rate and unlawful activities, it has become very mandatory that these surveillance
systems be fitted with some technology that has the capability to detect and identify
the person commiting the activity. Although multiple detection system has been
developed using basic software engineering methodologies, the model proposed here
makes a Deep Learning approach to the problem and makes use of OpenFace Neural
Network Architecture for the purpose of detecting, identifying and tracking the
person on frame.
Through this project, we would like to present a comprehensive study of all the
researches done on OpenFace Neural Network, compare it with other face
recognition system and provide a structured explanation of the implementation of
OpenFace neural network in creating a smart surveillance system.

**Keywords**: FaceNet, OpenFace, LFW Dataset, SVM Classifier, Embeddings
Extraction, Triplet Loss

## Structure

In this project we are combining two algorithms to detect and classify faces for digital detection. A FaceNet neural network model and a Support Vector Classifier.

Our dataset is a combination of images, divided into classes (folders). Each class contains 20 images and each class represents a person.

The FaceNet model consist of two major components. Detector and Embedder.

   • The detector takes the weights included with the FaceNet model which helps in localizing faces in an image.
    
   • The embedder works as a feature extraction tool. It extracts the embedding weights from the pre-trained model and uses those weights in extracting features from our own face database.

Once the feature extraction and detection are done, we use the extracted features from the image data to train our Support Vector Classifier (SVC). We use a linear kernel for the same. The trained weights are then stored in the form of pickle files.

After the SVC is trained, we move on to using those pickle files for the main detection and identification of the person in frame. Here we make use of openCV and using the weights, we carry out the proper detection along with the probability of occurrence.

## System Design

![System Design](https://github.com/borneelphukan/Smart-Surveillance-using-OpenFace-Face-Recognition/blob/master/implementation.png)

The system is designed using the following major components in mind:
**1. Database** – The database is a very important component that stores the image data
of the person who’s face is to be detected. Here, the images of every person is stored in
an individual sub-folder with the name of the person. Setting the sub-folder as the
name of the person is very important because we will be extracting the name of the sub
folder and train and use it for identification of the person on frame.
School of Computer Engineering, KIIT, BBSR
8Smart Surveillance System using OpenFace Face Recognition
2. **openface_nn4.small2.v1.t7** – This is the pytorch implementation of the FaceNet model. The
model makes use of Google’s FaceNet architecture for feature extraction and uses a triplet
loss function to test how accurate the neural net classifies a face. This model is trained using
50,000 images of the Labelled Faces in the Wild Home (LFW) Dataset and the weights stored as
a caffe model.
3. **res10_300x300_ssd_iter_140000.caffemodel** – The caffemodel is used for storing the
trained weights generated after training the openface neural network. It contains a
deploy.prototext file which stores the meta data of the model. In this project, we will be using
and overwriting these weights with our own custom face image data to generate an entirely new
facial recognition model.
4. **Support Vector Classifier** – Once the process of triplet loss calculation has been completed,
the process of classifying the different face images according to the person on frame is initiated.
For this, we can either use multiple classification, random forest classification or support vector
classification. Out of all these classification technique, the “sklearn-svc” or Support Vector
Classifier has proven to be the most accurate in terms of classification.
5. **Creating the FaceBlob** - Once the classification procedure has been accomplished, it is time
to initiate the creation of a Faceblob around the face on frame. This is done using the OpenCV
framework where we create a square around the face embeddings and implement it on a loop.
6. **Computing the Softmax probabilities** – Once the detection and creation of the face blob is
completed, we have to compute the softmax probabilities. A softmax probability is used for
determining the degree of accuracy of prediction by the system. It is a comparision between the
absolute value and the real value. Higher the softmax probability, more is the probability that
the prediction of the person is correct.

## Functional Requirements

   1. Download the Face Detector Caffe Model from here: https://drive.google.com/open?id=1hh4aAYVB3vYSCt91dB43j4XvgoSylJTi
   2. Run the embeddings_extraction.py using the command:
   `python embeddings_extraction.py --dataset data\ --embeddings output/embeddings.pickle --detector face_detector_caffe_model\ --embedding-model openface_nn4.small2.v1.t7`
   3. Run the train.py using the command:
   `python train.py --embeddings output/embeddings.pickle --recognizer output/recognizer.pickle --labelencoder output/le.pickle`
   4. Run the identification.py using the command:
   `python identification.py --detector face_detector_caffe_model\ --embedding-model openface_nn4.small2.v1.t7 --recognizer output/recognizer.pickle --labelencoder output/le.pickle`
