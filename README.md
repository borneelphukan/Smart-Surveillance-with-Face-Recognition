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

![Dataflow Diagram]()

## Functional Requirements

   1. Download the Face Detector Caffe Model from here: https://drive.google.com/open?id=1hh4aAYVB3vYSCt91dB43j4XvgoSylJTi
   2. Run the embeddings_extraction.py using the command:
   `python embeddings_extraction.py --dataset data\ --embeddings output/embeddings.pickle --detector face_detector_caffe_model\ --embedding-model openface_nn4.small2.v1.t7`
   3. Run the train.py using the command:
   `python train.py --embeddings output/embeddings.pickle --recognizer output/recognizer.pickle --labelencoder output/le.pickle`
   4. Run the identification.py using the command:
   `python identification.py --detector face_detector_caffe_model\ --embedding-model openface_nn4.small2.v1.t7 --recognizer output/recognizer.pickle --labelencoder output/le.pickle`
    
## Other Details

 **Language Written In:** Python 3.6
 
 **Environment (IDE):** Anaconda IDE (Jupyter Notebook/Spyder), PyCharm CE, VSCode
