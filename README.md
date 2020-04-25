# Face-Recognition-and-Identification

An ongoing project on Face Recognition and Identification, and a part of the academic minor project of 6th semester.

## Steps to run and test the model:

**Step 1:** Download and install the libraries using the command:
                pip install -r requirements.txt

**Step 2:** In the 'data' folder, create one folder for each person's face that you are going to identify.
        eg. data
              |-Person1
                |-00001.jpg
                |-00002.jpg
                |-00003.jpg
                |-00004.jpg
              |-Person2
                |-00001.jpg
                |-00002.jpg
                |-00003.jpg
                |-00004.jpg
                
**Steps 3:** Download the following files and folders from the given link and put it up in your directory. 
        https://drive.google.com/open?id=1hh4aAYVB3vYSCt91dB43j4XvgoSylJTi
        
**Step 4:** Run the first command for extracting embeddings from the image data: 
                python embeddings_extraction.py --dataset data\ --embeddings output/embeddings.pickle --detector face_detector_caffe_model\ --embedding-model openface_nn4.small2.v1.t7

**Step 5:** Run the second command to train the model: 
                python train.py --embeddings output/embeddings.pickle --recognizer output/recognizer.pickle --labelencoder output/le.pickle

**Step 6:** Run the third command to run the webcam application: 
                python identification.py --detector face_detector_caffe_model\ --embedding-model openface_nn4.small2.v1.t7 --recognizer output/recognizer.pickle --labelencoder output/le.pickle
