from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import argparse
import pickle
import matplotlib.pyplot as plt

argp = argparse.ArgumentParser()
argp.add_argument("-e", "--embeddings", required=True)
argp.add_argument("-r", "--recognizer", required=True)
argp.add_argument("-l", "--labelencoder", required=True)
args = vars(argp.parse_args())

#loading the face embeddings
print("Loading Face Embeddings...")
data = pickle.loads(open(args["embeddings"], "rb").read())

#encoding the labels
print("Encoding in process...")
le = LabelEncoder()
labels = le.fit_transform(data["names"])

#training the model
print("Model Training in progress...")
trainer = SVC(C=1.0, kernel="linear", probability=True)
trained_model = trainer.fit(data["embeddings"], labels)

#saving the configured face recognition model
file_object = open(args["recognizer"], "wb")
file_object.write(pickle.dumps(trained_model))
file_object.close()

#saving the label encoder
file_object = open(args["labelencoder"], "wb")
file_object.write(pickle.dumps(le))
file_object.close()