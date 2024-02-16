'''
TRAIN_CLASSIFIER.PY  - VERSION 1
COREY NOLAN
MCS CAPSTONE SPRING 2024
GROUP 6 - COREY NOLAN/CHRIS PATRELLA, YUYUAN LIU
ASL DETECTOR


GENERATE LANDMARK DATA FROM ASL PHOTOS

REQUIRES DOWNLOAD OF MEDIAPIPE MODELS FOR LANDMARK RECOGNITION:
--HANDLANDMARKER (FULL) AT https://developers.google.com/mediapipe/solutions/vision/hand_landmarker/index#models
----CONTAINS TWO MODELS USED FOR FINDING PALMS AND HAND LANDMARKS


IMAGE DATA DIRECTORIES MUST FOLLOW THE CORRECT NAMING CONVENTION
-ALL LETTER IMAGES MUST BE IN A DIRECTORY THAT IS NAMED FOR IT'S LETTER
---- ALL IMAGES FOR THE LETTER A, MUST BE IN A DIRECTORY NAMED 'A'

TAKES IMAGE FILES AS INPUT, PERFORMS HAND LANDMARK DETECTION ON EACH HAND IN THE PHOTO
EXPORTS THE LANDMARK DATA AS X/Y COORDS ALONG WITH A LETTER NAME LABEL TO 'DATA.PICKLE'


RESOURCES:
https://developers.google.com/mediapipe/solutions/vision/hand_landmarker
https://colab.research.google.com/github/googlesamples/mediapipe/blob/main/examples/hand_landmarker/python/hand_landmarker.ipynb#scrollTo=s3E6NFV-00Qt&uniqifier=1
https://www.youtube.com/watch?v=MJCSjXepaAM

'''


import pickle
# IMPORT THE CLASSIFIER TYPE FOR TRAINING THE MODEL
from sklearn.ensemble import RandomForestClassifier
# IMPORT FOR BREAKING DATA UP INTO TRAINING/TESTING SPLIT
from sklearn.model_selection import train_test_split
# IMPORT FOR SHOWING ACCURACY RESULTS
from sklearn.metrics import accuracy_score
import numpy as np
import os
import time

# SET THE TEST/TRAIN SPLIT PERCENTAGE
TEST_SIZE = .2
# SELECT THE CLASSIFIER TYPE FOR TRAINING
MODEL = RandomForestClassifier()
# BASE DATA DIR
DATA_DIR = 'data'
# DATA FILE NAME
DATA_FILE = 'data.pickle'
# BASE DIR FOR MODELS
MODEL_DIR = 'models'
# MODEL NAME
MODEL_FILE = 'aslModel.p'
# CURRENT RUNNING DIRECTORY/BASE DIRECTORY
BASE_DIR = os.getcwd()
# COMBINE TO CREATE ABSOLUTE PATHS FOR DATA AND MODELS
DATA_PATH = os.path.join(BASE_DIR, DATA_DIR,DATA_FILE)
MODEL_PATH = os.path.join(BASE_DIR,MODEL_DIR,MODEL_FILE)


def main():

    # GET THE START TIME
    startTime = time.time()

    # CHECK FOR MODELS DIRECTORY, CREATE IF NOT ALREADY CREATED.
    if not os.path.exists(os.path.join(BASE_DIR,MODEL_DIR)):
        os.makedirs(os.path.join(BASE_DIR,MODEL_DIR))

    # OPEN THE DATA FILE CONTAINING ALL THE HAND LANDMARK DETECTIONS AND LABELS
    data_dict = pickle.load(open(DATA_PATH, 'rb'))
    # data_dict = pickle.load(open(os.path.join(BASE_DIR,DATA_DIR, DATA_FILE), 'rb'))


    # EXTRACT DATA FROM IMPORTED FILE INTO A NUMPY ARRAY (RQUIRED BY SKLEARN MODULE)
    data = np.array(data_dict['data'])

    # EXTRACT THE LABELS FROM THE IMPORTED FILE INTO A NUMPY ARRAY
    labels = np.array(data_dict['labels'])

    print(f"\nSplitting data into testing and training with {TEST_SIZE*100}% reserved for testing.\n ")

    # SPLIT THE WHOLE DATA SET INTO TRAINING AND TESTING SETS
    x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=TEST_SIZE, shuffle=True, stratify=labels)

    print(f"Training the classifier using {MODEL}.\n ")
    # TRAIN THE MODEL USING THE TRAINING DATA/LABELS
    MODEL.fit(x_train, y_train)

    print(f"Testing the trained {MODEL}.\n ")
    # TEST THE PREDICTION OF THE TRAINED MODEL, BY SUPPLING DATA WITHOUT LABELS, STORE THE PREDICTION (OF LABEL) IN Y_PREDICT
    y_predict = MODEL.predict(x_test)


    # GET THE ACCURACY SCORE
    score = accuracy_score(y_predict, y_test)
    print(f"{score*100}% of samples were classified correctly")

    # EXPORT THE MODEL
    file = open(MODEL_PATH, 'wb')
    pickle.dump({'model': MODEL},file)
    file.close()

    # GET THE STOP TIME
    endTime = time.time()

    # PRINT EXECUTION TIME TO THE SCREEN
    print(f"\nExecution Time: {(((endTime-startTime)* 10**3)/1000)} Seconds")


    # # OPEN THE PICKLE FILE...IF NEEDED.
    # pickledFile = open(MODEL_PATH, 'rb')
    # pickledData = pickle.load(pickledFile)
    # print(f"The pickled model is: {pickledData['model']}\n")
    # pickledFile.close()

if __name__ == "__main__":
    main()