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
https://www.mygreatlearning.com/blog/gridsearchcv/

'''


import pickle
# IMPORT THE CLASSIFIER TYPE FOR TRAINING THE MODEL
from sklearn.ensemble import RandomForestClassifier
# IMPORT FOR BREAKING DATA UP INTO TRAINING/TESTING SPLIT
from sklearn.model_selection import train_test_split, GridSearchCV
# IMPORT FOR SKLEARN METRICS
from sklearn.metrics import classification_report,confusion_matrix
# IMPORT FOR SHOWING ACCURACY RESULTS
from sklearn.metrics import accuracy_score
import numpy as np
import os
import time
# FOR EXPORTING MODEL INTO THE FORMAT AWS EXPECTS
import joblib

# SET THE TEST/TRAIN SPLIT PERCENTAGE
TEST_SIZE = .2
# SELECT THE CLASSIFIER TYPE FOR TRAINING
# MODEL = RandomForestClassifier()

# CURRENT RUNNING DIRECTORY/BASE DIRECTORY
BASE_DIR = os.getcwd()

# BASE DATA DIR
DATA_DIR = 'data'
# DATA FILE NAME (may need to change this to a straight csv file instead of pickle file)
# DATA_FILE = 'data.joblib'

# INPUT DATA FILE NAME (may need to change this to a straight csv file instead of pickle file)
DATA_FILE = 'data.pickle'
# BASE DIR FOR MODELS
MODEL_DIR = 'models'

# OUTPUT JOBLIB FILE MODEL NAME
MODEL_FILE = 'aslModel.joblib'

# OUTPUT PICKLE FILE MODEL NAME
# MODEL_FILE = 'aslModel.p'

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
    # data_dict = joblib.load(open(DATA_PATH, 'rb'))
    data_dict = pickle.load(open(DATA_PATH, 'rb'))


    # EXTRACT DATA FROM IMPORTED FILE INTO A NUMPY ARRAY (RQUIRED BY SKLEARN MODULE)
    data = np.array(data_dict['data'])

    # EXTRACT THE LABELS FROM THE IMPORTED FILE INTO A NUMPY ARRAY
    labels = np.array(data_dict['labels'])

    print(f"\nSplitting data into testing and training with {TEST_SIZE*100}% reserved for testing.\n ")

    # SPLIT THE WHOLE DATA SET INTO TRAINING AND TESTING SETS
    x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=TEST_SIZE, shuffle=True, stratify=labels)


    # DEFINE THE PARAMETER GRID
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [None, 10],
        'min_samples_split': [2],
        'min_samples_leaf': [1, 2],
        'bootstrap': [True, False]
    }

    # INIT THE GRIDSEARCH OBJECT
    grid_search = GridSearchCV(estimator=RandomForestClassifier(), param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)

    # DO GRID SEARCH/FIND BEST HYPER PARAMS FOR OUR MODEL/DATA
    print("Starting Grid Search...")
    grid_search.fit(x_train, y_train)

    print("Here are the best params found:\n")
    print(grid_search.best_params_)

    # SAVE THE BEST MODEL
    best_model = grid_search.best_estimator_

    # TEST THE PREDICTION OF THE TRAINED MODEL, BY SUPPLING DATA WITHOUT LABELS, STORE THE PREDICTION (OF LABEL) IN Y_PREDICT
    y_predict = best_model.predict(x_test)


    print("CLASSIFICATION REPORT:\n")
    print(classification_report(y_test, y_predict))


    # GET THE ACCURACY SCORE
    score = accuracy_score(y_predict, y_test)
    print(f"\n\n{score*100}% of samples were classified correctly")

    # CREATE A NEW FILE TO STORE THE MODEL
    file = open(MODEL_PATH, 'wb')

    # # IF CREATING A PICKLE FILE...
    # pickle.dump({'model': MODEL}, file)
    # file.close()

    # IF CREATING A JOBLIB FILE...
    joblib.dump({'model': best_model},MODEL_PATH)
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

    # # OPEN THE JOBLIB FILE...IF NEEDED.
    # jobLibFile = open(MODEL_PATH, 'rb')
    # jobLibData = joblib.load(jobLibFile)
    # print(f"The joblib model is: {jobLibData['model']}\n")
    # jobLibFile.close()

if __name__ == "__main__":
    main()
