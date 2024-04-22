'''
INFERENCECLASSIFIER.PY  - VERSION 1
COREY NOLAN
MCS CAPSTONE SPRING 2024
GROUP 6 - COREY NOLAN/CHRIS PATRELLA, YUYUAN LIU
ASL FINGERSPELT LETTER TO WRITTEN ENGLISH LETTER TRANSLATION

DESCRIPTION:
-TAKES AS INPUT A JPG IMAGE OF AN ASL FINGERSPELLED LETTER.
-PERFORMS PALM, FINGER DETECTION
-PERFORMS HAND LANDMARKER DETECTION USING A PRE-TRAINED MODEL
----THIS MODEL WAS TRAINING USING THOUSANDS OF FINGERSPELLED LETTER IMAGES
-RETURNS SUCCESSCODE AND PREDICTED ENGLISH ALPHABET LETTER(EXCLUDING 'J' AND 'Z')
----TO 'asl_main.py' AS A JSON FORMATTED STRING


REQUIREMENTS:
-'InferenceClassifier.py' MUST BE IN THE SAME DIR AS AslMain.py


OUTPUTS:
-'AslMain.py' PERFORMS A CALL TO 'getResult()'
-FUNCTION BUILDS A JSON FORMATED STRING FROM THE 'self.predictedLetter' AND = 'self.successCode'(0 = Success/ 1 = Failure)
 ----VARIABLES.


RESOURCES:
https://developers.google.com/mediapipe/solutions/vision/hand_landmarker
https://colab.research.google.com/github/googlesamples/mediapipe/blob/main/examples/hand_landmarker/python/hand_landmarker.ipynb#scrollTo=s3E6NFV-00Qt&uniqifier=1
https://www.youtube.com/watch?v=MJCSjXepaAM

DATASETS:
https://www.kaggle.com/datasets/danrasband/asl-alphabet-test
https://www.kaggle.com/datasets/ayuraj/asl-dataset/data
https://www.kaggle.com/datasets/grassknoted/asl-alphabet


'''
import json
import os
import mediapipe as mp
import cv2
import matplotlib.pyplot as plt
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import pickle
import random
import time
import json
import joblib

class Inference:

# SET THE CLASS CONSTANTS
    # LABELS TO CHECK FOR MATCHES
    LABELS = {"A":"A","B":"B","C":"C","D":"D","E":"E","F":"F","G":"G","H":"H","I":"I","K":"K",
              "L":"L","M":"M","N":"N","O":"O","P":"P","Q":"Q","R":"R","S":"S","T":"T","U":"U",
              "V":"V","W":"W","X":"X","Y":"Y",'call': 'call', 'dislike': 'dislike', 'fist': 'fist',
              'four': 'four', 'like': 'like', 'mute': 'mute', 'ok': 'ok', 'one': 'one', 'palm': 'palm',
              'peace': 'peace', 'peace_inverted': 'peace_inverted', 'rock': 'rock', 'stop': 'stop',
              'stop_inverted': 'stop_inverted', 'three': 'three', 'three2': 'three2', 'two_up': 'two_up',
              'two_up_inverted': 'two_up_inverted'}
    # GET THE CURRENT WORK DIRECTORY, USE AS BASE PATH
    BASE_DIR = os.getcwd()
    # LOCATION OF MODEL
    MODEL_DIR = 'models'
    # NAME OF MODEL FILE
    # MODEL_FILE = 'aslModel3.p'
    # MODEL_FILE = 'aslKnnModel.p'
    MODEL_FILE = 'aslModel.joblib'

# LOCATION OF USER UPLOADED IMAGES (WILL BE IN S3 BUCKET LOCATION IN AWS)
    USER_DIR = 'user_image_dir'

    # FULL PATH TO THE INFERENCE MODEL
    MODEL_PATH = os.path.join(BASE_DIR,MODEL_DIR,MODEL_FILE)

    # FULL PATH TO THE USER IMAGE DIRECTORY
    USER_IMAGE_PATH = os.path.join(BASE_DIR, USER_DIR)

    # FULL PATH TO TO HANDLANDMARKER TASK, CONTAINS PALM AND FINGER LANDMARK DETECTION MODELS
    MODEL_TASK_PATH =r'S:\Program\GitHub\ASL-Recognition\aslml\hand_landmarker.task'

    def __init__(self, user_img_file):

        self.img_file = os.path.join(self.USER_IMAGE_PATH,user_img_file)

        # SET TASK OPTIONS FOR HAND LANDMARKER
        self.baseOptions = mp.tasks.BaseOptions
        self.handLandMarker = mp.tasks.vision.HandLandmarker
        self.handLandMarkerOptions = mp.tasks.vision.HandLandmarkerOptions
        self.visionRunningMode = mp.tasks.vision.RunningMode
        self.predictedLetter = ''
        self.successCode = 0 # 0 = Success/ 1 = Failure

    def inferenceClassify(self):

        self.predictedLetter = ''

        # LOAD THE TRAINED MODEL FROM THE PATH (**FOR JOBLIB FILES**)
        aslModelDict = joblib.load(open(self.MODEL_PATH, 'rb'))

        # LOAD THE TRAINED MODEL FROM THE PATH (**FOR PICKLE FILES**)
        # aslModelDict = pickle.load(open(self.MODEL_PATH,'rb'))

        aslModel = aslModelDict['model']
        # input("press to continue..")

        # SET THE OPTIONS FOR THE LANDMARKER INSTANCE WITH THE IMAGE MODE
        options = self.handLandMarkerOptions(
            base_options = self.baseOptions(self.MODEL_TASK_PATH),
            running_mode = self.visionRunningMode.IMAGE,
            num_hands=2)

        # CREATE A HAND LANDMARKER INSTANCE
        detector = self.handLandMarker.create_from_options(options)

        # READ IN THE IMAGE FROM THE FILE PATH
        userImage = mp.Image.create_from_file(self.img_file)

        # DETECT THE LANDMARKS
        detection_result = detector.detect(userImage)
        # print(f'detection result: {detection_result.hand_landmarks}')

        # IF HANDS WERE DETECTED
        if detection_result.hand_landmarks:
            # success += 1
            detected = []
            # print("inside the detecttion result loop\n")
            # FOR EACH OF THE HANDS DETECTED, ITERATE THROUGH THEM
            # for idx in range(len(detection_result.hand_landmarks))[:1]:
            for idx in range(len(detection_result.hand_landmarks)):
                # FOR EACH LANDMARK, GET THE X AND Y COORDINATE
                for i in detection_result.hand_landmarks[idx]:
                    # print('x is', i.x, 'y is', i.y, 'z is', i.z, 'visibility is', i.visibility)
                    x = i.x
                    y = i.y

                    # STORE X AND Y IN THE TEMP ARRAY
                    detected.append(x)
                    detected.append(y)

            print(f"Detected: {detected}")

            # RUN THE INFERENCE MODEL AGAINST THE LANDMARKS DETECTED
            prediction = aslModel.predict([np.asarray(detected)])

            # OUTPUT THE RESULT BASED ON MATCHES IN THE LABELS DICTIONARY
            predictedLetter = self.LABELS[(prediction[0])]

            # SET THE LOCAL VAR WITH THE RESULT FROM THE INFERENCE MODEL
            self.predictedLetter = predictedLetter
            self.successCode = 0

        # IF NO LANDMARKS WERE DETECTED IN THE IMAGE
        else:
            print(f"Inside failed inference classifier")
            # input("press to continue...")
            self.predictedLetter = 'None'
            self.successCode = 1

    # GET THE RESULT OF THE INFERENCE
    def getResult(self):

        # CREATE DICTIONARY OF VARIABLES TO JSONIFY
        pSon = {
            'SuccessCode' : self.successCode,
            'InferResult' : self.predictedLetter,
        }

        # CONVERT FROM PYTHON DICT TO JSON OBJECT
        jSon = json.dumps(pSon)

        # RETURN JSON OBJECT
        return jSon