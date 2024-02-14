'''
INFERENCE_CLASSIFIER.PY  - VERSION 2
COREY NOLAN
MCS CAPSTONE SPRING 2024
GROUP 6 - COREY NOLAN/CHRIS PATRELLA, YUYUAN LIU
ASL DETECTOR

DETECT THE CORRECT ENGLISH LANGUAGE ALPHABET LETTER (EXCLUDING J AND Z) IN AN AMERICAN SIGN LANGUAGE
FINGER SPELT IMAGE.

TAKES IMAGE FILES AS INPUT, PERFORMS...THE INPUT FILE OF THE ...

REQUIREMENTS:--


IMAGE DATA DIRECTORIES MUST FOLLOW THE CORRECT NAMING CONVENTION
--

OUTPUTS:
RESULTS OF THE PREDICTION ARE (CURRENTLY) PRINTED TO SCREEN


RESOURCES:
https://developers.google.com/mediapipe/solutions/vision/hand_landmarker
https://colab.research.google.com/github/googlesamples/mediapipe/blob/main/examples/hand_landmarker/python/hand_landmarker.ipynb#scrollTo=s3E6NFV-00Qt&uniqifier=1
https://www.youtube.com/watch?v=MJCSjXepaAM

'''


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



# GET THE CURRENT WORK DIRECTORY, USE AS BASE PATH
BASE_DIR = os.getcwd()
MODEL_DIR = 'models'
MODEL_FILE = 'aslModel.p'
USER_DIR = 'user_image'
# USER_DIR = r'images\train_full\X'

# USER_IMAGE = 'A1910.jpg'
# USER_IMAGE = 'Ayush_A.jpg'
# USER_IMAGE = '95.jpg'
# USER_IMAGE = 'B0.jpg'
# USER_IMAGE = 'A0.jpg'
# USER_IMAGE = 'Ayush_B.jpg'
# USER_IMAGE = 'X4.jpg'
# USER_IMAGE = 'X245.jpg'

MODEL_PATH = os.path.join(BASE_DIR,MODEL_DIR,MODEL_FILE)

# USER_IMAGE_PATH = os.path.join(BASE_DIR, USER_DIR, USER_IMAGE)
USER_IMAGE_PATH = os.path.join(BASE_DIR, USER_DIR)

MODEL_TASK_PATH =r'C:\Users\corey\PycharmProjects\ASL1\venv\hand_landmarker.task'


LABELS = {"A":"A","B":"B","C":"C","D":"D","E":"E","F":"F","G":"G","H":"H","I":"I","K":"K",
          "L":"L","M":"M","N":"N","O":"O","P":"P","Q":"Q","R":"R","S":"S","T":"T","U":"U",
          "V":"V","X":"X","Y":"Y"}



def main():

    # (IF NEED TO LIMIT SIZE OF INPUT DATA) SET THE SAME SIZE AS A PERCENTAGE OF THE OVERALL DATA
    # sampleSizePercentage = 5

    # LOAD THE IMAGE FROM THE PATH
    # userImage = mp.Image.create_from_file(USER_IMAGE_PATH)

    # LOAD THE TRAINED MODEL FROM THE PATH
    aslModelDict = pickle.load(open(MODEL_PATH,'rb'))
    aslModel = aslModelDict['model']

    # SET TASK OPTIONS FOR HAND LANDMARKER
    baseOptions = mp.tasks.BaseOptions
    handLandMarker = mp.tasks.vision.HandLandmarker
    handLandMarkerOptions = mp.tasks.vision.HandLandmarkerOptions
    visionRunningMode = mp.tasks.vision.RunningMode


    # SET THE OPTIONS FOR THE LANDMARKER INSTANCE WITH THE IMAGE MODE
    options = handLandMarkerOptions(
        base_options = baseOptions(MODEL_TASK_PATH),
        running_mode=visionRunningMode.IMAGE,
        num_hands=2)

    # CREATE A HAND LANDMARKER INSTANCE
    detector = handLandMarker.create_from_options(options)

    # TEMP ARRAY TO HOLD ONE HANDS WORTH OF X/Y HANDMARK COORDS
    # tempDetected = []
    success = 0
    fail = 0

    # dirImgCount = len(os.listdir(os.path.join(USER_IMAGE_PATH)))
    # sampleSize = int((sampleSizePercentage / 100) * dirImgCount)

    print(f'user image path is: {USER_IMAGE_PATH}')
    for img_file in os.listdir(USER_IMAGE_PATH):
    # for img_file in random.sample(os.listdir(USER_IMAGE_PATH),sampleSize):
        print(f'user image path is: {USER_IMAGE_PATH}')
        print(f"Loading image: {img_file}")


        userImage = mp.Image.create_from_file(os.path.join(USER_IMAGE_PATH,img_file))


        # DETECT THE LANDMARKS
        detection_result = detector.detect(userImage)
        # print(f'detection result: {detection_result.hand_landmarks}')

        # IF HANDS WERE DETECTED
        if detection_result.hand_landmarks:
            success += 1
            tempDetected = []
            print("inside the detecttion result loop\n")
            # FOR EACH OF THE HANDS DETECTED, ITERATE THROUGH THEM
            # for idx in range(len(detection_result.hand_landmarks))[:1]:
            for idx in range(len(detection_result.hand_landmarks)):
                # FOR EACH LANDMARK, GET THE X AND Y COORDINATE
                for i in detection_result.hand_landmarks[idx]:
                    # print('x is', i.x, 'y is', i.y, 'z is', i.z, 'visibility is', i.visibility)
                    x = i.x
                    y = i.y

                    # STORE X AND Y IN THE TEMP ARRAY
                    tempDetected.append(x)
                    tempDetected.append(y)


            prediction = aslModel.predict([np.asarray(tempDetected)])

            predictedLetter = LABELS[(prediction[0])]

            print(f'{predictedLetter}\n')
        else:
            print(f'Failed to detect {img_file}\n')
            fail += 1
    print(f"sucesses: {success}")
    print(f'fails: {fail}\n')


if __name__=="__main__":
    main()

