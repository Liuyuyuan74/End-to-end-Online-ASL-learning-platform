'''
CREATE_DATASET.PY  - VERSION 3
COREY NOLAN
MCS CAPSTONE SPRING 2024
GROUP 6 - COREY NOLAN/CHRIS PATRELLA, YUYUAN LIU
ASL DETECTOR

GENERATE LANDMARK DATA FROM ASL PHOTOS

TAKES IMAGE FILES AS INPUT, PERFORMS HAND LANDMARK DETECTION ON EACH HAND IN THE PHOTO
EXPORTS THE LANDMARK DATA AS X/Y COORDS ALONG WITH A LETTER NAME LABEL TO 'DATA.PICKLE'
THIS FILE IS USED AS THE INPUT FILE OF THE train_classifier.py FILE.

REQUIREMENTS:
DOWNLOAD OF MEDIAPIPE MODELS FOR LANDMARK RECOGNITION:
--HANDLANDMARKER (FULL) AT https://developers.google.com/mediapipe/solutions/vision/hand_landmarker/index#models
----CONTAINS TWO MODELS USED FOR FINDING PALMS AND HAND LANDMARKS


IMAGE DATA DIRECTORIES MUST FOLLOW THE CORRECT NAMING CONVENTION
-ALL LETTER IMAGES MUST BE IN A DIRECTORY THAT IS NAMED FOR IT'S LETTER
---- ALL IMAGES FOR THE LETTER A, MUST BE IN A DIRECTORY NAMED 'A'
------./IMAGES/A/A1.JPG

SET THE DESIRED SAMPLE SIZE BASED BASED ON PERCENTAGE USING THE INT VARIABLE 'sampleSizePercentage'.
EACH IMAGE SUBFOLDER (EACH LETTER) IS RANDOMLY SAMPLED BASED ON THIS SETTING.
-EXAMPLE - FOR 7% OF THE TOTAL AVAILABLE DATA TO BE SAMPLED...
----sampleSizePercentage = 7

OUTPUTS:
RESULTS OF THE HAND LANDMARK DETECTION ARE PRINTED TO SCREEN
LANDMARKS AND LABELS ARE OUTPUT TO data.pickle FILE



FULL PIPELINE:
-CREATE_DATASET.PY (TO CREATE A FILL CONTAINING A LARGE ARRAY OF X/Y COORDS OF HAND LANDMARKS DETECTED FOR EACH ASL FINGERSPELLED LETTER)
-TRAIN_CLASSIFIER.PY (TO CREATE A RANDOM FOREST TRAINED MODEL USING THE X/Y COORDS AND LABELS FROM THE DATASET CREATED WITH CREATE_DATASET.PY)
-ASL_MAIN.PY (IS THE CONTROLLER SCRIPT FOR RUNNING INFERENCE ON ONE IMAGE AT A TIME. IT CALLS THE INFERENCE_CLASSIFIER.PY INFERENCECLASSIFY FUNCTION)
-INFERENCE_CLASSIFER.PY - (TO PERFORM INFERENCE ON IMAGES)
-**INFERENCETESTER.PY - (TO TEST THE ACCURACY OF THE TRAINED MODEL AGAINST A LARGE SET OF NEW/NEVER SEEN DATA. IT CALLS INFERENCE_CLASSIFIER IN A LOOP FOR EACH IMAGE IN THE DIR)

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



# PATH TO THE MODEL USED FOR DETECTION
MODEL_PATH ='./hand_landmarker.task'
# GET THE CURRENT WORK DIRECTORY, USE AS BASE PATH
BASE_DIR = os.getcwd()
# PATH TO THE IMAGE DATA
# IMAGE_DIR = './images/ASL_Set'
# IMAGE_DIR = './images/train_full'
# IMAGE_DIR = './images/Ayush_set/asl_dataset'
# IMAGE_DIR = './images/DIY_Signs'
# IMAGE_DIR = './images/A_SubSet-ASLDataset'
IMAGE_DIR = './images/Full_Training_Dataset'
# DIR FOR OUTPUTTING THE HAND LANDMARK DATA
DATA_DIR = 'data'

# # PICKLE FILE NAME FOR HAND LANDMARK DATA FILE
DATA_FILE = 'data/data.pickle'

# JOBLIB FILE NAME FOR HAND LANDMARK DATA FILE
# DATA_FILE = 'data.joblib'

# FULL PATH TO IMAGE DIRECTORY
IMAGE_PATH = os.path.join(BASE_DIR, IMAGE_DIR)
# FULL PATH TO DATA DIRECTORY
DATA_PATH = os.path.join(BASE_DIR, DATA_DIR, DATA_FILE)

# SET THE SAME SIZE AS A PERCENTAGE OF THE OVERALL DATA
sampleSizePercentage = 10

def main():

    # GET THE START TIME
    startTime = time.time()

    # PRINT EXECUTION TIME TO THE SCREEN
    # print(f"Execution Time: {(endTime-startTime)* 10**3} ms")


    # CHECK FOR OUTPUTS DIRECTORY, CREATE IF NOT ALREADY CREATED.
    if not os.path.isdir(DATA_DIR):
        os.makedirs(os.path.join(BASE_DIR,DATA_DIR))

    # ARRAY TO HOLD THE X/Y COORDS OF THE LANDMARKS
    data = []
    # LABELS FOR EACH SIGN
    labels = []
    # VAR TO HOLD UNSUCCESSFUL DETECTION IMAGEPATHS
    failedLandmarks = []
    # VAR FOR IMAGE COUNT
    totalImageCount = 0

    # SET TASK OPTIONS FOR HAND LANDMARKER
    baseOptions = mp.tasks.BaseOptions
    handLandMarker = mp.tasks.vision.HandLandmarker
    handLandMarkerOptions = mp.tasks.vision.HandLandmarkerOptions
    visionRunningMode = mp.tasks.vision.RunningMode


    # SET THE OPTIONS FOR THE LANDMARKER INSTANCE WITH THE IMAGE MODE
    options = handLandMarkerOptions(
        base_options = baseOptions(MODEL_PATH),
        running_mode=visionRunningMode.IMAGE,
        num_hands=2)

    # CREATE A HAND LANDMARKER INSTANCE
    detector = handLandMarker.create_from_options(options)
    # print(options)

    # ITERATE THROUGH EACH LETTER SUB-DIRECTORY IN THE DATA_DIR DIRECTORY,
    for dir in os.listdir(IMAGE_DIR):
        print(f'Currently working on directory {dir}...\n\n')

        # GET THE TOTAL NUMBER OF IMAGES IN THIS DIRECTORY
        dirImgCount = len(os.listdir(os.path.join(IMAGE_DIR, dir)))

        # USE DESIRED PERCENTAGE TO CALCULATE CORRECT SAMPLE SIZE
        sampleSize = int((sampleSizePercentage/100)*dirImgCount)


        # ITERATE THROUGH EACH IMAGE FILE AND READ IN USING OPENCV, RANDOM SELECTION OF IMAGES FROM LETTER DIR
        # BASED ON SAMPLE SIZE
        for img_file in random.sample(os.listdir(os.path.join(IMAGE_DIR, dir)),sampleSize):

            # ITERATE NUMBER OF FILES PROCESSED, USED FOR FINAL RESULT DETAILS
            totalImageCount += 1

            # TEMP ARRAY TO HOLD ONE HANDS WORTH OF X/Y HANDMARK COORDS
            tempDetected = []

            # LOAD THE IMAGE FROM THE PATH
            img = mp.Image.create_from_file(os.path.join(IMAGE_DIR,dir,img_file))

            # DETECT THE LANDMARKS
            detection_result = detector.detect(img)

            # IF HANDS WERE DETECTED
            if detection_result.hand_landmarks:
                # FOR EACH OF THE HANDS DETECTED, ITERATE THROUGH THEM
                for idx in range(len(detection_result.hand_landmarks))[:1]:
                    # FOR EACH LANDMARK, GET THE X AND Y COORDINATE
                    for i in detection_result.hand_landmarks[idx]:
                        # print('x is', i.x, 'y is', i.y, 'z is', i.z, 'visibility is', i.visibility)
                        x = i.x
                        y = i.y

                        # STORE X AND Y IN THE TEMP ARRAY
                        tempDetected.append(x)
                        tempDetected.append(y)

                # ADD CONTENTS OF TEMP ARRAY TO DATA ARRAY BEFORE TEMP IS OVERWRITTEN BY THE NEXT IMAGE
                # print(temp)
                data.append(tempDetected)

                # ADD THE LABEL TO THE LABEL ARRAY
                labels.append(dir)

            else:
                failedLandmarks.append(img_file)

    # PRINT RESULTS TO SCREEN IF NEEDED
    print(f"Dataset sample size selected: {sampleSizePercentage}%")
    print(f"Total number of images processed ({sampleSizePercentage}% of Full Dataset): {totalImageCount}")
    print(f"Successful detections ({(len(data)/totalImageCount*100)}%): {len(data)}")
    print(f"Failed detections: {len(failedLandmarks)}\n\n")


    # OUTPUT LANDMARK COORDINATES AND LABELS TO A PICKLE FILE. CREATES A DICTIONARY FILE WITH X,Y COORDS AND LABEL FOR EACH HAND DECTECTED
            # {"data": [[x,y,x,y,x,y...],[x,y,x,y,x,y...]...], "labels: ["A","A","A","A"...]}  - THERE ARE 21 LANDMARKS PER HAND DETECTED (42 COORDS PER HAND)

    # IF EXPORTING TO .PICKLE FILE
    print("Landmark Detection Complete...Exporting x/y coords and labels to 'data.pickle'")
    f = open(DATA_PATH, 'wb')
    # f = open((os.path.join(DATA_DIR, DATA_FILE)),'wb')
    pickle.dump({'data': data, 'labels': labels}, f)
    f.close()

    # # IF EXPORTING TO JOBLIB FILE
    # print("Landmark Detection Complete...Exporting x/y coords and labels to 'data.pickle'")
    # f = open(DATA_PATH, 'wb')
    # joblib.dump({'data': data, 'labels': labels}, f)
    # f.close()

    # GET THE STOP TIME
    endTime = time.time()

    # PRINT EXECUTION TIME TO THE SCREEN
    print(f"\nExecution Time: {(((endTime - startTime) * 10 ** 3) / 1000)} Seconds")
    # print(f"Execution Time: {(endTime-startTime)* 10**3} ms")


    # # # OPEN THE PICKLE FILE...IF NEEDED.
    # # pickledFile = open(DATA_PATH, 'rb')
    # # # pickledFile = open((os.path.join(DATA_DIR, DATA_FILE)), 'rb')
    # # pickledData = pickle.load(pickledFile)
    # # print(f"The pickled data is: {pickledData['data']}\n\n The labels are: {pickledData['labels']}")



if __name__=="__main__":
    main()