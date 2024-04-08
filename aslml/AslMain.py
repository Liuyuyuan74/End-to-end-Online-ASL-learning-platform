'''
ASLMAIN.PY  - VERSION 1
COREY NOLAN
MCS CAPSTONE SPRING 2024
GROUP 6 - COREY NOLAN/CHRIS PATRELLA, YUYUAN LIU

DESCRIPTION:
-CONTROLLER FUNCTION FOR ASL TO WRITTEN ENGLISH ALPHABET LETTERS
-DETECTS THE CORRECT ENGLISH LANGUAGE ALPHABET LETTER (EXCLUDING J AND Z) IN AN AMERICAN SIGN LANGUAGE
----FINGER SPELT IMAGE.

-AT THE COMMAND LINE, ASLMAIN.PY TAKES AN IMAGE FILE NAME AS INPUT AND CALLS 'InferenceClassify()' from
----'Inference' CLASS TO PERFORM TRANSLATION.  RETURNS A JSON PAYLOAD WITH SUCCESS CODE AND TRANSLATED LETTER.

REQUIREMENTS:
-INFERENCECLASSIFIER.PY IN THE SAME DIRECTORY
-AT THE COMMAND LINE, MUST PROVIDE A VALID IMAGE FILENAME.
-FROM THE CURRENT DIRECTORY WHERE ASL_MAIN.PY IS LOCATED, THE FILE MUST BE IN THE 'user_image_dir' SUB-DIRECTORY
-FILES MUST BE IN .JPG FORMAT

OUTPUTS:
-RETURNS A JSON FORMATED STRING WITH THE SUCCESSCODE THE TRANSLATED LETTER.
-STORED IN THE 'inferResult' VARIABLE AND FORMATTED:
----RESULTS OF THE PREDICTION ARE (CURRENTLY) PRINTED TO SCREEN
        {
          SuccessCode: int_value,
          InferResult: string_value
        }
-SUCCESSCODES(0 = Success/ 1 = Failure)

RESOURCES:
https://developers.google.com/mediapipe/solutions/vision/hand_landmarker
https://colab.research.google.com/github/googlesamples/mediapipe/blob/main/examples/hand_landmarker/python/hand_landmarker.ipynb#scrollTo=s3E6NFV-00Qt&uniqifier=1
https://www.youtube.com/watch?v=MJCSjXepaAM

'''

import os
import sys

import mediapipe as mp
import cv2
import matplotlib.pyplot as plt
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import pickle
import random
import time
from InferenceClassifier import *
import json
import argparse

# MODEL_DIR = 'models'
# MODEL_FILE = 'aslModel.p'
# USER_DIR = r'images\train_full\X'

# GET THE CURRENT WORK DIRECTORY, USE AS BASE PATH
BASE_DIR = os.getcwd()

# SET THE DIRECTORY NAME WHERE THE USER IMAGES WILL BE PULLED FROM
USER_DIR = 'user_image_dir'

#
# USER_IMAGE = 'A1910.jpg'
# USER_IMAGE = 'Ayush_A.jpg'
# USER_IMAGE = '95.jpg'
# USER_IMAGE = 'B0.jpg'
# USER_IMAGE = 'A0.jpg'
# USER_IMAGE = 'Ayush_B.jpg'
# USER_IMAGE = 'X4.jpg'
USER_IMAGE = 'call1.jpg'

# USER_IMAGE_PATH = os.path.join(BASE_DIR, USER_DIR, USER_IMAGE)
#
def main(img):

    userImage = img

# def main():

    # userImage = USER_IMAGE

    # INSTANTIATE THE
    inferClassifier = Inference(userImage)

    # CALL THE INFERENCE CLASSIFIER
    try:
        inferClassifier.inferenceClassify()

        # RETURNS A JSON OBJECT WITH  {SuccessCode: int_value, InferResult: string_value}
        inferResult = inferClassifier.getResult()

        # IF FAILS TO DETECT LANDMARKS
        if (json.loads(inferResult))["SuccessCode"] == 1:

            print("\nFailed to detect landmarks in user image1")
            print(f"\nJSON Payload:\n{inferResult}")

        # IF SUCCESSFULLY DETECTS LANDMARKS
        elif (json.loads(inferResult))["SuccessCode"] == 0:

            print("\nSuccessfully detected landmarks in user image")
            print((json.loads(inferResult))["InferResult"])
            print(f"\nJSON Payload:\n{inferResult}\n")

    # HANDLE IF VALUE ERROR RAISE (x has n features, but RandomForestClassifier was expecting 42 features as input)
    except ValueError as ve:
        print(f"Value error: Failed to detect landmarks in user image2 {ve}")

        # BUILD DICT FORMATTED AS JSON STRING
        pSon = {
            'SuccessCode': 1,
            'InferResult': 'None',
        }

        # CONVERT FROM PYTHON DICT TO JSON OBJECT
        inferResult = json.dumps(pSon)
        print(f"\nJSON Payload:\n{inferResult}\n")

    ###### CALL TO SEND JSON BACK TO WEBFRONT OR API GATEWAY/LAMBDA IN AWS ######


if __name__=="__main__":

    lenArgs = len(sys.argv)

    if lenArgs < 2:
        print("Please provide the image filename as an argument\n EXAMPLE: python asl_main.py file.jpg")
        exit()
    elif lenArgs > 2:
        print("Please provide only one argument: image filename (file.jpg)")
        exit()
    else:
        main(sys.argv[1])

    # main()
