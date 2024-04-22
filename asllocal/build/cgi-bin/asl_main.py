'''
ASL_MAIN.PY  - VERSION 1
COREY NOLAN
MCS CAPSTONE SPRING 2024
GROUP 6 - COREY NOLAN/CHRIS PATRELLA, YUYUAN LIU

DESCRIPTION:
-CONTROLLER FUNCTION FOR ASL TO WRITTEN ENGLISH ALPHABET LETTERS
-DETECTS THE CORRECT ENGLISH LANGUAGE ALPHABET LETTER (EXCLUDING J AND Z) IN AN AMERICAN SIGN LANGUAGE
----FINGER SPELT IMAGE.

-AT THE COMMAND LINE, ASL_MAIN.PY TAKES AN IMAGE FILE NAME AS INPUT AND CALLS 'InferenceClassify()' from
----'Inference' CLASS TO PERFORM TRANSLATION.  RETURNS A JSON PAYLOAD WITH SUCCESS CODE AND TRANSLATED LETTER.

REQUIREMENTS:
-INFERENCE_CLASSIFIER.PY IN THE SAME DIRECTORY
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
from inference_classifier import *
import json
import argparse
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  
# MODEL_DIR = 'models'
# MODEL_FILE = 'aslModel.p'
# USER_DIR = r'images\train_full\X'

# GET THE CURRENT WORK DIRECTORY, USE AS BASE PATH
BASE_DIR = os.getcwd()

# SET THE DIRECTORY NAME WHERE THE USER IMAGES WILL BE PULLED FROM
USER_DIR = 'user_image_dir'

USER_IMAGE = '1.jpg'

# USER_IMAGE_PATH = os.path.join(BASE_DIR, USER_DIR, USER_IMAGE)
#
# def main(img):
#
#     userImage = img
UPLOAD_DIR = r'S:\Program\GitHub\ASL-Recognition\aslfront\temp_store_image'

def asl_main_launch(dir, name):

    response = {
            'SuccessCode': 5,
            'InferResult': 'None',
        }

    # HACK because we know we are on windows
    img = dir + "\\" + name
    userImage = img
    
    # CALL THE INFERENCE CLASSIFIER
    try:
        # INSTANTIATE THE
        inferClassifier = Inference(userImage)

        inferClassifier.inferenceClassify()

        # RETURNS A JSON OBJECT WITH  {SuccessCode: int_value, InferResult: string_value}
        inferResult = inferClassifier.getResult()

        # IF FAILS TO DETECT LANDMARKS
        if (json.loads(inferResult))["SuccessCode"] == 1:

            response = inferResult

        # IF SUCCESSFULLY DETECTS LANDMARKS
        elif (json.loads(inferResult))["SuccessCode"] == 0:

            response = inferResult

        return response

    except RuntimeError as ve:
        error_message = str(ve)
        # print(f"Runtime error: {error_message}")
        # Build dict formatted as JSON string
        response = {
            'SuccessCode': 2,
            'InferResult': 'None',
            'ErrorMessage': error_message  # Add the error message to the response for debugging
        }
        return response

    ###### CALL TO SEND JSON BACK TO WEBFRONT OR API GATEWAY/LAMBDA IN AWS ######


if __name__=="__main__":
    print(asl_main_launch(UPLOAD_DIR, USER_IMAGE))


