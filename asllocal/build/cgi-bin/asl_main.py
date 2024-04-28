'''
ASL_MAIN.PY  - PRODUCTION INFERENCE VERSION
COREY NOLAN / YUYUAN LIU
MCS CAPSTONE SPRING 2024
GROUP 6 - COREY NOLAN, CHRIS PATRELLA, YUYUAN LIU

DESCRIPTION:
-CONTROLLER FUNCTION FOR ASL TO WRITTEN ENGLISH ALPHABET LETTERS
-DETECTS THE CORRECT ENGLISH LANGUAGE ALPHABET LETTER/PHASE IN AN AMERICAN SIGN LANGUAGE
----FINGER SPELT IMAGE.

-CALLED FROM WEBSERVER HOSTED UPLOAD.PY FILE.
-RECEIVES A PATH TO A USER UPLOADED IMAGE
-CALLS INFERENCE_CLASSIFIER.PY - PASSES IMAGE DIRECTORY TO NEXT FUNCTION
-RECEIVES SUCCESS CODE AND RESULT FROM INFERENCE_CLASSIFIER.PY
-RETURNS A JSON FORMATED RESULT TO UPLOAD.PY

INPUTS:
-FILEPATH TO UPLOADED IMAGE

OUTPUTS:
-RETURNS A JSON FORMATED STRING WITH THE SUCCESSCODE THE TRANSLATED LETTER.
-STORED IN THE 'inferResult' VARIABLE AND FORMATTED:
----RESULTS OF THE PREDICTION ARE (CURRENTLY) PRINTED TO SCREEN
        {
          SuccessCode: int_value,
          InferResult: string_value
        }
-SUCCESSCODES(0 = Success/ 1 = Failure)


REQUIREMENTS:
-ALL REQUIREMENTS INSTALLED FROM REQUIREMENTS.TXT
-PYTHON VERSION 3.8 TO 3.11
-REPOSITORY CLONED FROM: https://github.com/cpetrella-sketch/ASL-Recognition.git
-UPLOAD.PY/ASL_MAIN.PY/INFERENCE_CLASSIFIER.PY INSIDE THE "CGI-BIN" DIRECTORY
-"MODELS" DIRECTORY AT SAME LEVEL AS CGI-BIN DIRECTORY (NOT INSIDE IT). "MODELS" DIRECTORY HAS "ASLMODEL.JOBLIB" FILE INSIDE
-"HAND_LANDMARKER.TASK" FILE IS LOCATED AT SAME LEVEL AS "CGI-BIN" DIRECTORY (NOT INSIDE IT)
-"TEMP_STORE_IMAGE" DIRECTORY IS AT SAME LEVEL AS "BUILD" AND "MOVE-INSIDE-FILE-INTO-BUILD" DIRECTORIES...JUST INSIDE "ASLLOCAL" DIRECTORY
-"NPM RUN BUILD" FROM INSIDE "BUILD" FOLDER
-COPY FILES FROM 'MOVE-INSIDE-FILE-INTO-BUILD' TO "BUILD" DIRECTORY
-FROM "BUILD" DIRECTORY RUN : python -m http.server --cgi 8990
-ACCESS THE WEB FRONT AT : http://localhost:8990
-UPLOADED FILES MUST BE IN .JPG FORMAT


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

if __name__ == "__main__":
    print(asl_main_launch(directory, imageName))