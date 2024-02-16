'''
INFERENCETESTER.PY  - VERSION 1
COREY NOLAN
MCS CAPSTONE SPRING 2024
GROUP 6 - COREY NOLAN/CHRIS PATRELLA, YUYUAN LIU

DESCRIPTION:
-CONTROLLER FUNCTION FOR AUTOMATED TESTING OF TRANSLATION OF ASL TO WRITTEN ENGLISH ALPHABET LETTERS
-DETECTS THE CORRECT ENGLISH LANGUAGE ALPHABET LETTER (EXCLUDING J AND Z) IN AN AMERICAN SIGN LANGUAGE
----FINGER SPELT IMAGE.


REQUIREMENTS:
-PROVIDE THE USER_IMAGE DIRECTORY LOCATION.
----THIS IS WHERE IMAGES TO TEST ARE STORED IN THEIR SUBDIRECTORIES BASED ON THEIR ASL SIGN LETTER
----EACH OF THESE SUBDIRECTORIES SHOULD BE NAMED FOR ITS LETTER.  'A', 'B', 'C', ETC.
----INDIVIDUAL FILE NAMES CAN BE ANYTHING, BUT IMAGES MUST BE IN .JPG FORMAT.
--SET THE DESIRED SAMPLE RATE.
----THIS SETS THE PERCENTAGE OF ALL IMAGES THAT WILL BE RANDOMLY SELECTED FOR TESTING FROM EACH LETTER DIR.


OUTPUTS:
-METRICS OF THE OVERALL SUCCESS/FAILURE RATES ARE PRINTED TO SCREEN AT THE END OF TESTING IMAGES.

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

# GET THE CURRENT WORK DIRECTORY, USE AS BASE PATH
BASE_DIR = os.getcwd()

# SET THE DIRECTORY NAME WHERE THE USER IMAGES WILL BE PULLED FROM
USER_DIR = r'images/Full_Testing_Dataset'
IMAGE_DIR = os.path.join(BASE_DIR, USER_DIR)

def main():

    sampleSizePercentage = 50

    # COUNT OF CORRECT PREDICTIONS
    successPrediction = 0
    failedPrediction = 0
    successLandmarks = 0
    failedLandmarks = 0
    totalNumImages = 0


    # GET THE TOTAL NUMBER OF IMAGES IN THIS DIRECTORY
    dirImgCount = len(os.listdir(IMAGE_DIR))
    # USE DESIRED PERCENTAGE TO CALCULATE CORRECT SAMPLE SIZE
    # sampleSize = int((sampleSizePercentage / 100) * dirImgCount)
    # sampleSize = int((sampleSizePercentage / 100))
    sampleSize = sampleSizePercentage


    # ITERATE THROUGH EACH LETTER SUB-DIRECTORY IN THE IMAGE_DIR DIRECTORY,
    for dir in os.listdir(IMAGE_DIR):

        # GET THE LETTER TO BE PREDICTED, BY TAKING THE NAME OF THE CURRENT DIRECTORY
        dirName = dir
        print(f'Currently working on directory {dirName}...\n')

        totalNumImages += len(os.listdir(os.path.join(IMAGE_DIR,dir)))

        # TAKE A RANDOM SAMPLE OF ALL THE FILES IN THE LETER SUB-DIR BASED ON SAMPLESIZE
        for img_file in random.sample(os.listdir(os.path.join(IMAGE_DIR,dir)), sampleSize):

            userImage = img_file
            print(f"\nImage file: {userImage}")

            # INSTANTIATE THE
            inferClassifier = Inference(os.path.join(IMAGE_DIR,dir,userImage))

            # CALL THE INFERENCE CLASSIFIER
            try:
                inferClassifier.inferenceClassify()

                # RETURNS A JSON OBJECT WITH  {SuccessCode: int_value, InferResult: string_value}
                inferResult = inferClassifier.getResult()
                successCode = (json.loads(inferResult))["SuccessCode"]
                letterResult =(json.loads(inferResult))["InferResult"]

                # IF FAILS TO DETECT LANDMARKS
                if successCode == 1:
                    failedLandmarks += 1
                    print(f"Failed to detect landmarks in user image: {userImage}\n")

                # IF SUCCESSFULLY DETECTS LANDMARKS
                elif successCode == 0:
                    successLandmarks += 1
                    print(f"Successfully detected landmarks in user image: {userImage}\n")
                    print(f"The model predicted an {letterResult}")
                    print(f"dirName is: {dirName}")


                    # USED FOR TESTING: CHECKS THE EXPECTED RESULT AGAINST THE ACTUAL RESULT
                    # COMMENT OUT NEXT THREE LINS FOR PRODUCTION USE
                    if letterResult == dirName:
                        print(f"CORRECT!!\n")
                        successPrediction += 1
                        # input("Press any key to continue...")
                    elif letterResult != dirName:
                        print(f"WRONG!!")
                        failedPrediction += 1
                        # input("Press any key to continue...")

            except ValueError as ve:
                failedLandmarks += 1
                print(f"Value error: Failed to detect landmarks in user image\n")
                print("Press any key to continue...")


    # PRINT RESULTS TO SCREEN
    print(f"\n\nUsing RandomForestClassifer trained model:")
    print(f"Percentage Successful Landmark Detection: {int((successLandmarks/(successLandmarks+failedLandmarks))*100)}%")
    print(f"Percentage Successful Letter Predictions Detection: {int((successPrediction / successLandmarks) * 100)}%\n")
    print(f"Total number of Testing Images Available: {totalNumImages}")
    print(f"{sampleSize}% random sampling.")
    print(f"Total number of Images Processed: {successLandmarks+failedLandmarks}")
    print(f"Total number of Correct predictions: {successPrediction}")
    print(f"Total number of Incorrect predictions: {failedPrediction}")
    print(f"Total number of Successful Landmark detections: {successLandmarks}")
    print(f"Total number of Unsuccessful Landmark detections: {failedLandmarks}")

if __name__ == "__main__":
    main()


