'''
UPLOAD.PY  - PRODUCTION CGI-BIN SCRIPT
COREY NOLAN / YUYUAN LIU
MCS CAPSTONE SPRING 2024
GROUP 6 - COREY NOLAN, CHRIS PATRELLA, YUYUAN LIU

DESCRIPTION:
-CGI-BIN SCRIPT FOR CONNECTING PYTHON SCRIPT/MACHINE LEARNING CODE WITH WEB FRONT JAVASCRIPT CODE
-RECEIVES IMAGE FILE NAME FROM 'INPUTFILEUPLOAD.JS'
-CHECKS IMAGE STORAGE FOR NEW FILE
-CONCATS FILE NAME AND CURRENT WORKING DIRECTORY OF THE WEB SERVER TO GET PATH THE FILE
-CALLS "ASL_MAIN.PY" WITH FILE PATH TO NEWLY UPLOADED FILE
-RECEIVES JSON FORMATED STRING WITH SUCCESS CODE AND RESULT OF INFERENCE CALL (A LETTER)
-RETURNS JSON FORMATTED STRING TO CALLING FUNCTION FOR DISPLAY BY WEB SERVICE.

REQUIREMENTS:
-IF USING PRETRAINED MODEL SUPPLIED BY GROUP 6, BE SURE REQUIREMENTS.TXT IS INSTALLING MEDIApip
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


#!/usr/bin/env python
import cgi
import os
from http import cookiesgit
import json
from asl_main import asl_main_launch

# GET THE CURRENT WORK DIRECTORY, USE AS BASE PATH
BASE_DIR = os.getcwd()

# GET THE REL PATH TO WHERE WEB SERVER IS STARTED (~\ASL-RECOGNITION\asllocal\build). PARENT DIR OF UPLOADED IMAGES ARE STORED ONE LEVEL UP IN THE DIRECTORY
IMAGES_PARENT_DIR = os.path.dirname(BASE_DIR)

# NAME OF THE TEMP UPLOAD DIR
IMAGES_DIR = r'temp_store_image'

# CREATE THE FULL PATH THE TEMP IMAGES DIRECTORY
UPLOAD_DIR = os.path.join(IMAGES_PARENT_DIR, IMAGES_DIR)

def process(fname):

    return asl_main_launch(UPLOAD_DIR, fname)


def main():

    # Create instance of FieldStorage
    form = cgi.FieldStorage()

    # Check if the file was uploaded
    if 'photo' in form:
        fileitem = form['photo']
        if fileitem.filename:
            fname = os.path.basename(fileitem.filename)
            open(os.path.join(UPLOAD_DIR, fname), 'wb').write(fileitem.file.read())

            js = process(fname)
            return js
        else:
            return {"SuccessCode": 3, "InferResult": "None"}
    else:
        return {"SuccessCode": 3, "InferResult": "None"}


if __name__ == "__main__":
    print("Content-Type: application/json\n")
    result = main()
    print(json.dumps(result))

