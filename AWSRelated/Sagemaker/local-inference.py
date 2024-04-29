'''
LOCAL-INFERENCE.PY (FORMERLY ASL_MAIN.PY - VERSION 1)
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
labels = {"A":"A","B":"B","C":"C","D":"D","E":"E","F":"F","G":"G","H":"H","I":"I","K":"K",
          "L":"L","M":"M","N":"N","O":"O","P":"P","Q":"Q","R":"R","S":"S","T":"T","U":"U",
          "V":"V","W":"W","X":"X","Y":"Y"}

successCode = 0 # 0 = Success/ 1 = Failure

predictedLetter = ''

# LOAD THE TRAINED MODEL FROM THE PATH (**FOR JOBLIB FILES**)
aslModelDict = joblib.load(open('./aslModel.joblib', 'rb'))

# LOAD THE TRAINED MODEL FROM THE PATH (**FOR PICKLE FILES**)
# aslModelDict = pickle.load(open(self.MODEL_PATH,'rb'))

aslModel = aslModelDict['model']
# input("press to continue..")

# SET THE OPTIONS FOR THE LANDMARKER INSTANCE WITH THE IMAGE MODE
options = mp.tasks.vision.HandLandmarkerOptions(
    base_options = mp.tasks.BaseOptions('./code/hand_landmarker.task'),
    running_mode = mp.tasks.vision.RunningMode.IMAGE,
    num_hands=2)

# CREATE A HAND LANDMARKER INSTANCE
detector = mp.tasks.vision.HandLandmarker.create_from_options(options)

# READ IN THE IMAGE FROM THE FILE PATH (THIS IS AN IMAGE OF A LETTER 'A')
# userImage = mp.Image.create_from_file('./Local-Data/A0023_test.jpg')
userImage = mp.Image.create_from_file('./Local-Data/4.jpg')

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
#             print('x is', i.x, 'y is', i.y, 'z is', i.z, 'visibility is', i.visibility)
            x = i.x
            y = i.y

            # STORE X AND Y IN THE TEMP ARRAY
            detected.append(x)
            detected.append(y)



    # RUN THE INFERENCE MODEL AGAINST THE LANDMARKS DETECTED
    prediction = aslModel.predict([np.asarray(detected)])

    # OUTPUT THE RESULT BASED ON MATCHES IN THE LABELS DICTIONARY
    predictedLetter = labels[(prediction[0])]
    
    print(f"The predicted letter is : {predictedLetter}")

    # SET THE LOCAL VAR WITH THE RESULT FROM THE INFERENCE MODEL
    successCode = 0
    
    # CREATE JSON RETURN
    pSon = {'SuccessCode': successCode,
           'InferResult': predictedLetter}
    
    jSon = json.dumps(pSon)
    
    print(f"JSON result string: {jSon}")

# IF NO LANDMARKS WERE DETECTED IN THE IMAGE
else:
    print(f"Inside failed inference classifier")
    predictedLetter = 'None'
    successCode = 1
          
    # CREATE JSON RETURN
    pSon = {'SuccessCode': successCode,
           'InferResult': predictedLetter}
    
    jSon = json.dumps(pSon)
    
    print(f"JSON result string: {jSon}")