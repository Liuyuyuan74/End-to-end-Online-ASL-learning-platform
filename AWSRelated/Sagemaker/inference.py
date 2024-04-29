'''
import os
import joblib

def predict_fn(input_object, model):
    ###########################################
    # Do your custom preprocessing logic here #
    ###########################################

    print("calling model")
    predictions = model.predict(input_object)
    return predictions


def model_fn(model_dir):
    print("loading model.joblib from: {}".format(model_dir))
    loaded_model = joblib.load(os.path.join(model_dir, "model.joblib"))
    return loaded_model


RESOURCES:
https://developers.google.com/mediapipe/solutions/vision/hand_landmarker
https://colab.research.google.com/github/googlesamples/mediapipe/blob/main/examples/hand_landmarker/python/hand_landmarker.ipynb#scrollTo=s3E6NFV-00Qt&uniqifier=1
https://www.youtube.com/watch?v=MJCSjXepaAM
https://towardsdatascience.com/deploying-a-pre-trained-sklearn-model-on-amazon-sagemaker-826a2b5ac0b6

'''
import os
import joblib
import mediapipe as mp
import numpy as np
# from inference_classifier import *
import json
import sys




# DESERIALIZE AND LOAD THE JOBLIB FILE
def model_fn(model_dir):
    print("Inside the model_fn()\n\nloading model.joblib from: {}".format(model_dir))
    model = joblib.load(os.path.join(model_dir, "model.joblib"))
    return model












        
        

# PASS IN THE DATA THE MODEL WILL EXPECT OF INPUT
def input_fn(request_body, request_content_type):
    print(f"Inside the input_fn()\n\nCurrent request_body:{request_body}\nCurrent content type: {request_content_type}")
          
#     if request_content_type == 'image/jpg':
#         request_body = json.loads(request_body)
#         imageVar = request_body['Input']        
# #         userImage = mp.Image.create_from_file(imageVar)
#         image_array = np.frombuffer(imageVar, dtype=np.uint8)
#         imageDecode = cv2.imdecode(image_array, cv2.IMREAD_COLOR)        
#         return imageDecode
# #         return imageVar
#     else:
#         raise ValueError("This model only supports image/jpg input")














# # THE MODEL PREDICTION FUNCTION
# def predict_fn(input_data, model):   
#     print("calling model")   
    
#     print(f"Inside the predict_fn()\n\nCurrent input data:{input_data}\nCurrent aslModel {model}")
    
#     # LIKELY NEED TO CONVERT INPUT DATA BACK TO AN IMAGE BEFORE CONTINUING...
        

#     labels = {"A":"A","B":"B","C":"C","D":"D","E":"E","F":"F","G":"G","H":"H","I":"I","K":"K",
#               "L":"L","M":"M","N":"N","O":"O","P":"P","Q":"Q","R":"R","S":"S","T":"T","U":"U",
#               "V":"V","W":"W","X":"X","Y":"Y"}

#     successCode = 0 # 0 = Success/ 1 = Failure

#     predictedLetter = ''

#     # LOAD THE TRAINED MODEL FROM THE PATH (**FOR JOBLIB FILES**)
#     aslModelDict = joblib.load(open(model, 'rb'))

#     # LOAD THE TRAINED MODEL FROM THE PATH (**FOR PICKLE FILES**)
#     # aslModelDict = pickle.load(open(self.MODEL_PATH,'rb'))

#     aslModel = aslModelDict['model']
#     # input("press to continue..")  
       

#     # SET THE OPTIONS FOR THE LANDMARKER INSTANCE WITH THE IMAGE MODE
#     options = mp.tasks.vision.HandLandmarkerOptions(
#         base_options = mp.tasks.BaseOptions('./code/hand_landmarker.task'),
#         running_mode = mp.tasks.vision.RunningMode.IMAGE,
#         num_hands=2)

#     # CREATE A HAND LANDMARKER INSTANCE
#     detector = mp.tasks.vision.HandLandmarker.create_from_options(options)
#     print("created handlandmark detection instance\n")


#     # READ IN THE IMAGE FROM THE FILE PATH (THIS IS AN IMAGE OF A LETTER 'A')
#     # userImage = mp.Image.create_from_file('./Local-Data/A0023_test.jpg')
#     userImage = mp.Image.create_from_file(input_data)

#     # DETECT THE LANDMARKS
#     detection_result = detector.detect(userImage)
#     # print(f'detection result: {detection_result.hand_landmarks}')

#     # IF HANDS WERE DETECTED
#     if detection_result.hand_landmarks:
#         # success += 1
#         detected = []
#         # print("inside the detecttion result loop\n")
#         # FOR EACH OF THE HANDS DETECTED, ITERATE THROUGH THEM
#         # for idx in range(len(detection_result.hand_landmarks))[:1]:
#         for idx in range(len(detection_result.hand_landmarks)):
#             # FOR EACH LANDMARK, GET THE X AND Y COORDINATE
#             for i in detection_result.hand_landmarks[idx]:
#     #             print('x is', i.x, 'y is', i.y, 'z is', i.z, 'visibility is', i.visibility)
#                 x = i.x
#                 y = i.y

#                 # STORE X AND Y IN THE TEMP ARRAY
#                 detected.append(x)
#                 detected.append(y)



#         # RUN THE INFERENCE MODEL AGAINST THE LANDMARKS DETECTED
#         prediction = aslModel.predict([np.asarray(detected)])
        
#         # OUTPUT THE LOCAL VAR WITH THE RESULT FROM THE INFERENCE MODEL
#         predictedLetter = labels[(prediction[0])]
        
# #         # UPDATE SUCCESS CODE
# #         successCode = 0
        
# #       # CREATE JSON RETURN
# #         pSon = {'SuccessCode': successCode,
# #            'InferResult': predictedLetter}
    
# #         jSon = json.dumps(pSon)
    
# #         print(f"JSON result string: {jSon}")

#         # RETURN THE PREDICTION
#         return predictedLetter
        
        
    
#     # IF NO LANDMARKS WERE DETECTED
#     else:
#         print("No landmarks detected")
#         predictedLetter = 'None'
# #         successCode = 1
        
# #         # CREATE JSON RETURN
# #         pSon = {'SuccessCode': successCode,
# #            'InferResult': predictedLetter}
    
# #         jSon = json.dumps(pSon)
    
# #         print(f"JSON result string: {jSon}")
        
#         # RETURN THE PREDICTION
#         return predictedLetter
    
    
    
    
    
    
    
    
    
    
    

    
    
# # PROCESSES RETURNED VALUE FROM THE PREDICT_FN AND THE TYPE OF RESPONSE THE ENDPOINT WILL GET
# def output_fn(prediction, content_type):
#     print(f"Inside the output_fn()\n\nCurrent prediction:{prediction}\nCurrent content type: {content_type}")

#     # IF NO LANDMARKS WERE DETECTED IN THE IMAGE
#     if prediction == 'None':

#         predictedLetter = 'None'
#         successCode = 1

#         # CREATE JSON RETURN
#         pSon = {'SuccessCode': successCode,
#                'InferResult': predictedLetter}

#         jSon = json.dumps(pSon)

#         return jSon


#     else:

#         # OUTPUT THE RESULT BASED ON MATCHES IN THE LABELS DICTIONARY
#         predictedLetter = labels[(prediction[0])]

#         # SET THE LOCAL VAR WITH THE RESULT FROM THE INFERENCE MODEL
#         successCode = 0

#         # CREATE JSON RETURN
#         pSon = {'SuccessCode': successCode,
#                'InferResult': predictedLetter}

#         jSon = json.dumps(pSon)

#         return jSon
