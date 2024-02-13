'''
TRAIN_CLASSIFIER.PY  - VERSION 1
MCS CAPSTONE SPRING 2024
GROUP 6 - COREY NOLAN/CHRIS PATRELLA, YUYUAN LIU
ASL DETECTOR


GENERATE LANDMARK DATA FROM ASL PHOTOS

REQUIRES DOWNLOAD OF MEDIAPIPE MODELS FOR LANDMARK RECOGNITION:
--HANDLANDMARKER (FULL) AT https://developers.google.com/mediapipe/solutions/vision/hand_landmarker/index#models
----CONTAINS TWO MODELS USED FOR FINDING PALMS AND HAND LANDMARKS


IMAGE DATA DIRECTORIES MUST FOLLOW THE CORRECT NAMING CONVENTION
-ALL LETTER IMAGES MUST BE IN A DIRECTORY THAT IS NAMED FOR IT'S LETTER
---- ALL IMAGES FOR THE LETTER A, MUST BE IN A DIRECTORY NAMED 'A'

TAKES IMAGE FILES AS INPUT, PERFORMS HAND LANDMARK DETECTION ON EACH HAND IN THE PHOTO
EXPORTS THE LANDMARK DATA AS X/Y COORDS ALONG WITH A LETTER NAME LABEL TO 'DATA.PICKLE'


RESOURCES:
https://developers.google.com/mediapipe/solutions/vision/hand_landmarker
https://colab.research.google.com/github/googlesamples/mediapipe/blob/main/examples/hand_landmarker/python/hand_landmarker.ipynb#scrollTo=s3E6NFV-00Qt&uniqifier=1
https://www.youtube.com/watch?v=MJCSjXepaAM

'''


import pickle
# IMPORT THE CLASSIFIER TYPE FOR TRAINING THE MODEL
from sklearn.ensemble import RandomForestClassifier
# IMPORT FOR BREAKING DATA UP INTO TRAINING/TESTING SPLIT
from sklearn.model_selection import train_test_split
# IMPORT FOR SHOWING ACCURACY RESULTS
from sklearn.metrics import accuracy_score
import numpy as np


testSize = .2
model = RandomForestClassifier()

data_dict = pickle.load(open('./data.pickle', 'rb'))

# EXTRACT DATA FROM IMPORTED FILE INTO A NUMPY ARRAY (RQUIRED BY SKLEARN MODULE)
data = np.array(data_dict['data'])
# EXTRACT THE LABELS FROM THE IMPORTED FILE INTO A NUMPY ARRAY
labels = np.array(data_dict['labels'])

print(f"\nSplitting data into testing and training with {testSize*100}% reserved for testing.\n ")

# SPLIT THE WHOLE DATA SET INTO TRAINING AND TESTING SETS
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=testSize, shuffle=True, stratify=labels)


print(f"Training the classifier using {model}.\n ")
# TRAIN THE MODEL USING THE TRAINING DATA/LABELS
model.fit(x_train, y_train)

print(f"Testing the trained {model}.\n ")
# TEST THE PREDICTION OF THE TRAINED MODEL, BY SUPPLING DATA WITHOUT LABELS, STORE THE PREDICTION (OF LABEL) IN Y_PREDICT
y_predict = model.predict(x_test)


# GET THE ACCURACY SCORE
score = accuracy_score(y_predict, y_test)
print(f"{score*100}% of samples were classified correctly")

# EXPORT THE MODEL
file = open('mlModel.pickle', 'wb')
pickle.dump({'data':data, 'labels':labels},file)
file.close()

# OPEN THE PICKLE FILE...IF NEEDED.
pickledFile = open('mlModel.pickle', 'rb')
pickledData = pickle.load(pickledFile)
print(f"The pickled data is: {pickledData['data']}\n\n The labels are: {pickledData['labels']}")


