MCS CAPSTONE SPRING 2024 GROUP 6 - COREY NOLAN/CHRIS PATRELLA, YUYUAN LIU ASL DETECTOR

MACHINE LEARNING TRAINING AND TESTING SCRIPTS/FILES:
CollectImages.PY, CreateDataset.py, TrainClassifier.py, AslMain.py, InferenceTester.py, InferenceClassifier.py, requirements.txt

DATASET MANIPULATION HELPER SCRIPTS:
  DirCombine.py, FileRename.py

REQUIREMENTS:
  -Python version 64bit 3.8 to 3.11 (mediapipe will not work with other versions installed)
  -required packages installed (preferrably in a virtual environment) from requirements.txt
  -[add javascript requirements]

IF STARTING FROM SCRATCH, FIRST DOWNLOAD YOUR DATA, NORMA-IZE/CLEAN IT THEN FOLLOW THIS WORK FLOW:
1a - CollectImages.py - to create your own dataset using your computers web cam. Will take 100 images of you doing ASL gestures.
1b - Download ASL datasets, normalize, clean and put into labeled directories as required by CreateDataset.py.
2 - CreateDataset.py - convert your dataset images into a large array of x/y coordinate Hand Landmarks.
3 - TrainClassifier.py - use the data.pickle file created from CreateDataset.py to train a Random Forest Classifier model. Outputs aslModel.joblib
4.1 - AslMain.py - run from the command line and provide one testing image filepath as the one required argument.  Will attempt to predict the Asl gesture in the image.
    - ...calls InferenceClassifier.py as a class and performs inference on the image. Results are output to screen.
4.2 - InferenceTester.py - runs inferences on many images from a provided directory in a loop.  Used for automated model testing in bulk. Outputs results and performance metrics to screen.
    - ... calls InferenceClassifier.py as a class and performs inference in a loop. 
    
RESOURCES: 
https://developers.google.com/mediapipe/solutions/vision/hand_landmarker 
https://colab.research.google.com/github/googlesamples/mediapipe/blob/main/examples/hand_landmarker/python/hand_landmarker.ipynb#scrollTo=s3E6NFV-00Qt&uniqifier=1 
https://www.youtube.com/watch?v=MJCSjXepaAM


Training Datasets:
-Nagaraj, A, (2018). ASL Alphabet, https://www.kaggle.com/dsv/29550
-sigNN (2021). ASL Sign Language Alphabet Pictures [Minus J, Z]. https://www.kaggle.com/datasets/signnteam/asl-sign-language-pictures-minus-j-z/data



Testing Datasets:
-https://www.kaggle.com/datasets/ayuraj/asl-dataset/data
-https://www.kaggle.com/datasets/danrasband/asl-alphabet-test
-DIY Self Produced datasets
