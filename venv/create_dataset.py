# GENERATE LANDMARK DATA FROM ASL PHOTOS

# RESOURCES:
# https://developers.google.com/mediapipe/solutions/vision/hand_landmarker
# https://www.youtube.com/watch?v=MJCSjXepaAM

import os
import mediapipe as mp
import cv2
import matplotlib.pyplot as plt

# VARS FOR MEDIAPIPE FUNCTIONS
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# MEDIA PIPE MODEL USED FOR HAND DETECTION
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)


# ITERATE THROUGH ALL THE IMAGES, EXTRACT LANDMARKS, SAVE TO FILE FOR CLASSIFIER TRAINING

# DATA_DIR = './data/train'
# DATA_DIR = './data/Ayush_set/asl_dataset'
DATA_DIR = './data/DIY_Signs'



# ITERATE THROUGH EACH DIRECTORY IN THE DATA_DIR DIRECTORY
for dir in os.listdir(DATA_DIR):

    # ITERATE THROUGH EACH IMAGE AND READ IN USING OPENCV
    for img_file in os.listdir(os.path.join(DATA_DIR, dir))[:1]:
        img = cv2.imread(os.path.join(DATA_DIR,dir,img_file))

        # ADD HAND/FINGER LANDMARKS ONTO HAND IN THE IMAGE
        results = hands.process(img)

        if results.multi_hand_landmarks:
            # ITERATE OVER EACH HAND IN THE IMAGE, OVERLAY LANDMARKS ON EACH HAND
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    img, # IMAGE TO DRAW
                    hand_landmarks, # OUTPUT LANDMARKS FROM MODEL
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())



        # PLOT THE IMAGE FOR TESTING PURPOSES
        # SET UP THE FIGURE(S)
        plt.figure()
        # SET UP EACH IMAGE
        plt.imshow(img)

# SHOW THE IMAGE(S)
plt.show()
