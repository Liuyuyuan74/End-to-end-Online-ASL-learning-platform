'''
CREATE_DATASET.PY  - VERSION 2
MCS CAPSTONE SPRING 2024
GROUP 6 - COREY NOLAN/CHRIS PATRELLA, YUYUAN LIU
ASL DETECTOR


GENERATE LANDMARK DATA FROM ASL PHOTOS

REQUIRES DOWNLOAD OF MEDIAPIPE MODELS FOR LANDMARK RECOGNITION:
--HANDLANDMARKER (FULL) AT https://developers.google.com/mediapipe/solutions/vision/hand_landmarker/index#models
----CONTAINS TWO MODELS USED FOR FINDING PALMS AND HAND LANDMARKS

RESOURCES:
https://developers.google.com/mediapipe/solutions/vision/hand_landmarker
https://colab.research.google.com/github/googlesamples/mediapipe/blob/main/examples/hand_landmarker/python/hand_landmarker.ipynb#scrollTo=s3E6NFV-00Qt&uniqifier=1
https://www.youtube.com/watch?v=MJCSjXepaAM

'''
import os
import mediapipe as mp
import cv2
import matplotlib.pyplot as plt
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import pickle


# PATH TO THE MODEL USED FOR DETECTION
model_path =r'C:\Users\corey\PycharmProjects\ASL1\venv\hand_landmarker.task'

# PATH TO THE IMAGE DATA
# DATA_DIR = './data/train'
# # DATA_DIR = './data/Ayush_set/asl_dataset'
DATA_DIR = './data/DIY_Signs'

# SET IMAGE FRONT OVERLAY ATTRIBUTES
MARGIN = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54) # vibrant green


# FUNCTION TO OVERLAY TEXT AND LANDMARKS ON IMAGES
def draw_landmarks_on_image(rgb_image, detection_result):
  hand_landmarks_list = detection_result.hand_landmarks
  handedness_list = detection_result.handedness
  annotated_image = np.copy(rgb_image)

  # LOOP THROUGH EACH DETECTED HAND.
  for idx in range(len(hand_landmarks_list)):
    hand_landmarks = hand_landmarks_list[idx]
    handedness = handedness_list[idx]

    # DRAW LANDMARKS
    hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    hand_landmarks_proto.landmark.extend([
      landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
    ])
    solutions.drawing_utils.draw_landmarks(
      annotated_image,
      hand_landmarks_proto,
      solutions.hands.HAND_CONNECTIONS,
      solutions.drawing_styles.get_default_hand_landmarks_style(),
      solutions.drawing_styles.get_default_hand_connections_style())

    #GET TOP LEFT CORNER OF DETECTED HANDS BOUNDING BOX
    height, width, _ = annotated_image.shape
    x_coordinates = [landmark.x for landmark in hand_landmarks]
    y_coordinates = [landmark.y for landmark in hand_landmarks]
    text_x = int(min(x_coordinates) * width)
    text_y = int(min(y_coordinates) * height) - MARGIN

    # OVERLAY HANDEDNESS OF THE DETECTED HAND.
    cv2.putText(annotated_image, f"{handedness[0].category_name}",
                (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX,
                FONT_SIZE, HANDEDNESS_TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)

  return annotated_image


def main():

    # ARRAY TO HOLD THE X/Y COORDS OF THE LANDMARKS
    data = []
    # LABELS FOR EACH SIGN
    labels = []

    # SET TASK OPTIONS FOR HAND LANDMARKER
    baseOptions = mp.tasks.BaseOptions
    handLandMarker = mp.tasks.vision.HandLandmarker
    handLandMarkerOptions = mp.tasks.vision.HandLandmarkerOptions
    visionRunningMode = mp.tasks.vision.RunningMode


    # SET THE OPTIONS FOR THE LANDMARKER INSTANCE WITH THE IMAGE MODE
    options = handLandMarkerOptions(
        base_options = baseOptions(model_path),
        running_mode=visionRunningMode.IMAGE,
        num_hands=2)

    # CREATE A HAND LANDMARKER INSTANCE
    detector = handLandMarker.create_from_options(options)
    # print(options)

    # ITERATE THROUGH EACH DIRECTORY IN THE DATA_DIR DIRECTORY
    for dir in os.listdir(DATA_DIR):
        print(f'This is the Current directory {dir}\n\n')

        # ITERATE THROUGH EACH IMAGE AND READ IN USING OPENCV
        for img_file in os.listdir(os.path.join(DATA_DIR, dir)):

            # TEMP ARRAY TO HOLD ONE HANDS WORTH OF X/Y HANDMARK COORDS
            temp = []

            # LOAD THE IMAGE FROM THE PATH
            img = mp.Image.create_from_file(os.path.join(DATA_DIR,dir,img_file))

            # DETECT THE LANDMARKS
            detection_result = detector.detect(img)

            # IF HANDS WERE DETECTED
            if detection_result.hand_landmarks:
                # FOR EACH OF THE HANDS DETECTED, ITERATE THROUGH THEM
                for idx in range(len(detection_result.hand_landmarks)):
                    # FOR EACH LANDMARK, GET THE X AND Y COORDINATE
                    for i in detection_result.hand_landmarks[idx]:
                        # print('x is', i.x, 'y is', i.y, 'z is', i.z, 'visibility is', i.visibility)
                        x = i.x
                        y = i.y
                        # print(f'x is {x} and y is {y}')

                        # STORE X AND Y IN THE TEMP ARRAY
                        temp.append(x)
                        temp.append(y)

                # ADD CONTENTS OF TEMP ARRAY TO DATA ARRAY BEFORE TEMP IS OVERWRITTEN BY THE NEXT IMAGE
                # print(temp)
                data.append(temp)

                # ADD THE LABEL TO THE LABEL ARRAY
                labels.append(dir)
                # print(labels)

    # OUTPUT LANDMARK COORDINATES AND LABELS TO A PICKLE FILE.
    print("Landmark Detection Complete...Exporting x/y coords and labels to 'data.pickle'\n\n")
    f = open('data.pickle', 'wb')
    pickle.dump({'data': data, 'labels': labels}, f)
    f.close()


    #         # OVERLAY THE LANDMARKS ON THE IMAGE (CALL DRAWING FUNCTION)
    #         mappedImage = draw_landmarks_on_image(img.numpy_view(), detection_result)
    #
    #         # PLOT THE IMAGE FOR TESTING PURPOSES / SET UP THE FIGURE(S)
    #         plt.figure()
    #         # SET UP EACH IMAGE
    #         plt.imshow(mappedImage)
    #
    # # SHOW THE IMAGE(S)
    # plt.show()


if __name__=="__main__":
    main()