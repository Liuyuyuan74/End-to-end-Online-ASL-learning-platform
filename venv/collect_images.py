'''
COLLECT IMAGES TO CREATE A DATASET

# THIS CODE IS LARGELY PULLED FROM COMPUTERVISIONENG GITHUB PROJECT
# https://github.com/computervisioneng/sign-language-detector-python.git


-Change the contents of 'alphabet' to match the letters/words you intend to train

-Change line 33 to reference the index of which ever camera attached to your computer you'd like to use
Currently set to 0...which might be the default for the forward camera on most laptops.

cap = cv2.VideoCapture(0)


-Change DATA_DIR to your base directory where you want the photos output.
Fullpath to a photo will be as such:
BaseDir\Letter\number.jpg


DATA IS OUTPUT TO FILES NAMED FOR THE LETTER OF THEIR CAPTURED ASL SIGN
---- ALL IMAGES FOR THE LETTER A, WILL BE IN A DIRECTORY NAMED 'A'
'''

import os
import cv2

# SET THE BASE OUTPUT DIR, CHECK IF EXISTS, CREATE IF NOT THERE
DATA_DIR = './data/DIY_Signs'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# NUMBER OF STILL IMAGES TO CAPTURE
dataset_size = 100

# WHICH SYMBOLS TO COLLECT
# alphabet = ["A","B","C","D","E","F","G","H","I","K","L","M","N","O","P","Q","R","S","T","U","V","X","Y","Z"]
# alphabet = ["A","B","C"]
alphabet = ["A","B"]
# alphabet = ["A"]


# SET TO YOUR CAMERA INDEX. IF YOU ONLY HAVE ONE...USE INDEX 0
cap = cv2.VideoCapture(0)

for j in alphabet:
    print(f'currently at letter{j}\n')
    # CHECK OF DIRECTORY OF THIS LETTER IS CREATED OR NOT, CREATE IF IS NOT YET CREATED
    if not os.path.exists(os.path.join(DATA_DIR, str(j))):
        os.makedirs(os.path.join(DATA_DIR, str(j)))

    print(f'Collecting data for class {j}\n')


    while True:

        # READ A FRAME FROM THE CAPTURED VIDEO, STORE IN TUPLE OF {BOOL, FRAME}
        ret, frame = cap.read()
        # # ADD TEXT TO THE FRAME
        # cv2.putText(frame, f'Press "w" to collect [{j}] / Press "q" to quit', (20, 30), cv2.FONT_HERSHEY_SIMPLEX, .75, (0, 0, 0), 2,
        #             cv2.LINE_AA)
        # FLIP IMAGE HORIZONTAL
        frame = cv2.flip(frame,1)
        # ADD TEXT TO THE FRAME
        cv2.putText(frame, f'Press "w" to collect [{j}] / Press "q" to quit', (20, 30), cv2.FONT_HERSHEY_SIMPLEX, .75, (0, 0, 0), 2,
                    cv2.LINE_AA)
        # SHOW THE FRAME
        cv2.imshow('frame', frame)
        # WAIT UNTIL KEY PRESS...CONTINUE TO SAVING FRAMES
        keyPress= cv2.waitKey(25)
        if keyPress == ord('w'):
            break
        elif keyPress == ord('q'):
            quit()


    counter = 0
    # CONTINUE COLLECTING IMAGES UNTIL AT LIMIT OF DATASET_SIZE
    while counter < dataset_size:
        # READ A FRAME FROM THE VIDEO CAPTURE
        ret, frame = cap.read()
        # FLIP FRAME HORIZONTAL
        frame = cv2.flip(frame, 1)
        # SHOW THE FRAME ON SCREEN
        cv2.imshow('frame', frame)
        # WAIT 25 MILLISECONDS
        cv2.waitKey(25)
        # SAVE THE IMAGE TO THE LETTER'S DIRECTORY, NAME IT BY COUNTER NUMBR
        cv2.imwrite(os.path.join(DATA_DIR, str(j), f'{counter}.jpg'), frame)

        counter += 1

cap.release()
cv2.destroyAllWindows()