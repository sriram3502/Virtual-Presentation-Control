import os
from cvzone.HandTrackingModule import HandDetector
import cv2
import numpy as np

# Variables (height and width for image
width, height = 1280, 720
folderPath = "ppt"
# camera setup
cap = cv2.VideoCapture(0)
cap.set(3, width)
cap.set(4, height)

# Get the list of presentation images
# if we don't sort we will get like this ['1.png', '10.png', '11.png', '12.png', '2.png', '3.png', '4.png',...,'9.png']
# for use sorted() func and set key as length of list(folderPath)
pathImages = sorted(os.listdir(folderPath), key=len)
print(pathImages)

# variables
imgNumber = 0
hs, ws = int(120 * 1), int(213 * 1)  # to determine size of slide on clubbed webcam image
gestureThreshold = 300
buttonpress = False
buttoncounter = 0
buttondelay = 20
annotations = [[]]  # To Storing all drawings
annotationNumber = -1  # To count number of drawings
annotationStart = False
# Hand Detector
detector = HandDetector(detectionCon=0.8, maxHands=1)

while True:
    # Import images
    success, img = cap.read()
    img = cv2.flip(img, 1)
    pathFullImage = os.path.join(folderPath, pathImages[imgNumber])
    imgCurrent = cv2.imread(pathFullImage)
    hands, img = detector.findHands(img)
    cv2.line(img, (0, gestureThreshold), (width, gestureThreshold), (0, 255, 0), 10)

    # we don't want 2 different images so we will club both webcam and slide into one
    # hand Gestures
    if hands and buttonpress is False:
        hand = hands[0]
        fingers = detector.fingersUp(hand)  # it will check  how many fingers are up
        cx, cy = hand['center']
        # print(fingers)
        lmList = hand['lmList']
        # Constrain Values for easier Drawing
        # indexFinger = lmList[0][0], lmList[0][1]
        # Converting one range to another range to cover entire ppt with limited movement
        xVal = int(np.interp(lmList[0][0], [width // 2, w], [0, width]))
        yVal = int(np.interp(lmList[0][1], [300, height - 50], [0, height]))
        indexFinger = xVal, yVal

        if cy <= gestureThreshold:  # if hand is at height of face
            # annotationStart = False
            # Gesture 1- left
            if fingers == [1, 0, 0, 0, 0]:
                print('left')

                if imgNumber > 0:
                    buttonpress = True
                    annotations = [[]]  # To Storing all drawings
                    annotationNumber = -1  # To count number of drawings
                    annotationStart = False
                    imgNumber -= 1

            # Gesture 2- right
            if fingers == [0, 0, 0, 0, 1]:
                print('right')

                if imgNumber < len(pathImages) - 1:
                    buttonpress = True
                    annotations = [[]]  # To Storing all drawings
                    annotationNumber = -1  # To count number of drawings
                    annotationStart = False
                    imgNumber += 1

        # let's make all these finger gestures active only above head level
        # Gesture 3 - Show Pointer
        if fingers == [0, 1, 1, 0, 0]:
            cv2.circle(imgCurrent, indexFinger, 12, (0, 0, 255), cv2.FILLED)
            # annotationStart = False

        # Gesture 4 - Draw Pointer
        if fingers == [0, 1, 0, 0, 0]:
            if annotationStart is False:
                annotationStart = True
                annotationNumber += 1
                annotations.append([])
            cv2.circle(imgCurrent, indexFinger, 12, (0, 0, 255), cv2.FILLED)
            annotations[annotationNumber].append(indexFinger)
        else:
            annotationStart = False

        # Gesture 5 - Erase
        if fingers == [0, 1, 1, 1, 0]:
            if not annotations:
                buttonpress = True
            elif annotations.pop(-1):
                annotationNumber -= 1
                buttonpress = True

    # Button Pressed Iterations
    if buttonpress:
        buttoncounter += 1
        if buttoncounter > buttondelay:
            buttoncounter = 0
            buttonpress = False

    # to draw lines between signals and differentiation of each drawing
    for i in range(len(annotations)):  # combinations of all drawings
        for j in range(len(annotations[i])):  # individual drawing
            if j != 0:
                cv2.line(imgCurrent, annotations[i][j-1], annotations[i][j], (0, 0, 200), 12)

    # Adding webcam image on slide
    imgSmall = cv2.resize(img, (ws, hs))
    h, w, _ = imgCurrent.shape
    imgCurrent[0:hs, w - ws:w] = imgSmall  # next hand tracking part
    cv2.imshow("Image", img)
    cv2.imshow("Slides", imgCurrent)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
