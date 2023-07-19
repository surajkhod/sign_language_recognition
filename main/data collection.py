import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands = 1)

offset =20
imgSize =300

folder = "Data/A"
counter = 0

while True:
    success, img =cap.read()
    hands, img = detector.findHands(img)

    if hands:
        hand = hands[0]
        x, y, width, height = hand['bbox']
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8)*255
        
        imgCrop = img[y-offset:y +height+ offset, x-offset:x +width + offset]
        imgCropShape = imgCrop.shape
        # imgWhite[0:imgCropShape[0], 0:imgCropShape[1]] = imgCrop

        aspectRatio = height/width

        if aspectRatio >1:
            # height = imgSize
            wCal = math.ceil((imgSize*width)/height)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            imgResizeShape = imgResize.shape
            wGap = math.ceil((imgSize-wCal)/2)
            imgWhite[0:imgResizeShape[0], wGap:wCal+wGap] =imgResize
        else:
            hCal = math.ceil((imgSize*height)/width)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            imgResizeShape = imgResize.shape
            hGap = math.ceil((imgSize-hCal)/2)
            imgWhite[hGap:hCal+hGap ,0:imgResizeShape[1]] =imgResize
            
        # cv2.imshow("ImageCrop", imgCrop)
        cv2.imshow("ImageWhite", imgWhite)
    else:
        pass
    cv2.imshow("Image", img)
    key = cv2.waitKey(1)
    if key == ord("s"):
        counter += 1
        cv2.imwrite (f'{folder}/Image_{time.time()}.jpg', imgWhite)
        print(counter)
