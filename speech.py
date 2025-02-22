import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import time
import pyttsx3
import asyncio
import gtts
import playsound
import keyboard


labels = ["A","B", "C"]

async def predict(imgWhite):
    classifier = Classifier("Train Model/keras_model.h5", "Train Model/labels.txt")
    prediction, index = classifier.getPrediction(imgWhite)
    print(prediction, index)
    return index

# labels = ["A","B", "C"]

async def func1():
    cap = cv2.VideoCapture(0)
    detector = HandDetector(maxHands=1)
    classifier = Classifier("Train Model/keras_model.h5", "Train Model/labels.txt")
    # text_speech = pyttsx3.init()

    offset =20
    imgSize =300
    # f = open('labels.txt', 'r')
    # labels = ["A","B", "C"]
    # labels =[]
    # for line in  f:
    #     line_stip = line.stip()
    #     line_split = line.strip.split()
    #     labels.append(line_split)

    # f.close()    


    while True:
        success, img =cap.read()
        imgOutput = img.copy()
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
                prediction, index = classifier.getPrediction(imgWhite)
                print(prediction, index)        


            else:
                hCal = math.ceil((imgSize*height)/width)
                imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                imgResizeShape = imgResize.shape
                hGap = math.ceil((imgSize-hCal)/2)
                imgWhite[hGap:hCal+hGap ,0:imgResizeShape[1]] =imgResize
                prediction, index = classifier.getPrediction(imgWhite)
                print(prediction, index)

            cv2.putText(imgOutput, labels[index], (x, y - 26), cv2.FONT_HERSHEY_COMPLEX, 1.7, (255, 0, 255), 2)
            cv2.rectangle(imgOutput, (x-offset, y-offset), (x+ width +offset, y+ height +offset), (255, 0, 255), 4)

            # speech = labels[index]
            # text_speech.say(speech)
            # text_speech.runAndWait()
            # time.sleep(1)

            # cv2.imshow("ImageCrop", imgCrop)
            cv2.imshow("ImageWhite", imgWhite)
        
        cv2.imshow("Image", imgOutput)
        cv2.waitKey(1)
        
    return imgWhite    

async def func2(index):
    if cv2.waitKey(1)== ord("s") :
        # speech = labels[index]
        # sound = gtts.gTTS(speech, lang ="en")
        # sound.save("gtts.mp3")
        # playsound.playsound("gtts.mp3")

        text_speech = pyttsx3.init()
        speech = labels[index]
        text_speech.say(speech)
        text_speech.runAndWait()
        # time.sleep(1)

async def main():
    while True:
        task1 = asyncio.create_task(func1())    
        task2 = asyncio.create_task(predict())
        task3 = asyncio.create_task(func2())
        await asyncio.gather(task1, task2, task3)
        continue
        await asyncio.wait([task1, task2, task3])
        
asyncio.run(main())