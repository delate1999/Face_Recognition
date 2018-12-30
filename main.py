import cv2
import sqlite3
import numpy as np
from database_ops import *

faceDetect = cv2.CascadeClassifier("haarcascade_frontalface_alt2.xml")
cam = cv2.VideoCapture(0)
recognizer = cv2.face.LBPHFaceRecognizer_create()

recognizer.read('recognizer/training_data.yml')


while True:
    ret, img = cam.read()
    img = cv2.resize(img,(640,480))
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = faceDetect.detectMultiScale(gray,scaleFactor=1.5,minNeighbors=5)

    for(x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,0),2)
        id,conf = recognizer.predict(gray[y:y+h+3, x:x+w+3])
        profile = 0
        if conf >= 45 and conf <= 85:
            profile = get_profile(id)
        if profile != 0:
            cv2.putText(img, profile[1], (x, y-40), 2, 0.8, (255, 255, 255), 1)
            cv2.putText(img, 'Age : ' + str(profile[2]), (x, y-20), 2, 0.8, (255, 255, 255), 1)
            cv2.putText(img, 'Gender : ' + str(profile[3]), (x, y), 2, 0.8, (255, 255, 255), 1)
        else:
            cv2.putText(img,'Unknown',(x,y),2,0.8,(255,255,255),1)

    cv2.imshow("Face_Detection_Window",img)
    if cv2.waitKey(1) == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()