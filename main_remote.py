import cv2
import sqlite3
import numpy as np
from database_ops import *
import requests

faceDetect = cv2.CascadeClassifier("haarcascade_frontalface_alt2.xml")
recognizer = cv2.face.LBPHFaceRecognizer_create()
url = 'http://192.168.43.1:8080/shot.jpg'

recognizer.read('recognizer/training_data.yml')


while True:
    img_resp = requests.get(url)
    img_array = np.array(bytearray(img_resp.content),dtype = np.uint8)
    img = cv2.imdecode(img_array, -1)
    img = cv2.resize(img,(1280,720))
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = faceDetect.detectMultiScale(gray,scaleFactor=1.5,minNeighbors=5)

    for(x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,0),2)
        id,conf = recognizer.predict(gray[y:y+h, x:x+w])
        profile = 0
        if conf >= 45 and conf <= 85:
            profile = get_profile(id)
        if profile != 0:
            cv2.putText(img, profile[1], (x, y-40), 2, 0.8, (255, 255, 255), 1)
            cv2.putText(img, 'Age : ' + str(profile[2]), (x, y-20), 2, 0.8, (255, 255, 255), 1)
            cv2.putText(img, 'Gender : ' + str(profile[3]), (x, y), 2, 0.8, (255, 255, 255), 1)
        else:
            cv2.putText(img, 'Unknown', (x, y), 2, 0.8, (255, 255, 255), 1)

    cv2.imshow("Face_Detection_Window",img)
    if cv2.waitKey(1) == ord('q'):
        break

cv2.destroyAllWindows()