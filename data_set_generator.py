import cv2
import sqlite3
import numpy as np
from database_ops import *

faceDetect = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
cam = cv2.VideoCapture(0)



id = raw_input('Enter id : \n')
name = raw_input('Enter your name : \n')
age = raw_input('Enter your age : \n')
gender = raw_input('Enter your gender(M/F) : \n')
insert(id,name,age,gender)
sample = 0

while True:
    ret, img = cam.read()
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = faceDetect.detectMultiScale(gray,scaleFactor=1.5,minNeighbors=5)

    for(x,y,w,h) in faces:
        sample += 1
        cv2.imwrite('images/User.' + str(id) + '.' + str(sample) + '.jpg',gray[y:y+h, x:x+w])
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,0),2)
        cv2.waitKey(100)

    cv2.imshow("Face_Detection_Window",img)
    cv2.waitKey(1)
    if sample > 55:
        break

cam.release()
cv2.destroyAllWindows()