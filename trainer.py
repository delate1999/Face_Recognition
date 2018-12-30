import os
import cv2
from PIL import Image
import numpy as np

recognizer = cv2.face.LBPHFaceRecognizer_create()
path = 'images'

def get_images_and_ids(path):
    imagePaths = [os.path.join(path,f) for f in os.listdir(path)]
    faces = []
    IDs = []
    for imagePath in imagePaths:
        faceImg = Image.open(imagePath).convert('L')
        faceNp = np.array(faceImg,'uint8')
        id = int(os.path.split(imagePath)[-1].split('.')[1])
        faces.append(faceNp)
        IDs.append(id)
        cv2.imshow('trainer',faceNp)
        cv2.waitKey(10)
    return np.array(IDs), faces


IDs, faces = get_images_and_ids(path)
recognizer.train(faces,IDs)
recognizer.save('recognizer/training_data.yml')
cv2.destroyAllWindows()

