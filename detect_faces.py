import cv2
import os
import tensorflow as tf

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def find_faces(imgname): 
    img = cv2.imread(imgname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.14, 3) # Detect faces
    face_crop=[]  
    for f in faces:
        x, y, w, h =f
        sub_face = img[y:y+h, x:x+w]
        face_crop.append(sub_face)
    return face_crop

picture = './images/img1.jpg'
print(picture)
faces = find_faces(picture)
for face in faces:
    face = cv2.resize(face,(160,160))
    cv2.imshow('result',face)
    cv2.waitKey(0)
    cv2.destroyAllWindows()