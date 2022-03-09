from flask import Flask, render_template,request
import shutil
import os

def recog():
    import os
    import cv2
    import numpy as np
    import tensorflow as tf
    from scipy.spatial import distance
    face_cascade = cv2.CascadeClassifier('./model/haarcascade_frontalface_default.xml')     #人臉檢測
    model = tf.keras.models.load_model('./model/facenet_keras.h5')      #人臉辨識
    shutil.rmtree('./static/selected_images')
    os.makedirs('./static/selected_images')
    def find_faces(imgname):    #找人臉 
        img = cv2.imread(imgname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.14, 3) # Detect faces
        face_crop=[]  
        for f in faces:
            x, y, w, h =f
            sub_face = img[y:y+h, x:x+w]
            face_crop.append(sub_face)
        return face_crop

    data = os.listdir('./goal_picture')     #只會有1張且1人臉
    picture = './goal_picture/'+data[0]
    faces = find_faces(picture)
    for face in faces:
        face = cv2.resize(face,(160,160))
        picture = np.reshape(face,(-1,160,160,3))

    picture = picture.astype('float32')	 	
    mean, std = picture.mean(), picture.std() 	
    picture = (picture - mean) / std
    main_embs = model.predict(picture)

    data = os.listdir('./images')
    test_embs = []
    a = 1
    for i in range(len(data)):
        name = './images/'+data[i]
        faces = find_faces(name)
        for face in faces:
            face = cv2.resize(face,(160,160))
            picture = np.reshape(face,(-1,160,160,3))
            picture = picture.astype('float32')	 	
            mean, std = picture.mean(), picture.std() 	
            picture = (picture - mean) / std
            test_embs = model.predict(picture)
            distanceNum = distance.euclidean(main_embs,test_embs)
            print(data[i],distanceNum)
            if distanceNum < 9:
                img = cv2.imread(name)
                n = './static/selected_images/'+str(a)+'.jpg'
                a += 1
                cv2.imwrite(n,img)
                break


app = Flask(__name__)
 

@app.route("/")
def index():
    return render_template('index.html')

@app.route("/image",methods=['post'])
def image():
    num = 1
    shutil.rmtree('./goal_picture')
    os.makedirs('./goal_picture')
    shutil.rmtree('./images')
    os.makedirs('./images')

    target = request.files.get('target')
    imgs = request.files.getlist('myfiles[]')
    target.save('./goal_picture/target.jpg')
    for im in imgs:
        im.save('./images/img'+str(num)+'.jpg')
        num += 1

    recog()

    recog_dir = os.listdir('./static/selected_images')
    return render_template('image.html',length = len(recog_dir))


if __name__ == '__main__':
    app.run(debug=True, use_reloader=True)