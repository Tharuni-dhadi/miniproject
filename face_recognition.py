# import  tensorflow as tf
# from tensorflow import  keras
# import  numpy as np
# import  cv2
# from keras.models import  load_model
#
# face_detect=cv2.CascadeClassifier('cas.xml')
# cap=cv2.VideoCapture(0)
# cap.set(3,640)
# cap.set(4,480)
# font=cv2.FONT_HERSHEY_COMPLEX
# model=load_model('keras_model.h5')
# def get_className(classNo):
#     if classNo==0:
#         return 'sathvik'
#     elif classNo==1:
#         return 'Class 2'
#
# while True:
#     sucess,imgOriginal=cap.read()
#     faces=face_detect.detectMutliScale(imgOriginal,1.3,5)
#     for x,y,w,h in faces:
#         crop_img=imgOriginal[y:y+h,x:x+h]
#         img=img.reshape(1,224,224,3)
#         prediction=model.predict(img)
#         classIndex=model.predict_classes(img)
#         probabilityValue=np.amax(prediction)
#         if classIndex==0:
#             cv2.rectangle(imgOriginal,(x,y),(x+w,y+h),(0,255,0),2)
#             cv2.rectangle(imgOriginal,(x,y-40),(x+w,y),(0,255,0),-2)
#             cv2.putText(imgOriginal,str(get_className(classIndex)),(x,y-10),font,0.75)
#         elif classIndex==1:
#             cv2.rectangle(imgOriginal, (x, y), (x + w, y + h), (0, 255, 0), 2)
#             cv2.rectangle(imgOriginal, (x, y - 40), (x + w, y), (0, 255, 0), -2)
#             cv2.putText(imgOriginal, str(get_className(classIndex)), (x, y - 10), font, 0.75)
#         cv2.putText(imgOriginal,str(round(probabilityValue*100,2))+'%',(180,75),font,0,75)
#     cv2.imshow('Result',imgOriginal)
#     k=cv2.waitKey(1)
# cv2.destroyAllWindows()
#
import time

import pandas
from PIL import Image
from keras.applications.vgg16 import preprocess_input
import base64
from io import BytesIO
import json
import random
import  time
import  pandas
import cv2
from keras.models import load_model
import numpy as np

from keras.preprocessing import image
df = pandas.DataFrame(columns=['name', 'time'])
model = load_model('keras_model.h5')

# Loading the cascades
face_cascade = cv2.CascadeClassifier('cas.xml')


def face_extractor(img):
    # Function detects faces and returns the cropped face
    # If no face detected, it returns the input image

    # gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(img, 1.3, 5)

    if faces is ():
        return None

    # Crop all faces found
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 255), 2)
        cropped_face = img[y:y + h, x:x + w]

    return cropped_face


# Doing some Face Recognition with the webcam
video_capture = cv2.VideoCapture(0)
while True:
    _, frame = video_capture.read()
    # canvas = detect(gray, frame)
    # image, face =face_detector(frame)

    face = face_extractor(frame)
    if type(face) is np.ndarray:
        face = cv2.resize(face, (224, 224))
        im = Image.fromarray(face, 'RGB')
        # Resizing into 128x128 because we trained the model with this image size.
        img_array = np.array(im)
        # Our keras model used a 4D tensor, (images x height x width x channel)
        # So changing dimension 128x128x3 into 1x128x128x3
        img_array = np.expand_dims(img_array, axis=0)
        pred = model.predict(img_array)
        print(pred)

        name = "None matching"
        print(pred[0][1])
        print(pred[0])
        if (pred[0][0] > 0.5):
            name = 'krupakar '+'attendence has been registered'
            new_row = {'name': name[:8], 'time': time.localtime()}
            df.loc[len(df)] = new_row



        cv2.putText(frame, name, (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

        df.to_csv('attendence.csv')
    else:
        cv2.putText(frame, "No face found", (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Video', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):

        break
video_capture.release()
cv2.destroyAllWindows()
df=df.drop_duplicates(subset=['name'])
df.to_csv('final.csv')
