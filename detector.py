

import argparse
import imutils
from imutils.video import VideoStream
import time
import cv2
import numpy as np
ap=argparse.ArgumentParser()
ap.add_argument('-p','--prototxt',required=True,help='path to protoxt file')
ap.add_argument('-m','--model',required=True,help='path to Caffe model file')
ap.add_argument('-c','--confidence',type=float,default=0.5,help='min probility to filter weak detections')

args=vars(ap.parse_args())
net=cv2.dnn.readNetFromCaffe(args['prototxt'],args['model'])
vs=VideoStream(src=0).start()
time.sleep(2)

while True:
    frame=vs.read()

    frame=imutils.resize(frame,width=800)
    (h,w)=frame.shape[:2]
    blob=cv2.dnn.blobFromImage(cv2.resize(frame,(300,300)),1.0,(300,300),(104.0,177.0,123.0))
    # cv2.imwrite(blob, 'sathvik.jpg')

    net.setInput(blob)
    detections=net.forward()
    for i in range(0,detections.shape[2]):
        confidence=detections[0,0,i,2]
        if confidence<args['confidence']:
            continue
        box=detections[0,0,i,3:7] * np.array([w,h,w,h])
        (startX,startY,endX,endY)=box.astype('int')
        text='krupakar attendence has been noted'
        y=startY-10 if startY-10>10 else startY+10
        cv2.rectangle(frame,(startX,startY),(endX,endY),(0,0,255),2)
        cv2.putText(frame,text,(startX,y),cv2.FONT_HERSHEY_SIMPLEX,0.45,(0,0,255),2)

    cv2.imshow('Frame',frame)

    key=cv2.waitKey(1) & 0xFF
    if key==ord('q'):
        break
cv2.destroyAllWindows()
vs.stop()


