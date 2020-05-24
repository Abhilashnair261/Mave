import cv2
import dlib
import numpy as numpy
import imutils.video as vidstream
import time
import argparse
from imutils import utils

#using argparse to parse all the arguments from the command line

arguments=argparse.ArgumentParser()
arguments.add_argument('--landmark_predictor',required=True)#loading the pretrained landmark detection model
arguments.add_argument('--cam',type=int,default=0)#index of the webcam to be used in videostream 
args=vars(arguments.parse_args())

#initialising face detector from dlib module which is based on HOG 

facedetector=dlib.get_frontal_face_detector()
print("Loading face and landmark detectors")

#now loading the landmark detection model to find landmarks and in our case eyes for drowsiness detection

fac_landmarks=dlib.shape_predictor(args["landmark_predictor"])

#landmark predictor marks all landmarks and outline as an index list for our pupose we need just the eyes

(left_start,left_end)=utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(right_start,right_end)=utils.FACIAL_LANDMARKS_IDXS["left_eye"]
print("Initializing videostream")
vid=vidstream(src=args["cam"]).start()
time.sleep(1.0)

#now we can loop over all the frames and apply our drowsiness function based on the eye aspect ratio

while (1):
    frame=vid.read()#reading the frame
    frame-utils.resize(frame,width=400)
    gray_scale=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)#converting into grayscale for detection purpose
    

