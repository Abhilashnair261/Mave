import cv2
import dlib
import numpy as numpy
import argparse

#using argparse to parse all the arguments from the command line
arguments=argparse.ArgumentParser()
arguments.add_argument('--landmark_predictor',required=True)#loading the pretrained landmark detection model
arguments.add_argument('--webcam',type=int,default=0)#index of the webcam to be used in videostream 
args=vars(arguments.parse_args())

#initialising face detector from dlib module which is based on HOG 
facedetector=dlib.get_frontal_face_detector()
print("Loading face and landmark detectors")
#now loading the landmark detection model to find landmarks and in our case eyes for drowsiness detection
fac_landmarks=dlib.shape_predictor(args["landmark_predictor"])