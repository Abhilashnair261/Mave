import cv2
from scipy.spatial import distance
import dlib
import numpy as numpy
import imutils
from imutils.video import VideoStream as vidstream
import time
import argparse
from imutils import face_utils as utils

def draw(lefteye,righteye):
    #calculating the curve for drawing the outline
    left_eye_hull = cv2.convexHull(lefteye)
    right_eye_hull = cv2.convexHull(righteye)
    cv2.drawContours(frame, [left_eye_hull],-1,(0,255,0),1)#using gree color to outline
    cv2.drawContours(frame, [right_eye_hull],-1,(0,255,0),1)

def ear(eye):#function to calculate the eye aspect ratio
    #we are gonna use the euclidiean distance formula for calculating the distance between the horizontal and vertical coordinates
    #and then use them for calculating the aspect ratio
    #vetical 
    v1=distance.euclidean(eye[1],eye[5])
    v2=distance.euclidean(eye[2],eye[4])
    #horizontal
    h=distance.euclidean(eye[0],eye[3])
    #eye aspect ratio calculation
    eye_aspect_ratio = (v1+v2)/(2*h)
    return eye_aspect_ratio
    
frame_count=0
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

#landmark predictor marks all landmarks and outline as an index list for our pupose we need just the eyes

(left_start,left_end)=utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(right_start,right_end)=utils.FACIAL_LANDMARKS_IDXS["right_eye"]
print("Initializing videostream")
vid=vidstream(src=args["webcam"]).start()
time.sleep(1.0)

#now we can loop over all the frames and apply our drowsiness function based on the eye aspect ratio
while (1):
    frame=vid.read()#reading the frame
    frame=imutils.resize(frame,width=450)
    gray_frame=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)#converting into grayscale for detection purpose

    faces=facedetector(gray_frame,0)
    #now we can iterate over all the faces detected and draw bounding outline of their eye
    # and then calculate the eye aspect ratio for the nearer one as nearer one would be the driver
    for face in faces:
        landmarks=fac_landmarks(gray_frame,face)
        landmarks=utils.shape_to_np(landmarks)#convering into np aray to carve out the eye parts from the landmarks
        lefteye=landmarks[left_start:left_end]
        righteye=landmarks[right_start:right_end]
        
        #now drawing the outline using the above index np array
        draw(lefteye,righteye)
        #now we go for finding the eye aspect ration for determining drowsiness
        left_ratio = ear(lefteye)
        right_ratio = ear(righteye)
        #we can take an average and then use a threshold for drowsiness alert
        avg_ratio=(left_ratio+right_ratio)/2

        if(avg_ratio<0.3):
            frame_count=frame_count+1
            if frame_count>40:
                cv2.putText(frame, "DROWSINESS ALERT!", (10, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            frame_count=0
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF#to wait for key press so that user can exit
 
	# press q to exist 
    if key == ord("q"):
        break

#cleaning at exit condition
cv2.destroyAllWindows()
vid.stop()
