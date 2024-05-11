##############################################################################################################################################
# Alexandra Papadaki
# Last update: 11/05/2024
#
# Detects 2D aruco markers on RGB image 
#
#### Before running adjust aruco dictionary, calibration and distortion parameters, aruco parameters
#
##############################################################################################################################################

import cv2
from cv2 import aruco
import cv2.aruco as cvAruco
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib as mpl

# Check opencv version
print(cv2. __version__)

frame = cv2.imread("Data/rgb/0000965.png", -1)

cv2.namedWindow('RGB', cv2.WINDOW_NORMAL)
cv2.imshow('RGB', frame)
key = cv2.waitKey(0)

aruco_dict = cv2.aruco.Dictionary_get(aruco.DICT_6X6_50) # TODO adjust dictionaty used
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
parameters = aruco.DetectorParameters_create()
# TODO define parameters see script for video
parameters.adaptiveThreshConstant = 10
parameters.maxMarkerPerimeterRate = 200
parameters.polygonalApproxAccuracyRate = 0.08
parameters.perspectiveRemovePixelPerCell = 10
parameters.minOtsuStdDev =10.0

# Calibration and distortion matrices # TODO define calibration and distortion parameters accordingly
mtx = np.array([[635.366, 0, 631.149], [0, 633.921, 363.119], [0, 0, 1]])
dist = np.array([-0.0559442974627018,  	0.0631732493638992,  	-0.000752611667849123,  	-0.0012245811522007,  	-0.0193959288299084])

corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters, cameraMatrix=mtx, distCoeff=dist)
frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
frame_markers = aruco.drawDetectedMarkers(frame_rgb.copy(), corners)


# font for displaying text (below)
font = cv2.FONT_HERSHEY_SIMPLEX

if np.all(ids != None):

    # estimate pose of each marker wrt the camera (world points) and return the values
    # rvet and tvec-different from camera coefficients
    rvec, tvec, _ = aruco.estimatePoseSingleMarkers(corners, 0.05, mtx, dist) # 2nd parameter is the marker size in m. tvec is in the same units
    # (rvec-tvec).any() # get rid of that nasty numpy value array error

    # # convert rotation matrix to usual 3x3 mat
    # rotationMatrix = cv2.Rodrigues(rvec)
    # print(rvec)
    # print(rotationMatrix)

    # project the world points of the aruco markers (rvet and tvec) on the image (uses cv.projectPoints)
    for i in range(0, ids.size):
        # draw axis for the aruco markers
        aruco.drawAxis(frame, mtx, dist, rvec[i], tvec[i], 0.1)

    # draw a square around the markers
    aruco.drawDetectedMarkers(frame, corners)

    # code to show ids of the marker found
    strg = ''
    for i in range(0, ids.size):
        strg += str(ids[i][0]) + ', '

    cv2.putText(frame, "Id: " + strg, (0, 64), font, 1, (0, 255, 0), 2, cv2.LINE_AA)


else:
    # code to show 'No Ids' when no markers are found
    cv2.putText(frame, "No Ids", (0, 64), font, 1, (0, 255, 0), 2, cv2.LINE_AA)

#rvec, tvec ,_ = aruco.estimatePoseSingleMarkers(corners[0], 0.02)



plt.figure()
plt.imshow(frame_markers)
for i in range(len(ids)):
    c = corners[i][0]
    plt.plot([c[:, 0].mean()], [c[:, 1].mean()], "o", label = "id={0}".format(ids[i]))
plt.legend()
plt.show()

# display the resulting frame
cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
cv2.imshow('frame', frame)
key = cv2.waitKey(0)
#if cv2.waitKey(1) & 0xFF == ord('q'):
 #   break
