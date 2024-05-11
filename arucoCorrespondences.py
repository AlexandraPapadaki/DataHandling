##############################################################################################################################################
# Alexandra Papadaki
# Last update: 11/05/2024
#
# Detects aruco markers on RGB images and calculates their 3D-2D correspondences for a planar aruco board (known 3D coordinates)
# Outputs txt with the 3D-2D correspondences 
#
#### Before running adjust aruco dictionary, calibration and distortion parameters, aruco size (mm), aruco parameters
#
##############################################################################################################################################

import cv2
from cv2 import aruco
import cv2.aruco as cvAruco
import argparse
import glob
import os
import math
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
from numpy.random import permutation
import time

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--imagesFolderPath", type=str, required=True,
	help="path to input RGB images folder"),
ap.add_argument("-f", "--filePath", type=str, required=False,
	default="test.csv", help="3D aruco coords csv file")
ap.add_argument("-o", "--outputFolder", type=str, required=True,
	help="path to output 32D.txt folder"),
args = vars(ap.parse_args())

## For 3D aruco coordinates
# Load the absolute 3D coordinates of the aruco markers on the board. Origin is the UL corner of the aruco in 3D space. Otherwise hardcode.
# File is structured like: id, x, y, z=0 (mm).
# z=0 because we have flat board.
# x ,y are manually measured with a scale on the printed aruco board.
markerULCorners = np.loadtxt(args["filePath"], delimiter=',', skiprows=0)

# Calculate the 3D Coordinates of the remaining 3 corners of each aruco w.r.t. the UL corner - clockwise" in 3D space (mm) (UL, UR, BR, BL)
# TODO adjust aruco size (mm)
markerCornersRel=[]
markerCornersRel.append(np.array([0.0, 0.0, 0.0])) #UL
markerCornersRel.append(np.array([40.0, 0.0, 0.0])) #UR
markerCornersRel.append(np.array([40.0, -40.0, 0.0])) #BR
markerCornersRel.append(np.array([0.0, -40.0, 0.0])) #BL
#print(markerCornersRel[1])

# Number of aruco
nTargets = int(markerULCorners.size/4)
#print("Number of aruco: ", nTargets)

# Calculate absolute 3D coordinates for all 4 corners of all aruco
marker3DIds=[]
markerCorners3DModel=[]
for i in range(0, nTargets):
    tULCorner=np.array([markerULCorners[i][1], markerULCorners[i][2], markerULCorners[i][3]])
    #print("tULCorner: ", tULCorner)

    tCorner=[]
    tCorner.append(tULCorner + markerCornersRel[0])
    tCorner.append(tULCorner + markerCornersRel[1])
    tCorner.append(tULCorner + markerCornersRel[2])
    tCorner.append(tULCorner + markerCornersRel[3])
    #print("tCorner: ", tCorner)

    marker3DIds.append(int(markerULCorners[i][0]))
    markerCorners3DModel.append(tCorner)

#for i in range(0, nTargets):
    #print("Marker ", marker3DIds[i], " is in: ", markerCorners3DModel[i])

# Detect markers and calculate 2D image coordinates
# Define used dictionary
aruco_dict = cv2.aruco.Dictionary_get(aruco.DICT_4X4_250) # TODO adjust dictionary used

## For 2D image coordinates
parameters = aruco.DetectorParameters_create()
# TODO define parameters if it doesnt see the markers
# These parameters determine the minimum and maximum size of a marker, specifically the minimum and maximum marker perimeter.
# They are not specified in absolute pixel values, instead they are specified relative to the maximum dimension of the input image.
parameters.adaptiveThreshConstant = 10
parameters.maxMarkerPerimeterRate = 200
parameters.minMarkerPerimeterRate = 0.025
parameters.polygonalApproxAccuracyRate = 0.03
parameters.perspectiveRemovePixelPerCell = 10 #5
parameters.minOtsuStdDev =10.0
parameters.adaptiveThreshWinSizeStep  = 3

# Calibration and distortion matrices # TODO define calibration and distortion parameters accordingly
mtx = np.array([[635.366, 0, 631.149], [0, 633.921, 363.119], [0, 0, 1]])
dist = np.array([-0.0559442974627018,  	0.0631732493638992,  	-0.000752611667849123,  	-0.0012245811522007,  	-0.0193959288299084])

images = glob.glob(os.path.join(args["imagesFolderPath"], './*.png'))
images.sort()
#print("Images found: ", len(images))

for i in range(0, len(images)):
    frame = cv2.imread(images[i], -1)
    # image = "/home/lele/PycharmProjects/realsense/Data/20220323_121728.jpg"  # TODO
    # frame = cv2.imread(image, -1) # TODO

    imageName0 = (images[i].rsplit('/'))[-1] # cut off path
    imageName = (imageName0.split('.'))[0] # cut off format

    # cv2.namedWindow('RGB', cv2.WINDOW_NORMAL)
    # cv2.imshow('RGB', frame)
    # key = cv2.waitKey(0)

    # Detect markers on gray scale image
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners0, ids0, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters, cameraMatrix=mtx, distCoeff=dist)

    # Do the following if you dont use all 3D aruco coordinates. Check whether 3D coord of each detected aruco is used. If yes then keep it, if not throw the detected marker
    # If you use 3D coords of all aruco then dont use. Instead rename the lists above corners0, ids0 to corner, ids.
    ids=[]
    corners=[]
    k=0
    for i in range(0, ids0.size):
        #print(ids0[i])
        if ids0[i] in marker3DIds:
            corners.append(corners0[i])
            ids.append(ids0[i])
            k+=1
    #print("ids0.size=", ids0.size)
    #print("len(ids)=", len(ids))

    # shuffle to use random aruco
    #corners, ids = shuffle(corners, ids, random_state=0)

    #print("markers found: ", len(corners))
    # Show detected markers on image
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_markers = aruco.drawDetectedMarkers(frame_rgb.copy(), corners)
    font = cv2.FONT_HERSHEY_SIMPLEX # font for displaying text

    # change nArucoEncountered to a specific number if i want to take into account a certain amount of detected 2D aruco. Use shuffle (see above) before to select random aruco
    nArucoEncountered=len(ids)
    #print("nArucoEncountered= ", nArucoEncountered)
    if np.all(ids != None):
        # # estimate pose of each marker wrt the camera (world points) and return the values
        # # rvet and tvec-different from camera coefficients
        # rvec, tvec, _ = aruco.estimatePoseSingleMarkers(corners, 0.05, mtx, dist) # 2nd parameter is the marker size in m. tvec is in the same units
        # (rvec-tvec).any() # get rid of that nasty numpy value array error
        #
        # # convert rotation matrix to usual 3x3 mat
        # rotationMatrix = cv2.Rodrigues(rvec)
        # print(rvec)
        # print(rotationMatrix)
        #
        # # project the world points of the aruco markers (rvet and tvec) on the image (uses cv.projectPoints)
        # for i in range(0, ids.size):
        #     # draw axis for the aruco markers
        #     aruco.drawAxis(frame, mtx, dist, rvec[i], tvec[i], 0.1)

        # draw a square around the markers
        aruco.drawDetectedMarkers(frame, corners)

        # code to show ids of the marker found
        strg = ''
        for i in range(0, nArucoEncountered):
            if ids[i] in marker3DIds:
                strg += str(ids[i][0]) + ', '

            cv2.putText(frame, "Id: " + strg, (0, 64), font, 1, (0, 255, 0), 2, cv2.LINE_AA)


    else:
        # code to show 'No Ids' when no markers are found
        cv2.putText(frame, "No Ids", (0, 64), font, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # rvec, tvec ,_ = aruco.estimatePoseSingleMarkers(corners[0], 0.02)

    #Create output folder if doesnt exist
    outputFolder = args["outputFolder"]
    if not os.path.exists(outputFolder):
        # Create a new directory because it does not exist
        os.makedirs(outputFolder)

    encounteredArucoInfoInName = ""
    if nArucoEncountered<len(ids):
        encounteredArucoInfoInName = "_" + str(nArucoEncountered) + "aruco"


    # TODO uncomment to show
    fig = plt.figure()
    plt.imshow(frame_markers)
    for i in range(0, nArucoEncountered):
        c = corners[i][0]
        plt.plot([c[:, 0].mean()], [c[:, 1].mean()], "o", label="id={0}".format(ids[i]))
    plt.legend(loc=2, prop={'size': 5}, ncol=3)
    # first save then plot otherwise saves blank
    plt.savefig(outputFolder+ "/_" + imageName + encounteredArucoInfoInName + '.png')
    plt.show(block=False) # add block=False to enable plt.close() so that the window closes automatically. Delete it to be able to see figures while running
    #time.sleep(5)
    plt.close()


    # display the resulting frame
    # TODO uncomment to show
    cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
    cv2.imshow('frame', frame)
    #key = cv2.waitKey(0)
    cv2.imwrite(outputFolder+ "/" + imageName + encounteredArucoInfoInName + '.png', frame)
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #   break

    #print("ids: ", ids)
    #print("ids3D: ", marker3DIds)

    # Define correspondences and save resulting correspondences into txt
    output=[]
    for i in range(0, nArucoEncountered):
        #print(ids[i])
        if ids[i] in marker3DIds:
            for j in range(4):
                corrs3D2D = np.concatenate((markerCorners3DModel[marker3DIds.index(int(ids[i]))][j], corners[i][0][j]))
                #print("Correspondences for marker ", ids[i], ": ", corrs3D2D)
                output.append(corrs3D2D)

    outputFile = outputFolder + "/" + imageName + encounteredArucoInfoInName + '.txt'
    np.savetxt(outputFile, output, delimiter=' ', fmt='%d')

