# import numpy as np
# import cv2
# from matplotlib import pyplot as plt


# mainImg = cv2.imread("photo1.jpg", 0)

# for file in onlyfiles:
#     currentImg = cv2.imread(file, 0)

#     orb = cv2.ORB_create()

#     kp1, des1 = orb.detectAndCompute(mainImg, None)
#     kp2, des2 = orb.detectAndCompute(currentImg, None)

#     bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

#     matches = bf.match(des1, des2)

#     matchesImg = cv2.drawMatches(
#         mainImg, kp1, currentImg, kp2, matches, None, flags=2)

#     plt.imshow(matchesImg), plt.show()

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import xlsxwriter
import time
from os import listdir
from os.path import isfile, join
onlyfiles = [f for f in listdir("assets") if isfile(join("assets", f))]

mainImg = cv.imread("mainphoto.jpg", 0)
orb = cv.ORB_create()
bf = cv.BFMatcher()
kp1, des1 = orb.detectAndCompute(mainImg, None)
goodMatchesCountArray = []
allMatchesCountArray = []
processingTimeArray = []
imageNamesArray = []

for fileName in onlyfiles:

    startTime = time.time()
    currentImg = cv.imread("assets/" + fileName, 0)
    kp2, des2 = orb.detectAndCompute(currentImg, None)

    endTime = time.time()

    matches = bf.knnMatch(des1, des2, k=2)

    goodMatchesCurrent = []

    for m, n in matches:
        if m.distance < 0.8*n.distance:
            goodMatchesCurrent.append([m])
    # img3 = cv.drawMatchesKnn(mainImg, kp1, currentImg, kp2, goodMatchesCurrent,
    #                          None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    # plt.imshow(img3), plt.show()

    goodMatchesCountArray.append(len(goodMatchesCurrent))
    allMatchesCountArray.append(len(matches))
    processingTimeArray.append(endTime-startTime)
    imageNamesArray.append(fileName)

workbook = xlsxwriter.Workbook('data.xlsx')

worksheet1 = workbook.add_worksheet()

worksheet1.write_string('A1', "Filename")
worksheet1.write_row('B1', imageNamesArray)
worksheet1.write_string('A2', "Number of good matches")
worksheet1.write_row('B2', goodMatchesCountArray)
worksheet1.write_string('A3', "Percent of good matches")

worksheet1.write_row('B3', np.multiply(np.divide(
    goodMatchesCountArray, allMatchesCountArray), 100))
worksheet1.write_string('A4', "Calculations time")
worksheet1.write_row('B4', processingTimeArray)


mainImg = cv.imread("mainphoto2.jpg", 0)
orb = cv.ORB_create()
bf = cv.BFMatcher()
kp1, des1 = orb.detectAndCompute(mainImg, None)
goodMatchesCountArray = []
allMatchesCountArray = []
processingTimeArray = []
imageNamesArray = []

for fileName in onlyfiles:

    startTime = time.time()
    currentImg = cv.imread("assets/" + fileName, 0)
    kp2, des2 = orb.detectAndCompute(currentImg, None)

    endTime = time.time()

    matches = bf.knnMatch(des1, des2, k=2)

    goodMatchesCurrent = []

    for m, n in matches:
        if m.distance < 0.8*n.distance:
            goodMatchesCurrent.append([m])

    goodMatchesCountArray.append(len(goodMatchesCurrent))
    allMatchesCountArray.append(len(matches))
    processingTimeArray.append(endTime-startTime)
    imageNamesArray.append(fileName)

worksheet1.write_string('A6', "Filename")
worksheet1.write_row('B6', imageNamesArray)
worksheet1.write_string('A7', "Number of good matches")
worksheet1.write_row('B7', goodMatchesCountArray)
worksheet1.write_string('A8', "Percent of good matches")

worksheet1.write_row('B8', np.multiply(np.divide(
    goodMatchesCountArray, allMatchesCountArray), 100))
worksheet1.write_string('A9', "Calculations time")
worksheet1.write_row('B9', processingTimeArray)

workbook.close()
