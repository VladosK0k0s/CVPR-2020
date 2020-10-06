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
from os import listdir
from os.path import isfile, join
onlyfiles = [f for f in listdir("assets") if isfile(join("assets", f))]

mainImg = cv2.imread("assets/photo1.jpg", 0)
orb = cv.ORB_create()
bf = cv.BFMatcher()
kp1, des1 = orb.detectAndCompute(mainImg, None)
goodMatchesArray = []
processingTimeArray = []

for file in onlyfiles:

    startTime = time.time()

    currentImg = cv2.imread(file, 0)
    kp2, des2 = orb.detectAndCompute(currentImg, None)

    endTime = time.time()

    processingTimeArray.append(endTime-startTime)

    matches = bf.knnMatch(des1, des2, k=2)

    goodMatchesCurrent = []

    for m, n in matches:
        if m.distance < 0.8*n.distance:
            good.append([m])
    img3 = cv.drawMatchesKnn(img1, kp1, img2, kp2, good,
                             None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    plt.imshow(img3), plt.show()
    print(len(good)/len(matches))
