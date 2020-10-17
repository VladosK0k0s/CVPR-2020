import cv2

# taking a photo from camera and saving it locally on disk
capturedData = cv2.VideoCapture(0)
ret, img = capturedData.read()
img = cv2.cvtColor(img, 0)
cv2.imshow("Image", img)
cv2.imwrite("./image.jpg", img)

# getting photo from disk, converting to gray and adding figures on it
imgLoaded = cv2.imread("./image.jpg")
imgGray = cv2.cvtColor(imgLoaded, cv2.COLOR_BGR2GRAY)
img1 = cv2.cvtColor(imgGray, cv2.COLOR_GRAY2BGR)
cv2.rectangle(img1, (300, 300), (200, 200), (255, 0, 255), 5)
cv2.line(img1, (200, 20), (300, 500), (0, 255, 255), 5)
cv2.imshow("window_1", img1)
cv2.imwrite("./img_with_drawings.jpg", img1)

# closing all windows
capturedData.release()
cv2.waitKey(0)
cv2.destroyAllWindows()
