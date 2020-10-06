import cv2

cap = cv2.VideoCapture(0)
ret, frame = cap.read()

# make photo, show it and save on the disk
image_0 = cv2.cvtColor(frame, 0)
cv2.imshow("window_0", image_0)
cv2.imwrite("./image_0.jpg", image_0)

# download photo from disk
image_0_downloaded = cv2.imread("./image_0.jpg")

# draw rect and line on gray photo, and show it
image_1_gray = cv2.cvtColor(image_0_downloaded, cv2.COLOR_BGR2GRAY)
image_1 = cv2.cvtColor(image_1_gray, cv2.COLOR_GRAY2BGR)
cv2.rectangle(image_1, (50, 50), (200, 200), (0, 0, 152), 5)
cv2.line(image_1, (60, 20), (400, 200), (255, 0, 0), 5)
cv2.imshow("window_1", image_1)
cv2.imwrite("./image_1.jpg", image_1)

# close everything and delete windows
cap.release()
cv2.waitKey(0)
cv2.destroyAllWindows()


# managing video
cap = cv2.VideoCapture(0)

# creating two writers to save two videos
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
size = (frame_width, frame_height)
writer_0 = cv2.VideoWriter('video_0.avi',
                           cv2.VideoWriter_fourcc(*'MJPG'),
                           10, size)

writer_1 = cv2.VideoWriter('video_1.avi',
                           cv2.VideoWriter_fourcc(*'MJPG'),
                           10, size)

# creating videos till user click 'q', saving them on disk
while(True):
    ret, frame = cap.read()

    if ret == True:
        # save frame using writer_0
        cv2.imshow('video_0', frame)
        writer_0.write(frame)

        # drawing on the same frame and save using writer_1
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = cv2.cvtColor(frame_gray, cv2.COLOR_GRAY2BGR)
        cv2.rectangle(frame, (50, 50), (200, 200), (0, 0, 152), 5)
        cv2.line(frame, (60, 20), (400, 200), (255, 0, 0), 5)
        cv2.imshow('video_1', frame)
        writer_1.write(frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

# close everything and delete windows
cap.release()
writer_0.release()
writer_1.release()
cv2.destroyAllWindows()
