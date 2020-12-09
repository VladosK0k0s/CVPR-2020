from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import cross_val_score
import xlsxwriter
from cv2 import cv2
import numpy as np
import time

from os import listdir
from os.path import isfile, join


# name for data folder that contains truly images

trueDataName = 'car'

# name for data folder that contains false images

falseDataName = 'not_car'

truefiles = [f for f in listdir(trueDataName) if isfile(join(trueDataName, f))]

falsefiles = [f for f in listdir(
    falseDataName) if isfile(join(falseDataName, f))]


def main():
    workbook = xlsxwriter.Workbook('data.xlsx')

    worksheet1 = workbook.add_worksheet()
    for descriptor_n in [1, 2, 3]:
        if descriptor_n == 1:
            descriptor_name = "ORB"
            nfeatures = 500
            pog = 32
            descriptor = cv2.ORB_create(nfeatures=nfeatures)
        if descriptor_n == 2:
            descriptor_name = "AKAZE"
            nfeatures = 500
            pog = 61
            descriptor = cv2.AKAZE_create()
        if descriptor_n == 3:
            descriptor_name = "BRISK"
            nfeatures = 500
            pog = 64
            descriptor = cv2.BRISK_create(thresh=30)

        def proccess_data(filesArr, folderName, isTrue):
            train_data = []
            y = []
            for fileName in filesArr:
                print(fileName)
                img = cv2.imread(folderName + "/" + fileName, 0)
                k, d = descriptor.detectAndCompute(img, None)
                # try:
                dest_matches = np.zeros((nfeatures, pog))
                for j in range(min(len(d), len(dest_matches))):
                    dest_matches[j, :] = d[j, :]
                train_data.append(dest_matches.ravel() / 256)
                if isTrue:
                    y.append(1)
                else:
                    y.append(0)
                # except:
                #     print("error")
            return train_data, y

        start_time = time.time()
        train_data_true, y_true = proccess_data(truefiles, trueDataName, True)
        train_data_false, y_false = proccess_data(
            falsefiles, falseDataName, False)

        train_data = np.array(np.concatenate(
            [train_data_true, train_data_false]))
        y = np.array(np.concatenate([y_true, y_false]))

        train_data = np.array(train_data)
        y = np.array(y)

        X_train, X_test, Y_train, Y_test = train_test_split(
            train_data, y, random_state=0, test_size=0.2)

        print("Train shape: ")
        print(X_train.shape)
        print("Test shape: ")
        print(X_test.shape)

        clf = AdaBoostClassifier(n_estimators=100)

        clf.fit(X_train, Y_train)
        clf.fit(train_data, y)
        scores = cross_val_score(clf, train_data, y, cv=5)
        print(scores.mean())

        times = round(time.time() - start_time, 0)
        print(
            f"Total time of learning: {times // 60} minutes {times % 60} seconds")

        learning_time = times

        ########################

        # TP:
        t1 = 0
        t2 = 0
        t3 = 0
        t4 = 0
        # (TP + FP):
        a1 = 0
        a2 = 0
        a3 = 0
        a4 = 0
        times = 0
        for i in range(X_test.shape[0]):
            start_time = time.time()
            yp = clf.predict(np.expand_dims(X_test[i], axis=0))
            times += (time.time() - start_time)

            yt = Y_test[i]

            if yp == yt == 1:
                t1 += 1
            elif yp == yt == 2:
                t2 += 1
            elif yp == yt == 3:
                t3 += 1
            elif yp == yt == 0:
                t4 += 1

            if yp == 1:
                a1 += 1
            elif yp == 2:
                a2 += 1
            elif yp == 3:
                a3 += 1
            elif yp == 0:
                a4 += 1

        # P:
        sY = [0, 0, 0, 0]
        sY[0] = np.count_nonzero(Y_test == 1)
        sY[1] = np.count_nonzero(Y_test == 0)

        worksheet1.write(f"A{descriptor_n*3+1}", "Datasetname")
        worksheet1.write(f"A{descriptor_n*3+2}", trueDataName)
        worksheet1.write(f"B{descriptor_n*3+1}", "Descriptor")
        worksheet1.write(f"B{descriptor_n*3+2}", descriptor_name)
        # TPR = TP/P
        worksheet1.write(f"C{descriptor_n*3+1}", "TPR")
        worksheet1.write(f"C{descriptor_n*3+2}", t1/sY[0])
        worksheet1.write(f"D{descriptor_n*3+1}", "TPR nothing")
        worksheet1.write(f"D{descriptor_n*3+2}", t4/sY[1])
        # FNR = 1 - TPR
        worksheet1.write(f"E{descriptor_n*3+1}", "FNR")
        worksheet1.write(f"E{descriptor_n*3+2}", 1 - t1/sY[0])
        worksheet1.write(f"F{descriptor_n*3+1}", "FNR nothing")
        worksheet1.write(f"F{descriptor_n*3+2}", 1 - t4/sY[1])
        # All = P + N
        All = X_test.shape[0]
        # FPR = FP/N = ((TP + FP) - TP)/(All - P)
        worksheet1.write(f"G{descriptor_n*3+1}", "FPR")
        worksheet1.write(f"G{descriptor_n*3+2}", (a1 - t1)/(All - sY[0]))
        worksheet1.write(f"H{descriptor_n*3+1}", "FPR nothing")
        worksheet1.write(f"H{descriptor_n*3+2}", (a4 - t4)/(All - sY[1]))

        worksheet1.write(f"I{descriptor_n*3+1}", "Total learning time, s")
        worksheet1.write(f"I{descriptor_n*3+2}", learning_time)
        worksheet1.write(f"J{descriptor_n*3+1}", "Average learning time, ms")
        worksheet1.write(f"J{descriptor_n*3+2}",
                         learning_time/X_test.shape[0]*1000)
        worksheet1.write(f"K{descriptor_n*3+1}", "Total prediction time, s")
        worksheet1.write(f"K{descriptor_n*3+2}", times)
        worksheet1.write(f"L{descriptor_n*3+1}", "Average prediction time, ms")
        worksheet1.write(f"L{descriptor_n*3+2}", times/X_test.shape[0]*1000)

        print(f"TPR:{t1/sY[0]}")
        print(f"TPR nothing:{t4/sY[1]}")

        print(f"FNR:{1 - t1/sY[0]}")
        print(f"FNR nothing:{1 - t4/sY[1]}")

        All = X_test.shape[0]
        print(f"FPR:{(a1 - t1)/(All - sY[0])}")
        print(f"FPR nothing:{(a4 - t4)/(All - sY[1])}")

        print(f"mean time:{times/X_test.shape[0]*1000}ms")

        # video filename (car.mp4)
        cap = cv2.VideoCapture(f'{trueDataName}.avi')

        images_arr = []

        font = cv2.FONT_HERSHEY_SIMPLEX
        output_size = (640, 360)  # dimension of video

        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                try:
                    k, d = descriptor.detectAndCompute(frame, None)
                    dest_matches = np.zeros((nfeatures, pog))
                    for j in range(min(len(d), len(dest_matches))):
                        dest_matches[j, :] = d[j, :]
                    x_data = dest_matches.ravel() / 256

                    yp = clf.predict(np.expand_dims(x_data, axis=0))
                    if yp == 1:
                        obj = trueDataName
                    elif yp == 0:
                        obj = falseDataName

                    cv2.putText(frame, obj, (10, 50), font, 2,
                                (0, 255, 0), 2, cv2.LINE_AA)
                except:
                    cv2.putText(frame, "none", (10, 50), font,
                                2, (0, 255, 0), 2, cv2.LINE_AA)

                output_im = cv2.resize(frame, output_size)
                images_arr.append(output_im)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                break

        fourcc = cv2.VideoWriter_fourcc(*'FMP4')
        fps = 24
        out = cv2.VideoWriter(
            f'{trueDataName}_{descriptor_name}.avi', fourcc, fps, output_size)
        for frame in images_arr:
            out.write(frame)

        cap.release()
        out.release()

    cv2.destroyAllWindows()

    workbook.close()

########################


main()