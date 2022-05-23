import os
from datetime import datetime

import cv2
import face_recognition as fr
import numpy as np

images = []
path = "Attendence_Images"
faces = os.listdir(path)
faceNames = [face.split('.jpg')[0] for face in faces]
print(faceNames)

for face in faces:
    curface = cv2.imread(f"{path}/{face}")
    images.append(curface)


def findEncoding(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = fr.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList




studentslist = []
def attendenceMarkings(name):

    with open("attendence.csv", 'r+') as file:
        data = file.readlines()
        for line in data:
            entry = line.split(',')
            studentslist.append(entry[0])
            if name not in studentslist:
                today = datetime.now()
                today = today.strftime("%H:%M:%S")
                file.writelines(f'\n{name},{today}')
            else:
                pass


# attendenceMarkings('Helloworld')
encodeList = findEncoding(images)
print("encodedCompleted")

videocap = cv2.VideoCapture(0)

while True:
    success, img = videocap.read()
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
    facesCurrFrame = fr.face_locations(imgS)
    currFacesEncoding = fr.face_encodings(imgS, facesCurrFrame)

    for faceloc, encodings in zip(facesCurrFrame, currFacesEncoding):
        result = fr.compare_faces(encodeList, encodings)
        facedis = fr.face_distance(encodeList, encodings)

        matchindex = np.argmin(facedis)
        # print(facedis)
        # print(result)
        if result[matchindex]:
            name = faceNames[matchindex]
            y1, x2, y2, x1 = faceloc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (255, 255, 255), cv2.FILLED)
            cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_ITALIC, 0.5, 2)
            print(name.lower())
            if name not in studentslist:
                attendenceMarkings(name.lower())
        else:
            name="Unknown"
            y1, x2, y2, x1 = faceloc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (255, 255, 255), cv2.FILLED)
            cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_ITALIC, 0.5, 2)
            print(name.lower())
    cv2.imshow('webcam', img)
    cv2.waitKey(1)
