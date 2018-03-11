import cv2

cap = cv2.VideoCapture('testfile.mpg')

while (True):
    ret, frame = cap.read()
