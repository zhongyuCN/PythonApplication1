import numpy as np
import cv2

cap=cv2.VideoCapture(0)
while True:
    ret,frame=cap.read()
    cv2.imshow('img',frame)
    cv2.waitKey(0)
    break
cap.release()
cv2.destroyAllWindows()


