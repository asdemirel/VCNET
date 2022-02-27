import numpy as np
import cv2

video = cv2.VideoCapture('highway.avi')

kind = "image"
filename ="deneme/"
ext = ".jpg"
counter = 1

while(True):
    ret,frame = video.read()
    file_patch =filename + kind +str(counter) +ext
    cv2.imwrite(file_patch,frame)
    counter= counter + 1
    