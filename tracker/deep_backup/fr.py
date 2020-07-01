import cv2
import numpy as np
import os

"""
cap = cv2.VideoCapture("/nfs4/ajaym/Downloads/cendeep_sort_pytorch-master/deep_sort/deep_backup/MOT16-11/MOT_moving.mp4")
frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

buf = np.empty((frameCount, frameHeight, frameWidth, 3), np.dtype('uint8'))

fc = 0
ret = True
frameCount = 10
while (fc < frameCount  and ret):
    ret, buf[fc] = cap.read()
    fc += 1

cap.release()

cv2.namedWindow('frame 10')
#cv2.imshow('frame 10', buf[9])
#cv2.waitKey(0)
"""

img_dir = "/nfs4/ajaym/Downloads/cendeep_sort_pytorch-master/deep_sort/deep_backup/MOT16-11/Data/train/MOT16-11/img1"
dir_list = os.listdir(img_dir)

dir_list.sort()
#print(dir_list)

img_path = os.path.join(img_dir,dir_list[0])

frameCount = len(dir_list)
frameHeight = (cv2.imread(img_path).shape)[0]
frameWidth =  (cv2.imread(img_path).shape)[1]

buf = np.empty((frameCount, frameHeight, frameWidth, 3), np.dtype('uint8'))

i = 0

for img in dir_list:
    img_path = os.path.join(img_dir,img)
    buf[i] = cv2.imread(img_path)
    i = i+1

for i in buf:
    cv2.imshow('name',i)
    cv2.waitKey(0)
