import cv2
import numpy as np
import os
from pprint import pprint
import random
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
from PIL import Image

#####################################################################################
"""
cap = cv2.VideoCapture("/nfs4/ajaym/Downloads/cendeep_sort_pytorch-master/deep_sort/deep_backup/MOT16-11/MOT16-11-raw.webm")
frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

buf = np.empty((frameCount, frameHeight, frameWidth, 3), np.dtype('uint8'))


print(frameCount) 
print(frameWidth)
print(frameHeight)

#exit()

fc = 0
ret = True
#frameCount = 10
while (fc < frameCount  and ret):
    ret, buf[fc] = cap.read()
    fc += 1

cap.release()
"""
#########################################################################################



img_dir = "/nfs4/ajaym/Downloads/cendeep_sort_pytorch-master/deep_sort/deep_backup/MOT16-11/Data/train/MOT16-11/img1"
dir_list = os.listdir(img_dir)

dir_list.sort()

img_path = os.path.join(img_dir,dir_list[0])

frameCount = len(dir_list)
frameHeight = (cv2.imread(img_path).shape)[0]
frameWidth =  (cv2.imread(img_path).shape)[1]

buf = np.empty((frameCount, frameHeight, frameWidth, 3), np.dtype('uint8'))

i=0
for img in dir_list:
    img_path = os.path.join(img_dir,img)
    buf[i] = np.asarray(Image.open(img_path))
    i = i+1

##############################################################################################

thres = 0.7
file1 =  open("/nfs4/ajaym/Downloads/cendeep_sort_pytorch-master/deep_sort/deep_backup/MOT16-11/gt/gt.txt","r") 

mainlist = []
sublist = []
preid=1

i=0
while True:
    rline = file1.readline()

    if len(rline) == 0:
        mainlist.append(sublist[:])
        break

    line = [float(x) for x in rline.split(',')]
    
    if line[8]>thres and line[7]==1 and line[6]==1:
        ins_line = line[0:-3]+line[-1:]
    else:
        continue


    if line[1]!= preid:
        mainlist.append(sublist[:])
        sublist.clear()
        preid = line[1]
    
    sublist.append(ins_line)
    
    i=i+1

file1.close()


mainlist1 = []
for sub_list in mainlist:
    #print("lensub",len(sub_list))
    if len(sub_list)>7:
        mainlist1.append(sub_list[:])
    else:
        #pass
        print("len<7", len(sub_list))

mainlist = mainlist1

#pprint(mainlist)
#for i in mainlist:
#    print(i[0][0])


#cv2.namedWindow('image',cv2.WINDOW_NORMAL)

"""
for i in mainlist[0]:
    image = cv2.rectangle(buf[int(i[0]-1)], (int(i[2]),int(i[3])), (int(i[2]+i[4]),int(i[3]+i[5])), (0,0,255), 10) 
    #image = cv2.rectangle(buf[int(i[0]-1)], (int(i[2]/2),int(i[3]/2)), (int(i[2]/2+i[4]/2),int(i[3]/2+i[5]/2)), (0,0,255), 10) 
    cv2.imshow('image',image)
    cv2.waitKey(5)
"""
##################################################################################


transform_train = torchvision.transforms.Compose([
    torchvision.transforms.Resize((128,64)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

class reid_dataset(Dataset):
    def __init__(self,mainlist,buf,transform):
        self.mainlist = mainlist
        self.buf = buf
        self.idcount = len(mainlist)
        self.transform = transform

    def __getitem__(self,index):
        no_frame_id = len(mainlist[index])
        frame_1 = random.randint(0,no_frame_id-5)
        frame_2 = frame_1+3
        i = mainlist[index][frame_1]
        j = mainlist[index][frame_2]
        
        index_range = list(range(0,index)) + list(range(index+1,self.idcount))
        index_dissimilar = random.choice(index_range)
        k = random.choice(mainlist[index_dissimilar])

        
        img_1 = self.buf[int(i[0]-1)][int(i[3]):int(i[3]+i[5]),int(i[2]):int(i[2]+i[4]),:] 
        img_2 = self.buf[int(j[0]-1)][int(j[3]):int(j[3]+j[5]),int(j[2]):int(j[2]+j[4]),:] 
        img_3 = self.buf[int(k[0]-1)][int(k[3]):int(k[3]+k[5]),int(k[2]):int(k[2]+k[4]),:]

        img_1 = Image.fromarray(img_1)
        img_2 = Image.fromarray(img_2)
        img_3 = Image.fromarray(img_3)
        
        
        img_1 = self.transform(img_1)
        img_2 = self.transform(img_2)
        img_3 = self.transform(img_3)


        #cv2.imshow('image',img_1)
        #cv2.waitKey(5000)
        #cv2.imshow('image',img_2)
        #cv2.waitKey(5000) 
        #cv2.imshow('image',img_3)
        #cv2.waitKey(5000)
      
        return img_1, img_2, img_3 

    def __len__(self):
        return len(self.mainlist)

datas = reid_dataset(mainlist,buf,transform_train)
i1,i2,i3 = datas.__getitem__(8)
print(type(i1),i1.shape)
print(type(i2),i2.shape)



