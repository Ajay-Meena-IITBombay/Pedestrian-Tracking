import argparse
import os
import time

import cv2
from pprint import pprint
import random

from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.backends.cudnn as cudnn
import torchvision
import torch.nn as nn
from model1 import Net
from utils1 import TripletLoss, reid_dataset
from PIL import Image
import pickle

################################directory images to numpy array######################
"""
img_dir_list = ["/nfs4/ajaym/Downloads/cendeep_sort_pytorch-master/deep_sort/deep_backup/MOT16-11/Data/train/MOT16-02/img1",
                "/nfs4/ajaym/Downloads/cendeep_sort_pytorch-master/deep_sort/deep_backup/MOT16-11/Data/train/MOT16-04/img1",
                "/nfs4/ajaym/Downloads/cendeep_sort_pytorch-master/deep_sort/deep_backup/MOT16-11/Data/train/MOT16-09/img1",
                "/nfs4/ajaym/Downloads/cendeep_sort_pytorch-master/deep_sort/deep_backup/MOT16-11/Data/train/MOT16-10/img1",
                "/nfs4/ajaym/Downloads/cendeep_sort_pytorch-master/deep_sort/deep_backup/MOT16-11/Data/train/MOT16-11/img1",
                "/nfs4/ajaym/Downloads/cendeep_sort_pytorch-master/deep_sort/deep_backup/MOT16-11/Data/train/MOT16-13/img1"]


buf_list = []

for dir_ind in range(len(img_dir_list)):
    img_dir = img_dir_list[dir_ind]
    
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
    
    buf_list.append(buf)


with open('buf_list', 'wb') as f:
    pickle.dump(buf_list, f,protocol=4)

exit()
"""
with open('buf_list', 'rb') as f:
    buf_list = pickle.load(f)

print('pickle Loading Complete')
buf = buf_list[-1]


#################################creating list of ground truth#########################
file_list = ["/nfs4/ajaym/Downloads/cendeep_sort_pytorch-master/deep_sort/deep_backup/MOT16-02/gt/gt.txt",
            "/nfs4/ajaym/Downloads/cendeep_sort_pytorch-master/deep_sort/deep_backup/MOT16-04/gt/gt.txt",
            "/nfs4/ajaym/Downloads/cendeep_sort_pytorch-master/deep_sort/deep_backup/MOT16-09/gt/gt.txt",
            "/nfs4/ajaym/Downloads/cendeep_sort_pytorch-master/deep_sort/deep_backup/MOT16-10/gt/gt.txt",
            "/nfs4/ajaym/Downloads/cendeep_sort_pytorch-master/deep_sort/deep_backup/MOT16-11/gt/gt.txt",
            "/nfs4/ajaym/Downloads/cendeep_sort_pytorch-master/deep_sort/deep_backup/MOT16-13/gt/gt.txt"]
mainlist_list = []

for file_ind in range(len(file_list)):

    file1 =  open(file_list[file_ind],"r")
    mainlist = []
    sublist = []
    preid=1
    
    
    thres = 0.35
    i=0
    while True:
        rline = file1.readline()
    
        if len(rline) == 0:
            mainlist.append(sublist[:])
            break
        line = [float(x) for x in rline.split(',')]
        
        if line[8]>thres and line[7]==1 and line[6]==1 and (line[5]*line[4] > 12000):
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
        if len(sub_list)>7:
            mainlist1.append(sub_list[:])
        else:
            pass
            #print("len<7", len(sub_list))
    
    mainlist = mainlist1
    print("len_mainlist:",len(mainlist))
    
    mainlist_list.append(mainlist)



#cv2.namedWindow('image',cv2.WINDOW_NORMAL)

# device
device = "cuda:1"

# data loading
transform_train = torchvision.transforms.Compose([
    torchvision.transforms.Resize((128,64)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


datas = [reid_dataset(mainlist_list[i],buf_list[i],transform_train) for i in list(range(6))] 

trainloader = [torch.utils.data.DataLoader(datas[i],batch_size=16,shuffle=False,drop_last=True) for i in list(range(6))]


# net definition
start_epoch = 0
net = Net()
#net = Net(reid=True)
#checkpoint_path = "/nfs4/ajaym/Downloads/cendeep_sort_pytorch-master/deep_sort/deep_backup/checkpoint/ckpt.t7"
#net.load_state_dict(torch.load(checkpoint_path)["net_dict"])
net.to(device)

criterion = TripletLoss(10)
#optimizer = torch.optim.SGD(net.parameters(), 0.001, momentum=0.5, weight_decay=5e-4)
optimizer = torch.optim.SGD(net.parameters(), 0.001, momentum=0.5)
#optimizer = torch.optim.SGD(net.parameters(), 0.00, momentum=0.5)
best_acc = 0.


#for i,j,k in trainloader:
#    print(i,j,k)
#    pass

# train function for each epoch
def train(epoch):
    print("\nEpoch : %d"%(epoch+1))
    
    net.train()
    training_loss = 0.
    total=0

    for d_ind in range(6):
    
        for img1,img2,img3 in trainloader[d_ind]:
            
            # forward
            img1 = img1.to(device)
            img2 = img2.to(device)
            img3 = img3.to(device)

            o1,o2,o3 = net(img1,img2,img3)
            loss = criterion(o1,o2,o3)

            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # accumurating
            training_loss += loss.item()
    
    print("train_loss: ", training_loss)
    
    return

# lr decay
def lr_decay():
    global optimizer
    for params in optimizer.param_groups:
        params['lr'] *= 0.5
        lr = params['lr']
        print("Learning rate adjusted to {}".format(lr))

def main():
    for epoch in range(1,700):
        train(epoch)
        if (epoch+1)%40==0:
            lr_decay()
    PATH_CK =  "/nfs4/ajaym/Downloads/cendeep_sort_pytorch-master/deep_sort/deep_backup/checkpoint/cknew.t7"
    torch.save(net.state_dict(), PATH_CK)


if __name__ == '__main__':
    main()
