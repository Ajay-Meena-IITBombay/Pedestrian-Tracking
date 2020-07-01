import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import random
import numpy as np
from pprint import pprint
from PIL import Image
import time

#Dataset
class reid_dataset(Dataset):
    def __init__(self,mainlist,buf,transform):
        self.mainlist = mainlist
        self.buf = buf
        self.idcount = len(mainlist)
        self.transform = transform
        self.frame_h,self.frame_w,_ = buf[0].shape


    def __getitem__(self,index):
        no_frame_id = len(self.mainlist[index])
        frame_1 = random.randint(0,no_frame_id-5)
        frame_2 = frame_1+3
        i = self.mainlist[index][frame_1]
        j = self.mainlist[index][frame_2]
        
        index_range = list(range(0,index)) + list(range(index+1,self.idcount))
        index_dissimilar = random.choice(index_range)
        k = random.choice(self.mainlist[index_dissimilar])


        
        img_1 = self.buf[int(i[0]-1)][int(max(0,i[3])):int(min(self.frame_h,i[3]+i[5])),int(max(0,i[2])):int(min(self.frame_w,i[2]+i[4])),:]
        img_2 = self.buf[int(j[0]-1)][int(max(0,j[3])):int(min(self.frame_h,j[3]+j[5])),int(max(0,j[2])):int(min(self.frame_w,j[2]+j[4])),:] 
        img_3 = self.buf[int(k[0]-1)][int(max(0,k[3])):int(min(self.frame_h,k[3]+k[5])),int(max(0,k[2])):int(min(self.frame_w,k[2]+k[4])),:]
        
        #img_2 = self.buf[int(j[0]-1)][int(j[3]):int(j[3]+j[5]),int(j[2]):int(j[2]+j[4]),:] 
        #img_3 = self.buf[int(k[0]-1)][int(k[3]):int(k[3]+k[5]),int(k[2]):int(k[2]+k[4]),:]


        #print("index:", index)
        #print("i:",i)
        img_1 = Image.fromarray(img_1)
        #print("j:",j)
        img_2 = Image.fromarray(img_2)
        #print("k:",k)
        img_3 = Image.fromarray(img_3)

#        img_1.show()


        img_1 = self.transform(img_1)
        img_2 = self.transform(img_2)
        img_3 = self.transform(img_3)


      
        return img_1, img_2, img_3 

    def __len__(self):
        return len(self.mainlist) -1


#TripletLoss
class TripletLoss(nn.Module):
    """
    Triplet loss
    Takes embeddings of an anchor sample, a positive sample and a negative sample
    """

    def __init__(self, margin):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        distance_positive = (anchor - positive).pow(2).sum(1)  # .pow(.5)
        distance_negative = (anchor - negative).pow(2).sum(1)  # .pow(.5)
        
        #distance_positive = F.cosine_similarity(anchor,positive)
        #distance_negative = F.cosine_similarity(anchor,negative)
        losses = F.relu(distance_positive - distance_negative + self.margin)
        #return losses
        return losses.sum()


