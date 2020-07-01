import os
import cv2
import time
import argparse
import numpy as np
import sys
import copy
import warnings




from YOLOv3 import YOLOv3
from tracker import DeepSort
from util import COLORS_10, draw_bboxes, draw_simple_bboxes, x1y1_x2y2_conf, crop_simple_bboxes

from centernet import _init_paths
from centernet.lib.opts import opts
from detectors.detector_factory import detector_factory
from centernet.lib.detectors.ctdet import CtdetDetector

image_ext = ['jpg', 'jpeg', 'png', 'webp']
video_ext = ['mp4', 'mov', 'avi', 'mkv']
time_stats = ['tot', 'load', 'pre', 'net', 'dec', 'post', 'merge']

class Detector(object):
    def __init__(self, args):
        self.args = args
        warnings.filterwarnings("ignore")
#        if args.display:
#            cv2.namedWindow("test", cv2.WINDOW_NORMAL)
#            cv2.resizeWindow("test", args.display_width, args.display_height)

        self.vdo = cv2.VideoCapture()
        self.start_frame_number = 50
        self.vdo.set(cv2.CAP_PROP_POS_FRAMES, self.start_frame_number)
        #self.yolo3 = YOLOv3(args.yolo_cfg, args.yolo_weights, args.yolo_names, is_xywh=True, conf_thresh=args.conf_thresh, nms_thresh=args.nms_thresh)
        #self.deepsort = DeepSort(args.deepsort_checkpoint, args.max_dist, args.nms_thresh)
        self.deepsort = DeepSort(args.deepsort_checkpoint)
        
        #self.class_names = self.yolo3.class_names

        self.single_display = args.single_d
        self.pause = args.pause_d
        self.only_detect = args.only_detect
        self.from_video = args.from_video
        self.path = args.external_video_path 
        self.save_per = 100


    def __enter__(self):
        if(self.from_video):
            assert os.path.isfile(self.args.VIDEO_PATH), "Error: path error"
            self.vdo.open(self.args.VIDEO_PATH)
            self.im_width = int(self.vdo.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.im_height = int(self.vdo.get(cv2.CAP_PROP_FRAME_HEIGHT))
            dim = (int(self.im_width*self.save_per/100),int(self.im_height*self.save_per/100))  
            if self.args.save_path:
                fourcc =  cv2.VideoWriter_fourcc(*'MJPG')
                self.output = cv2.VideoWriter(self.args.save_path, fourcc, 20, dim)
            assert self.vdo.isOpened()
            return self
        else:
            path = self.path
            list_dir = os.listdir(path)
            slist = sorted(list_dir)
            join = os.path.join(path,slist[0])
            ori_im = cv2.imread(join)
            self.im_width = int(ori_im.shape[1])
            self.im_height = int(ori_im.shape[0])
            dim = (int(self.im_width*self.save_per/100),int(self.im_height*self.save_per/100))
            if self.args.save_path:
                fourcc =  cv2.VideoWriter_fourcc(*'MJPG')
                self.output = cv2.VideoWriter(self.args.save_path, fourcc, 20, dim)
            return self

    
    def __exit__(self, exc_type, exc_value, exc_traceback):
        if exc_type:
            print(exc_type, exc_value, exc_traceback)
        
    def save_res(self,ori_im):
        dim = (int(ori_im.shape[1]*self.save_per/100),int(ori_im.shape[0]*self.save_per/100))
        return cv2.resize(ori_im, dim, interpolation = cv2.INTER_AREA)

    def detect(self, opt):
        os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
        opt.debug = max(opt.debug, 1)
        #Detector = detector_factory[opt.task]
        
        #"""
        Detector = CtdetDetector
        detector = Detector(opt)
        detector.pause = False
        #"""

        img_index = 0
        while True:
            img_index = img_index+1
            print("img_imdex: ",img_index) 
            if self.from_video:
                _, ori_im = self.vdo.read()
                start = time.time()
                #_, ori_im = self.vdo.retrieve()
                #im = cv2.cvtColor(ori_im, cv2.COLOR_BGR2RGB)
            else:
                start = time.time()
                path = self.path
                list_dir = os.listdir(path)
                slist = sorted(list_dir)
                join = os.path.join(path,slist[img_index])
                ori_im = cv2.imread(join)
                img_index = img_index+1
            
                
            im = ori_im
            dm = copy.deepcopy(ori_im)


            #"""
            ret = detector.run(im)

            xc = abs(ret[:,0]+ret[:,2])/2.0
            yc = abs(ret[:,1]+ret[:,3])/2.0
            w = abs(ret[:,0]-ret[:,2])
            h = abs(ret[:,1]-ret[:,3])
            cls_conf = ret[:,4]
            bbox_xcycwh = np.column_stack((xc,yc,w,h))
            bbox_x1y1x2y2_conf = np.concatenate((np.floor(ret[:,0:4]),np.transpose(np.expand_dims(ret[:,4],axis=0))),axis=1)
            
            #"""
                       


            #bbox_xcycwh, cls_conf, cls_ids = self.yolo3(im)
            #print(type(bbox_xcycwh),':',type(cls_conf),':',type(cls_ids))



            if bbox_xcycwh is not None:
                #uncomment for using yolo
                """   
                mask = cls_ids==0
                
                bbox_xcycwh = bbox_xcycwh[mask]
                bbox_xcycwh[:,2] *= 1.1

                cls_conf = cls_conf[mask]
                """

                outputs = self.deepsort.update(bbox_xcycwh, cls_conf, im)
                if len(outputs) > 0:
                    bbox_xyxy = outputs[:,:4]
                    identities = outputs[:,-1]
                    ori_im = draw_bboxes(ori_im, bbox_xyxy, identities)

            end = time.time()
            print("time: {}s, fps: {}".format(end-start, 1/(end-start)))
            s_size = 1.3
            d_size = 2

            ##double window resolution adjusting
            screen_res = 768, 1366
            #screen_res = 900, 1650
            screen_res = screen_res[0]/1.2,screen_res[1]/1.2
            scale = max(screen_res[0]/ori_im.shape[0],screen_res[1]/ori_im.shape[1])
            ####################################################


            if self.single_display:
                
                #cv2.namedWindow('image',cv2.WINDOW_NORMAL)
                #cv2.resizeWindow('image',(int((ori_im.shape[1])*scale),int((ori_im.shape[0])*scale)))
                
                if not(self.only_detect):

                    ## ori_img :  not detections, our tracking ones
                    
                    #cv2.imshow('image', ori_im)
                    #if cv2.waitKey(0 if self.pause else 1) == 27:
                    #  sys.exit(0)

                    if self.args.save_path:
                        re_ori_im = self.save_res(ori_im)
                        self.output.write(re_ori_im)

                else:
                    ## dm : detection only 
                    dm = draw_simple_bboxes(dm,bbox_x1y1x2y2_conf)

                    cv2.imshow('image', dm)
                    if cv2.waitKey(0 if self.pause else 1) == 27:
                      sys.exit(0)

                    if self.args.save_path:
                        self.output.write(dm)

                
                
            else:
                
                
                both = np.concatenate((dm,ori_im),axis=1)
                cv2.namedWindow('image',cv2.WINDOW_NORMAL)
                cv2.resizeWindow('image',(int((both.shape[1]*scale/1.7)),int((both.shape[0]*scale/1.7))))
                
                
                cv2.imshow('image', both)
                if cv2.waitKey(0 if self.pause else 1) == 27:
                  sys.exit(0)
                
                if self.args.save_path:
                    re_ori_im = self.save_res(ori_im)
                    self.output.write(ori_im)
                

#            if self.args.save_path:
#                self.output.write(ori_im)
            


if __name__=="__main__":
   
    # args = parse_args()
    # print(args.display)
    # print(args.store_false)
    opt = opts().init()
    with Detector(opt) as det:
         det.detect(opt)
    # det  = Detector(args)
    # det.detect()


