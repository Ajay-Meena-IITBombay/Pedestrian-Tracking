from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import numpy as np
from progress.bar import Bar
import time
import torch

try:
  from external.nms import soft_nms
except:
  print('NMS not imported! If you need it,'
        ' do \n cd $CenterNet_ROOT/src/lib/external \n make')
from models.decode import ctdet_decode
from models.utils import flip_tensor
from utils.image import get_affine_transform
from utils.post_process import ctdet_post_process
from utils.debugger import Debugger

from .base_detector import BaseDetector

class CtdetDetector(BaseDetector):
  def __init__(self, opt):
    super(CtdetDetector, self).__init__(opt)
    self.conf_thresh = opt.conf_thresh
  
  def process(self, images, return_time=False):
    with torch.no_grad():
      output = self.model(images)[-1]
      hm = output['hm'].sigmoid_()
      wh = output['wh']
      # print(wh.shape)
      reg = output['reg'] if self.opt.reg_offset else None
      # print('reg: ', reg)
      # print('reg shape: ', reg.shape)
      if self.opt.flip_test:
        # print("meaaya")
        hm = (hm[0:1] + flip_tensor(hm[1:2])) / 2
        wh = (wh[0:1] + flip_tensor(wh[1:2])) / 2
        reg = reg[0:1] if reg is not None else None
      torch.cuda.synchronize()       # Waits for all kernels in all streams on a CUDA device to complete
      forward_time = time.time()
      #det = torch.cat([bboxes, scores, clses], dim=2)
      dets = ctdet_decode(hm, wh, reg=reg, cat_spec_wh=self.opt.cat_spec_wh, K=self.opt.K)
      
    if return_time:
      return output, dets, forward_time
    else:
      return output, dets

  def post_process(self, dets, meta, scale=1):
    dets = dets.detach().cpu().numpy()
    dets = dets.reshape(1, -1, dets.shape[2])
    dets = ctdet_post_process(
        dets.copy(), [meta['c']], [meta['s']],
        meta['out_height'], meta['out_width'], self.opt.num_classes)
    for j in range(1, self.num_classes + 1):
      dets[0][j] = np.array(dets[0][j], dtype=np.float32).reshape(-1, 5)
      dets[0][j][:, :4] /= scale
    return dets[0]

  def merge_outputs(self, detections):
    results = {}
    per_classes = 1
    for j in range(1, per_classes + 1):
      results[j] = np.concatenate(
        [detection[j] for detection in detections], axis=0).astype(np.float32)
        #we could also directly use nms of deep sort but than it would cost us feature extraction of 
      #if len(self.scales) > 1 or self.opt.nms:
      #  soft_nms(results[j], Nt=0,threshold=0, method=2)
         #print("nmsaaya")
    scores = np.hstack(
      [results[j][:, 4] for j in range(1, per_classes + 1)])
    #if len(scores) > self.max_per_image:
    if True:
      #kth = len(scores) - self.max_per_image
      #thresh = np.partition(scores, kth)[kth]
      thresh = self.conf_thresh 
      #thresh = 0.47
      for j in range(1, per_classes + 1):
        #print('meaayathresh')
        ##the below operation only valid for array, while understanding if you are taking example dont generalize to list obly also consider numpy
        keep_inds = (results[j][:, 4] >= thresh)
        results[j] = results[j][keep_inds]

        #detection_mean = np.mean(results[j][:,2]*results[j][:,3])
        #keep_inds_2 = (results[j][:,2]*results[j][:,3] >= detection_mean/2)
        #results[j] = results[j][keep_inds_2]


    
    return results

  def debug(self, debugger, images, dets, output, scale=1):
    detection = dets.detach().cpu().numpy().copy()
    detection[:, :, :4] *= self.opt.down_ratio
    for i in range(1):
      img = images[i].detach().cpu().numpy().transpose(1, 2, 0)
      img = ((img * self.std + self.mean) * 255).astype(np.uint8)
      pred = debugger.gen_colormap(output['hm'][i].detach().cpu().numpy())
      debugger.add_blend_img(img, pred, 'pred_hm_{:.1f}'.format(scale))
      debugger.add_img(img, img_id='out_pred_{:.1f}'.format(scale))
      for k in range(len(dets[i])):
      ##center_threshold is like confidence_threshold and detection[i,k] i==1 and k is the index of detection 
        if detection[i, k, 4] > self.opt.center_thresh:
          debugger.add_coco_bbox(detection[i, k, :4], detection[i, k, -1],
                                 detection[i, k, 4], 
                                 img_id='out_pred_{:.1f}'.format(scale))
    #debugger.show_all_imgs(pause=self.pause)

  def show_results(self, debugger, image, results):
    debugger.add_img(image, img_id='ctdet')
    for j in range(1, self.num_classes + 1):
      for bbox in results[j]:
        if bbox[4] > self.opt.vis_thresh:
          debugger.add_coco_bbox(bbox[:4], j - 1, bbox[4], img_id='ctdet')
    debugger.show_all_imgs(pause=self.pause)
