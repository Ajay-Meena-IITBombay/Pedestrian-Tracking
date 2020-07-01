import numpy as np

from .appearance_features.feature_extractor import Extractor
from .associate.nn_matching import NearestNeighborDistanceMetric
from .associate.preprocessing import non_max_suppression
from .associate.detection import Detection
from .associate.tracker import Tracker
from datetime import datetime
import cv2

__all__ = ['DeepSort']

class DeepSort(object):
    def __init__(self, model_path, max_dist=0.09, nms_thresh=0.5):
    #def __init__(self, model_path, max_dist=0.13):
        #max_dist optimal for default reid = 0.09
        #max_dist optimat for ranked reid = 0.3
        #max_dist : max possible distance b/t features t get associated
        #max_dist = 0.109, 0.050 found optimum for default reid 
        #max_dist = 0.16 found some what optimum for ranked reid
        #self.min_confidence = 0.2
        #self.nms_max_overlap = 0.5
        self.nms_max_overlap = nms_thresh


        self.extractor = Extractor(model_path, use_cuda=True)

        max_cosine_distance = max_dist
        nn_budget = 200
        metric = NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
        # metric = NearestNeighborDistanceMetric("euclidean", max_cosine_distance, nn_budget)

        self.tracker = Tracker(metric)

    def update(self, bbox_xywh, confidences, ori_img):
        self.height, self.width = ori_img.shape[:2]
        
        # my codeeeeeeee
        box_con = zip(bbox_xywh, confidences) 
        features = self._get_features(box_con, ori_img)
        bbox_tlwh = self._xywh_to_tlwh(bbox_xywh)
        detections = [Detection(bbox_tlwh[i], conf, features[i]) for i,conf in enumerate(confidences)]

        # run on non-maximum supression
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        indices = non_max_suppression( boxes, self.nms_max_overlap, scores)
        detections = [detections[i] for i in indices]
        
        # update tracker
        self.tracker.predict()
        self.tracker.update(detections)

        # output bbox identities
        outputs = []
        for track in self.tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            box = track.to_tlwh()
            x1,y1,x2,y2 = self._tlwh_to_xyxy(box)
            track_id = track.track_id
            outputs.append(np.array([x1,y1,x2,y2,track_id], dtype=np.int))
        if len(outputs) > 0:
            outputs = np.stack(outputs,axis=0)
        return outputs


    @staticmethod
    def _xywh_to_tlwh(bbox_xywh):
        bbox_xywh[:,0] = bbox_xywh[:,0] - bbox_xywh[:,2]/2.
        bbox_xywh[:,1] = bbox_xywh[:,1] - bbox_xywh[:,3]/2.
        return bbox_xywh


    def _xywh_to_xyxy(self, bbox_xywh):
        x,y,w,h = bbox_xywh
        x1 = max(int(x-w/2),0)
        x2 = min(int(x+w/2),self.width-1)
        y1 = max(int(y-h/2),0)
        y2 = min(int(y+h/2),self.height-1)
        return x1,y1,x2,y2

    def _tlwh_to_xyxy(self, bbox_tlwh):
        x,y,w,h = bbox_tlwh
        x1 = max(int(x),0)
        x2 = min(int(x+w),self.width-1)
        y1 = max(int(y),0)
        y2 = min(int(y+h),self.height-1)
        return x1,y1,x2,y2
    
    #def _get_features(self, bbox_xywh, ori_img):
    def _get_features(self,box_con,ori_img):
        im_crops = []
        for box, confidence in box_con:
            x1,y1,x2,y2 = self._xywh_to_xyxy(box)
            im = ori_img[y1:y2,x1:x2]
            ### my code ######
            #outfile = '%s/%s.jpg' % ("/nfs4/ajaym/Downloads/cendeep_sort_pytorch-master/debugresult/detectorimages_protest/", "yolo" + str(datetime.now())+"con:"+str(confidence))
            #cv2.imwrite(outfile, im)
            #################
            im_crops.append(im)
        if im_crops:
            features = self.extractor(im_crops)
        else:
            features = np.array([])
        return features


