import torch
import torchvision.transforms as transforms
import numpy as np
import cv2

class Extractor(object):
    def __init__(self, model_path, use_cuda=True):
       self.model_no = 2
       self.device = "cuda" if torch.cuda.is_available() and use_cuda else "cpu"
       if(self.model_no==1):
            from tracker.appearance_features.model import Net
            #from model import Net

            self.net = Net(reid=True)
            state_dict = torch.load(model_path)['net_dict']
            self.net.load_state_dict(state_dict)
            print("Using deepsort default person Reid")
            self.net.to(self.device)
            self.net.eval()
            self.size = (64, 128)
            self.norm = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])
       elif(self.model_no==2):
           print("Using Ranked Person Reid")
           from tracker.appearance_features.ranked.modeling.baseline import Baseline
           #from ranked.modeling.baseline import Baseline
           self.net = Baseline(num_classes = 3,last_stride =1, model_path = ' ',\
                               stn_flag = 'no', model_name = 'resnet50_ibn_a', pretrain_choice = ' ')
           self.net.load_param('/nfs4/ajaym/Downloads/Ranked_Person_ReID-master/demo_data/mar_resnet50_ibn_a_model.pth')
           self.net.to(self.device)
           self.net.eval()
           normalize_transform = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
           self.transform = transforms.Compose([transforms.ToPILImage(),transforms.Resize([256, 128]),transforms.ToTensor(),normalize_transform])

       else:
           pass



        

    def _preprocess(self, im_crops):
         if(self.model_no==1):
             def _resize(im, size):
                 return cv2.resize(im.astype(np.float32)/255., size)

             im_batch = torch.cat([self.norm(_resize(im, self.size)).unsqueeze(0) for im in im_crops], dim=0).float()
             return im_batch
         elif(self.mode_no==2):
             im_batch = torch.cat([self.transform(im).unsqueeze(0) for im in im_crops], dim=0).float()
             return im_batch


    def __call__(self, im_crops):
        im_batch = self._preprocess(im_crops)
        with torch.no_grad():
            im_batch = im_batch.to(self.device)
            features = self.net(im_batch)
            a = features.cpu().numpy()
            la = len(a)
            k=np.random.randint(2,10,(la,320))
            k = k/100

        return a 


if __name__ == '__main__':
    img1 = cv2.imread("image1_path")
    img2 = cv2.imread("image2_path")
    
    im_crops = []
    im_crops.append(img1)
    im_crops.append(img2)
    extr = Extractor("checkpoint/ckpt.t7")
    feature = extr(im_crops)
    print(feature.shape)
    print(feature)

