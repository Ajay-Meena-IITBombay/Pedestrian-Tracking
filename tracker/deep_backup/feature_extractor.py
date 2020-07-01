import torch
import torchvision.transforms as transforms
import numpy as np
import cv2
#from torchsummary import summary
#from .sort.nn_matching import NearestNeighborDistanceMetric
#from deep_sort.deep.ranked.modeling.baseline import Baseline
#from model import Net

class Extractor(object):
    def __init__(self, model_path, use_cuda=True):
       self.model_no = 0
       self.device = "cuda" if torch.cuda.is_available() and use_cuda else "cpu"
       if(self.model_no==1):
            #from deep_sort.deep.model import Net
            from model import Net

            self.net = Net(reid=True)
            state_dict = torch.load(model_path)['net_dict']
            self.net.load_state_dict(state_dict)
            print("Using deepsort default person Reid")
            self.net.to(self.device)
            self.size = (64, 128)
            self.norm = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])
       else:
           print("Using Ranked Person Reid")
           from deep_sort.deep.ranked.modeling.baseline import Baseline
           #from ranked.modeling.baseline import Baseline
           self.net = Baseline(num_classes = 3,last_stride =1, model_path = ' ',\
                               stn_flag = 'no', model_name = 'resnet50_ibn_a', pretrain_choice = ' ')
           self.net.load_param('/nfs4/ajaym/Downloads/Ranked_Person_ReID-master/demo_data/mar_resnet50_ibn_a_model.pth')
           self.net.to(self.device)
           self.net.eval()
           normalize_transform = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
           self.transform = transforms.Compose([transforms.ToPILImage(),transforms.Resize([256, 128]),transforms.ToTensor(),normalize_transform])


        

#    def _preprocess(self, im_crops):
#        """
#        TODO:
#            1. to float with scale from 0 to 1
#            2. resize to (64, 128) as Market1501 dataset did
#            3. concatenate to a numpy array
#            3. to torch Tensor
#            4. normalize
#        """
#        def _resize(im, size):
#            return cv2.resize(im.astype(np.float32)/255., size)
#
#        im_batch = torch.cat([self.norm(_resize(im, self.size)).unsqueeze(0) for im in im_crops], dim=0).float()
#        return im_batch


    def _preprocess(self, im_crops):
         if(self.model_no==1):
             def _resize(im, size):
                 return cv2.resize(im.astype(np.float32)/255., size)

             im_batch = torch.cat([self.norm(_resize(im, self.size)).unsqueeze(0) for im in im_crops], dim=0).float()
             return im_batch
         else:
             im_batch = torch.cat([self.transform(im).unsqueeze(0) for im in im_crops], dim=0).float()
             #img = Image.open(img_path).convert('RGB')
             #img = transform(img).unsqueeze(0)
             #img = img.to(device)
             return im_batch


    def __call__(self, im_crops):
        im_batch = self._preprocess(im_crops)
        with torch.no_grad():
            im_batch = im_batch.to(self.device)
            features = self.net(im_batch)
            #print("im_batch::::::::::::", im_batch.shape)
            #summary(self.net,(im_batch[0]).shape)
            #print("type of feature is::::::::::::", (features.shape))
            #print(features)
            #print("type of features.cpu().numpy()::::::::", (features.cpu().numpy()).shape)
            a = features.cpu().numpy()
            la = len(a)
            k=np.random.randint(2,10,(la,320))
            k = k/100
            #b = np.ones((la,320))*k
            #print(type(a))
            #print(type(np.append(a,b,axis=1)))
            #a = np.append(a,k,axis=1)

            #print(features.cpu().numpy())
        return a 


if __name__ == '__main__':
    img1 = cv2.imread("/nfs4/ajaym/Downloads/cendeep_sort_pytorch-master/debugresult/im_protest/75_2.jpg")
    img2 = cv2.imread("/nfs4/ajaym/Downloads/cendeep_sort_pytorch-master/debugresult/im_protest/78_3.jpg")
    
    im_crops = []
    im_crops.append(img1)
    im_crops.append(img2)
    extr = Extractor("checkpoint/ckpt.t7")
    feature = extr(im_crops)
    print(feature.shape)
    print(feature)
    print(1-np.dot(feature[0],feature[1].T)/(np.linalg.norm(feature[0])*np.linalg.norm(feature[1])))

