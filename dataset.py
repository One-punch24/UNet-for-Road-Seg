"""Datasets"""

import os

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
import torchvision.transforms as transforms
import torchvision
import glob
import PIL
import scipy
import scipy.io
import numpy as np
import torch.functional as F

class Geodataset(Dataset):
    """Carla Dataset"""

    def __init__(self, dataset_path,train):
        super().__init__()
        self.train=train # bool
        if self.train:
            self.img_path=os.path.join(dataset_path,"images/*.png")
            self.label_path=os.path.join(dataset_path,"groundtruth/")
            self.data = glob.glob(self.img_path)
        else:
            self.img_path=os.path.join(dataset_path,"test_**/*.png")
            self.data=glob.glob(self.img_path)
        assert len(self.data) > 0, "Can't find data; make sure you specify the path to your dataset"
        self.transform = transforms.Compose(
                    [ transforms.ToTensor(), transforms.Normalize(std=(0.5,0.5,0.5),mean=(0.5,0.5,0.5))])

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):

        X = PIL.Image.open(self.data[index])
        
        (path,img_name)=os.path.split(self.data[index])
        # print(img_name)
        X = self.transform(X)
        if self.train:
            label_path_=os.path.join(self.label_path,img_name)
            Y=PIL.Image.open(label_path_)
            Y=np.asarray(Y)
            Y=torch.tensor(Y,dtype=torch.int64)
            # Debug !
            # print(torch.sum((Y>=1).int()))
            # print(torch.sum((Y>=1)*Y)/torch.sum((Y>=1)).float())
            # print(torch.sum((Y==2).int()))
            Y=torch.ones(Y.shape)*(Y>0.1*torch.max(Y))
            Y=Y.long()
            print(torch.sum(Y))
            # Y=Y.long().float()
            # transform=transforms.ToPILImage()
            # Y_pil=transform(Y)
            # Y_pil.save("nmd.png")
            return X,Y
        else:
            return X,self.data[index]



def get_dataloader(dataset, batch_size=1,shuffle=True,**kwargs):

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
        pin_memory=True,
        num_workers=4
    )
    return dataloader


