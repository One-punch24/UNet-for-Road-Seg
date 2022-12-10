import torch
import torch.nn as nn
import torch.nn.functional as F
import PIL
from PIL import Image
import os
import numpy as np
def Conv_twice(in_channel,out_channel):
    return nn.Sequential(
        nn.Conv2d(in_channel,out_channel,kernel_size=3,padding=1),
        nn.ReLU(),
        nn.Conv2d(out_channel,out_channel,kernel_size=3,padding=1),
        nn.ReLU(),

    )

class Unet(nn.Module):
    def __init__(self):
        super(Unet,self).__init__()
        self.dw_conv1=Conv_twice(3,64)
        self.dw_conv2=Conv_twice(64,128)
        self.dw_conv3=Conv_twice(128,256)
        self.dw_conv4=Conv_twice(256,512)
        self.dw_conv5=Conv_twice(512,1024)

        self.rev_conv1=nn.ConvTranspose2d(1024,512,stride=2,kernel_size=2)
        self.rev_conv2=nn.ConvTranspose2d(512,256,stride=2,kernel_size=2)
        self.rev_conv3=nn.ConvTranspose2d(256,128,stride=2,kernel_size=2)
        self.rev_conv4=nn.ConvTranspose2d(128,64,stride=2,kernel_size=2)

        self.up_conv1=Conv_twice(1024,512)
        self.up_conv2=Conv_twice(512,256)
        self.up_conv3=Conv_twice(256,128)
        self.up_conv4=Conv_twice(128,64)

        self.out=nn.Conv2d(64,2,kernel_size=1)



    def forward(self,x):
        x1=self.dw_conv1(x)
        # x1_pool=
        x2=self.dw_conv2(F.max_pool2d(x1,kernel_size=2))
        x3=self.dw_conv3(F.max_pool2d(x2,kernel_size=2))
        x4=self.dw_conv4(F.max_pool2d(x3,kernel_size=2))
        x5=self.dw_conv5(F.max_pool2d(x4,kernel_size=2))

        x6=torch.cat((self.rev_conv1(x5),x4),dim=1)  # Torch, NCHW
        x6=self.up_conv1(x6)

        x7=torch.cat((self.rev_conv2(x6),x3),dim=1)  # Torch, NCHW
        x7=self.up_conv2(x7)

        x8=torch.cat((self.rev_conv3(x7),x2),dim=1)  # Torch, NCHW
        x8=self.up_conv3(x8)


        x9=torch.cat((self.rev_conv4(x8),x1),dim=1)  # Torch, NCHW
        x9=self.up_conv4(x9)

        return self.out(x9)

if __name__=='__main__':
    img=torch.randn((1,3,400,400))
    model=Unet()
    predict=model(img)
    print(predict.shape)
    Images_Path="training/images/"
    X = PIL.Image.open(os.path.join(Images_Path,"satImage_001.png"))
    X_np=np.asarray(X)
    X_torch=torch.tensor(X_np,dtype=int)
    X_torch=X_torch.permute(2,0,1)
    print(X_torch.dtype)





        
        





