import argparse
import os

import torch
import torch.nn as nn

import torchvision
from torchvision import transforms

import torch.optim as optim

from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from U_Net import Unet
from dataset import *

CUDA_VISIBLE_DEVICES=0
def train(epoch,train_loader,optimizer,criterion,model,device,writer):
    for k in range(100):
        running_loss=0.0
        accu=0.0
        model.train()
        optimizer.zero_grad()
        transform=transforms.ToPILImage()
        for i,data in enumerate(train_loader):
            inputs,labels=data
            inputs=inputs.to(device)
            labels=labels.to(device)
            outputs=model(inputs)
            out_tensor=outputs[:,1,:,:]<outputs[:,0,:,:].long()
            img=transform(out_tensor.float())
            img.show()
            accuracy=torch.sum((labels==out_tensor).float())
            accuracy=accuracy/(400*400.0)
            # print(torch.max(labels))
            # print(torch.sum(labels>0)-torch.sum(labels>1))
            loss=criterion(outputs,labels)
            loss.backward()
            running_loss+=loss.item()
            accu+=accuracy
            # True Batch Size: opt.bsz*opt.accum_step
            if (i+1)% opt.accum_step==0:
                
                optimizer.step()
                optimizer.zero_grad()
                print("Train Epoch: {0} Step: {1} Running Loss: {2} Accu: {3}".format(epoch,
                (i+1)/ opt.accum_step,running_loss/(i+1),accu/(i+1)))
        
        # Optimize for the last incomplete batch of data
        optimizer.step()
        optimizer.zero_grad()        
        writer.add_scalar('Loss/train', running_loss/(i+1),epoch)
        print('Train Epoch: {0} Loss: {running_loss:.5f}'.format(
            epoch,running_loss=running_loss/(i+1)
        ))

def val(epoch,val_loader,criterion,model,device,writer):
    model.to(device)
    model.eval()
    running_loss=0
    for i, (input,img_path) in enumerate(val_loader):
        path,img_name=os.path.split(img_path)
        input=input.to(device)
        output=output.to(device)
        output=model(input)
        
        loss=criterion(output,input)
        running_loss+=loss
    writer.add_scalar('Loss/val', running_loss/(i+1),epoch)
    print('Val Epoch: {0} Loss: {running_loss:.5f}'.format(
        epoch,running_loss=running_loss/(i+1)
    )) 

def test(test_loader,criterion,model,device,epoch=0):
    model.eval()
    transform=transforms.ToPILImage()
    for i, (input,img_path) in enumerate(test_loader):
        # print(epoch, img_path[0])
        path,img_name=os.path.split(img_path[0])
        input=input.to(device)
        output=model(input)
        print(output)
        out_tensor=(output[:,1,:,:]>output[:,0,:,:])
        print(torch.sum(out_tensor))
        out_tensor=out_tensor.int().float()
        #print(torch.sum(out_tensor))
        
        out_img=transform(out_tensor[0])
        save_path=os.path.join(opt.output_path,img_name[:-4]+'_ep_%.3d' % (epoch+1)+"_mask.png")
        out_img.save(save_path)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--traindata_path",type=str,default="training_/")
    parser.add_argument("--testdata_path",type=str,default="testing/")
    parser.add_argument("--output_path",type=str,default="out/")
    parser.add_argument("--save_path",type=str,default="ckpt/")
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--accum_step", type=int, default=1)
    parser.add_argument("--bsz",type=int,default=1)
    parser.add_argument("--lr",type=float,default=1e-4)
    parser.add_argument("--weight_decay",type=float,default=5e-5)
    parser.add_argument("--record_interval",type=int,default=1)
    parser.add_argument("--port",type=str, default='11112')

    parser.add_argument("--train",type=bool, default=True)
    parser.add_argument("--load_path",type=str,default="ckpt/model_ep9.pth")
    opt = parser.parse_args()

    writer = SummaryWriter()
    train_dataset=Geodataset(opt.traindata_path,True)
    test_dataset=Geodataset(opt.testdata_path,False)
    train_loader=get_dataloader(train_dataset,opt.bsz,shuffle=True)
    test_loader=get_dataloader(test_dataset,shuffle=False)
    model=Unet()
    
    criterion=nn.CrossEntropyLoss()
    optimizer=optim.SGD(model.parameters(),lr=opt.lr,momentum=0.9,weight_decay=opt.weight_decay)
    if opt.train:
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        device='cpu'
        model.to(device,non_blocking=False)
        for epoch in range(opt.epochs):
            train(epoch,train_loader,optimizer,criterion,model,device,writer)
            # if (epoch+1)%opt.record_interval==0:
            #     torch.save(model.state_dict(),
            #     os.path.join(opt.save_path,"model_ep_"+str(epoch+1)+'.pth'))
        torch.save(model.state_dict(),
        os.path.join(opt.save_path,"model_ep_"+str(epoch+1)+'.pth'))
        test(test_loader,criterion,model,device,epoch) 
    else:
        device='cpu'

        f=torch.load(opt.load_path)
        model.load_state_dict(f)
        
        model.to(device,non_blocking=False)
        test(test_loader,criterion,model,device,epoch=0) 


        

        