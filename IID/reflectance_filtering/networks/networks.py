import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets, models, transforms
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import argparse
import sys
import time
import os
import copy
import PIL

def init_weights(m):
    if type(m) == nn.Linear \
       or type(m) == nn.Conv2d \
       or type(m) == nn.ConvTranspose2d:
        torch.nn.init.xavier_uniform_(m.weight.data)

class Direct_Reflectance_Prediction_Net(nn.Module):
    def __init__(self):
        super(Direct_Reflectance_Prediction_Net,self).__init__()
        
        self.layer1=nn.Sequential(
            nn.Conv2d(3,32,kernel_size=1,padding=0),
            nn.Softplus())
        self.layer1.apply(init_weights)

        self.layer2=nn.Sequential(
            nn.Conv2d(32,32,kernel_size=1,padding=0),
            nn.Softplus())
        self.layer2.apply(init_weights)

        self.layer3=nn.Sequential(
            nn.Conv2d(32,32,kernel_size=1,padding=0),
            nn.Softplus())
        self.layer3.apply(init_weights)

        self.layer4=nn.Sequential(
            nn.Conv2d(32,32,kernel_size=1,padding=0),
            nn.Softplus())
        self.layer4.apply(init_weights)

        self.layer5=nn.Sequential(
            nn.Conv2d(32,32,kernel_size=1,padding=0),
            nn.Softplus())
        self.layer5.apply(init_weights)

        # Need to test 3 channel output and 1 channel output,
        # to see which is better
        # In case of 3 channel output, you will use mean of predicted intensity image
        # for exampel, refl_img=torch.mean(o_p_ref,dim=0,keepdim=True).squeeze()
        self.last_conv=nn.Sequential(
            nn.Conv2d(160,1,kernel_size=1,padding=0),
            nn.Softplus())
        self.last_conv.apply(init_weights)

        self.sigmoid=nn.Sequential(
            nn.Sigmoid())

    def forward(self,x):
        o_1=self.layer1(x)
        o_2=self.layer2(o_1)
        o_3=self.layer3(o_2)
        o_4=self.layer4(o_3)
        o_5=self.layer5(o_4)

        cat_features=torch.cat((o_1,o_2,o_3,o_4,o_5),dim=1)

        out=self.last_conv(cat_features)

        # Need to test with sigmoid and without sigmoid,
        # to see which is better
        # out=self.sigmoid(out)
       
        return out
