from PIL import Image
import PIL.ImageOps
import scipy.misc
import cv2
import json
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction import image
import timeit
import sys,os
import glob
import natsort 
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

# ======================================================================
network_dir="./networks"
sys.path.insert(0,network_dir)
import networks as networks

# ======================================================================
if torch.cuda.is_available():
    device="cuda:0"
else:
    device="cpu"

# device=torch.device(device)
device=torch.device("cuda:0")

# ======================================================================
def print_network(net,struct=False):
    """
    Args
      net: created network
      struct (False): do you want to see structure of entire network?
    Print
      Structure of entire network
      Total number of parameters of network
    """
    if struct==True:
        print(net)
    
    num_params=0
    for param in net.parameters():
        num_params+=param.numel()

    print('Total number of parameters: %d' % num_params)

def net_generator(batch_size,args):

    gen_net=networks.Direct_Reflectance_Prediction_Net().cuda()
    # gen_net=networks.Residual_Net().cuda()
    
    # --------------------------------------------------
    lr=0.001
    optimizer=torch.optim.Adam(gen_net.parameters(),lr=lr)
    # optimizer=torch.optim.Adadelta(gen_net.parameters())
    # optimizer=torch.optim.SGD(gen_net.parameters(),lr=0.01,momentum=0.9)

    # --------------------------------------------------
    # scheduler=torch.optim.lr_scheduler.StepLR(optimizer,step_size=20000,gamma=0.1)

    # --------------------------------------------------
    print_network(gen_net)

    # --------------------------------------------------
    if args.continue_training=="True":
        checkpoint_Direct_Reflectance_Prediction_Net=torch.load(args.checkpoint_file_path)
        gen_net.load_state_dict(checkpoint_Direct_Reflectance_Prediction_Net['state_dict'])
        optimizer.load_state_dict(checkpoint_Direct_Reflectance_Prediction_Net['optimizer'])

        # checkpoint_Residual_Net=torch.load(
        #     "/home/young/Downloads/test-master/update_CNN/checkpoint/Residual_Net.pth")
        # gen_net.load_state_dict(checkpoint_Residual_Net['state_dict'])
        # optimizer.load_state_dict(checkpoint_Residual_Net['optimizer'])
    
    # return gen_net,optimizer,scheduler
    return gen_net,optimizer
    
def save_checkpoint(state,filename):
    torch.save(state,filename)
