from PIL import Image
import PIL.ImageOps
import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage.transform import resize
import sys,os
import scipy.misc

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
from torchvision import transforms
from torch.autograd import Variable

# ======================================================================
utils_dir="./utils"
sys.path.insert(0,utils_dir)
import utils_common as utils_common

# ======================================================================
def resize_img(img,h,w):
    rs_img=resize(img,(h,w))
    return rs_img

def load_img(path,gray=False):
    if gray==True:
        img=Image.open(path).convert("L")
        img=np.array(img)
    else:
        img=Image.open(path)
        img=np.array(img)
    return img

def lightness(r):
    num_channels=len(r)

    # RGB r prediction
    if num_channels == 3:
        L = max(eps, np.mean(r))
        dLdR=np.ones(3)/3.
    elif num_channels == 1:
        L = max(eps, r)
        dLdR = np.ones(1)
    else:
        raise Exception("Expecting 1 or 3 channels to compute lightness!")
    return L,dLdR

def colorize(intensity,image,eps=1e-3):
    norm_input=np.mean(image,axis=2)

    shading=np.nan_to_num(norm_input/intensity)

    reflectance=image/np.maximum(shading,eps)[:,:,np.newaxis]

    return reflectance,shading

def get_sha_from_ref_ori(ref,ori):
    sha=ori/ref
    sha=sha[:,:,0]
    return sha

def rgb_to_srgb(rgb):
    """Taken from bell2014: RGB -> sRGB."""
    ret = np.zeros_like(rgb)
    idx0 = rgb <= 0.0031308
    idx1 = rgb > 0.0031308
    ret[idx0] = rgb[idx0] * 12.92
    ret[idx1] = np.power(1.055 * rgb[idx1], 1.0 / 2.4) - 0.055
    return ret

def guided_f(src_img,output_img,radius=20,eps=20):
    guided_img=cv2.ximgproc.guidedFilter(src_img,output_img,radius,eps)
    return guided_img

def bilateral_f(src_img,output_img,d=-1,sigmaColor=20,sigmaSpace=20):
    filtered=cv2.ximgproc.jointBilateralFilter(
        src_img,output_img,d,sigmaColor,sigmaSpace)
    return filtered
