# source activate py36gputorch041
# cd /home/young/Downloads/test-master/update_CNN/utils/
# rm e.l && python utils_image.py 2>&1 | tee -a e.l && code e.l

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
    # 512 512
    # 640 640
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
    # plt.imshow(ret)
    # plt.show()
    return ret

def guided_f(src_img,output_img,radius=20,eps=20):
    guided_img=cv2.ximgproc.guidedFilter(src_img,output_img,radius,eps)
    return guided_img

def bilateral_f(src_img,output_img,d=-1,sigmaColor=20,sigmaSpace=20):
    filtered=cv2.ximgproc.jointBilateralFilter(
        src_img,output_img,d,sigmaColor,sigmaSpace)
    return filtered

# ======================================================================
# Raw intensity into bilateral filter affected intensity
# call
# raw_inten_p="/home/young/Downloads/test-master/update_CNN/result/54_000_raw_intensity.png"
# fn=raw_inten_p.split("/")[-1].split(".")[0]
# raw_inten_img=load_img(raw_inten_p)
# ori_p="/mnt/1T-5e7/mycodehtml/data_col/cv/IID_f_w_w/iiw-dataset/train_with_100100patches/original_tr/test/temp/54_000.png"
# ori_img=load_img(raw_inten_p)
# sigmaColor=20
# sigmaSpace=20
# bi_f_img=bilateral_f(ori_img,raw_inten_img,d=-1,sigmaColor=sigmaColor,sigmaSpace=sigmaSpace)
# scipy.misc.imsave(fn+"_bi"+str(sigmaColor)+str(sigmaSpace)+".png",bi_f_img)

# ======================================================================
# Create reflectance by using bilateral filter affected intensity and original image
# call
# bi_inten_p="/home/young/Downloads/test-master/update_CNN/utils/54_000_raw_intensity_bi2020.png"
# fn=bi_inten_p.split("/")[-1].split(".")[0]
# bi_inten_img=load_img(bi_inten_p,gray=True)/255.0
# # Importantly, intensity image needs to be [0.5,1]
# bi_inten_img=cv2.normalize(bi_inten_img,None,alpha=0.5,beta=1.0,norm_type=cv2.NORM_MINMAX,dtype=cv2.CV_32F)

# ori_p="/mnt/1T-5e7/mycodehtml/data_col/cv/IID_f_w_w/iiw-dataset/train_with_100100patches/original_tr/test/temp/54_000.png"
# ori_img=load_img(ori_p)/255.0

# ref,sha=colorize(bi_inten_img,ori_img,eps=1e-3)

# ref=rgb_to_srgb(ref)
# ref=np.clip(ref,0.001,2.0)

# sha=get_sha_from_ref_ori(ref,ori_img)
# sha=np.clip(sha,0.0,2.0)

# ref=cv2.normalize(ref,None,alpha=0,beta=1,norm_type=cv2.NORM_MINMAX,dtype=cv2.CV_32F)
# sha=cv2.normalize(sha,None,alpha=0,beta=1,norm_type=cv2.NORM_MINMAX,dtype=cv2.CV_32F)
# scipy.misc.imsave(fn+"_ref.png",ref)
# scipy.misc.imsave(fn+"_sha.png",sha)

# ======================================================================
# Create reflectance by using bilateral filter affected intensity and original image
# And apply guided filter above reflectance
# call
# bi_inten_p="/home/young/Downloads/test-master/update_CNN/utils/54_000_raw_intensity_bi2020.png"
# fn=bi_inten_p.split("/")[-1].split(".")[0]
# bi_inten_img=load_img(bi_inten_p,gray=True)/255.0
# # Importantly, intensity image needs to be [0.5,1]
# bi_inten_img=cv2.normalize(bi_inten_img,None,alpha=0.5,beta=1.0,norm_type=cv2.NORM_MINMAX,dtype=cv2.CV_32F)

# ori_p="/mnt/1T-5e7/mycodehtml/data_col/cv/IID_f_w_w/iiw-dataset/train_with_100100patches/original_tr/test/temp/54_000.png"
# ori_img=load_img(ori_p)/255.0
# ori_img_255=load_img(ori_p)

# ref,sha=colorize(bi_inten_img,ori_img,eps=1e-3)
# ref=rgb_to_srgb(ref)
# ref=np.clip(ref,0.001,2.0)
# ref=(255*(ref-np.min(ref))/np.ptp(ref)).astype("uint8")

# radius=2
# eps=2
# ref=guided_f(ori_img_255,ref,radius=radius,eps=eps)
# ref=cv2.normalize(ref,None,alpha=0.001,beta=1.0,norm_type=cv2.NORM_MINMAX,dtype=cv2.CV_32F)

# sha=get_sha_from_ref_ori(ref,ori_img)
# sha=np.clip(sha,0.0,2.0)

# sha=cv2.normalize(sha,None,alpha=0,beta=1,norm_type=cv2.NORM_MINMAX,dtype=cv2.CV_32F)
# scipy.misc.imsave(fn+"ref_guided"+str(radius)+str(eps)+".png",ref)
# scipy.misc.imsave(fn+"sha_guided"+str(radius)+str(eps)+".png",sha)
