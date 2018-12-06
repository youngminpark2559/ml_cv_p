# source activate py36gputorch041
# cd /home/young/Downloads/test-master/update_CNN/utils/
# rm e.l && python utils_image.py 2>&1 | tee -a e.l && code e.l

import matplotlib as mpl
from PIL import Image
import PIL.ImageOps
import cv2
import json
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction import image
import timeit
import sys
import scipy.misc
from skimage.transform import resize

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
