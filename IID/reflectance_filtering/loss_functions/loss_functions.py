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
import timeit
import sys
import glob
from skimage.transform import resize

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable,Function
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets, models, transforms

# ======================================================================
# utils_dir="/home/young/Downloads/test-master/update_CNN/utils"
# sys.path.insert(0,utils_dir) 
# import utils

# ======================================================================
delta=0.12  # threshold for "more or less equal"
xi=0.08  # margin in Hinge
ratio=1.0  # ratio of evaluated comparisons
eval_dense=1  # evaluate dense labels?
# m=0.45
m=0.25

# ======================================================================
class HumanReflectanceJudgements(object):

    def __init__(self, judgements):
        if not isinstance(judgements, dict):
            raise ValueError("Invalid judgements: %s" % judgements)

        self.judgements = judgements
        self.id_to_points = {p['id']: p for p in self.points}

    @staticmethod
    def from_file(filename):
        judgements = json.load(open(filename))
        # c hrj: human reflectance judgement json file
        hrj=HumanReflectanceJudgements(judgements)
        return hrj

    @property
    def points(self):
        return self.judgements['intrinsic_points']

    @property
    def comparisons(self):
        return self.judgements['intrinsic_comparisons']

# region original iiw loss
# This loss function is based on Eq 4 of tnestmeyer paper
# https://arxiv.org/pdf/1612.05062.pdf
def eval_single_comparison_with_SVM_hinge_loss(o_p_ref,comparisons,hrjf,o_ori_img):
    # print("o_p_ref",o_p_ref.shape)
    # o_p_ref torch.Size([3, 3, 384, 512])

    # c refl_img: mean of one predicted reflectance
    refl_img=torch.mean(o_p_ref,dim=0,keepdim=True).squeeze()
    # print("m_o_p_ref",m_o_p_ref.shape)
    # m_o_p_ref torch.Size([341, 512])

    # --------------------------------------------------
    rows=o_ori_img.shape[0]
    cols=o_ori_img.shape[1]

    # --------------------------------------------------
    # c error_sum: sum all errors from all comparisons in one image
    error_sum=0.0
    # c weight_sum: sum all weights from all comparisons in one image
    weight_sum=0.0

    # --------------------------------------------------
    # JSON GT for 1 image, 
    # containing all relative reflectance comparisons information
    # c c: one comparison from 1 image's all comparisons
    for c in comparisons:
        # c n_po1: number of point1
        n_po1=c['point1']
        # c point1: Point1 from one comparison
        point1 = hrjf.id_to_points[n_po1]

        n_po2=c['point2']
        # c point2: Point2 from one comparison
        point2=hrjf.id_to_points[n_po2]

        # c darker: Darker information from one comparison
        darker=c['darker']
        
        # Weight information from one comparison
        weight=c['darker_score'] # 1.14812035203497
        # print("weight",weight)

        # --------------------------------------------------
        # Check exception
        if not point1['opaque'] or not point2['opaque']:
            # Pass this judgement
            continue
        # weight<0 or weight is None -> invalid darker_score so pass
        if weight<0 or weight is None:
            raise ValueError("Invalid darker_score: %s" % weight)
        if darker not in ('1','2','E'):
            raise ValueError("Invalid darker: %s" % darker)
        
        # --------------------------------------------------
        x1,y1,x2,y2,darker=int(point1['x']*cols),\
                           int(point1['y']*rows),\
                           int(point2['x']*cols),\
                           int(point2['y']*rows),\
                           darker

        # --------------------------------------------------        
        # c R1: scalar value of point1 from predicted intensity image
        R1=refl_img[y1,x1]
        R2=refl_img[y2,x2]
        
        # --------------------------------------------------
        div_R1_R2=torch.div(R1,R2)
        
        # c dx_inv: 1+delta+xi inverse
        dx_inv=(1.0/(1.0+delta+xi))
        # c dx: 1+delta+xi
        dx=(1.0+delta+xi)
        # c dx_m_inv: 1+delta-i inverse
        dx_m_inv=(1.0/(1.0+delta-xi))
        # c dx_m: 1+delta-xi
        dx_m=(1.0+delta-xi)

        # --------------------------------------------------
        if darker=='1':
            # c ersp: error of single pair
            ersp=torch.max(torch.Tensor([0.0]).cuda(),div_R1_R2-dx_inv)
            error_sum+=ersp
            weight_sum+=weight
        elif darker=='2':
            ersp=torch.max(torch.Tensor([0.0]).cuda(),dx-div_R1_R2)
            error_sum+=ersp
            weight_sum+=weight
        elif darker=='E':
            if xi<=delta:
                ersp=torch.max(torch.Tensor([0.0]).cuda(),dx_m_inv-div_R1_R2)
                error_sum+=ersp
                weight_sum+=weight
            else:
                ersp=torch.max(torch.Tensor([0.0]).cuda(),div_R1_R2-dx_m)
                error_sum+=ersp
                weight_sum+=weight

    # Now, you have processed all comparisons in one image
    # If weight_sum exist
    if weight_sum:
        # c whdr: calculated whdr of one image
        whdr=error_sum/weight_sum

    # If weight_sum=0, it means there's no comparisons
    # In that case, you assign 0 into whdr
    else:
        whdr=0.0

    # Return whdr score of one image
    return whdr
# endregion original iiw loss

# region original iiw loss, updated with log
# This loss function is based on Eq 7 of CGINTRINSICS paper
# https://arxiv.org/pdf/1808.08601.pdf
# def eval_single_comparison_with_SVM_hinge_loss(o_p_ref,comparisons,hrjf,o_ori_img):

#     # c refl_img: mean of one predicted reflectance
#     refl_img=torch.mean(o_p_ref,dim=0,keepdim=True).squeeze()
#     # print("m_o_p_ref",m_o_p_ref.shape)
#     # m_o_p_ref torch.Size([341, 512])

#     # --------------------------------------------------
#     rows=o_ori_img.shape[0]
#     cols=o_ori_img.shape[1]

#     # --------------------------------------------------
#     # c error_sum: sum all errors from all comparisons in one image
#     # error_sum=0.0

#     # --------------------------------------------------
#     num_valid_comparisons=0.0

#     num_valid_comparisons_ineq=0.0
#     num_valid_comparisons_eq=0.0

#     total_loss_eq=torch.cuda.FloatTensor(1)
#     total_loss_eq[0]=0.0
#     total_loss_ineq=torch.cuda.FloatTensor(1)
#     total_loss_ineq[0]=0.0

#     # --------------------------------------------------
#     # JSON GT for 1 image, 
#     # containing all relative reflectance comparisons information
#     # c c: one comparison from 1 image's all comparisons
#     for c in comparisons:
#         # c n_po1: number of point1
#         n_po1=c['point1']
#         # c point1: Point1 from one comparison
#         point1 = hrjf.id_to_points[n_po1]

#         n_po2=c['point2']
#         # c point2: Point2 from one comparison
#         point2=hrjf.id_to_points[n_po2]

#         # c darker: Darker information from one comparison
#         darker=c['darker']
        
#         # Weight information from one comparison
#         weight=c['darker_score'] # 1.14812035203497
#         weight=torch.Tensor([weight]).cuda()

#         # --------------------------------------------------
#         # Check exception
#         if not point1['opaque'] or not point2['opaque']:
#             # Pass this judgement
#             continue
#         # weight<0 or weight is None -> invalid darker_score so pass
#         if weight<0 or weight is None:
#             raise ValueError("Invalid darker_score: %s" % weight)
#         if darker not in ('1','2','E'):
#             raise ValueError("Invalid darker: %s" % darker)
        
#         # --------------------------------------------------
#         x1,y1,x2,y2,darker=int(point1['x']*cols),int(point1['y']*rows),int(point2['x']*cols),int(point2['y']*rows),darker

#         # --------------------------------------------------        
#         # c R1: scalar value of point1 from predicted intensity image
#         # R1=(torch.log(refl_img[y1,x1]))
#         # R2=(torch.log(refl_img[y2,x2]))
#         R1=refl_img[y1,x1]
#         R2=refl_img[y2,x2]

#         # --------------------------------------------------
#         if darker=='1':
#             # c ersp: error of single pair
#             total_loss_ineq+=weight*torch.mean(torch.pow(m-(R2-R1),2))
#             num_valid_comparisons_ineq+=1.
#         elif darker=='2':
#             total_loss_ineq+=weight*torch.mean(torch.pow(m-(R1-R2),2))
#             num_valid_comparisons_ineq+=1.
#         elif darker=='E':
#             total_loss_eq+=weight*torch.mean(torch.pow(R1-R2,2))
#             num_valid_comparisons_eq+=1.
    
#     # --------------------------------------------------
#     total_loss=total_loss_ineq+total_loss_eq
#     num_valid_comparisons=num_valid_comparisons_eq+num_valid_comparisons_ineq

#     total_loss_real=total_loss/(num_valid_comparisons+1e-6)
#     # print("total_loss_real",total_loss_real)

#     return total_loss_real
# endregion original iiw loss, updated with log

def SVM_hinge_loss(o_p_ref,hrjf,o_ori_img):
    """
    Act
      Mu function based on Eq 2 from supplementary material
      which is used with confidence to calculate domain filter loss L_df
    """
    # c comparisons: JSON comparison file for 1 GT image
    comparisons=hrjf.judgements['intrinsic_comparisons']

    # c o_whdr: whdr score from one image
    o_whdr=eval_single_comparison_with_SVM_hinge_loss(o_p_ref,comparisons,hrjf,o_ori_img)

    return o_whdr
