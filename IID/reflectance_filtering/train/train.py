# source activate py36gputorch041
# cd /home/young/Downloads/test-master/update_CNN/train/

# Train
# rm e.l && python train.py \
# --epoch=20 \
# --batch_size=15 \
# --train_mode=True \
# --load_data_from_text_file=False \
# --continue_training=False \
# --iiw_dataset_dir="/mnt/1T-5e7/mycodehtml/data_col/cv/IID_f_w_w/iiw-dataset/data/temp" \
# --iiw_dataset_img_text_file="/mnt/1T-5e7/image/whole_dataset/iiw_data_img.txt" \
# --iiw_dataset_gt_text_file="/mnt/1T-5e7/image/whole_dataset/iiw_data_json.txt" \
# --checkpoint_save_dir="/home/young/Downloads/test-master/update_CNN/checkpoint" \
# --checkpoint_file_path="./checkpoint/Direct_Reflectance_Prediction_Net.pth" \
# 2>&1 | tee -a e.l && code e.l

# Test
# rm e.l && python train.py \
# --epoch=20 \
# --batch_size=15 \
# --train_mode=False \
# --load_data_from_text_file=False \
# --continue_training=True \
# --iiw_dataset_dir="/mnt/1T-5e7/mycodehtml/data_col/cv/IID_f_w_w/iiw-dataset/data/temp" \
# --iiw_dataset_img_text_file="/mnt/1T-5e7/image/whole_dataset/iiw_data_img.txt" \
# --iiw_dataset_gt_text_file="/mnt/1T-5e7/image/whole_dataset/iiw_data_json.txt" \
# --checkpoint_save_dir="/home/young/Downloads/test-master/update_CNN/checkpoint" \
# --checkpoint_file_path="./checkpoint/Direct_Reflectance_Prediction_Net.pth" \
# 2>&1 | tee -a e.l && code e.l

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
import glob
import cv2
import natsort 
from PIL import Image
from skimage.transform import resize
from random import shuffle
import scipy.misc
import gc

# ======================================================================
if torch.cuda.is_available():
    device="cuda:0"
else:
    device="cpu"

# device=torch.device(device)
device=torch.device("cuda:0")

# ======================================================================
network_dir="../networks"
sys.path.insert(0,network_dir)
import networks as networks

loss_function_dir="../loss_functions"
sys.path.insert(0,loss_function_dir)
import loss_functions as loss_functions

utils_dir="../utils"
sys.path.insert(0,utils_dir)
import utils_common as utils_common
import utils_image as utils_image
import utils_net as utils_net
import utils_paper as utils_paper

# ======================================================================
def train():
    batch_size=int(args.batch_size)
    epoch=int(args.epoch)

    if args.load_data_from_text_file=="True":
        # ======================================================================
        # Data load by bringing images from text files which contain paths
        # c iiw_tr_img_p: iiw train images path
        iiw_tr_img_p=args.iiw_dataset_img_text_file
        # c iiw_gt_json_p: iiw ground truth images path
        iiw_gt_json_p=args.iiw_dataset_gt_text_file

        # --------------------------------------------------
        iiw_tr_p,iiw_tr_num=utils_common.return_path_list_from_txt(iiw_tr_img_p)
        iiw_gt_p,iiw_gt_num=utils_common.return_path_list_from_txt(iiw_gt_json_p)

        num_imgs=len(tr_img_li)
        num_imgs_float=float(num_imgs)

        # --------------------------------------------------
        # Shuffle list
        # c iiw_d_list: iiw dataset list
        iiw_d_list=list(zip(iiw_tr_p,iiw_gt_p))
        shuffle(iiw_d_list)
        # print("iiw_d_list",iiw_d_list)

        # --------------------------------------------------
        print("Current number of images:",num_imgs)
        print("Current batch size:",args.batch_size)
        print("Possible batch size:",list(utils_common.divisorGenerator(num_imgs)))
        assert str(num_imgs/batch_size).split(".")[-1]==str(0),"Check batch size"

    else:
        # ======================================================================
        # Data load by bringing images from directories
        # c tr_img_p: train images path
        tr_img_p=args.iiw_dataset_dir+"/*.png"
        # c gt_json_p: ground truth images path
        gt_json_p=args.iiw_dataset_dir+"/*.json"

        # --------------------------------------------------
        # c tr_img_li: train images list
        tr_img_li=utils_common.get_file_list(tr_img_p)
        # c gt_json_li: ground truth json files list
        gt_json_li=utils_common.get_file_list(gt_json_p)

        num_imgs=len(tr_img_li)
        num_imgs_float=float(num_imgs)

        # --------------------------------------------------
        # c d_list: dataset list
        d_list=list(zip(tr_img_li,gt_json_li))
        shuffle(d_list)

        # --------------------------------------------------
        print("Current number of images:",num_imgs)
        print("Current batch size:",args.batch_size)
        print("Possible batch size:",list(utils_common.divisorGenerator(num_imgs)))
        assert str(num_imgs/batch_size).split(".")[-1]==str(0),"Check batch size"

    # ======================================================================
    if args.continue_training==True:
        # c gen_net: generated network
        gen_net,optimizer=utils_net.net_generator(batch_size,args)
    else:
        gen_net,optimizer=utils_net.net_generator(batch_size,args)

    # ======================================================================
    # List for loss visualization
    loss_temp=[]
    lo_list=[
        ("loss",loss_temp)]
    num_lo=len(lo_list)

    # ======================================================================
    if args.train_mode=="True":
        # Iterates all epochs
        for one_ep in range(epoch):
            # Iterates all train images
            for bs in range(0,len(d_list)-batch_size+1,batch_size):
                # c bs_pa: batch size paths
                bs_pa=d_list[bs:bs+batch_size]
                # print("bs_pa",bs_pa)
                
                # --------------------------------------------------
                # c ori_ph: original image placeholder
                ori_ph=[]
                # c tr_ph: train image placeholder
                tr_ph=[]
                # c json_ph: json file placeholder
                json_ph=[]

                for one_file in bs_pa:
                    # c one_tr_img_p: one train image path
                    one_tr_img_p=one_file[0]
                    # c one_json_gt_p: one json gt path
                    one_json_gt_p=one_file[1]

                    # --------------------------------------------------
                    # c o_l_tr_i: one loaded train image
                    o_l_tr_i=utils_image.load_img(one_tr_img_p)/255.0
                    # c o_l_r_tr_i: one loaded resized train image
                    o_l_r_tr_i=utils_image.resize_img(o_l_tr_i,256,256)

                    ori_ph.append(o_l_tr_i)
                    tr_ph.append(o_l_r_tr_i)
                    json_ph.append(one_json_gt_p)

                # c tr_imgs: train images
                tr_imgs=np.stack(tr_ph,axis=0).transpose(0,3,1,2)
                
                # --------------------------------------------------
                # c O_img_tc: original image in pytorch tensor
                O_img_tc=Variable(torch.Tensor(tr_imgs).cuda())
                
                # --------------------------------------------------
                # Initialize all gradients of Variables as zero
                optimizer.zero_grad()

                # c pred_i_img: predicted intensity image
                pred_i_img_direct=gen_net(O_img_tc)

                # --------------------------------------------------
                # c resized_p_img_li: resized predicted images list
                resized_p_img_li=[]
                for one_img in range(pred_i_img_direct.shape[0]):
                    one_ori=ori_ph[one_img]
                    one_pred=pred_i_img_direct[one_img,:,:,:].unsqueeze(1)
                    one_pred=torch.nn.functional.interpolate(
                        one_pred,size=(one_ori.shape[0],one_ori.shape[1]),
                        scale_factor=None,mode='bilinear',align_corners=True)
                    resized_p_img_li.append(one_pred)
                
                # ======================================================================
                # c en_i_loss: entire image of this batch loss
                en_i_loss=0.0
                for one_file_loss in range(len(resized_p_img_li)):
                    # c o_ori_img: one original image
                    o_ori_img=ori_ph[one_file_loss]
                    
                    # c o_gt: one gt file
                    o_gt=json_ph[one_file_loss]
                    
                    # c o_pred_in: one predicted intensity image
                    o_pred_in=resized_p_img_li[one_file_loss]

                    # c hrjf: human relative reflectance judgement file
                    hrjf=loss_functions.HumanReflectanceJudgements.from_file(o_gt)

                    # o_i_loss=loss_functions.Direct_Intrinsic_Net_Loss_F.forward(o_p_ref,hrjf,ori_img_s)
                    o_i_loss=loss_functions.SVM_hinge_loss(o_pred_in,hrjf,o_ori_img)

                    en_i_loss+=o_i_loss

                # ======================================================================
                en_i_loss_div=en_i_loss/num_imgs_float
                # Update network based on loss
                loss_temp.append(en_i_loss_div)
                # Backpropagation based on loss
                en_i_loss.backward()
                # Update network based on backpropagation
                optimizer.step()

            # ======================================================================
            # Save trained parameters at every epoch
            checkpoint_path=args.checkpoint_save_dir+"/Direct_Reflectance_Prediction_Net_"+str(one_ep)+".pth"
            utils_net.save_checkpoint(
                {'state_dict':gen_net.state_dict(),
                 'optimizer':optimizer.state_dict()}, 
                checkpoint_path)

            # ======================================================================
            # scheduler.step()

        # ======================================================================
        # Loss visualization
        for i,tup in enumerate(lo_list):
            plt.subplot(num_lo,1,i+1)
            plt.title(tup[0])
            plt.plot(tup[1])
        plt.savefig("loss.png")
        plt.show()

    # ======================================================================
    # Test trained model
    else:
        with torch.no_grad():
            tr_img_p="/mnt/1T-5e7/mycodehtml/data_col/cv/IID_f_w_w/iiw-dataset/data/temp/*.png"
            tr_img_li=utils_common.get_file_list(tr_img_p)
            iteration=int(len(tr_img_li))
            
            # Iterates all train images
            for itr in range(iteration):
                fn=tr_img_li[itr].split("/")[-1].split(".")[0]

                o_l_tr_i_0_255=utils_image.load_img(tr_img_li[itr])
                o_l_tr_i=utils_image.load_img(tr_img_li[itr])/255.0
                ori_h=o_l_tr_i.shape[0]
                ori_w=o_l_tr_i.shape[1]
                o_l_r_tr_i=utils_image.resize_img(o_l_tr_i,256,256)

                # ======================================================================
                o_l_r_tr_i=o_l_r_tr_i.transpose(2,0,1)
                o_l_r_tr_i=o_l_r_tr_i[np.newaxis,:,:,:]
                con_O_img_arr=torch.Tensor(o_l_r_tr_i).cuda()
                pred_inten_imgs=gen_net(con_O_img_arr)
                
                for one_pred_img in range(pred_inten_imgs.shape[0]):
                    # --------------------------------------------------
                    # c pred_one_inten_img: prediction one intensity image
                    pred_one_inten_img=pred_inten_imgs[one_pred_img,:,:,:].unsqueeze(0)
                    pred_one_inten_img=torch.nn.functional.interpolate(
                        pred_one_inten_img,size=(ori_h,ori_w),scale_factor=None,mode='bilinear',align_corners=True)
                    pred_one_inten_img=pred_one_inten_img.squeeze().detach().cpu().numpy()
                    one_ori_img=o_l_tr_i_0_255/255.0
                    # print("one_ori_img",one_ori_img.shape)
                    # one_ori_img (720, 1280, 3)

                    # --------------------------------------------------
                    ref,sha=utils_image.colorize(pred_one_inten_img,one_ori_img)
                    
                    # ref=np.clip(ref,0.0,1.2)
                    # ref=cv2.normalize(ref,None,alpha=0,beta=1,norm_type=cv2.NORM_MINMAX,dtype=cv2.CV_32F)
                    ref=utils_image.rgb_to_srgb(ref)
                    ref[np.abs(ref)<0.0001]=0.0001
                    sha=one_ori_img/ref
                    sha=np.clip(sha,0.0,1.3)[:,:,0]
                    # sha=cv2.normalize(sha,None,alpha=0,beta=1,norm_type=cv2.NORM_MINMAX,dtype=cv2.CV_32F)

                    print("np.min(pred_one_inten_img)",np.min(ref))
                    print("np.max(pred_one_inten_img)",np.max(ref))
                    print("np.min(pred_one_inten_img)",np.min(sha))
                    print("np.max(pred_one_inten_img)",np.max(sha))

                    # --------------------------------------------------
                    scipy.misc.imsave('./result/'+fn+'_raw_intensity.png',pred_one_inten_img)
                    scipy.misc.imsave('./result/'+fn+'_ref.png',ref)
                    scipy.misc.imsave('./result/'+fn+'_sha.png',sha)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="This parses arguments and then runs appropriate mode based on arguments")

    # --------------------------------------------------
    parser.add_argument(
        "--train_mode",
        default=True)
    
    parser.add_argument(
        "--load_data_from_text_file",
        default=True)

    parser.add_argument(
        "--epoch",
        default=2)
    
    parser.add_argument(
        "--batch_size",
        default=2,)

    parser.add_argument(
        "--continue_training",
        default=False,)

    parser.add_argument(
        "--iiw_dataset_dir")
    
    parser.add_argument(
        "--iiw_dataset_text_file")
    
    parser.add_argument(
        "--checkpoint_save_dir")
    
    parser.add_argument(
        "--checkpoint_file_path")
    
    parser.add_argument(
        "--iiw_dataset_img_text_file")
    
    parser.add_argument(
        "--iiw_dataset_gt_text_file")
    
    parser.add_argument(
        "--project_dir")

    # --------------------------------------------------
    # args = parser.parse_args(sys.argv[1:])
    args=parser.parse_args()
    
    global global_args
    global_args=args

    train()
