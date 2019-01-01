# PyTorch implementation of paper "Reflectance Adaptive Filtering Improves Intrinsic Image Estimation"
---

## Cite original paper authors
```
@inproceedings{nestmeyer2017reflectanceFiltering,
  title={Reflectance Adaptive Filtering Improves Intrinsic Image Estimation},
  author={Nestmeyer, Thomas and Gehler, Peter V},
  booktitle={Computer Vision and Pattern Recognition (CVPR)},
  year={2017}
}
```

## Original caffe implementation
https://github.com/tnestmeyer/reflectance-filtering

## Dependencies
Python 3.6  
PyTorch 0.4.0  
CUDA 0.8  
OpenCV3 and external contrib module for guided and bilateral filters  
Natsort  
Glob  
and so on.

## IIW dataset
http://opensurfaces.cs.cornell.edu/publications/intrinsic/

## Change project directories to import project modules 
1. Find all /home/young/Downloads/test-master/update_CNN in this project 
2. Replace all /home/young/Downloads/test-master/update_CNN with your path, 
for example, /home/user/your_dir

## Run example
```
python train.py \
--epoch=20 \
--batch_size=15 \
--train_mode=True \
--load_data_from_text_file=False \
--continue_training=False \
--iiw_dataset_dir="/mnt/1T-5e7/mycodehtml/data_col/cv/IID_f_w_w/iiw-dataset/data/temp" \
--iiw_dataset_img_text_file="/mnt/1T-5e7/image/whole_dataset/iiw_data_img.txt" \
--iiw_dataset_gt_text_file="/mnt/1T-5e7/image/whole_dataset/iiw_data_json.txt" \
--checkpoint_save_dir="/home/young/Downloads/test-master/update_CNN/checkpoint" \
--checkpoint_file_path="./checkpoint/Direct_Reflectance_Prediction_Net.pth"
```

## After getting predicted intensity image, you can apply bilater and guided filters in
utils_image.py

## Loss
Tnestmeyer loss
![alt text](https://github.com/youngminpark2559/ml_cv_p/blob/master/IID/reflectance_filtering/train/loss_one.png)

GCINTRINSICS loss
![alt text](https://github.com/youngminpark2559/ml_cv_p/blob/master/IID/reflectance_filtering/train/loss_cgintrinsic.png)

## Result
Prediction intensity image
/home/young/Downloads/test-master/update_CNN/result/54_000_raw_intensity.png
![alt text](https://github.com/youngminpark2559/ml_cv_p/blob/master/IID/reflectance_filtering/result/54_000_raw_intensity.png)

Apply bilater filter (sigmaColor=20, sigmaSpace=20) to above predicted raw intensity image
/home/young/Downloads/test-master/update_CNN/utils/54_000_raw_intensity_bi2020.png
![alt text](https://github.com/youngminpark2559/ml_cv_p/blob/master/IID/reflectance_filtering/utils/54_000_raw_intensity_bi2020.png)

Obtained predicted reflectance image by using 54_000_raw_intensity_bi2020.png and original image 54.png
/home/young/Downloads/test-master/update_CNN/utils/54_000_raw_intensity_bi2020_ref.png
![alt text](https://github.com/youngminpark2559/ml_cv_p/blob/master/IID/reflectance_filtering/utils/54_000_raw_intensity_bi2020_ref.png)

Obtained predicted shading image by using 54_000_raw_intensity_bi2020.png and original image 54.png
/home/young/Downloads/test-master/update_CNN/utils/54_000_raw_intensity_bi2020_sha.png
![alt text](https://github.com/youngminpark2559/ml_cv_p/blob/master/IID/reflectance_filtering/utils/54_000_raw_intensity_bi2020_sha.png)

Apply guided filter (radius=2, eps=2) to 54_000_raw_intensity_bi2020_ref.png
/home/young/Downloads/test-master/update_CNN/utils/54_000_raw_intensity_bi2020ref_guided22.png
![alt text](https://github.com/youngminpark2559/ml_cv_p/blob/master/IID/reflectance_filtering/utils/54_000_raw_intensity_bi2020ref_guided22.png)

Obtained predicted shading image by using 54_000_raw_intensity_bi2020ref_guided22.png and original image 54.png
/home/young/Downloads/test-master/update_CNN/utils/54_000_raw_intensity_bi2020sha_guided22.png
![alt text](https://github.com/youngminpark2559/ml_cv_p/blob/master/IID/reflectance_filtering/utils/54_000_raw_intensity_bi2020sha_guided22.png)
