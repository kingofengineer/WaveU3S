# WaveU3S

## Introduction

![framework](./figure/image.jpg) 

  **This repo is code of WaveU3S: A Lightweight Wavelet Dual-attention Unet For 3D Medical Image Segmentatio (ISBI2024).**

WaveU3S is a segmentation network for 3D medical image. Its purpose is to reducing the computation burden while maintaining the performance under 3D segmentation tasks. We use nnUNet as default data preprocessing method and training framework. Comparative experiments are conducted on Flare22 and ACDC datasets. For more details, please refer to our paper.

## Start
Our models are built based on [nnUNet V2](https://github.com/MIC-DKFZ/nnUNet )
### Clone repository
```shell
git clone git@github.com:kingofengineer/WaveU3S.git
cd WaveU3S/
pip install -e .
```
### Requirement
Install requirements.
```
pip install -r requirements.txt 
```
### Dataset and data processing
You can obtain the datasets via following links：
* Abdominal multi organ dataset: [Flare22](https://flare22.grand-challenge.org/Dataset/) 
* Multi-category cardiac MRI dataset: [ACDC](https://humanheart-project.creatis.insa-lyon.fr/database/#collection/637218c173e9f0047faa00fb) 


Please refer to nnUNetv2 for data storage rules and data preprocessing
Modify the patch size of the configuration file ```nnUNetPlans.json ``` by using ```modify_Plans.py``` 

