# WaveU3S

## Introduction

![framework](./figure/image.jpg) 

  **This repo is code of WaveU3S: A Lightweight Wavelet Dual-attention Unet For 3D Medical Image Segmentatio (ISBI2024).**

WaveU3S is a segmentation network for 3D medical image. Its purpose is to reducing the computation burden while maintaining the performance under 3D segmentation tasks. We use [nnUNet](https://github.com/MIC-DKFZ/nnUNet )as default data preprocessing method and training framework. Comparative experiments are conducted on [Flare22](https://flare22.grand-challenge.org/Dataset/) and [ACDC](https://humanheart-project.creatis.insa-lyon.fr/database/#collection/637218c173e9f0047faa00fb) datasets. For more details, please refer to our paper.

## start
Our models are built based on [nnUNet V2](https://github.com/MIC-DKFZ/nnUNet )
### Clone repository
```shell
git clone git@github.com:kingofengineer/WaveU3S.git
cd WaveU3S/
pip install -e .
```
### Requirement

