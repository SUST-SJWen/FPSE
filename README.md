## Title
Fully Perturbed Self-Ensemble Framework using Cascaded Parallel CNN-Transformer for Semi-Supervised Medical Image Segmentation

## Requirements
* [Pytorch]
* Some basic python packages such as Numpy, Scikit-image, SimpleITK, Scipy, Medpy ......

## Usage

1. Download the pre-processed data and put the data in `../data/ACDC`. In this project, we use the ACDC dataset for training demonstration, and the training process for the AbdomenCT-1K and ISIC2018 datasets is similar to that of ACDC. You can download the dataset with the list of labeled training, unlabeled training, validation, and testing slices as following:
ACDC from [Google Drive Link](https://drive.google.com/file/d/1F3JzBSIURtFJkfcExBcT6Hu7Ar5_f8uv/view?usp=sharing), or [Baidu Netdisk Link](https://pan.baidu.com/s/1LS6VHujD8kvuQikbydOibQ) with passcode: 'kafc'.

2. Train the model

```
cd code
```

You can choose model(unet/vnet/pnet...), dataset(ACDC/AbdomenCT-1K/ISIC2018), experiment name(the path of saving your model weights and inference), iteration number, batch size and etc in your command line, or leave it with default option.

FPSE 
```
python train_myHiFormer_and_Match_cross_pseudo_supervision_2D.py --root_path ../data/ACDC --exp ACDC/XXX --model XXX -max_iterations XXX -batch_size XXX --base_lr XXX --num_classes 4 --labeled_num XXX
```

3. Test the model
```
python test_2D_fully.py -root_path ../data/ACDC --exp ACDC/XXX -model XXX --num_classes 4 --labeled_num XXX
```
Check trained model and inference
```
cd model
```


## Acknowledgement

This code is mainly based on [SSL4MIS](https://github.com/HiLab-git/SSL4MIS).

Some of the other code is from [SegFormer](https://github.com/NVlabs/SegFormer), [SwinUNet](https://github.com/HuCaoFighting/Swin-Unet), [Segmentation Models](https://github.com/qubvel/segmentation_models.pytorch), [UAMT](https://github.com/yulequan/UA-MT), [nnUNet](https://github.com/MIC-DKFZ/nnUNet).