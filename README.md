# Vessel correspondence in pre-post-intervention DSA images of ischemic stroke patients

- [Quick Start](https://github.com/maberrospi/DSA-GraphMatching?tab=readme-ov-file#quick-start)
- [Description](https://github.com/maberrospi/DSA-GraphMatching?tab=readme-ov-file#description)
- [Usage](https://github.com/maberrospi/DSA-GraphMatching?tab=readme-ov-file#usage)
  - [Data Preparation (Optional)](https://github.com/maberrospi/DSA-GraphMatching?tab=readme-ov-file#data-preparation)
  - [RB-VM Method](https://github.com/maberrospi/DSA-GraphMatching?tab=readme-ov-file#run-rb-vm-method)
  - [GB-VM Methods](https://github.com/maberrospi/DSA-GraphMatching?tab=readme-ov-file#run-gb-vm-methods).

## Quick Start
```
# Clone repository
git clone https://gitlab.com/radiology/igit/icai-stroke-lab/msc-students/dsa-graphmatching.git

# Install the 'enviornment_nobuilds.yml' file
$ conda env create -f environment_nobuilds.yml

# Activate environment
$ conda activate DSA-GM
```

## Description
This code implements two methods to identify arterial vessel correspondence in pre- and post-EVT DSA images. 
They are Registration Based Vessel Matching (RB-VM) and Graph Based Vessel Matching (GB-VM).
The code for RB-VM is provided in 'sift_based.py' and the code for GB-VM is in 'Pipeline.py'

## Usage
Note: The methods were implemented using Python 3.12.2

### Data Preparation (Optional)
If using data from the MR CLEAN Registry, it is possible to create Nifti files and MinIP images from the
original DICOM data. We used these during development:

```
# Run the script to prepare the data
$ python prepareData.py -i dicom_directory -c csv_directory
```

Segmentations and UNet feature maps (if applicable) can also be generated and loaded afterwards
instead of calculating them on fly during matching:

```
python .\Segmentation\predict.py -h
usage: predict.py [-h] [--input-type INPUT_TYPE] [--input-format INPUT_FORMAT] [--label-type LABEL_TYPE] [--rnn RNN] [--rnn_kernel RNN_KERNEL] [--rnn_layers RNN_LAYERS] [--img_size IMG_SIZE]
                  [--amp] [--save_ft_maps]
                  in_img_path out_img_path ft_out_img_path model

example: python Segmentation/predict.py nifti_dir Segms/Sequence/ FeatMaps model.pt -i sequence -f nifti -t av --amp --save_ft_maps
This would create venous and arterial segmentations for the nifti files contained in nifti_dir and save them to Segms/Sequence
and save the extracted feature maps in FeatMaps directory.

Create segmentations and feature maps (if applicable)

positional arguments:
  in_img_path           Input images to be segmented.
  out_img_path          Directory to save segmentation result images.
  ft_out_img_path       Directory to save feature maps.
  model                 Load model from a .pth file

options:
  -h, --help            show this help message and exit
  --input-type INPUT_TYPE, -i INPUT_TYPE
                        Model input - minip or sequence.
  --input-format INPUT_FORMAT, -f INPUT_FORMAT
                        Input format - dicom or nifti
  --label-type LABEL_TYPE, -t LABEL_TYPE
                        Label type - vessel (binary) or av (4 classes).
  --rnn RNN, -r RNN     RNN type: convGRU or convLSTM.
  --rnn_kernel RNN_KERNEL, -k RNN_KERNEL
                        RNN kernel: 1 (1x1) or 3 (3x3).
  --rnn_layers RNN_LAYERS, -n RNN_LAYERS
                        Number of RNN layers.
  --img_size IMG_SIZE, -s IMG_SIZE
                        Targe image size for resizing images
  --amp                 Use mixed precision.
  --save_ft_maps        Save the feature maps.


```


### Run RB-VM method
```
usage: sift_baseline.py [-h] [--in_img_path IN_IMG_PATH] [--in_segm_path IN_SEGM_PATH] [--in_pre_path IN_PRE_PATH] [--in_post_path IN_POST_PATH] [--load-segs] [--pixel-wise] [--eval]

Find correspondeces using SIFT on a set of pre/post-EVT DSA images

options:
  -h, --help            show this help message and exit
  --in_img_path IN_IMG_PATH, -i IN_IMG_PATH
                        Directory of pre-post DSA sequences if data was prepared.
  --in_segm_path IN_SEGM_PATH, -is IN_SEGM_PATH
                        Directory of pre-post DSA segmentations if data was prepared.
  --in_pre_path IN_PRE_PATH, -pre IN_PRE_PATH
                        Path of pre-DSA sequence.
  --in_post_path IN_POST_PATH, -post IN_POST_PATH
                        Path of post-DSA sequence.
  --input-format INPUT_FORMAT, -f INPUT_FORMAT
                        Input format - dicom or nifti
  --load-segs           Load the segmentations.
  --pixel-wise          Use the pixel wise method for matching.
  --eval                Evaluate the method.

example 1 uses prepared data generated from Data Preparation
example1: python sift_baseline.py -i Niftisv2/R0002/0 -is Segms/Sequence --pixel-wise --load-segs -f nifti
This generates matching vessels  using the pixel wise method (reccommended)
and loading Nifti files and segmentations created via Data Preparation

example 2 uses pre- and post-EVT DSA series and calculates the segmentations on the fly.
example 2: python sift_baseline.py -pre Niftis/pre_evt.nii -post Niftis/post_evt.nii  --pixel-wise -f nifti
This generates matching vessels  using the pixel wise method (reccommended)
and loading arbitrary Nifti files and creating segmentations on the fly.
Equivalent dicom code:
python sift_baseline.py -pre Dicoms/pre_evt.dcm -post Dicoms/post_evt.dcm --pixel-wise -f dicom

```

### Run GB-VM method
```
usage: Pipeline.py [-h] [--in_img_path IN_IMG_PATH] [--in_segm_path IN_SEGM_PATH] [--in_ftmap_path IN_FTMAP_PATH] [--in_pre_path IN_PRE_PATH] [--in_post_path IN_POST_PATH] [--input-format INPUT_FORMAT]
                   [--match-type MATCH_TYPE] [--ftmap_type FTMAP_TYPE]

Find correspondences using graph matching on a set of pre/post-EVT DSA images

options:
  -h, --help            show this help message and exit
  --in_img_path IN_IMG_PATH, -i IN_IMG_PATH
                        Directory of pre-post DSA sequences if data was prepared.
  --in_segm_path IN_SEGM_PATH, -is IN_SEGM_PATH
                        Directory of pre-post DSA segmentations if data was prepared.
  --in_ftmap_path IN_FTMAP_PATH, -if IN_FTMAP_PATH
                        Directory of pre-post DSA UNet feature maps if data was prepared.
  --in_pre_path IN_PRE_PATH, -pre IN_PRE_PATH
                        Path of pre-DSA sequence.
  --in_post_path IN_POST_PATH, -post IN_POST_PATH
                        Path of post-DSA sequence.
  --input-format INPUT_FORMAT, -f INPUT_FORMAT
                        Input format - dicom or nifti
  --match-type MATCH_TYPE, -t MATCH_TYPE
                        Type of match to perform - single | multi | patched
  --ftmap_type FTMAP_TYPE, -fmt FTMAP_TYPE
                        Type of feature maps to use - unet | lm

example 1: use prepared data (Niftis, segmentations and UNet feature maps) to generate matches
python Pipeline.py -i Niftisv2/R0002/0 -is Segms/Sequence -if FeatMapsv2 -t patched -fmt unet
Alternative for using LM filter bank feature maps:
python Pipeline.py -i Niftisv2/R0002/0 -is Segms/Sequence -t patched -fmt lm

example 2: uses pre- and post-EVT DSA series and calculates the segmentations and Unet feature maps (if applicable) on the fly.
python Pipeline.py -pre Niftis/pre_evt.nii -post Niftis/post_evt.nii -t patched -f nifti -fmt lm
Equivalent dicom code:
python Pipeline.py -pre Dicoms/pre_evt.dcm -post Dicoms/post_evt.dcm -t patched -f dicom -fmt lm
These examples use LM feature maps and the 'patched' version of the methods.

```
