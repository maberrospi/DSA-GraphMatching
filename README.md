# Vessel correspondence in pre-post-intervention DSA images of ischemic stroke patients

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
Equivalent dicom code: python sift_baseline.py -pre Dicoms/pre_evt.dcm -post Dicoms/post_evt.dcm --pixel-wise -f dicom

```

### Run GB-VM method
```
usage: Pipeline.py [-h] [--match-type MATCH_TYPE] [--load-segs] [--load-ft-maps]

Find correspondences using graph matching on a set of pre/post-EVT DSA images

options:
  -h, --help            show this help message and exit
  --match-type MATCH_TYPE, -t MATCH_TYPE
                        Type of match to perform - single | multi | patched
  --load-segs           Load the segmentations and feature maps.
  --load-ft-maps        Load the feature maps corresponding to the segmentations.
```
