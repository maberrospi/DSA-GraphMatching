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
usage: sift_baseline.py [-h] [--load-segs] [--pixel-wise] [--eval]

Find correspondeces using SIFT on a set of pre/post-EVT DSA images

options:
  -h, --help    show this help message and exit
  --load-segs   Load the segmentations.
  --pixel-wise  Use the pixel wise method for matching.
  --eval        Evaluate the method.

example: python sift_baseline.py --pixel-wise --load-segs
This generates matching vessels  using the pixel wise method (reccommended)
and loading segments if previously created.
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
