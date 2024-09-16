import logging
from glob import glob
import os
import sys
from pathlib import Path
import argparse

import cv2
import numpy as np
from pydicom import dcmread
import pandas as pd
import nibabel as nib
import imageio
from pydicom import dcmread

# Add Segmentation package path to sys path to fix importing unet
sys.path.insert(1, os.path.join(sys.path[0], "Segmentation"))
from Segmentation import predict
from Segmentation.unet import UNet, TemporalUNet, ConvLSTM, ConvGRU
import torch

logger = logging.getLogger(__name__)


def cut_seq(seq, max_len):
    if seq.shape[0] > max_len:
        if np.sum(seq[0, ...]) >= np.sum(seq[-1, ...]):
            seq = seq[1:]
        else:
            seq = seq[:-1]
        seq = cut_seq(seq, max_len=max_len)
    return seq


def normalize(img):
    """Outputs image of type unsigned int"""
    image_minip_norm = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
    return image_minip_norm.astype(np.uint8)


def prepare_data(dicom_dir="Dicoms", nifti_dir="Niftis", minip_dir="Minip"):
    DICOM_DIR = dicom_dir
    NIFTI_DIR = nifti_dir
    MINIP_DIR = minip_dir

    if not os.path.isdir(DICOM_DIR):
        logger.warning("Path directory {} does not exist".format(DICOM_DIR))
        sys.exit(1)

    patient_folders = glob(DICOM_DIR + "/*")
    logger.info("Converting DSA DICOM files to MinIP NIFTI1 files.")
    for folder in patient_folders:
        patient_dcms = glob(os.path.join(folder, "*.dcm"))
        logger.info("Converting DSA DICOM files to NIFTI1 files.")
        for dicom in patient_dcms:
            ds = dcmread(dicom, defer_size="1 KB", stop_before_pixels=False, force=True)
            patient_filename = Path(dicom).stem
            patient_ID = ds["PatientID"].value

            # Get the correct index for the csv file
            idx = patient_info[
                patient_info["filename"] == patient_filename
            ].index.item()

            # If AP is assigned to the 0 dir otherwise to 1
            if patient_view[idx] == "ap":
                dir_number = str(0)
            elif patient_view[idx] == "lateral":
                dir_number = str(1)

            # If it is the preEVT image append pre otherwise append post
            if patient_pre[idx]:
                nii_dst_path = os.path.join(
                    NIFTI_DIR,
                    patient_ID,
                    dir_number,
                    "{}.nii".format(Path(dicom).stem + "_pre"),
                )
                minip_dst_path = os.path.join(
                    MINIP_DIR,
                    patient_ID,
                    dir_number,
                    "{}.png".format(Path(dicom).stem + "_pre"),
                )
            else:
                nii_dst_path = os.path.join(
                    NIFTI_DIR,
                    patient_ID,
                    dir_number,
                    "{}.nii".format(Path(dicom).stem + "_post"),
                )
                minip_dst_path = os.path.join(
                    MINIP_DIR,
                    patient_ID,
                    dir_number,
                    "{}.png".format(Path(dicom).stem + "_post"),
                )

            Path(nii_dst_path).parent.mkdir(parents=True, exist_ok=True)
            # print(ds.pixel_array.shape)
            if "FrameTimeVector" in ds:
                if ds.FrameTimeVector is None:
                    logger.warning(
                        "Missing time info in Frame Time Vector of type ({}) and Number of frames ({}) -> Skipping patient."
                        "".format(
                            ds.FrameTimeVector,
                            ds.NumberOfFrames,
                        )
                    )
                    break
                if len(ds.FrameTimeVector) != ds.NumberOfFrames:
                    logger.warning(
                        "Number of Frames ({}) does not match frame time vector length ({}): {}"
                        "".format(
                            ds.NumberOfFrames,
                            len(ds.FrameTimeVector),
                            ds.FrameTimeVector,
                        )
                    )
                    ds.FrameTimeVector = ds.FrameTimeVector[: ds.NumberOfFrames]
                cum_time_vector = np.cumsum(ds.FrameTimeVector)
            elif "FrameTime" in ds:
                cum_time_vector = int(ds.FrameTime) * np.array(range(ds.NumberOfFrames))
            else:
                logger.error("Missing time info: {}".format(dicom))
                break
            non_duplicated_frame_indices = np.where(
                ~pd.DataFrame(cum_time_vector).duplicated()
            )
            cum_time_vector = cum_time_vector[non_duplicated_frame_indices]
            seq = ds.pixel_array[non_duplicated_frame_indices]
            # remove the first frame as it is most likely a non-contrast frame or an un-subtracted frame
            cum_time_vector, seq = cum_time_vector[1:], seq[1:]

            MAX_LEN = 20  # Shorten unnecessarily long sequences.
            if seq.shape[0] > MAX_LEN:
                logger.warning(
                    "Sequence is unnecessarily long, "
                    "cutting it to {} frames based on minimum contrast.".format(MAX_LEN)
                )
            seq = cut_seq(seq, max_len=MAX_LEN)

            seq = normalize(seq)
            seq = seq.transpose((2, 1, 0))
            nii_image = nib.Nifti1Image(seq, np.eye(4))
            logger.info("Saving NIFTI1 to {}".format(nii_dst_path))
            nib.save(nii_image, nii_dst_path)

            logger.info("Creating MinIP file.")

            seq = seq.transpose((2, 1, 0))

            # Prepare MinIP images
            img_minip = np.min(seq, axis=0)
            # img_minip = normalize(img_minip)  # I have a feeling this is wrong
            Path(minip_dst_path).parent.mkdir(parents=True, exist_ok=True)
            logger.info("Saving minip to {}".format(minip_dst_path))
            imageio.imwrite(minip_dst_path, img_minip)

            # print(f"These are not duplicates:{non_duplicated_frame_indices}")
            # print(ds["FrameTimeVector"].value)
            # print(ds["FrameTime"].value)
            # print(ds["NumberOfFrames"].value)
            # print(ds["SeriesNumber"].value)
            # print(ds["PatientID"].value)


def save_feature_maps(
    in_img_path,
    out_img_path,
    model,
    input_type="minip",
    input_format="dicom",
    label_type="vessel",
    rnn="ConvGRU",
    rnn_kernel=1,
    rnn_layers=2,
    img_size=512,
    amp=False,
):
    """Global settings"""
    assert input_type in ["minip", "sequence"], "Invalid input image type"
    assert input_format in ["dicom", "nifti"], "Invalid input format"
    assert label_type in ["vessel", "av"], "Invalid label type"
    n_classes = (1, 2)[label_type == "av"]
    orig_out_img_path = out_img_path
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device {device}")

    """Set up the network"""
    if input_type == "minip":
        net = UNet(n_channels=1, n_classes=n_classes, bilinear=True)
    else:
        rnn = (ConvGRU, ConvLSTM)[rnn == "ConvLSTM"]
        kernel_size = (rnn_kernel, rnn_kernel)
        net = TemporalUNet(
            rnn,
            n_channels=1,
            kernel_size=kernel_size,
            num_layers=rnn_layers,
            n_classes=n_classes,
            bilinear=True,
        )
    net.to(device=device)

    """Load trained model."""
    net.load_state_dict(torch.load(model, map_location=device))
    logging.info(f"Model loaded from {model}")

    """Segmentation"""
    # test_img = load_image(in_img_path, img_size, img_type=input_type)
    # predict(net, test_img, out_img_path, device=device)

    if input_format == "nifti":
        dcm_fps = sorted(glob(os.path.join(in_img_path, "**", "*.nii"), recursive=True))
    elif input_format == "dicom":
        dcm_fps = sorted(glob(os.path.join(in_img_path, "**", "*.dcm"), recursive=True))

    elapsed_per_frame_list = []
    elapsed_per_sequence_list = []
    activation = {}
    # Register forward hooks on the layer(s) of choice
    h1 = net.up4.register_forward_hook(predict.get_activation("up4", activation))

    global patient_id
    for idx, fp in enumerate(dcm_fps):
        patient_id = Path(fp).parent.name
        # if patient_id not in patient_ids:
        #     continue
        logging.info(f"{idx+1}/{len(dcm_fps)}, segmenting: {fp}")
        test_img = predict.load_image(fp, img_size, img_type=input_type)
        if input_format == "nifti":
            out_img_path = fp.replace(in_img_path, orig_out_img_path).replace(
                ".nii", ".npz"
            )
        elif input_format == "dicom":
            out_img_path = fp.replace(in_img_path, orig_out_img_path).replace(
                ".dcm", ".npz"
            )
        Path(out_img_path).parent.mkdir(parents=True, exist_ok=True)

        # Run the forward pass (prediction)
        out_seg, elapsed = predict.segment(net, test_img, device=device)
        elapsed_per_frame_list.append(elapsed / len(test_img))
        elapsed_per_sequence_list.append(elapsed)
        # Save the feature maps to an npz file
        # Basically you save 64x512x512 feature maps per file
        np.savez_compressed(out_img_path, feat_maps=activation["up4"].squeeze())

    # Detach the hooks
    h1.remove()

    del patient_id
    logging.info(
        "Average time per frame: {}\u00B1{}".format(
            np.mean(elapsed_per_frame_list), np.std(elapsed_per_frame_list)
        )
    )
    logging.info(
        "Average time per sequence: {}\u00B1{}".format(
            np.mean(elapsed_per_sequence_list), np.std(elapsed_per_sequence_list)
        )
    )
    logging.info("Done!")


def load_and_preprocess_dicom(dcm_path):
    ds = dcmread(dcm_path, defer_size="1 KB", stop_before_pixels=False, force=True)
    # This function is not used in this file
    if "FrameTimeVector" in ds:
        if ds.FrameTimeVector is None:
            logger.warning(
                "Missing time info in Frame Time Vector of type ({}) and Number of frames ({}) -> Skipping patient."
                "".format(
                    ds.FrameTimeVector,
                    ds.NumberOfFrames,
                )
            )

        if len(ds.FrameTimeVector) != ds.NumberOfFrames:
            logger.warning(
                "Number of Frames ({}) does not match frame time vector length ({}): {}"
                "".format(
                    ds.NumberOfFrames,
                    len(ds.FrameTimeVector),
                    ds.FrameTimeVector,
                )
            )
            ds.FrameTimeVector = ds.FrameTimeVector[: ds.NumberOfFrames]
        cum_time_vector = np.cumsum(ds.FrameTimeVector)
    elif "FrameTime" in ds:
        cum_time_vector = int(ds.FrameTime) * np.array(range(ds.NumberOfFrames))
    else:
        logger.error("Missing time info: {}".format(dcm_path))
        raise "Patient is missing time info: {}.".format(dcm_path)
    non_duplicated_frame_indices = np.where(~pd.DataFrame(cum_time_vector).duplicated())
    cum_time_vector = cum_time_vector[non_duplicated_frame_indices]
    seq = ds.pixel_array[non_duplicated_frame_indices]
    # remove the first frame as it is most likely a non-contrast frame or an un-subtracted frame
    cum_time_vector, seq = cum_time_vector[1:], seq[1:]

    MAX_LEN = 20  # Shorten unnecessarily long sequences.
    if seq.shape[0] > MAX_LEN:
        logger.warning(
            "Sequence is unnecessarily long, "
            "cutting it to {} frames based on minimum contrast.".format(MAX_LEN)
        )
    seq = cut_seq(seq, max_len=MAX_LEN)

    seq = normalize(seq)
    # seq = seq.transpose((2, 1, 0))
    logger.info("Creating MinIP file.")

    # seq = seq.transpose((2, 1, 0))

    # Return MinIP image
    return np.min(seq, axis=0)
    # img_minip = normalize(img_minip)  # I have a feeling this is wrong


# fmt: off
def get_args():
    parser = argparse.ArgumentParser(description="Prepare data from MR CLEAN Registry.")
    parser.add_argument('--dcm_dir_path','-i',default=None, help='Directory of MR CLEAN DSA sequences.')
    parser.add_argument('--csv_path','-c',default=None, help='Path to MR CLEAN CSV file.')
    parser.add_argument('--out_nifti_path','-on',default='Niftis', help='Output directory for Nifti files.')
    parser.add_argument('--out_minip_path','-od',default='Minips', help='Output directory for MinIP files.')
#fmt:on

    return parser.parse_args()

def main(dcm_dir_path,csv_path,out_nifti_path,out_minip_path):
    log_filepath = "log/{}.log".format(Path(__file__).stem)
    if not os.path.isdir("log"):
        os.mkdir("log")
    logging.basicConfig(
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
        format="%(asctime)s %(levelname)-8s %(message)s",
        handlers=[
            logging.FileHandler(log_filepath, mode="w"),
            logging.StreamHandler(sys.stdout),
        ],
    )

    # DICOM_DIR = "E:/vessel_diff_first_50_patients/mrclean_part1_2_first_50"
    # CSV_DIR = "E:/vessel_diff_first_50_patients/mrclean_part1_2.csv"
    DICOM_DIR = dcm_dir_path
    CSV_DIR = csv_path
    
    global patient_info
    try:
        patient_info = pd.read_csv(CSV_DIR, nrows=174)  # Small dataset (50 pat)
    except FileNotFoundError:
        logger.warning("Path directory {} does not exist".format(CSV_DIR))
        sys.exit(1)

    # patient_ids = patient_info["patient_id"].unique().tolist()
    global patient_view
    global patient_pre
    patient_view = patient_info["view"].to_list()
    patient_pre = patient_info["preEVT"].fillna(False).to_list()
    # print(patient_view[10])
    # print(f'Patient ID: {patient_ids[-1]} \nSeries: {patient_info['series_number'].to_list()[-1]}')
    prepare_data(dicom_dir=DICOM_DIR, minip_dir=out_minip_path, nifti_dir=out_nifti_path)

    NIFTI_DIR = out_nifti_path
    FEAT_MAP_DIR = "FeatMaps"
    # save_feature_maps(
    #     in_img_path=NIFTI_DIR,
    #     out_img_path=FEAT_MAP_DIR,
    #     model="C:/Users/mab03/Desktop/RuSegm/TemporalUNet/models/1096-sigmoid-sequence-av.pt",
    #     input_type="sequence",
    #     input_format="nifti",
    #     label_type="av",
    #     amp=True,
    # )


if __name__ == "__main__":
    # main()
    args = get_args()
    main(**vars(args))
