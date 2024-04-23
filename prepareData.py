import logging
import glob
import os
import sys
from pathlib import Path

import cv2
import numpy as np
from pydicom import dcmread
import pandas as pd
import nibabel as nib
import imageio
from pydicom import dcmread

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
    patient_folders = glob.glob(DICOM_DIR + "/*")
    logger.info("Converting DSA DICOM files to MinIP NIFTI1 files.")
    for folder in patient_folders:
        patient_dcms = glob.glob(os.path.join(folder, "*.dcm"))
        logger.info("Converting DSA DICOM files to NIFTI1 files.")
        comp_series = None
        counter = 0
        for idx, dicom in enumerate(patient_dcms):
            ds = dcmread(dicom, defer_size="1 KB", stop_before_pixels=False, force=True)
            patient_ID = ds["PatientID"].value
            series_num = ds["SeriesNumber"].value
            if comp_series != series_num and counter < 2:
                nii_dst_path = os.path.join(
                    NIFTI_DIR, patient_ID, str(0), "{}.nii".format(Path(dicom).stem)
                )
                minip_dst_path = os.path.join(
                    MINIP_DIR, patient_ID, str(0), "{}.png".format(Path(dicom).stem)
                )
                counter += 1
            else:
                nii_dst_path = os.path.join(
                    NIFTI_DIR, patient_ID, str(1), "{}.nii".format(Path(dicom).stem)
                )
                minip_dst_path = os.path.join(
                    MINIP_DIR, patient_ID, str(1), "{}.png".format(Path(dicom).stem)
                )
            if idx == 0:
                comp_series = series_num
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
                return
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
            img_minip = normalize(img_minip)
            Path(minip_dst_path).parent.mkdir(parents=True, exist_ok=True)
            logger.info("Saving minip to {}".format(minip_dst_path))
            imageio.imwrite(minip_dst_path, img_minip)

            # print(f"These are not duplicates:{non_duplicated_frame_indices}")
            # print(ds["FrameTimeVector"].value)
            # print(ds["FrameTime"].value)
            # print(ds["NumberOfFrames"].value)
            # print(ds["SeriesNumber"].value)
            # print(ds["PatientID"].value)


def main():
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

    DICOM_DIR = "E:/vessel_diff_first_50_patients/mrclean_part1_2_first_50"
    # patient_info = pd.read_csv(
    #     "E:/vessel_diff_first_50_patients/mrclean_part1_2.csv", nrows=174
    # )  # Small dataset (50 pat)
    # patient_ids = patient_info["patient_id"].unique().tolist()
    # patient_view = patient_info["view"].to_list()
    # print(patient_view[10])
    # print(f'Patient ID: {patient_ids[-1]} \nSeries: {patient_info['series_number'].to_list()[-1]}')
    # prepare_data(dicom_dir=DICOM_DIR)


if __name__ == "__main__":
    main()
