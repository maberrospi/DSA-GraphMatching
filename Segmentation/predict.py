import argparse
import logging
import sys

# print(sys.path)
from pathlib import Path
import os
import nibabel as nib
import numpy as np
import torch
from torchinfo import summary
from PIL import Image
import matplotlib.pyplot as plt
from skimage.transform import resize
import pydicom
from unet import UNet, TemporalUNet, ConvLSTM, ConvGRU
from glob import glob
import pandas as pd
from scipy.interpolate import interp1d
import time


def cut_seq(seq, max_len):
    if seq.shape[0] > max_len:
        if np.sum(seq[0, ...]) >= np.sum(seq[-1, ...]):
            seq = seq[1:]
        else:
            seq = seq[:-1]
        seq = cut_seq(seq, max_len=max_len)
    return seq


def load_and_preprocess_dicom(img_path):
    ds = pydicom.dcmread(
        img_path, defer_size="1 KB", stop_before_pixels=False, force=False
    )
    assert (
        2 ** (ds.BitsStored - 1) < ds.pixel_array.max() < 2**ds.BitsStored
    ), "Error: bits stored: {}, pixel value max: {}".format(
        ds.BitsStored, ds.pixel_array.max()
    )

    cum_time_vector = None
    if ("FrameTimeVector" in ds) and (ds.FrameTimeVector is not None):
        if len(ds.FrameTimeVector) != ds.NumberOfFrames:
            print(
                "Warning: number of Frames ({}) does not match frame time vector length ({}): {}"
                "".format(
                    ds.NumberOfFrames, len(ds.FrameTimeVector), ds.FrameTimeVector
                )
            )
            ds.FrameTimeVector = ds.FrameTimeVector[: ds.NumberOfFrames]
        cum_time_vector = np.cumsum(ds.FrameTimeVector)
    elif "FrameTime" in ds:
        cum_time_vector = int(ds.FrameTime) * np.array(range(ds.NumberOfFrames))
    else:
        print("Error: missing time info: {}".format(img_path))

    seq = ds.pixel_array
    if seq.ndim == 2:
        seq = seq.reshape((1, seq.shape[0], seq.shape[1]))
    elif cum_time_vector is not None:
        non_duplicated_frame_indices = np.where(
            ~pd.DataFrame(cum_time_vector).duplicated()
        )
        cum_time_vector = cum_time_vector[non_duplicated_frame_indices]
        seq = ds.pixel_array[non_duplicated_frame_indices]
        # cum_time_vector = [e for i, e in enumerate(cum_time_vector) if i not in duplicated_frame_indices]
        # remove the first frame as it is most likely a non-contrast frame or an un-subtracted frame
        if seq.shape[0] > 2:
            cum_time_vector, seq = cum_time_vector[1:], seq[1:]
            desired_frame_interval = 1000  # ms
            if (
                patient_id != "R0365"
            ):  # The frame time info in dicom header of this patient is wrong.
                interp = interp1d(cum_time_vector, seq, axis=0)
                seq = interp(
                    np.arange(
                        cum_time_vector[0], cum_time_vector[-1], desired_frame_interval
                    )
                )

    MAX_LEN = 20  # Shorten unnecessarily long sequences.
    if seq.shape[0] > MAX_LEN:
        print(
            "Warning: sequence is unnecessarily long ({}), "
            "cutting it to {} frames based on minimum contrast.".format(
                seq.shape[0], MAX_LEN
            )
        )
    seq = cut_seq(seq, max_len=MAX_LEN)

    seq = np.transpose(
        255 * (seq.astype(np.float32) / (2**ds.BitsStored - 1)), (1, 2, 0)
    )

    return seq


def load_image(img_path, img_size, img_type):
    if ".nii" in img_path:
        img_obj = nib.load(img_path)
        img = np.transpose(img_obj.get_fdata(), (1, 0, 2))
    elif ".dcm" in img_path:
        img = load_and_preprocess_dicom(img_path)
    else:
        img = np.asarray(Image.open(img_path))

    newW, newH = img_size, img_size
    assert (
        newW > 0 and newH > 0
    ), "Image size is too small, resized images would have no pixel"

    if img.ndim == 2:
        img = img[np.newaxis, ...]
        img = resize(
            img, (img.shape[0], newW, newH), anti_aliasing=False, preserve_range=True
        )
    else:
        img = np.transpose(img, (2, 0, 1))
        img = resize(
            img, (img.shape[0], newW, newH), anti_aliasing=False, preserve_range=True
        )
        img = img[:, np.newaxis, ...]
    img = img / 255

    if img_type == "minip":
        img = np.min(img, axis=0)

    return img


def predict(net, img, out_img_path, device="cuda"):
    net.eval()
    img = (
        torch.as_tensor(img.copy())
        .float()
        .contiguous()
        .to(device=device, dtype=torch.float32)
    )
    img = torch.unsqueeze(img, 0)
    with torch.no_grad():
        masks_pred = net(img)
        mask_pred = Image.fromarray(
            masks_pred.argmax(dim=1)[0].cpu().detach().numpy().astype(np.uint8)
        )
        Path(out_img_path).parent.mkdir(parents=True, exist_ok=True)
        mask_pred.save(out_img_path)


def segment(net, img, device="cuda"):
    net.eval()
    img = (
        torch.as_tensor(img.copy())
        .float()
        .contiguous()
        .to(device=device, dtype=torch.float32)
    )
    img = torch.unsqueeze(img, 0)
    with torch.no_grad():
        t = time.time()
        masks_pred = net(img)
        elapsed = time.time() - t
        masks_pred = masks_pred > 0.5
        mask_pred = masks_pred[0].cpu().detach().numpy().astype(np.uint8)
    return mask_pred, elapsed


# fmt: off
def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('in_img_path', help='Input image to be segmented.')
    parser.add_argument('out_img_path', default='./out.png', help='Segmentation result image.')
    parser.add_argument('model', type=str, help='Load model from a .pth file')
    parser.add_argument('--input-type', '-i', default='minip', help='Model input - minip or sequence.')
    parser.add_argument('--input-format','-f',default='dicom',help='Input format - dicom or nifti')
    parser.add_argument('--label-type', '-t', default='vessel', help='Label type - vessel (binary) or av (4 classes).')
    parser.add_argument('--rnn', '-r', type=str, default='ConvGRU', help='RNN type: convGRU or convLSTM.')
    parser.add_argument('--rnn_kernel', '-k', type=int, default=1, help='RNN kernel: 1 (1x1) or 3 (3x3).')
    parser.add_argument('--rnn_layers', '-n', type=int, default=2, help='Number of RNN layers.')
    parser.add_argument('--img_size', '-s', type=float, default=512, help='Targe image size for resizing images')
    parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision.')

    return parser.parse_args()
# fmt: on


def get_activation(name, activation):
    # Function to get activation of a specific layer during the forward pass
    # The hook signature
    def hook(model, input, output):
        activation[name] = output.detach()

    return hook


def display_feature_maps(sequence: list):
    # Plot the feature maps of one seq for reference
    sq = 8
    fig, axs = plt.subplots(sq, sq, figsize=(16, 16))
    col = 0
    for idx in range(sequence[0].shape[0]):
        if idx % sq == 0:
            col += 1
        axs[col - 1, (idx % sq)].imshow(sequence[0][idx])
        axs[col - 1, (idx % sq)].set_xticks([])
        axs[col - 1, (idx % sq)].set_yticks([])
    plt.show()


def run_predict(
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
    log_filepath = "log/{}.log".format(Path(__file__).stem)
    logging.basicConfig(
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
        format="%(asctime)s %(levelname)-8s %(message)s",
        handlers=[
            logging.FileHandler(log_filepath, mode="w"),
            logging.StreamHandler(sys.stdout),
        ],
    )

    # Equivalent CLI code for running as a module

    # Debugging args
    # print(locals())

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

    # summary(model=net, input_size=(1, 20, 1, 512, 512))  # B,T,C,H,W

    """Segmentation"""
    # test_img = load_image(in_img_path, img_size, img_type=input_type)
    # predict(net, test_img, out_img_path, device=device)

    if input_format == "nifti":
        dcm_fps = sorted(glob(os.path.join(in_img_path, "**", "*.nii"), recursive=True))
    elif input_format == "dicom":
        dcm_fps = sorted(glob(os.path.join(in_img_path, "**", "*.dcm"), recursive=True))
    # df_patients = pd.read_csv("/mnt/data2/Dropbox/Ruisheng/PhD/MyManuscripts/CTA-DSA mapping/dsa_removal_annotation_Sijie.csv")
    # df_patients = df_patients[df_patients['operation'].isna()]
    # df_patients = df_patients[df_patients['operation'] == "temp"]
    # patient_ids = df_patients['patient_id'].unique()
    elapsed_per_frame_list = []
    elapsed_per_sequence_list = []
    up4_list = []
    activation = {}
    # Register forward hooks on the layer(s) of choice
    h1 = net.up4.register_forward_hook(get_activation("up4", activation))

    global patient_id
    for idx, fp in enumerate(dcm_fps):
        patient_id = Path(fp).parent.name
        # if patient_id not in patient_ids:
        #     continue
        logging.info(f"{idx+1}/{len(dcm_fps)}, segmenting: {fp}")
        test_img = load_image(fp, img_size, img_type=input_type)
        if input_format == "nifti":
            out_img_path = fp.replace(in_img_path, orig_out_img_path).replace(
                ".nii", ".png"
            )
        elif input_format == "dicom":
            out_img_path = fp.replace(in_img_path, orig_out_img_path).replace(
                ".dcm", ".png"
            )
        Path(out_img_path).parent.mkdir(parents=True, exist_ok=True)

        # Run the forward pass (prediction)
        out_seg, elapsed = segment(net, test_img, device=device)
        elapsed_per_frame_list.append(elapsed / len(test_img))
        elapsed_per_sequence_list.append(elapsed)
        if label_type == "av":
            out_artery_img_path = out_img_path.replace(".png", "_artery.png")
            out_vein_img_path = out_img_path.replace(".png", "_vein.png")
            Image.fromarray(out_seg[0]).save(out_artery_img_path)
            Image.fromarray(out_seg[1]).save(out_vein_img_path)
        else:
            Image.fromarray(out_seg[0]).save(out_img_path)

        # Collect the activations for all patient dicoms
        up4_list.append(activation["up4"].squeeze().numpy())
        # print(activation["up4"].squeeze().shape)

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

    return up4_list


if __name__ == "__main__":
    log_filepath = "log/{}.log".format(Path(__file__).stem)
    logging.basicConfig(
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
        format="%(asctime)s %(levelname)-8s %(message)s",
        handlers=[
            logging.FileHandler(log_filepath, mode="w"),
            logging.StreamHandler(sys.stdout),
        ],
    )

    """Global settings"""
    # Inspiration for passing args:
    # https://www.peterbe.com/plog/vars-argparse-namespace-into-a-function
    args = get_args()
    run_predict(**vars(args))
