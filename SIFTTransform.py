import logging
import glob
import os

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pydicom import dcmread
from matplotlib import colors

# Inspiration: https://www.sicara.fr/blog-technique/2019-07-16-image-registration-deep-learning
#              https://docs.opencv.org/4.x/da/df5/tutorial_py_sift_intro.html
#              https://theailearner.com/2020/11/06/perspective-transformation/

logger = logging.getLogger(__name__)


def feat_kp_dscr(preEVT, postEVT, feat_extr="sift"):
    # Create SIFT object
    if feat_extr == "sift":
        ft_extractor = cv2.SIFT_create()
    elif feat_extr == "orb":
        ft_extractor = cv2.ORB_create()
    preEVTkp, preEVTdescr = ft_extractor.detectAndCompute(preEVT, None)
    postEVTkp, postEVTdescr = ft_extractor.detectAndCompute(postEVT, None)
    preEVTwKP = np.copy(preEVT)
    postEVTwKP = np.copy(postEVT)
    cv2.drawKeypoints(preEVT, preEVTkp, preEVTwKP)
    cv2.drawKeypoints(postEVT, postEVTkp, postEVTwKP)

    # Visualise Keypoints on both images
    fig, axs = plt.subplots(1, 2, figsize=(10, 6))

    axs[0].set_title("Pre-EVT keypoints With Size")
    axs[0].imshow(preEVTwKP, cmap="gray")

    axs[1].set_title("Post-EVT keypoints With Size")
    axs[1].imshow(postEVTwKP, cmap="gray")
    # plt.show()

    return preEVTkp, preEVTdescr, postEVTkp, postEVTdescr


def find_feat_matches(preEVTdescr, postEVTdescr):
    # Initialize Brute Force matcher
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(preEVTdescr, postEVTdescr, k=2)

    # Apply ratio test as per Lowe - Original SIFT paper
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append([m])

    return good_matches


def plot_matches(preEVT, preEVTkp, postEVT, postEVTkp, matches):
    # Draw matches and plot matches images
    Matchesimg = cv2.drawMatchesKnn(
        preEVT,
        preEVTkp,
        postEVT,
        postEVTkp,
        matches,
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )

    plt.figure(figsize=(10, 6))
    plt.imshow(Matchesimg)
    plt.title("SIFT Matches")
    # plt.show()


def transform_post(preEVT, postEVT, preEVTkp, postEVTkp, matches):
    # Select good matched keypoints
    preEVT_matched_kpts = np.float32([preEVTkp[m[0].queryIdx].pt for m in matches])
    postEVT_matched_kpts = np.float32([postEVTkp[m[0].trainIdx].pt for m in matches])

    # Compute homography - RANSAC is built-in this function (Detects outliers and removes them)
    H, status = cv2.findHomography(
        postEVT_matched_kpts, preEVT_matched_kpts, cv2.RANSAC, 5.0
    )

    # Warp image
    warped_image = cv2.warpPerspective(postEVT, H, (postEVT.shape[1], postEVT.shape[0]))

    # Plot pre,post and transformed post images
    fig, axs = plt.subplots(1, 3, figsize=(12, 6))
    axs[0].imshow(preEVT, cmap="gray")
    axs[1].imshow(postEVT, cmap="gray")
    axs[2].imshow(warped_image, cmap="gray")
    axs[0].set_title("Pre-EVT")
    axs[1].set_title("Post-EVT")
    axs[2].set_title("Transformed Post-EVT")

    # Warped image to grayscale
    warped_image_gray = cv2.cvtColor(warped_image, cv2.COLOR_BGR2GRAY)
    preEVTorig = cv2.cvtColor(preEVT, cv2.COLOR_BGR2GRAY)

    # Overlay transformed and pre-EVT images
    cmap_artery_pre = colors.ListedColormap(["white", "red"])  # Not used after all.
    cmap_artery_post = colors.ListedColormap(["white", "green"])
    fig, axs = plt.subplots(1, 3, figsize=(12, 6))
    axs[0].imshow(preEVTorig, cmap="gray")
    axs[0].imshow(warped_image_gray, cmap="Purples", alpha=0.5)
    axs[1].imshow(preEVTorig, cmap="gray")
    axs[2].imshow(warped_image_gray, cmap="Purples")
    axs[0].set_title("Overlayed Transform")
    axs[1].set_title("Pre-EVT")
    axs[2].set_title("Transformed Post-EVT")
    # plt.show()


def main():
    # Read the images to be used by SIFT
    preEVT = cv2.imread("images/preEVTcrop.png", cv2.COLOR_BGR2GRAY)
    postEVT = cv2.imread("images/PostEVTcrop.png", cv2.COLOR_BGR2GRAY)
    # cv2.imshow('PreEVT',preEVT)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    DICOM_DIR = "Dicoms"
    patients = glob.glob(DICOM_DIR + "/*")
    patient_dcms = glob.glob(os.path.join(patients[1], "*.dcm"))
    pat = patient_dcms[0]
    ds = dcmread(pat, defer_size="1 KB", stop_before_pixels=False, force=True)
    print(ds.pixel_array.shape)
    if "FrameTimeVector" in ds:
        if len(ds.FrameTimeVector) != ds.NumberOfFrames:
            logger.warning(
                "Number of Frames ({}) does not match frame time vector length ({}): {}"
                "".format(
                    ds.NumberOfFrames, len(ds.FrameTimeVector), ds.FrameTimeVector
                )
            )
            ds.FrameTimeVector = ds.FrameTimeVector[: ds.NumberOfFrames]
        cum_time_vector = np.cumsum(ds.FrameTimeVector)
    elif "FrameTime" in ds:
        cum_time_vector = int(ds.FrameTime) * np.array(range(ds.NumberOfFrames))
    else:
        logger.error("Missing time info: {}".format(pat))
        return
    non_duplicated_frame_indices = np.where(~pd.DataFrame(cum_time_vector).duplicated())
    print(f"These are not duplicates:{non_duplicated_frame_indices}")
    print(ds["FrameTimeVector"].value)
    print(ds["NumberOfFrames"].value)
    # print(ds["FrameTime"].value)
    prekp, predsc, postkp, postdsc = feat_kp_dscr(preEVT, postEVT, feat_extr="sift")
    matches = find_feat_matches(predsc, postdsc)
    plot_matches(preEVT, prekp, postEVT, postkp, matches)
    transform_post(preEVT, postEVT, prekp, postkp, matches)
    plt.show()


if __name__ == "__main__":
    main()
