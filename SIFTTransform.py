import logging
import sys
import glob
import os
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import nibabel as nib
from prepareData import prepare_data

# Inspiration: https://www.sicara.fr/blog-technique/2019-07-16-image-registration-deep-learning
#              https://docs.opencv.org/4.x/da/df5/tutorial_py_sift_intro.html
#              https://theailearner.com/2020/11/06/perspective-transformation/
#              https://stackoverflow.com/questions/37771263/detect-text-area-in-an-image-using-python-and-opencv

logger = logging.getLogger(__name__)


def feat_kp_dscr(preEVT, postEVT, feat_extr="sift"):
    logger.info("Creating Keypoints and Descriptors")
    # Create SIFT/ORB object
    if feat_extr == "sift":
        ft_extractor = cv2.SIFT_create()
    elif feat_extr == "orb":
        ft_extractor = cv2.ORB_create()
    # Convert to RGB to display the keypoints
    preEVT = cv2.cvtColor(preEVT, cv2.COLOR_GRAY2RGB)
    postEVT = cv2.cvtColor(postEVT, cv2.COLOR_GRAY2RGB)
    preEVTkp, preEVTdescr = ft_extractor.detectAndCompute(preEVT, None)
    postEVTkp, postEVTdescr = ft_extractor.detectAndCompute(postEVT, None)
    preEVTwKP = np.copy(preEVT)
    postEVTwKP = np.copy(postEVT)

    # Disregard the text areas
    # If the keypoint is in the range of rectangles from the locs
    # Remove it from the list along with the descriptor
    # for loc in locs:
    #     for y in range(loc['ymin'], loc['ymin']+ loc['h']):
    #         for x in range(loc['xmin'],loc['xmin']+loc['w']):
    #             if (y,x) in preEVTkp:
    #               Remove the point and do the same for postEVT

    cv2.drawKeypoints(preEVT, preEVTkp, preEVTwKP)
    cv2.drawKeypoints(postEVT, postEVTkp, postEVTwKP)

    # Visualise Keypoints on both images
    fig, axs = plt.subplots(1, 2, figsize=(10, 6))

    axs[0].set_title("Pre-EVT keypoints")
    axs[0].imshow(preEVTwKP, cmap="gray")

    axs[1].set_title("Post-EVT keypoints")
    axs[1].imshow(postEVTwKP, cmap="gray")
    # plt.show()

    return preEVTkp, preEVTdescr, postEVTkp, postEVTdescr


def find_feat_matches(preEVTdescr, postEVTdescr):
    logger.info("Finding Descriptor Matches")
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


def transform_post(preEVTkp, postEVTkp, matches):
    logger.info("Calculating Transformation")
    # Select good matched keypoints
    preEVT_matched_kpts = np.float32([preEVTkp[m[0].queryIdx].pt for m in matches])
    postEVT_matched_kpts = np.float32([postEVTkp[m[0].trainIdx].pt for m in matches])
    # print(preEVT_matched_kpts.shape)
    # print(postEVT_matched_kpts.shape)

    # Compute homography - RANSAC is built-in this function (Detects outliers and removes them)
    H, status = cv2.findHomography(
        postEVT_matched_kpts, preEVT_matched_kpts, cv2.RANSAC, 5.0
    )

    # Tested the affine transformation as well -> Seemingly worse results
    # H, status = cv2.estimateAffine2D(
    #     postEVT_matched_kpts,
    #     preEVT_matched_kpts,
    #     method=cv2.RANSAC,
    #     ransacReprojThreshold=5.0,
    #     refineIters=0,
    # )

    return H


def display_tranformed(transform, preEVT, postEVT, ret=False):
    # Warp image
    warped_image = cv2.warpPerspective(
        postEVT, transform, (postEVT.shape[1], postEVT.shape[0])
    )
    # warped_image = cv2.warpAffine(
    #     postEVT, transform, (postEVT.shape[1], postEVT.shape[0])
    # )

    # Plot pre,post and transformed post images
    fig, axs = plt.subplots(1, 3, figsize=(12, 6))
    axs[0].imshow(preEVT, cmap="gray")
    axs[1].imshow(postEVT, cmap="gray")
    axs[2].imshow(warped_image, cmap="gray")
    axs[0].set_title("Pre-EVT")
    axs[1].set_title("Post-EVT")
    axs[2].set_title("Transformed Post-EVT")

    # Warped image to grayscale
    if preEVT.ndim == 3:
        warped_image = cv2.cvtColor(warped_image, cv2.COLOR_BGR2GRAY)
        preEVT = cv2.cvtColor(preEVT, cv2.COLOR_BGR2GRAY)

    # Overlay transformed and pre-EVT images
    cmap_artery_pre = colors.ListedColormap(["white", "red"])  # Not used after all.
    cmap_artery_post = colors.ListedColormap(["white", "green"])
    fig, axs = plt.subplots(1, 3, figsize=(12, 6))
    axs[0].imshow(preEVT, cmap="gray")
    axs[0].imshow(warped_image, cmap="Purples", alpha=0.5)
    axs[1].imshow(preEVT, cmap="gray")
    axs[2].imshow(warped_image, cmap="Purples")
    axs[0].set_title("Overlayed Transform")
    axs[1].set_title("Pre-EVT")
    axs[2].set_title("Transformed Post-EVT")

    if ret:
        return warped_image


def load_img_dir(img_dir_path, img_type="minip"):
    if not os.path.isdir(img_dir_path):
        logger.warning("Path directory {} does not exist".format(img_dir_path))
        return
    if img_type == "minip":
        images = sorted(glob.glob(os.path.join(img_dir_path, "*.png"), recursive=False))
    elif img_type == "dicom":
        # THIS NEEDS TO BE MODIFIED
        images = sorted(glob.glob(os.path.join(img_dir_path, "*.dcm"), recursive=False))
        images = None
    else:
        images = sorted(glob.glob(os.path.join(img_dir_path, "*.nii"), recursive=False))
    return images


def load_img(img_path):
    if ".nii" in img_path:
        img_obj = nib.load(img_path)
        img = np.transpose(img_obj.get_fdata(), (2, 1, 0))
        # This has to be done because initially it was Float64
        img = img.astype(np.uint8)
        img = np.min(img, axis=0)
    elif ".dcm" in img_path:
        # THIS NEEDS TO BE MODIFIED
        img = None
    else:
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)

    return img


def remove_borders(preevt, postevt):
    # If the top left pixel is not black there is likely no boundary
    if preevt[0, 0] != 0 and postevt[0, 0] != 0:
        return preevt, postevt

    pre = np.copy(preevt)
    post = np.copy(postevt)
    # Check if the pre-EVT images contains boundary
    if pre[0, 0] == 0:
        # Find the last column containing black pixels
        for idx, px in enumerate(pre[0, :]):
            if px != 0:
                px_val = px
                col_idx = idx
                break
        # Change value of all pixels in that column range to the first colored pixel value
        # print(f"Pixel value: {px_val}/Col_idx: {col_idx}")
        pre[:, :col_idx] = px_val
        # pre[:, -col_idx + 1 :] = px_val

    # Check if the post-EVT images contains boundary
    if post[0, 0] == 0:
        # Find the last column containing black pixels
        for idx, px in enumerate(post[0, :]):
            if px != 0:
                px_val = px
                col_idx = idx
                break
        # Change value of all pixels in that column range to the first colored pixel value
        # print(f"Pixel value: {px_val}/Col_idx: {col_idx}")
        post[:, :col_idx] = px_val
        # post[:, -col_idx + 1 :] = px_val

    # Do the same things in reverse for the opposite side
    # Check if the pre-EVT images contains boundary
    if pre[-1, -1] == 0:
        # Find the last column containing black pixels
        for idx, px in enumerate(pre[0, ::-1]):
            if px != 0:
                px_val = px
                col_idx = pre.shape[1] - idx
                break
        # Change value of all pixels in that column range to the first colored pixel value
        # print(f"Pixel value: {px_val}/Col_idx: {col_idx}")
        pre[:, col_idx:] = px_val

    # Check if the post-EVT images contains boundary
    if post[-1, -1] == 0:
        # Find the last column containing black pixels
        for idx, px in enumerate(post[0, ::-1]):
            if px != 0:
                px_val = px
                col_idx = post.shape[1] - idx
                break
        # Change value of all pixels in that column range to the first colored pixel value
        # print(f"Pixel value: {px_val}/Col_idx: {col_idx}")
        post[:, col_idx:] = px_val

    return pre, post


def remove_unwanted_text(preevtimg, postevtimg):
    preevt = np.copy(preevtimg)
    postevt = np.copy(postevtimg)
    # Threshold the pre-EVT image to find the text
    ret, thresh = cv2.threshold(preevt, 240, 255, cv2.THRESH_BINARY)
    # Dilate to combine adjacent text contours
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    dilate = cv2.dilate(thresh, kernel, iterations=4)
    # plt.imshow(thresh, cmap="gray")

    # Find contours, highlight text areas
    cnts, hierarchy = cv2.findContours(
        dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    # Find color to mask the text areas
    # Find the first column with non-black px value
    for px in preevt[0, :]:
        if px != 0:
            px_val_post = px
            break

    for px in postevt[0, :]:
        if px != 0:
            px_val_pre = px
            break

    # Use contours to mask the text areas
    locations = []
    for c in cnts:
        # area = cv2.contourArea(c)
        # if area > 10000:
        x, y, w, h = cv2.boundingRect(c)
        # cv2.rectangle(preevt, (x, y), (x + w, y + h), (36, 255, 12), 3)
        preevt[y : y + h, x : x + w] = px_val_pre
        postevt[y : y + h, x : x + w] = px_val_post
        loc_dict = {"xmin": x, "ymin": y, "w": w, "h": h}
        # locations.append((x,y,w,h))
        locations.append(loc_dict)

    return preevt, postevt, locations


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
    # Read the images to be used by SIFT
    # preEVT = cv2.imread("images/preEVTcrop.png", cv2.COLOR_BGR2GRAY)
    # postEVT = cv2.imread("images/PostEVTcrop.png", cv2.COLOR_BGR2GRAY)
    # preEVT = cv2.cvtColor(preEVT, cv2.COLOR_BGR2GRAY)
    # postEVT = cv2.cvtColor(postEVT, cv2.COLOR_BGR2GRAY)
    IMG_DIR_PATH = "Minip/R0008/0"
    # IMG_DIR_PATH = "Niftis/R0001/0"
    images_path = load_img_dir(IMG_DIR_PATH, img_type="minip")
    # Check if list is empty
    if not images_path:
        return
    images = []
    for path in images_path:
        images.append(load_img(path))
    OrigpreEVT = images[0]
    OrigpostEVT = images[1]

    # for path in images_path:
    #     print(f"Images: {path}")

    # Remove the text areas from the images
    # preEVT[30:130, :135] = preEVT[0, 0]
    # preEVT[890:990, :150] = preEVT[0, 0]
    # preEVT[925:990, 945:] = preEVT[0, 0]
    # postEVT[30:130, :135] = postEVT[0, 0]
    # postEVT[890:990, :150] = postEVT[0, 0]
    # postEVT[925:990, 945:] = postEVT[0, 0]

    # nobordpreEVT, nobordpostEVT = remove_borders(OrigpreEVT, OrigpostEVT)
    # preEVT, postEVT = remove_borders(OrigpreEVT, OrigpostEVT)
    # plt.imshow(nobordpreEVT, cmap="gray")
    # plt.show()
    # preEVT, postEVT, locations = remove_unwanted_text(nobordpreEVT, nobordpostEVT)

    # Replaced order of unwanted text and border
    notextpreEVT, notextpostEVT, locations = remove_unwanted_text(
        OrigpreEVT, OrigpostEVT
    )
    preEVT, postEVT = remove_borders(notextpreEVT, notextpostEVT)

    # preEVT, postEVT, locations = remove_unwanted_text(OrigpreEVT, OrigpostEVT)
    # preEVT = OrigpreEVT
    # postEVT = OrigpostEVT
    # print(locations[0]["xmin"])

    # cv2.imshow("PreEVT", preEVT)
    # plt.imshow(preEVT, cmap="gray")
    # plt.show()
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # prepare_data()
    prekp, predsc, postkp, postdsc = feat_kp_dscr(preEVT, postEVT, feat_extr="sift")
    matches = find_feat_matches(predsc, postdsc)
    plot_matches(preEVT, prekp, postEVT, postkp, matches)
    transformation = transform_post(prekp, postkp, matches)
    display_tranformed(transformation, preEVT, postEVT)
    plt.show()


if __name__ == "__main__":
    main()
