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
from prepareData import prepare_data, load_and_preprocess_dicom

# Inspiration: https://www.sicara.fr/blog-technique/2019-07-16-image-registration-deep-learning
#              https://docs.opencv.org/4.x/da/df5/tutorial_py_sift_intro.html
#              https://theailearner.com/2020/11/06/perspective-transformation/
#              https://stackoverflow.com/questions/37771263/detect-text-area-in-an-image-using-python-and-opencv
#              https://matthew-brett.github.io/teaching/mutual_information.html

logger = logging.getLogger(__name__)


def feat_kp_dscr(preEVT, postEVT, feat_extr="sift", vis=False):
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

    if vis:
        preEVTwKP = np.copy(preEVT)
        postEVTwKP = np.copy(postEVT)

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


def find_feat_matches(preEVTdescr, postEVTdescr, feat_extr="sift"):
    logger.info("Finding Descriptor Matches")
    # Initialize Brute Force matcher
    if feat_extr == "sift":
        bf = cv2.BFMatcher.create()
        matches = bf.knnMatch(preEVTdescr, postEVTdescr, k=2)

        # Apply ratio test as per Lowe - Original SIFT paper
        good_matches = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good_matches.append([m])
    elif feat_extr == "orb":
        bf = cv2.BFMatcher.create(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(preEVTdescr, postEVTdescr)

        # Sort them from lowest to highest distance
        good_matches = sorted(matches, key=lambda x: x.distance)

    # print(good_matches)
    return good_matches


def plot_matches(preEVT, preEVTkp, postEVT, postEVTkp, matches, feat_extr="sift"):
    # Draw matches and plot matches images
    if feat_extr == "sift":
        Matchesimg = cv2.drawMatchesKnn(
            preEVT,
            preEVTkp,
            postEVT,
            postEVTkp,
            matches,
            None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
        )
        title = "SIFT Matches"
    elif feat_extr == "orb":
        Matchesimg = cv2.drawMatches(
            preEVT,
            preEVTkp,
            postEVT,
            postEVTkp,
            matches,
            None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
        )
        title = "ORB Matches"

    plt.figure(figsize=(10, 6))
    plt.imshow(Matchesimg)
    plt.title(f"{title}")
    # plt.show()


def calculate_transform(preEVTkp, postEVTkp, matches, feat_extr="sift"):
    logger.info("Calculating Transformation")
    if feat_extr == "sift":
        # Select good matched keypoints
        preEVT_matched_kpts = np.float32([preEVTkp[m[0].queryIdx].pt for m in matches])
        postEVT_matched_kpts = np.float32(
            [postEVTkp[m[0].trainIdx].pt for m in matches]
        )
    elif feat_extr == "orb":
        preEVT_matched_kpts = np.float32([preEVTkp[m.queryIdx].pt for m in matches])
        postEVT_matched_kpts = np.float32([postEVTkp[m.trainIdx].pt for m in matches])
    # print(preEVT_matched_kpts.shape)
    # print(postEVT_matched_kpts.shape)

    # Need at least 4 matching points to calculate homography
    if preEVT_matched_kpts.shape[0] < 4:
        return np.eye(3, dtype=np.float32)
    # Compute homography - RANSAC is built-in this function (Detects outliers and removes them)
    H, status = cv2.findHomography(
        postEVT_matched_kpts, preEVT_matched_kpts, cv2.RANSAC, 5.0  # orig 5.0
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


def apply_transformation(transform, preEVT, postEVT, ret=False, vis=False):
    # Warp image
    warped_image = cv2.warpPerspective(
        postEVT,
        transform,
        (postEVT.shape[1], postEVT.shape[0]),
        # borderMode=cv2.BORDER_REPLICATE,
    )
    # warped_image = cv2.warpAffine(
    #     postEVT, transform, (postEVT.shape[1], postEVT.shape[0])
    # )

    if vis:
        # Plot pre,post and transformed post images
        fig, axs = plt.subplots(1, 3, figsize=(12, 6))
        axs[0].imshow(preEVT, cmap="gray")
        axs[1].imshow(postEVT, cmap="gray")
        axs[2].imshow(warped_image, cmap="gray")
        axs[0].set_title("Pre-EVT")
        axs[1].set_title("Post-EVT")
        axs[2].set_title("Transformed Post-EVT")
        for a in axs:
            a.set_xticks([])
            a.set_yticks([])

    # Warped image to grayscale
    if preEVT.ndim == 3:
        warped_image = cv2.cvtColor(warped_image, cv2.COLOR_BGR2GRAY)
        preEVT = cv2.cvtColor(preEVT, cv2.COLOR_BGR2GRAY)

    if vis:
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
        for a in axs:
            a.set_xticks([])
            a.set_yticks([])

    if ret:
        return warped_image


def check_transform(transform, preEVT, postEVT, verbose=False):
    # The Mutual Information score should be relatively higher
    # when comparing the original and transformed images if the
    # registration yields good results

    # Warp image
    warped_image = cv2.warpPerspective(
        postEVT,
        transform,
        (postEVT.shape[1], postEVT.shape[0]),
        # borderMode=cv2.BORDER_REPLICATE,
    )
    # Calculate 2D histogram of the preEVT and warped postEVT
    hist, x_edges, y_edges = np.histogram2d(
        preEVT.ravel(), warped_image.ravel(), bins=20
    )

    # Show log histogram, avoiding divide by 0
    # hist_log = np.zeros(hist.shape)
    # non_zeros = hist != 0
    # hist_log[non_zeros] = np.log(hist[non_zeros])
    # plt.imshow(hist_log.T, origin="lower")
    # plt.xlabel("pre-EVT bin")
    # plt.ylabel("post-EVT bin")

    # Calculate Mutual Information for the joint histogram
    mi_tr = calc_mutual_information(hist)

    # Calculate 2D histogram of the preEVT and original postEVT
    hist_or, x_edges, y_edges = np.histogram2d(preEVT.ravel(), postEVT.ravel(), bins=20)
    # Calculate Mutual Information for the joint histogram
    mi_orig = calc_mutual_information(hist_or)

    # Calculate 2D histogram of the preEVT
    hist_or, x_edges, y_edges = np.histogram2d(preEVT.ravel(), preEVT.ravel(), bins=20)
    # Calculate Mutual Information for the joint histogram
    mi_own = calc_mutual_information(hist_or)

    if verbose:
        print(f"Transformed Mutual information score: {mi_tr}")
        print(f"Original Mutual information score: {mi_orig}")
        print(f"Max Mutual information score: {mi_own}")

    # Compare the two scores
    if mi_orig < mi_tr:
        # print("The transformed image is better aligned")
        return True
    else:
        return False


def calc_mutual_information(histogram):
    # Calculate Mutual Information for the joint histogram
    # This function uses the Probability Mass function MI equation
    # Convert bins counts to probability values
    pxy = histogram / float(np.sum(histogram))
    px = np.sum(pxy, axis=1)  # marginal for x over y
    py = np.sum(pxy, axis=0)  # marginal for y over x
    px_py = px[:, None] * py[None, :]  # Broadcast to multiply marginals
    # Now we can do the calculation using the pxy, px_py 2D arrays
    nzs = pxy > 0  # Only non-zero pxy values contribute to the sum
    mi = np.sum(pxy[nzs] * np.log(pxy[nzs] / px_py[nzs]))
    return mi


def load_img_dir(img_dir_path, img_type="minip"):
    if not os.path.isdir(img_dir_path):
        logger.warning("Path directory {} does not exist".format(img_dir_path))
        return
    if img_type == "minip":
        images = sorted(glob.glob(os.path.join(img_dir_path, "*.png"), recursive=False))
    elif img_type == "dicom":
        images = sorted(glob.glob(os.path.join(img_dir_path, "*.dcm"), recursive=False))
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
        img = load_and_preprocess_dicom(img_path)
    else:
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)

    return img


def load_pre_post_imgs(img_paths: list) -> list:
    images = []
    for cnt, path in enumerate(img_paths):
        pre_or_post = Path(path).stem.split("_")[-1]
        if cnt == 1 and pre_or_post == "pre":
            images.insert(0, load_img(path))
        else:
            images.append(load_img(path))

    return images


def remove_black_borders(image):

    border_rows = []
    has_left_border = False
    has_right_border = False
    if image[0, 0] == 0:
        has_left_border = True
        # Find the last column containing black pixels
        for row in range(0, image.shape[0]):
            for idx, px in enumerate(image[row, :]):
                if px != 0:
                    # print(row, idx)
                    left_px_val = px
                    left_col_idx = idx
                    break
            else:
                border_rows.append(row)
                continue
            break

    # Do the same things in reverse for the opposite side
    # Check if the pre-EVT images contains boundary
    if image[-1, -1] == 0:
        has_right_border = True
        # Find the last column containing black pixels
        for row in range(image.shape[0] - 1, -1, -1):
            for idx, px in enumerate(image[row, ::-1]):
                if px != 0:
                    right_px_val = px
                    right_col_idx = image.shape[1] - idx
                    break
            else:
                border_rows.append(row)
                continue
            break

    # Change value of all pixels in that left and right column range to the first colored pixel value
    if has_left_border:
        image[:, :left_col_idx] = left_px_val
    if has_right_border:
        image[:, right_col_idx:] = right_px_val

    # Change the border rows if there are any
    if border_rows:
        image[border_rows, :] = left_px_val

    return image


def remove_borders(preevt, postevt):
    # If the top left pixel is not black there is likely no boundary
    if preevt[0, 0] != 0 and postevt[0, 0] != 0:
        return preevt, postevt

    pre = np.copy(preevt)
    post = np.copy(postevt)

    # fig, axs = plt.subplots(1, 2, figsize=(6, 6))
    # axs[0].imshow(pre, cmap="gray")
    # axs[1].imshow(post, cmap="gray")
    # plt.show()

    pre = remove_black_borders(pre)
    post = remove_black_borders(post)

    return pre, post


def remove_unwanted_text(preevtimg, postevtimg):
    preevt = np.copy(preevtimg)
    postevt = np.copy(postevtimg)
    # Threshold the pre-EVT image to find the text
    # Values are based on trial and error
    ret, thresh = cv2.threshold(preevt, 230, 255, cv2.THRESH_BINARY)
    # If the the given threshold only returns black
    if not np.any(thresh):
        ret, thresh = cv2.threshold(preevt, 200, 255, cv2.THRESH_BINARY)
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
    for row in range(0, preevt.shape[0]):
        for px in preevt[row, :]:
            if px != 0:
                px_val_pre = px
                break
        else:
            continue
        break

    for row in range(0, postevt.shape[0]):
        for px in postevt[row, :]:
            if px != 0:
                px_val_post = px
                break
        else:
            continue
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
    # prepare_data()
    # IMG_DIR_PATH = "Minipv2/R0030/0"
    IMG_DIR_PATH = "Niftisv2/R0002/1"
    images_path = load_img_dir(IMG_DIR_PATH, img_type="nifti")
    # Check if list is empty
    if not images_path:
        return
    # Load the pre and post images in a list
    images = load_pre_post_imgs(images_path)

    OrigpreEVT = images[0]
    OrigpostEVT = images[1]

    # Remove unwanted text and borders - the order must remain as is
    notextpreEVT, notextpostEVT, locations = remove_unwanted_text(
        OrigpreEVT, OrigpostEVT
    )
    preEVT, postEVT = remove_borders(notextpreEVT, notextpostEVT)

    # Calculate the keypoints and their feature descriptors
    feature_extractor = "sift"
    prekp, predsc, postkp, postdsc = feat_kp_dscr(
        preEVT, postEVT, feat_extr=feature_extractor, vis=True
    )
    matches = find_feat_matches(predsc, postdsc, feat_extr=feature_extractor)
    plot_matches(preEVT, prekp, postEVT, postkp, matches, feat_extr=feature_extractor)
    transformation = calculate_transform(
        prekp, postkp, matches, feat_extr=feature_extractor
    )
    apply_transformation(transformation, preEVT, postEVT, vis=True)
    plt.show()


if __name__ == "__main__":
    main()
