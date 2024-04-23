import logging
import os
import sys
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import cv2
import numpy as np
from skimage.transform import resize

import SIFTTransform as sift
from Skeletonization import (
    load_images,
    VisualizeSkeletons,
    get_skeletons,
    vis_skeleton_and_segm,
)


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

    # 1. SIFT Transformation

    # Read the images to be used by SIFT
    # preEVT = cv2.imread("images/preEVTcrop.png", cv2.COLOR_BGR2GRAY)
    # postEVT = cv2.imread("images/PostEVTcrop.png", cv2.COLOR_BGR2GRAY)
    IMG_DIR_PATH = "Minip/R0008/0"
    # IMG_DIR_PATH = "Niftis/R0001/0"
    images_path = sift.load_img_dir(IMG_DIR_PATH, img_type="minip")
    # Check if list is empty
    if not images_path:
        return

    # Load images from paths
    images = []
    for path in images_path:
        images.append(sift.load_img(path))
    OrigpreEVT = images[0]
    OrigpostEVT = images[1]

    # Remove unwanted text that comes with dicoms
    notextpreEVT, notextpostEVT, locations = sift.remove_unwanted_text(
        OrigpreEVT, OrigpostEVT
    )
    # Remove unwanted black borders if they exist
    preEVT, postEVT = sift.remove_borders(notextpreEVT, notextpostEVT)

    # Resize the images
    newW, newH = 512, 512
    preEVT = resize(preEVT, (newW, newH), anti_aliasing=True, preserve_range=True)
    postEVT = resize(postEVT, (newW, newH), anti_aliasing=True, preserve_range=True)
    preEVT = preEVT.astype(np.uint8)
    postEVT = postEVT.astype(np.uint8)

    prekp, predsc, postkp, postdsc = sift.feat_kp_dscr(
        preEVT, postEVT, feat_extr="sift"
    )
    matches = sift.find_feat_matches(predsc, postdsc)
    sift.plot_matches(preEVT, prekp, postEVT, postkp, matches)
    transformation = sift.transform_post(prekp, postkp, matches)

    # 2. Segmentation

    IMG_MIN_DIR_PATH = "C:/Users/mab03/Desktop/RuSegm/TemporalUNet/Outputs/Minip/R0001"
    IMG_SEQ_DIR_PATH = (
        "C:/Users/mab03/Desktop/RuSegm/TemporalUNet/Outputs/Sequence/R0008"
    )
    # segm_images = load_images(IMG_MIN_DIR_PATH)
    segm_images1 = load_images(IMG_SEQ_DIR_PATH)

    # Find corresponding pre and post images from the segmentations
    segm_pre_post = []
    for path in images_path:
        # print(f"Images: {Path(path).stem}")
        for segm in segm_images1:
            if (
                Path(path).stem == Path(segm).stem.rsplit("_", 1)[0]
                and Path(segm).stem.rsplit("_", 1)[1] == "artery"
            ):
                segm_pre_post.append(sift.load_img(segm))
                break

    # for segm in segm_images1:
    #     print(f"Segmentations: {Path(segm).stem.rsplit("_",1)[0]}")

    sift.display_tranformed(transformation, preEVT, postEVT)
    # Warp and display the segmentations
    warped_segm_post = sift.display_tranformed(
        transformation, segm_pre_post[0], segm_pre_post[1], ret=True
    )

    # 3. Skeletonization

    # Perform skeletonization
    skeleton_images, distance_transform = get_skeletons(
        [segm_pre_post[0], warped_segm_post], method="lee"
    )
    if not skeleton_images:
        return

    # Multiply skeleton images with the distance transform (optional)
    # skeleton_images = [
    #     sklt * dst for sklt, dst in zip(skeleton_images, distance_transform)
    # ]

    # vis_skeleton_and_segm(skeleton_image, segm)

    vis_skeletons = VisualizeSkeletons(
        skeleton_images, [segm_pre_post[0], warped_segm_post]
    )
    vis_skeletons.vis_images()
    bnext = Button(vis_skeletons.axnext, "Next")
    bnext.on_clicked(vis_skeletons.next)
    bprev = Button(vis_skeletons.axprev, "Previous")
    bprev.on_clicked(vis_skeletons.prev)

    plt.show()


if __name__ == "__main__":
    main()
