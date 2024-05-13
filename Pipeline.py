import logging
import os
import sys
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import cv2
import numpy as np
from skimage.transform import resize
import igraph as ig

import SIFTTransform as sift
from Skeletonization import (
    load_images,
    VisualizeSkeletons,
    get_skeletons,
    find_centerlines,
)
from graph_processing import create_graph, concat_extracted_features

# Add Segmentation package path to sys path to fix importing unet
sys.path.insert(1, os.path.join(sys.path[0], "Segmentation"))
from Segmentation import predict

# Inspiration: https://stackoverflow.com/questions/21019338/how-to-change-the-homography-with-the-scale-of-the-image/48915151#48915151


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
    # IMG_DIR_PATH = "Minip/R0011/0"
    IMG_DIR_PATH = "Niftis/R0002/0"
    images_path = sift.load_img_dir(IMG_DIR_PATH, img_type="nifti")
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
    # newW, newH = 512, 512
    # preEVT = resize(preEVT, (newW, newH), anti_aliasing=True, preserve_range=True)
    # postEVT = resize(postEVT, (newW, newH), anti_aliasing=True, preserve_range=True)
    # preEVT = preEVT.astype(np.uint8)
    # postEVT = postEVT.astype(np.uint8)

    prekp, predsc, postkp, postdsc = sift.feat_kp_dscr(
        preEVT, postEVT, feat_extr="sift"
    )
    matches = sift.find_feat_matches(predsc, postdsc)
    # sift.plot_matches(preEVT, prekp, postEVT, postkp, matches)
    transformation = sift.transform_post(prekp, postkp, matches)
    # print(transformation)

    # 2. Segmentation

    # IMG_MIN_DIR_PATH = "C:/Users/mab03/Desktop/RuSegm/TemporalUNet/Outputs/Minip/R0001"
    # IMG_SEQ_DIR_PATH = (
    #     "C:/Users/mab03/Desktop/RuSegm/TemporalUNet/Outputs/Sequence/R0002"
    # )

    # NOTE: IMG_DIR_PATH and in_img_path must be refering to the same patient (e.g. R0002)
    segm_output_folder = "Outputs/test"
    # Clear the segmentation output folder for every run
    # for root, dirs, files in os.walk(segm_output_folder):
    #     for f in files:
    #       os.remove(f)
    for path in Path(segm_output_folder).glob("*"):
        if path.is_file():
            path.unlink()
            # print(path)

    # Returns the sequence feature maps from the chosen layer (feature extraction)
    # In our case the chosen layer is "up4" form the model
    seq_features_list = predict.run_predict(
        # in_img_path="E:/vessel_diff_first_50_patients/mrclean_part1_2_first_50/R0002",
        in_img_path="Niftis/R0002/0",
        out_img_path=segm_output_folder,
        model="C:/Users/mab03/Desktop/RuSegm/TemporalUNet/models/1096-sigmoid-sequence-av.pt",
        input_type="sequence",
        label_type="av",
        amp=True,
    )

    # print(len(seq_features_list))
    # print(seq_features_list[0].size(0))

    # Plot the feature maps of one seq for reference
    predict.display_feature_maps(seq_features_list)
    # return
    # Could come up with a better way of doing this (ensure correct)
    # Sort it to maintain pre-post map correlation
    # The list contains two maps, where pre-post are first-second
    # The maps have a torch tensor format
    # seq_features_list = sorted(seq_features_list)

    IMG_SEQ_DIR_PATH = segm_output_folder
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

    # Check if the transformation quality is better than the original
    if not sift.check_transform(transformation, preEVT, postEVT, verbose=False):
        # The transformation is worse than the original
        print("Using the original post-EVT image")
        final_segm_post = segm_pre_post[1]
    else:
        # Scale transformation matrix for segmentation image size
        scale_matrix = np.array([[0.5, 0, 0], [0, 0.5, 0], [0, 0, 1]])
        transformation = scale_matrix @ transformation @ np.linalg.inv(scale_matrix)
        # Warp and display the segmentations
        final_segm_post = sift.display_tranformed(
            transformation, segm_pre_post[0], segm_pre_post[1], ret=True
        )

    # 3. Skeletonization

    # Perform skeletonization
    skeleton_images, distance_transform = get_skeletons(
        [segm_pre_post[0], final_segm_post], method="lee"
    )
    if not skeleton_images:
        return

    # Multiply skeleton images with the distance transform (optional)
    # skeleton_images = [
    #     sklt * dst for sklt, dst in zip(skeleton_images, distance_transform)
    # ]

    # vis_skeleton_and_segm(skeleton_image, segm)

    vis_skeletons = VisualizeSkeletons(
        skeleton_images, [segm_pre_post[0], final_segm_post]
    )
    vis_skeletons.vis_images()
    bnext = Button(vis_skeletons.axnext, "Next")
    bnext.on_clicked(vis_skeletons.next)
    bprev = Button(vis_skeletons.axprev, "Previous")
    bprev.on_clicked(vis_skeletons.prev)

    plt.show()

    # 4. Create Graphs
    skeleton_points = find_centerlines(skeleton_images[0])
    pre_graph = create_graph(
        skeleton_images[0],
        skeleton_points,
        distance_transform[0],
        g_name="pre_graph",
        vis=False,
        verbose=True,
    )
    skeleton_points = find_centerlines(skeleton_images[1])
    post_graph = create_graph(
        skeleton_images[1],
        skeleton_points,
        distance_transform[1],
        g_name="post_graph",
        vis=False,
        verbose=True,
    )

    # 5. Extract features from the segmenation model
    # and add them as graph attributes

    pre_feat_map = seq_features_list[0]
    post_feat_map = seq_features_list[1]
    # print(f"Feature map size: {pre_feat_map.size()}")

    # Pre-EVT graph
    concat_extracted_features(pre_graph, pre_feat_map, inplace=True)
    # Post-EVT graph
    concat_extracted_features(post_graph, post_feat_map, inplace=True)
    # print(pre_graph.vs[0].attributes())


if __name__ == "__main__":
    main()
