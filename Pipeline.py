import logging
import os
import sys
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import cv2
import numpy as np
from skimage.transform import resize
from scipy.spatial import distance
import igraph as ig

# Test Boxcox
from scipy import stats

import SIFTTransform as sift
from Skeletonization import (
    load_images,
    VisualizeSkeletons,
    get_skeletons,
    find_centerlines,
)
from graph_processing import (
    create_graph,
    concat_extracted_features,
    concat_extracted_features_v2,
    plot_pre_post,
    multiscale_graph,
)
import graph_matching as gm

# Add Segmentation package path to sys path to fix importing unet
sys.path.insert(1, os.path.join(sys.path[0], "Segmentation"))
from Segmentation import predict

# Inspiration: https://stackoverflow.com/questions/21019338/how-to-change-the-homography-with-the-scale-of-the-image/48915151#48915151


def main(load_segs=True):
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
    # Pat. 18 and 30 are early ICA blockages
    pat_id = "R0002"
    pat_ori = "1"
    IMG_DIR_PATH = "Niftisv2/" + pat_id + "/" + pat_ori
    images_path = sift.load_img_dir(IMG_DIR_PATH, img_type="nifti")
    # Check if list is empty
    if not images_path:
        return

    # Load images from paths
    # images = []
    # for path in images_path:
    #     images.append(sift.load_img(path))
    images = sift.load_pre_post_imgs(images_path)

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
    feature_extractor = "sift"  # choose sift or orb
    prekp, predsc, postkp, postdsc = sift.feat_kp_dscr(
        preEVT, postEVT, feat_extr=feature_extractor
    )
    matches = sift.find_feat_matches(predsc, postdsc, feat_extr=feature_extractor)
    # sift.plot_matches(preEVT, prekp, postEVT, postkp, matches,feat_extr=feature_extractor)
    transformation = sift.calculate_transform(
        prekp, postkp, matches, feat_extr=feature_extractor
    )
    # print(transformation)

    # 2. Segmentation

    if load_segs:
        # If True we assume that the feature maps will also be loaded
        IMG_MIN_DIR_PATH = (
            "C:/Users/mab03/Desktop/RuSegm/TemporalUNet/Outputs/Minip/" + pat_id
        )
        IMG_SEQ_DIR_PATH = (
            "C:/Users/mab03/Desktop/ThesisCode/Segms/Sequence/" + pat_id + "/" + pat_ori
        )

        FEAT_MAP_DIR_PATH = "FeatMapsv2/" + pat_id + "/" + pat_ori
        feat_map_pre_post = []

    else:
        # NOTE: IMG_DIR_PATH and IMG_SEQ_DIR_PATH must be refering to the same patient (e.g. R0002)
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
        # Doesn't work with mrclean dir structure yet.
        # Need to change this to make sure pre and post are not reversed
        feat_map_pre_post = predict.run_predict(
            # in_img_path="E:/vessel_diff_first_50_patients/mrclean_part1_2_first_50/R0002",
            in_img_path=IMG_DIR_PATH,
            out_img_path=segm_output_folder,
            model="C:/Users/mab03/Desktop/RuSegm/TemporalUNet/models/1096-sigmoid-sequence-av.pt",
            input_type="sequence",
            input_format="nifti",
            label_type="av",
            amp=True,
        )

        # Plot the feature maps of one seq for reference
        predict.display_feature_maps(feat_map_pre_post)
        # Could come up with a better way of doing this (ensure correct)
        # Sort it to maintain pre-post map correlation
        # The list contains two maps, where pre-post are first-second
        # The maps have a torch tensor format
        IMG_SEQ_DIR_PATH = segm_output_folder

    # segm_images = load_images(IMG_MIN_DIR_PATH)
    segm_images1 = load_images(IMG_SEQ_DIR_PATH)

    # Find corresponding pre and post images from the segmentations and feat maps
    segm_pre_post = []
    for segm in segm_images1:
        if Path(segm).stem.rsplit("_", 1)[1] == "artery":
            if Path(segm).stem.rsplit("_")[1] == "pre":
                pre = True
                segm_pre_post.insert(0, sift.load_img(segm))
            else:
                pre = False
                segm_pre_post.append(sift.load_img(segm))

            if load_segs:
                # The feature map dir has the same structure as the sequence dir
                # and the same file name. So replace the .png extension with npz
                # and read the appropriate feature maps
                feature_maps_path = segm.replace(
                    IMG_SEQ_DIR_PATH, FEAT_MAP_DIR_PATH
                ).replace("_artery.png", ".npz")
                # Load npz file
                feature_maps = np.load(feature_maps_path)
                if pre:
                    feat_map_pre_post.insert(0, feature_maps["feat_maps"])
                else:
                    feat_map_pre_post.append(feature_maps["feat_maps"])

    # Check if the transformation quality is better than the original
    if not sift.check_transform(transformation, preEVT, postEVT, verbose=False):
        # The transformation is worse than the original
        print("Using the original post-EVT image")
        final_segm_post = segm_pre_post[1]
        tr_postEVT = postEVT
    else:
        tr_postEVT = sift.apply_transformation(
            transformation, preEVT, postEVT, ret=True, vis=True
        )
        # Scale transformation matrix for segmentation image size
        scale_matrix = np.array([[0.5, 0, 0], [0, 0.5, 0], [0, 0, 1]])
        transformation = scale_matrix @ transformation @ np.linalg.inv(scale_matrix)
        # Warp and display the segmentations
        final_segm_post = sift.apply_transformation(
            transformation, segm_pre_post[0], segm_pre_post[1], ret=True, vis=True
        )

    # 3. Skeletonization

    # Perform skeletonization
    skeleton_images, distance_transform = get_skeletons(
        [segm_pre_post[0], final_segm_post], method="lee"
    )
    if not skeleton_images:
        return

    # Multiply skeleton images with the distance transform for visualization (optional)
    # skeleton_images = [
    #     sklt * dst for sklt, dst in zip(skeleton_images, distance_transform)
    # ]
    # vis_skeleton_and_segm(skeleton_image, segm)

    # Keep the skeleton image and the distance transform at a specific scale
    # So that we treat the problem as a multi-scale
    # The choice relies on the distance transform and the value is chosen heuristically
    # If the distance transform is less than 3 disregard it
    # for idx in range(len(distance_transform)):
    #     skeleton_images[idx] = np.where(
    #         distance_transform[idx] > 4, skeleton_images[idx], 0
    #     )
    #     distance_transform[idx] = np.where(
    #         distance_transform[idx] > 4, distance_transform[idx], 0
    #     )

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

    # Get the multiscale graph
    scale = [0, 2]
    pre_graph = multiscale_graph(pre_graph, scale)
    post_graph = multiscale_graph(post_graph, scale)

    # Plot the two graph side-by-side (optional)
    # plot_pre_post(pre_graph, post_graph, preEVT, tr_postEVT, overlay_orig=True)
    # plot_pre_post(
    #     pre_graph, post_graph, segm_pre_post[0], final_segm_post, overlay_seg=True
    # )

    # 5. Extract features from the segmenation model
    # and add them as graph attributes

    pre_feat_map = feat_map_pre_post[0]
    post_feat_map = feat_map_pre_post[1]
    # predict.display_feature_maps(feat_map_pre_post)

    # Use the feature descriptors calculated in a similar way to SIFT
    # Given that keypoints are the nodes of the two graphs
    # This does not offer improve performance qualitatively(Optional)
    # Save for imaging
    # pre_kpts = [(float(pt[0]), float(pt[1])) for pt in pre_graph.vs["coords"]]
    # post_kpts = [(float(pt[0]), float(pt[1])) for pt in post_graph.vs["coords"]]
    # pre_feat_matrix, post_feat_matrix = gm.create_sift_feat_matrix(
    #     preEVT, tr_postEVT, pre_graph, post_graph
    # )

    # return
    # Calculate feature map similarity (Optional)
    # predict.calc_ft_map_sim(pre_feat_map)
    # predict.calc_ft_map_sim(post_feat_map)

    # print(f"Feature map size: {pre_feat_map.size()}")
    # print(f"Feature map shape: {pre_feat_map.shape}")

    # # Pre-EVT graph
    # concat_extracted_features(pre_graph, pre_feat_map, inplace=True)
    # # Post-EVT graph
    # concat_extracted_features(post_graph, post_feat_map, inplace=True)
    # # print(pre_graph.vs[0].attributes())

    # Test averaging the neighborhood (3x3) or (5x5) values for the features
    # Pre-EVT graph
    concat_extracted_features_v2(pre_graph, pre_feat_map, inplace=True)
    # Post-EVT graph
    concat_extracted_features_v2(post_graph, post_feat_map, inplace=True)
    # print(pre_graph.vs[0].attributes())

    # 6. Preprocess features if needed

    # Save keypoints for imaging
    pre_kpts = [(float(pt[0]), float(pt[1])) for pt in pre_graph.vs["coords"]]
    post_kpts = [(float(pt[0]), float(pt[1])) for pt in post_graph.vs["coords"]]

    # Delete the coords attribute before calculating similarity
    del pre_graph.vs["coords"]
    del post_graph.vs["coords"]

    # Find min max radius of all nodes in the graphs for normalization
    pre_r_avg, pre_r_max, pre_r_min, pre_r_std = gm.calc_graph_radius_info(pre_graph)
    print(
        f"R-Avg: {pre_r_avg}, R-Max: {pre_r_max}, R-Min: {pre_r_min}, R-Std: {pre_r_std}"
    )
    post_r_avg, post_r_max, post_r_min, post_r_std = gm.calc_graph_radius_info(
        post_graph
    )
    print(
        f"R-Avg: {post_r_avg}, R-Max: {post_r_max}, R-Min: {post_r_min}, R-Std: {post_r_std}"
    )
    r_norm_max = pre_r_max if pre_r_max > post_r_max else post_r_max
    r_norm_min = pre_r_min if pre_r_min < post_r_min else post_r_min

    # Normalize x,y and radius features - Prevent blowup with similarity mat
    pre_graph.vs["x"] = [x / skeleton_images[0].shape[1] for x in pre_graph.vs["x"]]
    pre_graph.vs["y"] = [y / skeleton_images[0].shape[0] for y in pre_graph.vs["y"]]
    post_graph.vs["x"] = [x / skeleton_images[1].shape[1] for x in post_graph.vs["x"]]
    post_graph.vs["y"] = [y / skeleton_images[1].shape[0] for y in post_graph.vs["y"]]
    # Test assigning a larger weight to the position of the nodes
    # pre_graph.vs["x"] = [
    #     32 * (x / skeleton_images[0].shape[1]) for x in pre_graph.vs["x"]
    # ]
    # pre_graph.vs["y"] = [
    #     32 * (y / skeleton_images[0].shape[0]) for y in pre_graph.vs["y"]
    # ]
    # post_graph.vs["x"] = [
    #     32 * (x / skeleton_images[1].shape[1]) for x in post_graph.vs["x"]
    # ]
    # post_graph.vs["y"] = [
    #     32 * (y / skeleton_images[1].shape[0]) for y in post_graph.vs["y"]
    # ]
    # Min-max normalization of radii
    pre_graph.vs["radius"] = [
        (r - r_norm_min) / (r_norm_max - r_norm_min) for r in pre_graph.vs["radius"]
    ]
    post_graph.vs["radius"] = [
        (r - r_norm_min) / (r_norm_max - r_norm_min) for r in post_graph.vs["radius"]
    ]

    # Test using only the extracted features
    # Cannot delete the x,y attrs since they are needed for plotting
    # del pre_graph.vs["x"]
    # del post_graph.vs["x"]
    # del pre_graph.vs["y"]
    # del post_graph.vs["y"]
    del pre_graph.vs["radius"]
    del post_graph.vs["radius"]

    # 7. Calculate graph node feature matrices and perform matching

    # Create a feature matrices from all the node attributes
    pre_feat_matrix = gm.create_feat_matrix(pre_graph)
    post_feat_matrix = gm.create_feat_matrix(post_graph)

    # Log transformation of data since they are right skewed
    pre_feat_matrix = np.log(pre_feat_matrix + 1)  # 1e-4
    post_feat_matrix = np.log(post_feat_matrix + 1)
    # Box-cox transformation - Seems like very similar results to log tr.
    # pre_feat_matrix = pre_feat_matrix + 1
    # post_feat_matrix = post_feat_matrix + 1
    # for col in range(pre_feat_matrix.shape[1]):
    #     pre_feat_matrix[:, col] = stats.boxcox(pre_feat_matrix[:, col])[0]
    # for col in range(post_feat_matrix.shape[1]):
    #     post_feat_matrix[:, col] = stats.boxcox(post_feat_matrix[:, col])[0]

    pre_feat_avg, pre_feat_max, pre_feat_min, pre_feat_std = gm.calc_feat_matrix_info(
        pre_feat_matrix, vis=False
    )
    post_feat_avg, post_feat_max, post_feat_min, post_feat_std = (
        gm.calc_feat_matrix_info(post_feat_matrix, vis=False)
    )
    print(
        f"Avg: {pre_feat_avg}, Max: {pre_feat_max}, Min: {pre_feat_min}, Std: {pre_feat_std}"
    )
    print(
        f"Avg: {post_feat_avg}, Max: {post_feat_max}, Min: {post_feat_min}, Std: {post_feat_std}"
    )

    # Calculate the node similarity matrix
    sim_matrix = gm.calc_similarity(pre_feat_matrix, post_feat_matrix)

    # Calculate the soft assignment matrix via Sinkhorn
    # Differes quite a bit from the hungarian and does not give 1-1 mapping
    # in many cases, so it's probably best to not use it for now.
    # Orig tau: 100 Orig iter: 250 for sinkhorn
    # assignment_mat = gm.calc_assignment_matrix(
    #     sim_matrix, tau=100, iter=250, method="sinkhorn"
    # )

    assignment_mat = gm.calc_assignment_matrix(sim_matrix, method="hungarian")

    # Get the maximum argument for each row (node)
    # This complicates the drawing later if they have to be transposed.
    # if assignment_mat.shape[0] < assignment_mat.shape[1]:
    #     matchings = np.argmax(assignment_mat, axis=1)  # row
    # else:
    #     matchings = np.argmax(assignment_mat, axis=0)  # column
    matchings = np.argmax(assignment_mat, axis=1)  # row

    # Test masking based on feature vector Euclidean distance
    feat_dist = distance.cdist(pre_feat_matrix, post_feat_matrix, "euclidean")
    # Get matchings based on minimum euclidean distance
    # matchings = np.argmin(feat_dist, axis=1)

    # Test evaluation using positional Euclidean distance
    pre_pos = np.zeros((len(pre_graph.vs), 2))
    for idx, pt in enumerate(pre_kpts):
        pre_pos[idx, :] = pt
    post_pos = np.zeros((len(post_graph.vs), 2))
    for idx, pt in enumerate(post_kpts):
        post_pos[idx, :] = pt
    # print(pre_pos[:5, :])
    node_pos_dist = distance.cdist(pre_pos, post_pos, "euclidean")

    thresh = 15  # 15
    masked = np.where(
        [
            node_pos_dist[i, j] < thresh
            for i, j in zip([*range(len(matchings))], matchings)
        ],
        1,
        0,
    ).tolist()
    # masked = None

    # Draw keypoints and then their matches
    # Drawing the matches results in an incomprehensable visual because of the large number of nodes
    # gm.draw_keypoints(preEVT, tr_postEVT, pre_kpts, post_kpts)
    # gm.draw_matches(preEVT, tr_postEVT, pre_kpts, post_kpts, matchings, mask=masked)
    # gm.draw_matches_animated(
    #     preEVT, tr_postEVT, pre_kpts, post_kpts, matchings, save_fig=False
    # )
    # Draw the preEVT and postEVT Minips
    # gm.draw_matches_interactive(
    #     pre_graph,
    #     post_graph,
    #     preEVT,
    #     tr_postEVT,
    #     pre_kpts,
    #     post_kpts,
    #     matchings,
    #     mask=masked,
    # )
    # Draw the preEVT and postEVT segmentations
    gm.draw_matches_interactive(
        pre_graph,
        post_graph,
        segm_pre_post[0],
        final_segm_post,
        pre_kpts,
        post_kpts,
        matchings,
        mask=masked,
        segm=True,
    )

    ### Spectral matching - Cannot use atm because not enough memory to create aff matrix.

    # # Calculate affinity matrix
    # K = gm.calc_affinity_matrix(pre_graph, post_graph)

    # # Solve using spectral matching
    # matchings = gm.spectral_matcher(K)

    # gm.draw_matches_animated(
    #     preEVT, tr_postEVT, pre_kpts, post_kpts, matchings, save_fig=False
    # )


if __name__ == "__main__":
    main(load_segs=True)
