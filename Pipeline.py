import argparse
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

import SIFTTransform as sift
import Skeletonization as sklt
import graph_processing as GProc
import graph_matching as GMatch
import filter_banks as FilterBanks
import normalizations as Norms

# Add Segmentation package path to sys path to fix importing unet
sys.path.insert(1, os.path.join(sys.path[0], "Segmentation"))
from Segmentation import predict

# Inspiration: https://stackoverflow.com/questions/21019338/how-to-change-the-homography-with-the-scale-of-the-image/48915151#48915151


def main(match_type="single", load_segs=True, load_ft_maps=True):
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

    # Check the validity of arguments
    assert match_type in [
        "single",
        "multi",
        "patched",
        "rand_patch",
    ], "Invalid matching type"

    # 1. SIFT Transformation

    # Read the images to be used by SIFT
    # Pat. 18 and 30 are early ICA blockages
    pat_id = "R0002"
    pat_ori = "0"
    IMG_DIR_PATH = "Niftisv2/" + pat_id + "/" + pat_ori
    images_path = sift.load_img_dir(IMG_DIR_PATH, img_type="nifti")
    # Check if list is empty
    if not images_path:
        return

    # Load images from paths
    images = sift.load_pre_post_imgs(images_path)

    OrigpreEVT = images[0]
    OrigpostEVT = images[1]

    # Remove unwanted text that comes with dicoms
    notextpreEVT, notextpostEVT, locations = sift.remove_unwanted_text(
        OrigpreEVT, OrigpostEVT
    )
    # Remove unwanted black borders if they exist
    preEVT, postEVT = sift.remove_borders(notextpreEVT, notextpostEVT)

    feature_extractor = "sift"  # choose sift or orb
    prekp, predsc, postkp, postdsc = sift.feat_kp_dscr(
        preEVT, postEVT, feat_extr=feature_extractor
    )
    matches = sift.find_feat_matches(predsc, postdsc, feat_extr=feature_extractor)
    # sift.plot_matches(preEVT, prekp, postEVT, postkp, matches,feat_extr=feature_extractor)
    transformation = sift.calculate_transform(
        prekp, postkp, matches, feat_extr=feature_extractor
    )

    # 2. Segmentation

    if load_segs:
        # If True we assume that the feature maps will also be loaded
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

        # Plot the feature maps of one seq for reference (Optional)
        # predict.display_feature_maps(feat_map_pre_post)

        # The list contains two maps, where pre-post are first-second
        # The maps have a torch tensor format
        IMG_SEQ_DIR_PATH = segm_output_folder

    segm_images = sklt.load_images(IMG_SEQ_DIR_PATH)

    # Find corresponding pre and post images from the segmentations and feat maps
    segm_pre_post = []
    for segm in segm_images:
        if Path(segm).stem.rsplit("_", 1)[1] == "artery":
            if Path(segm).stem.rsplit("_", 2)[1] == "pre":
                pre = True
                segm_pre_post.insert(0, sift.load_img(segm))
            else:
                pre = False
                segm_pre_post.append(sift.load_img(segm))

            if load_segs and load_ft_maps:
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
        if load_ft_maps:
            # Warp the feature maps
            for i in range(feat_map_pre_post[1].shape[0]):
                feat_map_pre_post[1][i, :, :] = sift.apply_transformation(
                    transformation,
                    feat_map_pre_post[0][i, :, :],
                    feat_map_pre_post[1][i, :, :],
                    ret=True,
                    vis=False,
                )

    # Careful with this since arrays are mutable
    # Used final_segm_post in order to plot original and transformed post
    # After the plotting this variable is not necessary anymore.
    segm_pre_post[1] = final_segm_post

    # Leave these for reference since we tried it.
    # Opening (morphology) of the segmentations
    # Essentially erosion followed by dilation
    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    # segm_pre_post = [cv2.erode(segm, kernel, iterations=1) for segm in segm_pre_post]
    # segm_pre_post = [
    #     cv2.morphologyEx(segm, cv2.MORPH_OPEN, kernel) for segm in segm_pre_post
    # ]

    if not load_ft_maps:
        # Use Leung-Malik features
        LM_filter_banks = FilterBanks.makeLMfilters()
        pre_feat_map = FilterBanks.create_ft_map(LM_filter_banks, preEVT)
        post_feat_map = FilterBanks.create_ft_map(LM_filter_banks, tr_postEVT)
        feat_map_pre_post = [pre_feat_map, post_feat_map]
        # FilterBanks.visualize_ft_map(pre_feat_map)
        # FilterBanks.visualize_ft_map(post_feat_map)

    # Calculate feature map similarity (Optional)
    # predict.calc_ft_map_sim(pre_feat_map)
    # predict.calc_ft_map_sim(post_feat_map)

    # Run the fully patched version
    if match_type == "patched":
        patch_match(
            segm_pre_post, feat_map_pre_post, load_ft_maps, [preEVT, tr_postEVT]
        )
        return

    if match_type == "rand_patch":

        # Get patched images
        patched_imgs = GMatch.get_patches(segm_pre_post, 128, 128)

        # Visualize patched pre and post evt
        GMatch.visualize_patches(patched_imgs)

        # Randomly select a common patch from pre-post evt to match points
        r_idx = np.random.randint(0, patched_imgs[0].shape[1])
        c_idx = np.random.randint(0, patched_imgs[0].shape[0])
        print(r_idx, c_idx)

        # Load the specific patches
        segm_pre_post[0] = patched_imgs[0][r_idx, c_idx, ...]
        segm_pre_post[1] = patched_imgs[1][r_idx, c_idx, ...]

    # 3. Skeletonization

    # Perform skeletonization
    skeleton_images, distance_transform = sklt.get_skeletons(
        segm_pre_post,
        method="lee",
    )
    if not skeleton_images:
        return

    # Multiply skeleton images with the distance transform for visualization (optional)
    # skeleton_images = [
    #     sklt * dst for sklt, dst in zip(skeleton_images, distance_transform)
    # ]
    # vis_skeleton_and_segm(skeleton_image, segm)

    vis_skeletons = sklt.VisualizeSkeletons(
        skeleton_images,
        segm_pre_post,
    )
    vis_skeletons.vis_images()
    bnext = Button(vis_skeletons.axnext, "Next")
    bnext.on_clicked(vis_skeletons.next)
    bprev = Button(vis_skeletons.axprev, "Previous")
    bprev.on_clicked(vis_skeletons.prev)

    plt.show()

    # 4. Create Graphs
    skeleton_points = sklt.find_centerlines(skeleton_images[0])
    pre_graph = GProc.create_graph(
        skeleton_images[0],
        skeleton_points,
        distance_transform[0],
        g_name="pre_graph",
        vis=False,
        verbose=True,
    )
    skeleton_points = sklt.find_centerlines(skeleton_images[1])
    post_graph = GProc.create_graph(
        skeleton_images[1],
        skeleton_points,
        distance_transform[1],
        g_name="post_graph",
        vis=False,
        verbose=True,
    )

    # Plot the two graph side-by-side (optional)
    # plot_pre_post(pre_graph, post_graph, preEVT, tr_postEVT, overlay_orig=True)
    # plot_pre_post(
    #     pre_graph, post_graph, segm_pre_post[0], segm_pre_post[1], overlay_seg=True
    # )

    # 5. Extract features from the segmenation model
    # and add them as graph attributes

    pre_feat_map = feat_map_pre_post[0]
    post_feat_map = feat_map_pre_post[1]
    # predict.display_feature_maps(feat_map_pre_post)

    # Scales used for multi-scaled matching
    scales = [(0, 2), (2, 3), (3, 4), 4]

    if match_type == "single" or match_type == "rand_patch":
        # Set scale to [0] to perform single scale matching
        # If 'single' matching is performed on the original entire graphs
        scales = [0]

    if match_type == "rand_patch":
        # Patch the feature as already done with the segmentations
        patched_feat_maps = GMatch.get_patches([pre_feat_map, post_feat_map], 128, 128)
        # Load the specific patches - same as the segmentation
        pre_feat_map = patched_feat_maps[0][:, r_idx, c_idx, ...]
        post_feat_map = patched_feat_maps[1][:, r_idx, c_idx, ...]

    # Variables to store lines and keypoints for plotting
    lines = []
    keypoints1 = []
    keypoints2 = []
    # Loop through all the scales perform matching and finally combine them
    for _, scale in enumerate(scales):
        cur_pre_graph = GProc.multiscale_graph(pre_graph, scale)
        cur_post_graph = GProc.multiscale_graph(post_graph, scale)
        # GProc.plot_pre_post(
        #     cur_pre_graph,
        #     cur_post_graph,
        #     segm_pre_post[0],
        #     final_segm_post,
        #     overlay_seg=True,
        # )

        # Test averaging the neighborhood (3x3) or (5x5) or (7x7) or (9x9) values for the features
        # Pre-EVT graph
        GProc.concat_extracted_features_v2(cur_pre_graph, pre_feat_map, inplace=True)
        # Post-EVT graph
        GProc.concat_extracted_features_v2(cur_post_graph, post_feat_map, inplace=True)
        # print(cur_pre_graph.vs[0].attributes())

        # 6. Preprocess features if needed

        # Save keypoints for imaging
        pre_kpts = [(float(pt[0]), float(pt[1])) for pt in cur_pre_graph.vs["coords"]]
        post_kpts = [(float(pt[0]), float(pt[1])) for pt in cur_post_graph.vs["coords"]]

        # Delete the coords attribute before calculating similarity
        del cur_pre_graph.vs["coords"]
        del cur_post_graph.vs["coords"]

        # Find min max radius of all nodes in the graphs (Optional)
        # pre_r_avg, pre_r_max, pre_r_min, pre_r_std = GMatch.calc_graph_radius_info(
        #     cur_pre_graph
        # )
        # print(
        #     f"R-Avg: {pre_r_avg}, R-Max: {pre_r_max}, R-Min: {pre_r_min}, R-Std: {pre_r_std}"
        # )

        # Normalize x,y and radius features - Prevent blowup with similarity mat (Optional)
        # cur_pre_graph.vs["x"] = [
        #     x / skeleton_images[0].shape[1] for x in cur_pre_graph.vs["x"]
        # ]
        # cur_pre_graph.vs["y"] = [
        #     y / skeleton_images[0].shape[0] for y in cur_pre_graph.vs["y"]
        # ]
        # cur_post_graph.vs["x"] = [
        #     x / skeleton_images[1].shape[1] for x in cur_post_graph.vs["x"]
        # ]
        # cur_post_graph.vs["y"] = [
        #     y / skeleton_images[1].shape[0] for y in cur_post_graph.vs["y"]
        # ]

        # Delete the node radius info since I think it is misleading
        del cur_pre_graph.vs["radius"]
        del cur_post_graph.vs["radius"]
        # For now coordinates cannot be deleted since they are used for plotting.

        # 7. Calculate graph node feature matrices and perform matching

        # Create a feature matrices from all the node attributes
        pre_feat_matrix = GMatch.create_feat_matrix(cur_pre_graph)
        post_feat_matrix = GMatch.create_feat_matrix(cur_post_graph)

        if load_ft_maps:
            # Only instance based available since distribution parameters weren't calculated
            # The normalizations are optional
            pre_feat_matrix = Norms.instance_normalization(pre_feat_matrix)
            post_feat_matrix = Norms.instance_normalization(post_feat_matrix)
            # pass
        else:
            # The normalizations are optional
            pre_feat_matrix = Norms.instance_normalization(pre_feat_matrix)
            post_feat_matrix = Norms.instance_normalization(post_feat_matrix)

            # pre_feat_matrix = Norms.distribution_normalization(
            #     pre_feat_matrix, norm_type="minmax", fts=2
            # )
            # post_feat_matrix = Norms.distribution_normalization(
            #     post_feat_matrix, norm_type="minmax", fts=2
            # )
            # pass

        # Calculate feature info (optional)
        # pre_feat_avg, pre_feat_max, pre_feat_min, pre_feat_std = (
        #     GMatch.calc_feat_matrix_info(pre_feat_matrix, vis=False)
        # )
        # print(
        #     f"Avg: {pre_feat_avg}, Max: {pre_feat_max}, Min: {pre_feat_min}, Std: {pre_feat_std}"
        # )

        # Calculate the node similarity matrix
        sim_matrix = GMatch.calc_similarity(pre_feat_matrix, post_feat_matrix)

        # Calculate the soft assignment matrix via Sinkhorn
        # Differes quite a bit from the hungarian and does not give 1-1 mapping
        # in many cases, so it's probably best to not use it for now.
        # Orig tau: 100 Orig iter: 250 for sinkhorn
        # assignment_mat = GMatch.calc_assignment_matrix(
        #     sim_matrix, tau=100, iter=250, method="sinkhorn"
        # )

        assignment_mat = GMatch.calc_assignment_matrix(sim_matrix, method="hungarian")

        # Final matchings depend on pre and post graph node count
        if assignment_mat.shape[0] < assignment_mat.shape[1]:
            matchings = np.argmax(assignment_mat, axis=1)  # row
            transposed = False
        else:
            matchings = np.argmax(assignment_mat, axis=0)  # column
            transposed = True

        # Was not used but could be useful.
        # Test masking based on feature vector Euclidean distance
        # if feat_dist.shape[0] < feat_dist.shape[1]:
        #     matchings = np.argmin(feat_dist, axis=1)
        #     transposed = False
        # else:
        #     matchings = np.argmin(feat_dist, axis=0)
        #     transposed = True

        # Test evaluation using positional Euclidean distance
        pre_pos = np.zeros((len(cur_pre_graph.vs), 2))
        for idx, pt in enumerate(pre_kpts):
            pre_pos[idx, :] = pt
        post_pos = np.zeros((len(cur_post_graph.vs), 2))
        for idx, pt in enumerate(post_kpts):
            post_pos[idx, :] = pt

        node_pos_dist = distance.cdist(pre_pos, post_pos, "euclidean")

        # If post evt has less nodes than pre we change the following orders
        if transposed:
            node_pos_dist = node_pos_dist.T
            temp_kpts = pre_kpts.copy()
            pre_kpts = post_kpts.copy()
            post_kpts = temp_kpts.copy()

        thresh = 15  # ~10px
        masked = np.where(
            [
                node_pos_dist[i, j] < thresh
                for i, j in zip([*range(len(matchings))], matchings)
            ],
            1,
            0,
        ).tolist()
        # masked = None

        # Keep track of the matched keypoints and their lines to draw them after

        lines_kpts = GMatch.get_kpts_lines(
            pre_kpts,
            post_kpts,
            matchings,
            im_width=segm_pre_post[1].shape[1],
            inliers=masked,
            transposed=transposed,
        )

        # Get values from the dictionary and extend the lists
        lines.extend(lines_kpts["lines"])
        keypoints1.extend(lines_kpts["kpts1"])
        keypoints2.extend(lines_kpts["kpts2"])

    # When the loop ends we plot our original graphs and images with the matches
    GMatch.multiscale_draw_matches_interactive(
        pre_graph,
        post_graph,
        segm_pre_post[0],
        segm_pre_post[1],
        keypoints1,
        keypoints2,
        lines,
        segm=True,
    )

    return


# Function to see how I can implement patches properly
def patch_match(
    pre_post_evt: list[np.ndarray],
    pre_post_feat_maps: list[np.ndarray],
    load_ft_maps: bool,
    orig_pre_post_evt: list[np.ndarray] | None = None,
) -> None:

    # Keep track of pre and post segmentations for visualization
    vis_pre_segm = pre_post_evt[0].copy()
    vis_post_segm = pre_post_evt[1].copy()

    pre_feat_map = pre_post_feat_maps[0]
    post_feat_map = pre_post_feat_maps[1]
    # predict.display_feature_maps(feat_map_pre_post)
    # Calculate feature map similarity (Optional)
    # predict.calc_ft_map_sim(pre_feat_map)
    # predict.calc_ft_map_sim(post_feat_map)

    patch_size = (128, 128)

    # Get patched images
    patched_imgs = GMatch.get_patches(pre_post_evt, patch_size[0], patch_size[1])
    patches_shape = (patched_imgs[0].shape[0], patched_imgs[0].shape[1])

    # Visualize patched pre and post evt (Optional)
    # GMatch.visualize_patches(patched_imgs)

    # Patch the feature maps as already done with the segmentations
    patched_feat_maps = GMatch.get_patches(
        [pre_feat_map, post_feat_map], patch_size[0], patch_size[1]
    )

    for_range = patched_imgs[0].shape[0] * patched_imgs[0].shape[1]

    # Variables to store lines and keypoints for plotting
    lines = []
    keypoints1 = []
    keypoints2 = []

    # Loop through all the patches and calculate matches, lines and keypoints for plotting
    for p_idx in range(0, for_range):

        # In order to use all patches and combine afterwards
        # I convert from an index used in the loop to x,y positions to use for offsets.
        # Calculate the row and col indexes
        # The formula would be I = x*img_width + y
        # To get x we use I / width and for y I % width.
        r_idx, c_idx = (
            int(p_idx / patched_imgs[0].shape[1]),
            int(p_idx % patched_imgs[0].shape[1]),
        )
        # Load the specific patches
        pre_post_evt[0] = patched_imgs[0][r_idx, c_idx, ...]
        pre_post_evt[1] = patched_imgs[1][r_idx, c_idx, ...]
        pre_feat_map = patched_feat_maps[0][:, r_idx, c_idx, ...]
        post_feat_map = patched_feat_maps[1][:, r_idx, c_idx, ...]
        # FilterBanks.visualize_ft_map(post_feat_map)

        # 3. Skeletonization

        # Perform skeletonization
        skeleton_images, distance_transform = sklt.get_skeletons(
            # [segm_pre_post[0], final_segm_post], method="lee"
            pre_post_evt,
            method="lee",
        )
        if not skeleton_images:
            return

        # A bit unnecessary to visualize every patch here
        # But good for debugging and thinking for simplifications
        # vis_skeletons = sklt.VisualizeSkeletons(
        #     # skeleton_images, [segm_pre_post[0], final_segm_post]
        #     skeleton_images,
        #     segm_pre_post,
        # )
        # vis_skeletons.vis_images()
        # bnext = Button(vis_skeletons.axnext, "Next")
        # bnext.on_clicked(vis_skeletons.next)
        # bprev = Button(vis_skeletons.axprev, "Previous")
        # bprev.on_clicked(vis_skeletons.prev)

        # plt.show()

        # 4. Create Graphs
        # In the case of patches where nodes are not present
        # We need to catch the exception and let it pass
        try:
            skeleton_points = sklt.find_centerlines(skeleton_images[0])
            pre_graph = GProc.create_graph(
                skeleton_images[0],
                skeleton_points,
                distance_transform[0],
                g_name="pre_graph",
                vis=False,
                verbose=True,
            )
            skeleton_points = sklt.find_centerlines(skeleton_images[1])
            post_graph = GProc.create_graph(
                skeleton_images[1],
                skeleton_points,
                distance_transform[1],
                g_name="post_graph",
                vis=False,
                verbose=True,
            )
        except:
            # We skip this patch and move to the next one
            continue

        # Plot the two graph side-by-side (optional)
        # plot_pre_post(pre_graph, post_graph, preEVT, tr_postEVT, overlay_orig=True)
        # plot_pre_post(
        #     pre_graph, post_graph, segm_pre_post[0], segm_pre_post[1], overlay_seg=True
        # )

        # 5. Add extracted features from the segmenation model as graph attributes

        # Set scale to [0] to perform single scale matching
        # For now we assume patched matching is only done on one scale
        scales = [0]

        # Loop through all the scales perform matching and finally combine them
        for _, scale in enumerate(scales):
            cur_pre_graph = GProc.multiscale_graph(pre_graph, scale)
            cur_post_graph = GProc.multiscale_graph(post_graph, scale)
            # plot_pre_post(
            #     cur_pre_graph,
            #     cur_post_graph,
            #     segm_pre_post[0],
            #     final_segm_post,
            #     overlay_seg=True,
            # )

            # Test averaging the neighborhood (3x3) or (5x5) or (7x7) or (9x9) values for the features
            # Pre-EVT graph
            GProc.concat_extracted_features_v2(
                cur_pre_graph, pre_feat_map, inplace=True
            )
            # Post-EVT graph
            GProc.concat_extracted_features_v2(
                cur_post_graph, post_feat_map, inplace=True
            )
            # print(cur_pre_graph.vs[0].attributes())

            # 6. Preprocess features if needed

            # Save keypoints for visualization
            pre_kpts = [
                (float(pt[0]), float(pt[1])) for pt in cur_pre_graph.vs["coords"]
            ]
            post_kpts = [
                (float(pt[0]), float(pt[1])) for pt in cur_post_graph.vs["coords"]
            ]

            # Delete the coords attribute before calculating similarity
            del cur_pre_graph.vs["coords"]
            del cur_post_graph.vs["coords"]

            # Find min max radius of all nodes in the graphs for normalization (Optional)
            # pre_r_avg, pre_r_max, pre_r_min, pre_r_std = GMatch.calc_graph_radius_info(
            #     cur_pre_graph, verbose=True
            # )

            # Normalize x,y and radius features - Prevent blowup with similarity mat (Optional)
            # cur_pre_graph.vs["x"] = [
            #     x / skeleton_images[0].shape[1] for x in cur_pre_graph.vs["x"]
            # ]
            # cur_pre_graph.vs["y"] = [
            #     y / skeleton_images[0].shape[0] for y in cur_pre_graph.vs["y"]
            # ]
            # cur_post_graph.vs["x"] = [
            #     x / skeleton_images[1].shape[1] for x in cur_post_graph.vs["x"]
            # ]
            # cur_post_graph.vs["y"] = [
            #     y / skeleton_images[1].shape[0] for y in cur_post_graph.vs["y"]
            # ]

            # Delete the node radius info since I think it is misleading
            del cur_pre_graph.vs["radius"]
            del cur_post_graph.vs["radius"]

            # 7. Calculate graph node feature matrices and perform matching

            # Create a feature matrices from all the node attributes
            pre_feat_matrix = GMatch.create_feat_matrix(cur_pre_graph)
            post_feat_matrix = GMatch.create_feat_matrix(cur_post_graph)

            if load_ft_maps:
                # Only instance based available since distribution parameters weren't calculated
                # The normalizations are optional
                pre_feat_matrix = Norms.instance_normalization(pre_feat_matrix)
                post_feat_matrix = Norms.instance_normalization(post_feat_matrix)
                # pass
            else:
                # The normalizations are optional
                # pre_feat_matrix = Norms.instance_normalization(pre_feat_matrix)
                # post_feat_matrix = Norms.instance_normalization(post_feat_matrix)

                # pre_feat_matrix = Norms.distribution_normalization(
                #     pre_feat_matrix, norm_type="minmax", fts=2
                # )
                # post_feat_matrix = Norms.distribution_normalization(
                #     post_feat_matrix, norm_type="minmax", fts=2
                # )
                pass

            # Calculate feature info (Optional)
            # pre_feat_avg, pre_feat_max, pre_feat_min, pre_feat_std = (
            #     GMatch.calc_feat_matrix_info(pre_feat_matrix, vis=False)
            # )
            # print(
            #     f"Avg: {pre_feat_avg}, Max: {pre_feat_max}, Min: {pre_feat_min}, Std: {pre_feat_std}"
            # )

            # Calculate the node similarity matrix
            sim_matrix = GMatch.calc_similarity(pre_feat_matrix, post_feat_matrix)

            # Calculate the soft assignment matrix via Sinkhorn
            # Differes quite a bit from the hungarian and does not give 1-1 mapping
            # in many cases, so it's probably best to not use it for now.
            # Orig tau: 100 Orig iter: 250 for sinkhorn
            # assignment_mat = gm.calc_assignment_matrix(
            #     sim_matrix, tau=100, iter=250, method="sinkhorn"
            # )

            assignment_mat = GMatch.calc_assignment_matrix(
                sim_matrix, method="hungarian"
            )

            # This complicates the drawing later if they have to be transposed.
            if assignment_mat.shape[0] < assignment_mat.shape[1]:
                matchings = np.argmax(assignment_mat, axis=1)  # row
                transposed = False
            else:
                matchings = np.argmax(assignment_mat, axis=0)  # column
                transposed = True
            # print(len(matchings))

            # # Test matching based on feature vector Euclidean distance
            # feat_dist = distance.cdist(pre_feat_matrix, post_feat_matrix, "euclidean")
            # # Get matchings based on minimum euclidean distance
            # if feat_dist.shape[0] < feat_dist.shape[1]:
            #     matchings = np.argmin(feat_dist, axis=1)
            #     transposed = False
            # else:
            #     matchings = np.argmin(feat_dist, axis=0)
            #     transposed = True

            # Also tried this method but it had issues. Not sure where the problem is.
            # ### Spectral matching - Memory is no issue here but still has a problem

            # # Calculate affinity matrix
            # K = GMatch.calc_affinity_matrix(cur_pre_graph, cur_post_graph)

            # # Solve using spectral matching
            # assignment_mat = GMatch.spectral_matcher(
            #     K, len(cur_pre_graph.vs), len(cur_post_graph.vs)
            # )

            # if assignment_mat.shape[0] < assignment_mat.shape[1]:
            #     matchings = np.argmax(assignment_mat, axis=1)  # row
            #     transposed = False
            # else:
            #     matchings = np.argmax(assignment_mat, axis=0)  # column
            #     transposed = True

            # Use positional Euclidean distance kind of like a filter
            pre_pos = np.zeros((len(cur_pre_graph.vs), 2))
            for idx, pt in enumerate(pre_kpts):
                pre_pos[idx, :] = pt
            post_pos = np.zeros((len(cur_post_graph.vs), 2))
            for idx, pt in enumerate(post_kpts):
                post_pos[idx, :] = pt

            node_pos_dist = distance.cdist(pre_pos, post_pos, "euclidean")

            # If post evt has less nodes than pre we change the following orders
            if transposed:
                node_pos_dist = node_pos_dist.T
                temp_kpts = pre_kpts.copy()
                pre_kpts = post_kpts.copy()
                post_kpts = temp_kpts.copy()

            thresh = 15  # ~10px
            masked = np.where(
                [
                    node_pos_dist[i, j] < thresh
                    for i, j in zip([*range(len(matchings))], matchings)
                ],
                1,
                0,
            ).tolist()
            # masked = None

            # Keep track of kpts and lines to draw after
            lines_kpts = GMatch.get_kpts_lines_patched(
                pre_kpts,
                post_kpts,
                matchings,
                im_width=vis_pre_segm.shape[1],  # Has to be 512
                patch_size=patch_size,
                patches_shape=patches_shape,
                patch_index=p_idx,
                inliers=masked,
                transposed=transposed,
            )

            # Get intermediate patch matches for plotting
            tmp_lines_kpts = GMatch.get_kpts_lines(
                pre_kpts,
                post_kpts,
                matchings,
                im_width=pre_post_evt[1].shape[1],
                inliers=masked,
                transposed=transposed,
            )

            # This should  happen only if they have been normalized.
            # Reverse normalization of x,y for plotting
            cur_pre_graph.vs["x"] = [
                x * skeleton_images[0].shape[1] for x in cur_pre_graph.vs["x"]
            ]
            cur_pre_graph.vs["y"] = [
                y * skeleton_images[0].shape[0] for y in cur_pre_graph.vs["y"]
            ]
            cur_post_graph.vs["x"] = [
                x * skeleton_images[1].shape[1] for x in cur_post_graph.vs["x"]
            ]
            cur_post_graph.vs["y"] = [
                y * skeleton_images[1].shape[0] for y in cur_post_graph.vs["y"]
            ]

            # Plot intermediate patch results (Optional)
            # GMatch.multiscale_draw_matches_interactive(
            #     cur_pre_graph,
            #     cur_post_graph,
            #     pre_post_evt[0],
            #     pre_post_evt[1],
            #     tmp_lines_kpts["kpts1"],
            #     tmp_lines_kpts["kpts2"],
            #     tmp_lines_kpts["lines"],
            #     segm=True,
            # )

            # Extend the lists
            lines.extend(lines_kpts["lines"])
            keypoints1.extend(lines_kpts["kpts1"])
            keypoints2.extend(lines_kpts["kpts2"])

    # Calculate the original graphs to use for visualization
    # Perform skeletonization
    skeleton_images, distance_transform = sklt.get_skeletons(
        [vis_pre_segm, vis_post_segm],
        method="lee",
    )
    if not skeleton_images:
        return

    # Create Graphs
    skeleton_points = sklt.find_centerlines(skeleton_images[0])
    pre_graph = GProc.create_graph(
        skeleton_images[0],
        skeleton_points,
        distance_transform[0],
        g_name="pre_graph",
        vis=False,
        verbose=True,
    )
    skeleton_points = sklt.find_centerlines(skeleton_images[1])
    post_graph = GProc.create_graph(
        skeleton_images[1],
        skeleton_points,
        distance_transform[1],
        g_name="post_graph",
        vis=False,
        verbose=True,
    )

    # When the loop ends we plot our original graphs and images with the matches
    GMatch.multiscale_draw_matches_interactive(
        pre_graph,
        post_graph,
        orig_pre_post_evt[0],  # orig_pre_post_evt[0],  # vis_pre_segm,
        orig_pre_post_evt[1],  # orig_pre_post_evt[1],  # vis_post_segm,
        keypoints1,
        keypoints2,
        lines,
        segm=False,
    )


# fmt: off
def get_args():
    parser = argparse.ArgumentParser(description="Perform graph matching on a set of pre/post-EVT DSA images")
    # parser.add_argument('in_img_path', help='Input image to be segmented.')
    parser.add_argument("--match-type", "-t", help="Type of match to perform - single | multi | patched")
    parser.add_argument("--load-segs",action="store_true",default=False,help="Load the segmentations and feature maps.")
    parser.add_argument("--load-ft-maps",action="store_true",default=False,help="Load the feature maps corresponding to the segmentations.")
#fmt:on

    return parser.parse_args()


if __name__ == "__main__":
    # Inspiration for passing args:
    # https://www.peterbe.com/plog/vars-argparse-namespace-into-a-function
    args = get_args()
    main(**vars(args))
