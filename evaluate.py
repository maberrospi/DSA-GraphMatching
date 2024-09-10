from glob import glob
import os
from pathlib import Path
import sys
import logging
import argparse
import numpy as np
import imageio
import json
import random
import matplotlib.pyplot as plt
from time import perf_counter
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# My imports
import SIFTTransform as sift
import Skeletonization as sklt
import graph_matching as GMatch
import graph_processing as GProc
import filter_banks as FilterBanks
import norm_params

logger = logging.getLogger(__name__)

# This file includes the preparation of the evaluation data
# And the evaluation using a set of annotations.


def prepare_patches(img_dir, output_dir="PatchedImages"):
    # Loop through the patients images and segmentations
    if not os.path.isdir(img_dir):
        logger.warning("Path directory {} does not exist".format(img_dir))
        sys.exit(1)

    patient_folders = glob(img_dir + "/*")
    logger.info(f"Extracting patches to {os.path.join(Path.cwd(),output_dir)}")

    for folder in patient_folders:
        # Loop through the two patient orientation folders
        pat_id = os.path.basename(folder)
        orientations = glob(folder + "/*")
        for orientation in orientations:
            pat_ori = os.path.basename(orientation)
            images_path = sift.load_img_dir(orientation, img_type="nifti")

            # Check if list is empty
            if not images_path:
                break

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
            matches = sift.find_feat_matches(
                predsc, postdsc, feat_extr=feature_extractor
            )
            # sift.plot_matches(preEVT, prekp, postEVT, postkp, matches,feat_extr=feature_extractor)
            transformation = sift.calculate_transform(
                prekp, postkp, matches, feat_extr=feature_extractor
            )

            # Assume that segmentations are ALWAYS being loaded
            IMG_SEQ_DIR_PATH = (
                "C:/Users/mab03/Desktop/ThesisCode/Segms/Sequence/"
                + pat_id
                + "/"
                + pat_ori
            )

            segm_images = sklt.load_images(IMG_SEQ_DIR_PATH)

            # Find corresponding pre and post images from the segmentations
            segm_pre_post = []
            for segm in segm_images:
                if Path(segm).stem.rsplit("_", 1)[1] == "artery":
                    if Path(segm).stem.rsplit("_", 2)[1] == "pre":
                        segm_pre_post.insert(0, sift.load_img(segm))
                    else:
                        segm_pre_post.append(sift.load_img(segm))

            # Check if the transformation quality is better than the original
            if not sift.check_transform(transformation, preEVT, postEVT, verbose=False):
                # The transformation is worse than the original
                print("Using the original post-EVT image")
                final_segm_post = segm_pre_post[1]
                tr_postEVT = postEVT
            else:
                tr_postEVT = sift.apply_transformation(
                    transformation, preEVT, postEVT, ret=True, vis=False
                )
                # Scale transformation matrix for segmentation image size
                scale_matrix = np.array([[0.5, 0, 0], [0, 0.5, 0], [0, 0, 1]])
                transformation = (
                    scale_matrix @ transformation @ np.linalg.inv(scale_matrix)
                )
                # Warp and display the segmentations
                final_segm_post = sift.apply_transformation(
                    transformation,
                    segm_pre_post[0],
                    segm_pre_post[1],
                    ret=True,
                    vis=False,
                )

            # Careful with this since arrays are mutable
            # Used final_segm_post in order to plot original and transformed post
            # After the plotting this variable is not necessary anymore.
            segm_pre_post[1] = final_segm_post

            patch_size = (128, 128)

            # Get patched images
            patched_imgs = GMatch.get_patches(
                segm_pre_post, patch_size[0], patch_size[1]
            )
            patches_shape = (patched_imgs[0].shape[0], patched_imgs[0].shape[1])

            # Visualize patched pre and post evt
            # GMatch.visualize_patches(patched_imgs)

            n_patches_to_save = 2
            saved_patches = 0
            visited = []

            while saved_patches != n_patches_to_save:
                # Randomly select a common patch from pre-post evt to match points
                r_idx = np.random.randint(0, patched_imgs[0].shape[1])
                c_idx = np.random.randint(0, patched_imgs[0].shape[0])

                if (r_idx, c_idx) in visited:
                    continue
                visited.append((r_idx, c_idx))

                # Load the specific patches
                pre_post_patches = [None] * 2
                pre_post_patches[0] = patched_imgs[0][r_idx, c_idx, ...]
                pre_post_patches[1] = patched_imgs[1][r_idx, c_idx, ...]

                # Perform skeletonization
                skeleton_images, distance_transform = sklt.get_skeletons(
                    pre_post_patches,
                    method="lee",
                )
                if not skeleton_images:
                    continue

                pre_patch_w_skeleton = pre_post_patches[0][:]
                post_patch_w_skeleton = pre_post_patches[1][:]
                pre_patch_w_skeleton = np.where(
                    skeleton_images[0] == 1, 0.5, pre_patch_w_skeleton
                )
                post_patch_w_skeleton = np.where(
                    skeleton_images[1] == 1, 0.5, post_patch_w_skeleton
                )
                # Only for debugging
                # fig, axs = plt.subplots(1, 2, figsize=(6, 6))
                # axs[0].imshow(pre_patch_w_skeleton, cmap="gray")
                # axs[1].imshow(post_patch_w_skeleton, cmap="gray")
                # # fig, axs = plt.subplots(1, 2, figsize=(8, 6))
                # # axs[0].imshow(pre_post_patches[0], cmap="gray")
                # # axs[0].imshow(skeleton_images[0], cmap="Purples", alpha=0.5)
                # # axs[1].imshow(pre_post_patches[1], cmap="gray")
                # # axs[1].imshow(skeleton_images[1], cmap="Purples", alpha=0.5)
                # # axs[0].set_title("Pre-patch")
                # # axs[1].set_title("Post-patch")
                # plt.show()

                pre_patch_w_skeleton = pre_patch_w_skeleton * 255
                post_patch_w_skeleton = post_patch_w_skeleton * 255
                pre_patch_w_skeleton = pre_patch_w_skeleton.astype(np.uint8)
                post_patch_w_skeleton = post_patch_w_skeleton.astype(np.uint8)

                # Create Graphs
                # In the case of patches where nodes are not present
                # We need to catch the exception and get a different patch
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

                # If succesfull increase number of saved patch images
                saved_patches += 1

                # Save the segmentation patches
                patch_dst_path = os.path.join(
                    output_dir,
                    "{}.png".format(
                        "_".join([pat_id, pat_ori, "pre", str(r_idx), str(c_idx)])
                    ),
                )  # pat_id + "_" + pat_ori + "_pre" + '_' + r_idx + '_' + c_idx
                Path(patch_dst_path).parent.mkdir(parents=True, exist_ok=True)
                logger.info("Saving patch to {}".format(patch_dst_path))
                imageio.imwrite(patch_dst_path, pre_patch_w_skeleton)
                patch_dst_path = os.path.join(
                    output_dir,
                    "{}.png".format(
                        "_".join([pat_id, pat_ori, "post", str(r_idx), str(c_idx)])
                    ),
                )  # pat_id + "_" + pat_ori + "_pre" + '_' + r_idx + '_' + c_idx
                Path(patch_dst_path).parent.mkdir(parents=True, exist_ok=True)
                logger.info("Saving patch to {}".format(patch_dst_path))
                imageio.imwrite(patch_dst_path, post_patch_w_skeleton)

    logger.info("All patches in the directory saved succesfully!")


def visualize_matches(pre_kpts, post_kpts, img1, img2, true_matches, g1_idxs, g2_idxs):
    colors = [
        (0, 1, 0),
        (0, 0, 1),
        (1, 0, 0),
        (0, 1, 1),
        (1, 0, 1),
        (1, 1, 0),
        (1, 1, 1),
        (0, 0, 0),
        (0.5, 1, 0),
        (1, 0.5, 0),
        (1, 0, 0.5),
        (0.5, 0, 1),
        (0, 0.5, 1),
        (0, 1, 0.5),
        (0.5, 0.5, 0.5),
        (0.5, 0, 0),
        (0, 0.5, 0),
        (0, 0, 0.5),
        (0.5, 0.5, 0),
        (0.5, 0, 0.5),
    ]
    # Random shuffle the colors so we don't get similar colors next to eachother.
    random.shuffle(colors)

    fig, axs = plt.subplots(1, 2)
    axs[0].scatter([y for (x, y) in pre_kpts.keys()], [x for (x, y) in pre_kpts.keys()])
    axs[1].scatter(
        [y for (x, y) in post_kpts.keys()], [x for (x, y) in post_kpts.keys()]
    )
    axs[0].imshow(img1, cmap="gray")
    axs[1].imshow(img2, cmap="gray")

    for i, points in enumerate(true_matches):
        # Choose color based on the index
        color = colors[i % len(colors)]
        # Load annotated points - Integers
        x1 = round(points.get("x1"))
        y1 = round(points.get("y1"))
        x2 = round(points.get("x2"))
        y2 = round(points.get("y2"))

        # print(x1, y1, x2, y2, sep=" - ")
        axs[0].scatter(x1, y1, color=color)
        axs[1].scatter(x2, y2, color=color)

    plt.show()

    # for match in zip(g1_idxs,g2_idxs):
    #     ax.plot([x1, x2], [y1, y2], color=color, linewidth=1)


def instance_normalization(feature_matrix, norm_type="zscore"):
    assert norm_type in ["zscore", "minmax"]

    ft_matrix = feature_matrix.copy()

    if norm_type == "zscore":
        z_score = StandardScaler()
        ft_matrix = z_score.fit_transform(ft_matrix)
        # This was used when coords were not included in zscore norm
        # ft_matrix = np.concatenate(
        #     (
        #         ft_matrix[:, :2],
        #         z_score.fit_transform(ft_matrix[:, 2:]),
        #     ),
        #     axis=1,
        # )
    elif norm_type == "minmax":
        min_max = MinMaxScaler()
        ft_matrix = min_max.fit_transform(ft_matrix)

    return ft_matrix


def distribution_normalization(feature_matrix, norm_type="zscore", fts=1):
    assert norm_type in ["zscore", "minmax"]
    assert fts in [1, 2, 3, 4]

    # fts has four ids:
    # 1: Radius and LM fts
    # 2: Coordinates and LM fts
    # 3: Radius and Coordinates
    # 4: Only LM features

    ft_matrix = feature_matrix.copy()

    new_min = norm_params.distr_min
    new_max = norm_params.distr_max
    new_mean = norm_params.distr_mean
    new_std = norm_params.distr_std

    if norm_type == "zscore":

        # The zcore depends on what features are used
        # Since our normalization parameter matrix includes all fts

        match fts:
            case 1:
                # Not Coords and Radius
                ft_matrix = (ft_matrix - new_mean[:, 2:]) / new_std[:, 2:]
            case 2:
                # Coords and NOT Radius
                ft_matrix = np.concatenate(
                    (
                        (ft_matrix[:, :2] - new_mean[:, :2]) / new_std[:, :2],
                        (ft_matrix[:, 2:] - new_mean[:, 3:]) / new_std[:, 3:],
                    ),
                    axis=1,
                )
            case 3:
                # Coords and Radius
                ft_matrix = (ft_matrix - new_mean) / new_std
            case 4:
                # NOT coords NOT radius
                ft_matrix = (ft_matrix - new_mean[:, 3:]) / new_std[:, 3:]

    elif norm_type == "minmax":
        match fts:
            case 1:
                # Not Coords and Radius
                ft_matrix = (ft_matrix - new_min[:, 2:]) / (
                    new_max[:, 2:] - new_min[:, 2:]
                )
            case 2:
                # Coords and NOT Radius
                ft_matrix = np.concatenate(
                    (
                        (ft_matrix[:, :2] - new_min[:, :2])
                        / (new_max[:, :2] - new_min[:, :2]),
                        (ft_matrix[:, 2:] - new_min[:, 3:])
                        / (new_max[:, 3:] - new_min[:, 3:]),
                    ),
                    axis=1,
                )
            case 3:
                # Coords and Radius
                ft_matrix = (ft_matrix - new_min) / (new_max - new_min)
            case 4:
                # NOT coord NOT Radius
                ft_matrix = (ft_matrix - new_min[:, 3:]) / (
                    new_max[:, 3:] - new_min[:, 3:]
                )

    return ft_matrix


def bootstrap(data: list, n_splits=100):
    bootstrap_list = []
    for i in range(n_splits):
        random.seed(a=i)  # For reproducibility
        bootstrap_list.append(random.choices(data, k=len(data)))

    return bootstrap_list


def evaluate_patches(
    annot_dir, match_type="patched", load_ft_maps=False, calculate_ci=False
):

    # Check the validity of arguments
    assert match_type in [
        "single",
        "multi",
        "patched",
    ], "Invalid matching type"

    # Initialise metrics
    TP = FP = FN = 0
    total_time = 0
    total_time_loading = 0
    total_time_gr = 0

    # Load annotations
    json__files = sorted(glob(os.path.join(annot_dir, "**", "*.json"), recursive=True))

    # Calculate Confidence Interval
    if calculate_ci:
        logger.info("Calculating Recall with Confidence Interval using Bootstrap...")
        recall_values = []
        bootstrap_json_files = bootstrap(json__files, n_splits=20)
        LM_filter_banks = None
        if not load_ft_maps:
            # Create Leung-Malik filter bank
            LM_filter_banks = FilterBanks.makeLMfilters()
        for json_files in tqdm(
            bootstrap_json_files,
            total=len(bootstrap_json_files),
            desc="Bootstrap iteration",
            unit="set",
        ):
            recall_values.append(
                bootstrap_eval(
                    json_files,
                    lm_filters=LM_filter_banks,
                    match_type=match_type,
                    load_ft_maps=load_ft_maps,
                )
            )

        estimate = np.mean(recall_values)
        lower_bound = np.percentile(recall_values, 100 * (0.05 / 2))  # 95th Percentile
        upper_bound = np.percentile(recall_values, 100 * (1 - 0.05 / 2))
        stderr = np.std(recall_values)
        print(
            f"Estimate: {estimate} - Lower Bound: {lower_bound} - Upper Bound: {upper_bound} - Error: {stderr}"
        )

        # Plot histogram of recall values
        counts, bins = np.histogram(recall_values)
        plt.stairs(counts, bins)
        plt.xlabel("Recall")
        plt.ylabel("Frequency")
        plt.show()
        return

    logger.info("Calculating Recall of annotated dataset.")

    # Prepare for Leung-Malik
    if not load_ft_maps:
        # Create Leung-Malik filter bank
        LM_filter_banks = FilterBanks.makeLMfilters()

    for json_path in json__files:
        with open(json_path, "r") as json_file:
            data = json.load(json_file)
            true_matches = data["matches"]
            # print(idx, json_path, matches, sep=" - ")
            json_path_stem = Path(json_path).stem
            print("########################")
            print(json_path_stem)

            # Extract patient id, orientation and patch index from annotation name
            json_pre, _ = json_path_stem.split("-")
            json_pre_split = json_pre.split("_")

            pat_id = json_pre_split[0]
            pat_ori = json_pre_split[1]
            patch_row_idx, patch_col_idx = int(json_pre_split[3]), int(
                json_pre_split[4]
            )

            # Start timer
            t2_start = perf_counter()

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
            matches = sift.find_feat_matches(
                predsc, postdsc, feat_extr=feature_extractor
            )
            # sift.plot_matches(preEVT, prekp, postEVT, postkp, matches,feat_extr=feature_extractor)
            transformation = sift.calculate_transform(
                prekp, postkp, matches, feat_extr=feature_extractor
            )

            # 2. Segmentation
            # Since this is the evaluation we assume the segmentations and feature maps will always be loaded.

            IMG_SEQ_DIR_PATH = (
                "C:/Users/mab03/Desktop/ThesisCode/Segms/Sequence/"
                + pat_id
                + "/"
                + pat_ori
            )

            FEAT_MAP_DIR_PATH = "FeatMapsv2/" + pat_id + "/" + pat_ori
            feat_map_pre_post = []

            segm_images1 = sklt.load_images(IMG_SEQ_DIR_PATH)

            # Find corresponding pre and post images from the segmentations and feat maps
            segm_pre_post = []
            for segm in segm_images1:
                if Path(segm).stem.rsplit("_", 1)[1] == "artery":
                    if Path(segm).stem.rsplit("_", 2)[1] == "pre":
                        pre = True
                        segm_pre_post.insert(0, sift.load_img(segm))
                    else:
                        pre = False
                        segm_pre_post.append(sift.load_img(segm))

                    if load_ft_maps:
                        # Load feature maps
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
                    transformation, preEVT, postEVT, ret=True, vis=False
                )
                # Scale transformation matrix for segmentation image size
                scale_matrix = np.array([[0.5, 0, 0], [0, 0.5, 0], [0, 0, 1]])
                transformation = (
                    scale_matrix @ transformation @ np.linalg.inv(scale_matrix)
                )
                # Warp and display the segmentations
                final_segm_post = sift.apply_transformation(
                    transformation,
                    segm_pre_post[0],
                    segm_pre_post[1],
                    ret=True,
                    vis=False,
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

            if not load_ft_maps:
                # Use Leung-Malik features
                pre_feat_map = FilterBanks.create_ft_map(LM_filter_banks, preEVT)
                post_feat_map = FilterBanks.create_ft_map(LM_filter_banks, tr_postEVT)
                feat_map_pre_post = [pre_feat_map, post_feat_map]
                # FilterBanks.visualize_ft_map(pre_feat_map)
                # FilterBanks.visualize_ft_map(post_feat_map)

            # Start timer
            t1_start = perf_counter()
            # Timer for loading things
            load_elapsed = t1_start - t2_start
            total_time_loading += load_elapsed

            if match_type == "patched":
                # Get patched images
                patched_imgs = GMatch.get_patches(segm_pre_post, 128, 128)
                # Load the specific patches
                segm_pre_post[0] = patched_imgs[0][patch_row_idx, patch_col_idx, ...]
                segm_pre_post[1] = patched_imgs[1][patch_row_idx, patch_col_idx, ...]

            # 3. Skeletonization

            # Perform skeletonization
            skeleton_images, distance_transform = sklt.get_skeletons(
                segm_pre_post,
                method="lee",
            )
            if not skeleton_images:
                return

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

            # 5. Extract features from the segmenation model
            # and add them as graph attributes

            pre_feat_map = feat_map_pre_post[0]
            post_feat_map = feat_map_pre_post[1]

            if match_type == "multi":
                t_gr_stop = perf_counter()
                # # Timer for creating graphs
                load_elapsed = t_gr_stop - t1_start
                total_time_gr += load_elapsed

            # Scales used for multi-scaled matching
            scales = [(0, 2), (2, 3), (3, 4), 4]

            if match_type == "single" or match_type == "patched":
                # Set scale to [0] to perform single scale matching
                # If 'single' matching is performed on the original entire graphs
                scales = [0]

            if match_type == "patched":
                # Patch the feature as already done with the segmentations
                patched_feat_maps = GMatch.get_patches(
                    [pre_feat_map, post_feat_map], 128, 128
                )
                # Load the specific patches - same as the segmentation
                pre_feat_map = patched_feat_maps[0][
                    :, patch_row_idx, patch_col_idx, ...
                ]
                post_feat_map = patched_feat_maps[1][
                    :, patch_row_idx, patch_col_idx, ...
                ]

            cur_segm_pre_post = segm_pre_post.copy()
            # Loop through all the scales perform matching and finally combine them
            for _, scale in enumerate(scales):
                # Start timer
                t3_start = perf_counter()
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
                GProc.concat_extracted_features_v2(
                    cur_pre_graph, pre_feat_map, nb_size=24, inplace=True
                )
                # Post-EVT graph
                GProc.concat_extracted_features_v2(
                    cur_post_graph, post_feat_map, nb_size=24, inplace=True
                )

                # 6. Preprocess features if needed

                # Save keypoints for imaging
                pre_kpts = [
                    (float(pt[0]), float(pt[1])) for pt in cur_pre_graph.vs["coords"]
                ]
                post_kpts = [
                    (float(pt[0]), float(pt[1])) for pt in cur_post_graph.vs["coords"]
                ]

                # Delete the coords attribute before calculating similarity
                del cur_pre_graph.vs["coords"]
                del cur_post_graph.vs["coords"]

                # Calculate radius info if needed
                # pre_r_avg, pre_r_max, pre_r_min, pre_r_std = (
                #     GMatch.calc_graph_radius_info(cur_pre_graph, verbose=False)
                # )
                # print(
                #     f"R-Avg: {pre_r_avg}, R-Max: {pre_r_max}, R-Min: {pre_r_min}, R-Std: {pre_r_std}"
                # )

                # Normalize x,y and radius features - Prevent blowup with similarity mat
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

                # Delete the node radius features (optional)
                del cur_pre_graph.vs["radius"]
                del cur_post_graph.vs["radius"]
                # Delete x,y features if the graph visualization is not used (optional)
                # del cur_pre_graph.vs["x"]
                # del cur_pre_graph.vs["y"]
                # del cur_post_graph.vs["y"]
                # del cur_post_graph.vs["x"]

                # 7. Calculate graph node feature matrices and perform matching

                # Create a feature matrices from all the node attributes
                pre_feat_matrix = GMatch.create_feat_matrix(cur_pre_graph)
                post_feat_matrix = GMatch.create_feat_matrix(cur_post_graph)

                if load_ft_maps:
                    # Only instance based available since distribution parameters weren't calculated
                    # The normalizations are optional
                    pre_feat_matrix = instance_normalization(pre_feat_matrix)
                    post_feat_matrix = instance_normalization(post_feat_matrix)
                    # pass
                else:
                    # The normalizations are optional
                    pre_feat_matrix = instance_normalization(pre_feat_matrix)
                    post_feat_matrix = instance_normalization(post_feat_matrix)

                    # pre_feat_matrix = distribution_normalization(
                    #     pre_feat_matrix, norm_type="minmax", fts=2
                    # )
                    # post_feat_matrix = distribution_normalization(
                    #     post_feat_matrix, norm_type="minmax", fts=2
                    # )
                    # pass

                # Calculate feature info if needed
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

                # Final matchings depend on pre and post graph node count
                if assignment_mat.shape[0] < assignment_mat.shape[1]:
                    matchings = np.argmax(assignment_mat, axis=1)  # row
                    transposed = False
                else:
                    matchings = np.argmax(assignment_mat, axis=0)  # column
                    transposed = True
                # print(len(matchings))

                if match_type == "patched" or match_type == "single":
                    # Use this for patched and single-scale
                    t1_stop = perf_counter()
                    elapsed = t1_stop - t1_start
                    total_time += elapsed
                elif match_type == "multi":
                    # Use this for multi-scaled performance
                    # Start timer
                    t3_stop = perf_counter()
                    # Timer for running matching for scales
                    load_elapsed = t3_stop - t3_start
                    total_time += load_elapsed

                graph_node_idx = [*range(len(matchings))]

                # For every point in True Matches we need to check if there are any
                # matched points in the corresponding pre-post areas.
                # If a match is found TP+=1 else FN+=1
                # TN don't have any point and FP will be the count of points that
                # are matched but don't exist in the True Matches (Does not say much since True Matches are not perfect)

                # For VISUALIZATION:
                # get the patch slices from the original segmentation images
                if match_type == "patched":
                    pre_kpts = {
                        (round(x), round(y)): idx for idx, (x, y) in enumerate(pre_kpts)
                    }
                    post_kpts = {
                        (round(x), round(y)): idx
                        for idx, (x, y) in enumerate(post_kpts)
                    }
                elif match_type == "single" or match_type == "multi":
                    # Seems to be capturing the points okay
                    pre_kpts = {
                        (
                            round(x) - patch_row_idx * 128,
                            round(y) - (patch_col_idx) * 128,
                        ): idx
                        for idx, (x, y) in enumerate(pre_kpts)
                        if x in range(patch_row_idx * 128, (patch_row_idx + 1) * 128)
                        and y in range(patch_col_idx * 128, (patch_col_idx + 1) * 128)
                    }
                    post_kpts = {
                        (
                            round(x) - patch_row_idx * 128,
                            round(y) - (patch_col_idx) * 128,
                        ): idx
                        for idx, (x, y) in enumerate(post_kpts)
                        if x in range(patch_row_idx * 128, (patch_row_idx + 1) * 128)
                        and y in range(patch_col_idx * 128, (patch_col_idx + 1) * 128)
                    }

                    # Now the points have to be scaled down

                    cur_segm_pre_post[0] = segm_pre_post[0][
                        patch_row_idx * 128 : (patch_row_idx + 1) * 128,
                        patch_col_idx * 128 : (patch_col_idx + 1) * 128,
                    ]
                    cur_segm_pre_post[1] = segm_pre_post[1][
                        patch_row_idx * 128 : (patch_row_idx + 1) * 128,
                        patch_col_idx * 128 : (patch_col_idx + 1) * 128,
                    ]

                    # fig, axs = plt.subplots(1, 2)
                    # axs[0].scatter(
                    #     [y for (x, y) in pre_kpts], [x for (x, y) in pre_kpts]
                    # )
                    # axs[1].scatter(
                    #     [y for (x, y) in post_kpts], [x for (x, y) in post_kpts]
                    # )
                    # axs[0].imshow(segm_pre_post[0], cmap="gray")
                    # axs[1].imshow(segm_pre_post[1], cmap="gray")
                    # plt.show()

                patch_TP = 0

                # print(list(pre_kpts.items()))
                # print(list(post_kpts.items()))

                # Only for Visualization (didn't use them after all)
                g1_idxs = []
                g2_idxs = []

                for idx, points in enumerate(true_matches):
                    # Load annotated points - Integers
                    x1 = round(points.get("x1"))
                    y1 = round(points.get("y1"))
                    x2 = round(points.get("x2"))
                    y2 = round(points.get("y2"))

                    # print(x1, y1, x2, y2, sep=" - ")

                    pre_points_to_check = []
                    # Check if there are any matches in the point areas
                    for nb in GProc.candidate_neighbors((y1, x1), hood_size=81):
                        # If the neighboor is outside the bounds of the image skip
                        if (nb[0] < 0 or nb[0] >= cur_segm_pre_post[0].shape[0]) or (
                            nb[1] < 0 or nb[1] >= cur_segm_pre_post[0].shape[1]
                        ):
                            continue

                        if (nb[0], nb[1]) in pre_kpts:
                            # Get the points index
                            pt_index = pre_kpts[(nb[0], nb[1])]
                            pre_points_to_check.append(pt_index)

                    # Append the coordinates since they are not included in the neighbors
                    if (y1, x1) in pre_kpts:
                        pt_index = pre_kpts[(y1, x1)]
                        pre_points_to_check.append(pt_index)

                    post_points_to_check = []
                    # Check if there are any matches in the point areas
                    for nb in GProc.candidate_neighbors((y2, x2), hood_size=81):
                        # If the neighboor is outside the bounds of the image skip
                        if (nb[0] < 0 or nb[0] >= cur_segm_pre_post[1].shape[0]) or (
                            nb[1] < 0 or nb[1] >= cur_segm_pre_post[1].shape[1]
                        ):
                            continue

                        if (nb[0], nb[1]) in post_kpts:
                            # Get the points index
                            pt_index = post_kpts[(nb[0], nb[1])]
                            post_points_to_check.append(pt_index)

                    # Append the coordinates since they are not included in the neighbors
                    if (y2, x2) in post_kpts:
                        pt_index = post_kpts[(y2, x2)]
                        post_points_to_check.append(pt_index)

                    # print(pre_points_to_check)
                    # print(post_points_to_check)
                    if not pre_points_to_check or not post_points_to_check:
                        continue

                    # If there are, check if they match the ground truth matches
                    if transposed:
                        for idx in post_points_to_check:
                            g1_idxs.append(matchings[idx])
                            g2_idxs.append(graph_node_idx[idx])
                            # print(graph_node_idx[idx], matchings[idx])
                            if matchings[idx] in pre_points_to_check:
                                # print("Found!!!")
                                TP += 1
                                patch_TP += 1
                            else:
                                # print("NOT IN")
                                FN += 1
                    else:
                        for idx in pre_points_to_check:
                            g1_idxs.append(graph_node_idx[idx])
                            g2_idxs.append(matchings[idx])
                            # print(graph_node_idx[idx], matchings[idx])
                            if matchings[idx] in post_points_to_check:
                                # print("Found!!!")
                                TP += 1
                                patch_TP += 1
                            else:
                                # print("NOT IN")
                                FN += 1

                FP += min(len(pre_kpts), len(post_kpts)) - patch_TP
                print(f"TP: {TP} - FN: {FN} - FP: {FP}")

                # Optional:
                # visualize_matches(
                #     pre_kpts,
                #     post_kpts,
                #     cur_segm_pre_post[0],
                #     cur_segm_pre_post[1],
                #     true_matches,
                #     g1_idxs,
                #     g2_idxs,
                # )

    # Report on time performance
    avg_time = total_time / len(json__files)
    print(f"Average execution time: {avg_time} seconds.")
    avg_loading_time = total_time_loading / len(json__files)
    print(f"Average loading time: {avg_loading_time} seconds.")
    avg_graph_time = total_time_gr / len(json__files)
    print(f"Average graph creation time: {avg_graph_time} seconds.")
    if match_type == "patched":
        print(
            f"Average execution time for full patched: {16*avg_time+avg_loading_time} seconds."
        )
    elif match_type == "single":
        print(f"Average total time: {avg_time+avg_loading_time} second.")
    else:
        print(f"Average total time: {avg_time+avg_loading_time+avg_graph_time} second.")


### Not sure how to implement the bootsrap nicely
### I know this is ugly but I just need it to work for now.
def bootstrap_eval(
    json_files, lm_filters=None, match_type="patched", load_ft_maps=False
):
    TP = FP = FN = 0
    total_time = 0
    total_time_loading = 0
    total_time_gr = 0
    if not load_ft_maps:
        # Create Leung-Malik filter bank
        LM_filter_banks = lm_filters
    for json_path in json_files:
        with open(json_path, "r") as json_file:
            data = json.load(json_file)
            true_matches = data["matches"]
            # print(idx, json_path, matches, sep=" - ")
            json_path_stem = Path(json_path).stem
            print("########################")
            print(json_path_stem)

            # Extract patient id, orientation and patch index from annotation name
            json_pre, _ = json_path_stem.split("-")
            json_pre_split = json_pre.split("_")

            pat_id = json_pre_split[0]
            pat_ori = json_pre_split[1]
            patch_row_idx, patch_col_idx = int(json_pre_split[3]), int(
                json_pre_split[4]
            )

            # Start timer
            t2_start = perf_counter()

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
            matches = sift.find_feat_matches(
                predsc, postdsc, feat_extr=feature_extractor
            )
            # sift.plot_matches(preEVT, prekp, postEVT, postkp, matches,feat_extr=feature_extractor)
            transformation = sift.calculate_transform(
                prekp, postkp, matches, feat_extr=feature_extractor
            )

            # 2. Segmentation
            # Since this is the evaluation we assume the segmentations and feature maps will always be loaded.

            IMG_SEQ_DIR_PATH = (
                "C:/Users/mab03/Desktop/ThesisCode/Segms/Sequence/"
                + pat_id
                + "/"
                + pat_ori
            )

            FEAT_MAP_DIR_PATH = "FeatMapsv2/" + pat_id + "/" + pat_ori
            feat_map_pre_post = []

            segm_images1 = sklt.load_images(IMG_SEQ_DIR_PATH)

            # Find corresponding pre and post images from the segmentations and feat maps
            segm_pre_post = []
            for segm in segm_images1:
                if Path(segm).stem.rsplit("_", 1)[1] == "artery":
                    if Path(segm).stem.rsplit("_", 2)[1] == "pre":
                        pre = True
                        segm_pre_post.insert(0, sift.load_img(segm))
                    else:
                        pre = False
                        segm_pre_post.append(sift.load_img(segm))

                    if load_ft_maps:
                        # Load feature maps
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
                    transformation, preEVT, postEVT, ret=True, vis=False
                )
                # Scale transformation matrix for segmentation image size
                scale_matrix = np.array([[0.5, 0, 0], [0, 0.5, 0], [0, 0, 1]])
                transformation = (
                    scale_matrix @ transformation @ np.linalg.inv(scale_matrix)
                )
                # Warp and display the segmentations
                final_segm_post = sift.apply_transformation(
                    transformation,
                    segm_pre_post[0],
                    segm_pre_post[1],
                    ret=True,
                    vis=False,
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

            if not load_ft_maps:
                # Use Leung-Malik features
                pre_feat_map = FilterBanks.create_ft_map(LM_filter_banks, preEVT)
                post_feat_map = FilterBanks.create_ft_map(LM_filter_banks, tr_postEVT)
                feat_map_pre_post = [pre_feat_map, post_feat_map]
                # FilterBanks.visualize_ft_map(pre_feat_map)
                # FilterBanks.visualize_ft_map(post_feat_map)

            # Start timer
            t1_start = perf_counter()
            # Timer for loading things
            load_elapsed = t1_start - t2_start
            total_time_loading += load_elapsed

            if match_type == "patched":
                # Get patched images
                patched_imgs = GMatch.get_patches(segm_pre_post, 128, 128)
                # Load the specific patches
                segm_pre_post[0] = patched_imgs[0][patch_row_idx, patch_col_idx, ...]
                segm_pre_post[1] = patched_imgs[1][patch_row_idx, patch_col_idx, ...]

            # 3. Skeletonization

            # Perform skeletonization
            skeleton_images, distance_transform = sklt.get_skeletons(
                segm_pre_post,
                method="lee",
            )
            if not skeleton_images:
                return

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

            # 5. Extract features from the segmenation model
            # and add them as graph attributes

            pre_feat_map = feat_map_pre_post[0]
            post_feat_map = feat_map_pre_post[1]

            if match_type == "multi":
                t_gr_stop = perf_counter()
                # # Timer for creating graphs
                load_elapsed = t_gr_stop - t1_start
                total_time_gr += load_elapsed

            # Scales used for multi-scaled matching
            scales = [(0, 2), (2, 3), (3, 4), 4]

            if match_type == "single" or match_type == "patched":
                # Set scale to [0] to perform single scale matching
                # If 'single' matching is performed on the original entire graphs
                scales = [0]

            if match_type == "patched":
                # Patch the feature as already done with the segmentations
                patched_feat_maps = GMatch.get_patches(
                    [pre_feat_map, post_feat_map], 128, 128
                )
                # Load the specific patches - same as the segmentation
                pre_feat_map = patched_feat_maps[0][
                    :, patch_row_idx, patch_col_idx, ...
                ]
                post_feat_map = patched_feat_maps[1][
                    :, patch_row_idx, patch_col_idx, ...
                ]

            cur_segm_pre_post = segm_pre_post.copy()
            # Loop through all the scales perform matching and finally combine them
            for _, scale in enumerate(scales):
                # Start timer
                t3_start = perf_counter()
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
                GProc.concat_extracted_features_v2(
                    cur_pre_graph, pre_feat_map, nb_size=8, inplace=True
                )
                # Post-EVT graph
                GProc.concat_extracted_features_v2(
                    cur_post_graph, post_feat_map, nb_size=8, inplace=True
                )

                # 6. Preprocess features if needed

                # Save keypoints for imaging
                pre_kpts = [
                    (float(pt[0]), float(pt[1])) for pt in cur_pre_graph.vs["coords"]
                ]
                post_kpts = [
                    (float(pt[0]), float(pt[1])) for pt in cur_post_graph.vs["coords"]
                ]

                # Delete the coords attribute before calculating similarity
                del cur_pre_graph.vs["coords"]
                del cur_post_graph.vs["coords"]

                # Normalize x,y and radius features - Prevent blowup with similarity mat
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
                # We can also delete x,y features if the graph visualization is not used.
                del cur_pre_graph.vs["x"]
                del cur_pre_graph.vs["y"]
                del cur_post_graph.vs["y"]
                del cur_post_graph.vs["x"]

                # 7. Calculate graph node feature matrices and perform matching

                # Create a feature matrices from all the node attributes
                pre_feat_matrix = GMatch.create_feat_matrix(cur_pre_graph)
                post_feat_matrix = GMatch.create_feat_matrix(cur_post_graph)

                if load_ft_maps:
                    # Only instance based available since distribution parameters weren't calculated
                    # The normalizations are optional
                    pre_feat_matrix = instance_normalization(pre_feat_matrix)
                    post_feat_matrix = instance_normalization(post_feat_matrix)
                    # pass
                else:
                    pre_feat_matrix = instance_normalization(pre_feat_matrix)
                    post_feat_matrix = instance_normalization(post_feat_matrix)

                    # pre_feat_matrix = distribution_normalization(
                    #     pre_feat_matrix, norm_type="zscore", fts=1
                    # )
                    # post_feat_matrix = distribution_normalization(
                    #     post_feat_matrix, norm_type="zscore", fts=1
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

                # Final matchings depend on pre and post graph node count
                if assignment_mat.shape[0] < assignment_mat.shape[1]:
                    matchings = np.argmax(assignment_mat, axis=1)  # row
                    transposed = False
                else:
                    matchings = np.argmax(assignment_mat, axis=0)  # column
                    transposed = True
                # print(len(matchings))

                if match_type == "patched" or match_type == "single":
                    # Use this for patched and single-scale
                    t1_stop = perf_counter()
                    elapsed = t1_stop - t1_start
                    total_time += elapsed
                elif match_type == "multi":
                    # Use this for multi-scaled performance
                    # Start timer
                    t3_stop = perf_counter()
                    # Timer for running matching for scales
                    load_elapsed = t3_stop - t3_start
                    total_time += load_elapsed

                graph_node_idx = [*range(len(matchings))]

                # For every point in True Matches we need to check if there are any
                # matched points in the corresponding pre-post areas.
                # If a match is found TP+=1 else FN+=1
                # TN don't have any point and FP will be the count of points that
                # are matched but don't exist in the True Matches (Does not say much since True Matches are not perfect)

                # For VISUALIZATION:
                # get the patch slices from the original segmentation images
                if match_type == "patched":
                    pre_kpts = {
                        (round(x), round(y)): idx for idx, (x, y) in enumerate(pre_kpts)
                    }
                    post_kpts = {
                        (round(x), round(y)): idx
                        for idx, (x, y) in enumerate(post_kpts)
                    }
                elif match_type == "single" or match_type == "multi":
                    pre_kpts = {
                        (
                            round(x) - patch_row_idx * 128,
                            round(y) - (patch_col_idx) * 128,
                        ): idx
                        for idx, (x, y) in enumerate(pre_kpts)
                        if x in range(patch_row_idx * 128, (patch_row_idx + 1) * 128)
                        and y in range(patch_col_idx * 128, (patch_col_idx + 1) * 128)
                    }
                    post_kpts = {
                        (
                            round(x) - patch_row_idx * 128,
                            round(y) - (patch_col_idx) * 128,
                        ): idx
                        for idx, (x, y) in enumerate(post_kpts)
                        if x in range(patch_row_idx * 128, (patch_row_idx + 1) * 128)
                        and y in range(patch_col_idx * 128, (patch_col_idx + 1) * 128)
                    }

                    # Now the points have to be scaled down

                    cur_segm_pre_post[0] = segm_pre_post[0][
                        patch_row_idx * 128 : (patch_row_idx + 1) * 128,
                        patch_col_idx * 128 : (patch_col_idx + 1) * 128,
                    ]
                    cur_segm_pre_post[1] = segm_pre_post[1][
                        patch_row_idx * 128 : (patch_row_idx + 1) * 128,
                        patch_col_idx * 128 : (patch_col_idx + 1) * 128,
                    ]

                    # fig, axs = plt.subplots(1, 2)
                    # axs[0].scatter(
                    #     [y for (x, y) in pre_kpts], [x for (x, y) in pre_kpts]
                    # )
                    # axs[1].scatter(
                    #     [y for (x, y) in post_kpts], [x for (x, y) in post_kpts]
                    # )
                    # axs[0].imshow(segm_pre_post[0], cmap="gray")
                    # axs[1].imshow(segm_pre_post[1], cmap="gray")
                    # plt.show()

                patch_TP = 0

                # print(list(pre_kpts.items()))
                # print(list(post_kpts.items()))

                # Only for Visualization
                g1_idxs = []
                g2_idxs = []
                for idx, points in enumerate(true_matches):
                    # Load annotated points - Integers
                    x1 = round(points.get("x1"))
                    y1 = round(points.get("y1"))
                    x2 = round(points.get("x2"))
                    y2 = round(points.get("y2"))

                    # print(x1, y1, x2, y2, sep=" - ")

                    pre_points_to_check = []
                    # Check if there are any matches in the point areas
                    for nb in GProc.candidate_neighbors((y1, x1), hood_size=81):
                        # If the neighboor is outside the bounds of the image skip
                        if (nb[0] < 0 or nb[0] >= cur_segm_pre_post[0].shape[0]) or (
                            nb[1] < 0 or nb[1] >= cur_segm_pre_post[0].shape[1]
                        ):
                            continue

                        if (nb[0], nb[1]) in pre_kpts:
                            # Get the points index
                            # pt_index = pre_kpts.index((nb[0], nb[1]))
                            # pre_points_to_check.append(pt_index)
                            pt_index = pre_kpts[(nb[0], nb[1])]
                            pre_points_to_check.append(pt_index)

                    # Append the coordinates since they are not included in the neighbors
                    if (y1, x1) in pre_kpts:
                        pt_index = pre_kpts[(y1, x1)]
                        pre_points_to_check.append(pt_index)

                    post_points_to_check = []
                    # Check if there are any matches in the point areas
                    for nb in GProc.candidate_neighbors((y2, x2), hood_size=81):
                        # If the neighboor is outside the bounds of the image skip
                        if (nb[0] < 0 or nb[0] >= cur_segm_pre_post[1].shape[0]) or (
                            nb[1] < 0 or nb[1] >= cur_segm_pre_post[1].shape[1]
                        ):
                            continue

                        if (nb[0], nb[1]) in post_kpts:
                            # Get the points index
                            pt_index = post_kpts[(nb[0], nb[1])]
                            post_points_to_check.append(pt_index)

                    # Append the coordinates since they are not included in the neighbors
                    if (y2, x2) in post_kpts:
                        pt_index = post_kpts[(y2, x2)]
                        post_points_to_check.append(pt_index)

                    # print(pre_points_to_check)
                    # print(post_points_to_check)
                    if not pre_points_to_check or not post_points_to_check:
                        continue

                    # If there are, check if they match the ground truth matches
                    if transposed:
                        for idx in post_points_to_check:
                            g1_idxs.append(matchings[idx])
                            g2_idxs.append(graph_node_idx[idx])
                            # print(graph_node_idx[idx], matchings[idx])
                            if matchings[idx] in pre_points_to_check:
                                # print("Found!!!")
                                TP += 1
                                patch_TP += 1
                            else:
                                # print("NOT IN")
                                FN += 1
                    else:
                        for idx in pre_points_to_check:
                            g1_idxs.append(graph_node_idx[idx])
                            g2_idxs.append(matchings[idx])
                            # print(graph_node_idx[idx], matchings[idx])
                            if matchings[idx] in post_points_to_check:
                                # print("Found!!!")
                                TP += 1
                                patch_TP += 1
                            else:
                                # print("NOT IN")
                                FN += 1

                FP += min(len(pre_kpts), len(post_kpts)) - patch_TP
                # print(patch_TP)
                print(f"TP: {TP} - FN: {FN} - FP: {FP}")

    return TP / (TP + FN)

    # visualize_matches(
    #     pre_kpts,
    #     post_kpts,
    #     cur_segm_pre_post[0],
    #     cur_segm_pre_post[1],
    #     true_matches,
    #     g1_idxs,
    #     g2_idxs,
    # )


# fmt: off
def get_args():
    parser = argparse.ArgumentParser(description='Prepare and/or evaluate method using annotations')
    parser.add_argument('--in_img_path', '-i', default='Niftisv2/', help='Input directory for creating patches.')
    parser.add_argument('--out_patch_path', '-o', default='PatchedImages', help='Directory to save patches.')
    parser.add_argument('--prepare', action='store_true', default=False, help='Prepare the patched images for annotation.')
    parser.add_argument('--evaluate', action='store_true', default=False, help='Evaluate method using the annotated patched images.')
    parser.add_argument("--load-ft-maps",action="store_true",default=False,help="Load the feature maps corresponding to the segmentations.")
    parser.add_argument("--calculate-ci",action="store_true",default=False,help="Calculate confidence intervals using bootstrap.")
    parser.add_argument('--match_type', '-t', default='patched', help='Type of matching to evaluate.')
    
    return parser.parse_args()
# fmt: on


def main(
    in_img_path,
    out_patch_path,
    match_type,
    prepare=False,
    evaluate=False,
    load_ft_maps=False,
    calculate_ci=False,
):
    if not prepare and not evaluate:
        print("All arguments are False -> No actions taken.")
        return
    if prepare:
        sure = input(
            "Are you sure you want prepare the patches? If a patch exists it will be rewritten. (y/n): "
        )
        while sure.lower() not in ["y", "n"]:
            sure = input("give y/n: ")

        if sure == "y":
            IMG_DIR_PATH = in_img_path
            prepare_patches(IMG_DIR_PATH, output_dir=out_patch_path)
        else:
            print("Canceling preparation...")

    if evaluate:
        ANNOT_DIR_PATH = "C:/Users/mab03/Desktop/AnnotationTool/Output"
        evaluate_patches(ANNOT_DIR_PATH, match_type, load_ft_maps, calculate_ci)


if __name__ == "__main__":
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
    args = get_args()
    main(**vars(args))
