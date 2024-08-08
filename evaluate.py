from glob import glob
import os
from pathlib import Path
import sys
import logging
import argparse
import numpy as np
import imageio
import json
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from random import shuffle
from time import perf_counter

# My imports
import SIFTTransform as sift
import Skeletonization as sklt
import graph_matching as GMatch
import graph_processing as GProc

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

                # print(r_idx, c_idx)
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
                # imageio.imwrite(patch_dst_path, pre_post_patches[0])
                imageio.imwrite(patch_dst_path, pre_patch_w_skeleton)
                patch_dst_path = os.path.join(
                    output_dir,
                    "{}.png".format(
                        "_".join([pat_id, pat_ori, "post", str(r_idx), str(c_idx)])
                    ),
                )  # pat_id + "_" + pat_ori + "_pre" + '_' + r_idx + '_' + c_idx
                Path(patch_dst_path).parent.mkdir(parents=True, exist_ok=True)
                logger.info("Saving patch to {}".format(patch_dst_path))
                # imageio.imwrite(patch_dst_path, pre_post_patches[1])
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
    shuffle(colors)

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

    # for match in zip(g1_idxs,g2_idxs):
    #     ax.plot([x1, x2], [y1, y2], color=color, linewidth=1)


def evaluate_patches(annot_dir, match_type="patched"):

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
    for idx, json_path in enumerate(json__files):
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

            # print(pat_id, pat_ori)
            # print(patch_row_idx, patch_col_idx)

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

            # vis_skeletons = sklt.VisualizeSkeletons(
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

            # t_gr_stop = perf_counter()
            # # Timer for creating graphs
            # load_elapsed = t_gr_stop - t1_start
            # total_time_gr += load_elapsed

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

                # Test averaging the neighborhood (3x3) or (5x5) or (7x7) values for the features
                # Pre-EVT graph
                GProc.concat_extracted_features_v2(
                    cur_pre_graph, pre_feat_map, nb_size=48, inplace=True
                )
                # Post-EVT graph
                GProc.concat_extracted_features_v2(
                    cur_post_graph, post_feat_map, nb_size=48, inplace=True
                )
                # print(cur_pre_graph.vs[0].attributes())

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

                # Find min max radius of all nodes in the graphs for normalization
                pre_r_avg, pre_r_max, pre_r_min, pre_r_std = (
                    GMatch.calc_graph_radius_info(cur_pre_graph)
                )
                # print(
                #     f"R-Avg: {pre_r_avg}, R-Max: {pre_r_max}, R-Min: {pre_r_min}, R-Std: {pre_r_std}"
                # )
                post_r_avg, post_r_max, post_r_min, post_r_std = (
                    GMatch.calc_graph_radius_info(cur_post_graph)
                )
                # print(
                #     f"R-Avg: {post_r_avg}, R-Max: {post_r_max}, R-Min: {post_r_min}, R-Std: {post_r_std}"
                # )
                r_norm_max = pre_r_max if pre_r_max > post_r_max else post_r_max
                r_norm_min = pre_r_min if pre_r_min < post_r_min else post_r_min

                # Min-max normalization of radii
                cur_pre_graph.vs["radius"] = [
                    (r - r_norm_min) / (r_norm_max - r_norm_min)
                    for r in cur_pre_graph.vs["radius"]
                ]
                cur_post_graph.vs["radius"] = [
                    (r - r_norm_min) / (r_norm_max - r_norm_min)
                    for r in cur_post_graph.vs["radius"]
                ]

                # Normalize x,y and radius features - Prevent blowup with similarity mat
                cur_pre_graph.vs["x"] = [
                    x / skeleton_images[0].shape[1] for x in cur_pre_graph.vs["x"]
                ]
                cur_pre_graph.vs["y"] = [
                    y / skeleton_images[0].shape[0] for y in cur_pre_graph.vs["y"]
                ]
                cur_post_graph.vs["x"] = [
                    x / skeleton_images[1].shape[1] for x in cur_post_graph.vs["x"]
                ]
                cur_post_graph.vs["y"] = [
                    y / skeleton_images[1].shape[0] for y in cur_post_graph.vs["y"]
                ]

                # Delete the node radius info since I think it is misleading
                del cur_pre_graph.vs["radius"]
                del cur_post_graph.vs["radius"]
                # We can also delete x,y features if the graph visualization is not used.
                # del cur_pre_graph.vs["x"]
                # del cur_pre_graph.vs["y"]
                # del cur_post_graph.vs["y"]
                # del cur_post_graph.vs["x"]

                # 7. Calculate graph node feature matrices and perform matching

                # Create a feature matrices from all the node attributes
                pre_feat_matrix = GMatch.create_feat_matrix(cur_pre_graph)
                post_feat_matrix = GMatch.create_feat_matrix(cur_post_graph)

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

                # pre_feat_avg, pre_feat_max, pre_feat_min, pre_feat_std = (
                #     GMatch.calc_feat_matrix_info(pre_feat_matrix, vis=False)
                # )
                # post_feat_avg, post_feat_max, post_feat_min, post_feat_std = (
                #     GMatch.calc_feat_matrix_info(post_feat_matrix, vis=False)
                # )
                # print(
                #     f"Avg: {pre_feat_avg}, Max: {pre_feat_max}, Min: {pre_feat_min}, Std: {pre_feat_std}"
                # )
                # print(
                #     f"Avg: {post_feat_avg}, Max: {post_feat_max}, Min: {post_feat_min}, Std: {post_feat_std}"
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

                # Use this for patched and single-scale
                t1_stop = perf_counter()
                elapsed = t1_stop - t1_start
                total_time += elapsed

                # Use this for multi-scaled performance
                # Start timer
                # t3_stop = perf_counter()
                # # Timer for running matching for scales
                # load_elapsed = t3_stop - t3_start
                # total_time += load_elapsed

                graph_node_idx = [*range(len(matchings))]

                # print(len(graph_node_idx), len(matchings))
                # print(matchings)

                # Test masking based on feature vector Euclidean distance
                # feat_dist = distance.cdist(
                #     pre_feat_matrix, post_feat_matrix, "euclidean"
                # )
                # # Get matchings based on minimum euclidean distance
                # # matchings = np.argmin(feat_dist, axis=1)

                # For every point in True Matches we need to check if there are any
                # matched points in the corresponding pre-post areas.
                # If a match is found TP+=1 else FN+=1
                # TN don't have any point and FP will be the count of points that
                # are matched but don't exist in the True Matches (Does not say much since True Matches are not perfect)

                # pre_kpts = np.round(pre_kpts).astype(int)
                # post_kpts = np.round(post_kpts).astype(int)
                # pre_kpts = [(round(x), round(y)) for (x, y) in pre_kpts]
                # post_kpts = [(round(x), round(y)) for (x, y) in post_kpts]
                # print(pre_kpts)
                # print(post_kpts)

                #### If its SINGLE or MULTI
                # I somehow need to get the pre and post kpts in that specific patch
                # in order to test it. Most likely i need to scale them depending on the
                # patch position as well - to correspond to the patch annotations
                # I need to keep in mind that the matchings are also related to the indexes
                # of pre and post kpts
                # IDEA: get pre and post kpts that are in the specific patch based on row
                # and column index, use a dictionary where the keys are the kpts and the values
                # are the original indexes. So that we can use those indexes when evaluating from
                # the matchings list
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

                # FP += len(matchings) - patch_TP
                FP += min(len(pre_kpts), len(post_kpts)) - patch_TP
                # print(patch_TP)
                print(f"TP: {TP} - FN: {FN} - FP: {FP}")
                # for i, j in zip(graph_node_idx, matchings):
                #     if transposed:
                #         pt1 = pre_kpts[j]
                #         pt2 = post_kpts[i]
                #     else:
                #         pt1 = pre_kpts[i]
                #         pt2 = post_kpts[j]
                #     print(i, j)

                # break
                visualize_matches(
                    pre_kpts,
                    post_kpts,
                    cur_segm_pre_post[0],
                    cur_segm_pre_post[1],
                    true_matches,
                    g1_idxs,
                    g2_idxs,
                )

                # axs[0].invert_yaxis()
                # axs[1].invert_yaxis()
                # plt.show()

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


# fmt: off
def get_args():
    parser = argparse.ArgumentParser(description='Prepare and/or evaluate method using annotations')
    parser.add_argument('--in_img_path', '-i', default='Niftisv2/', help='Input directory for creating patches.')
    parser.add_argument('--out_patch_path', '-o', default='PatchedImages', help='Directory to save patches.')
    parser.add_argument('--prepare', action='store_true', default=False, help='Prepare the patched images for annotation.')
    parser.add_argument('--evaluate', action='store_true', default=False, help='Evaluate method using the annotated patched images.')
    parser.add_argument('--match_type', '-t', default='patched', help='Type of matching to evaluate.')
    
    return parser.parse_args()
# fmt: on


def main(in_img_path, out_patch_path, match_type, prepare=False, evaluate=False):
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
        evaluate_patches(ANNOT_DIR_PATH, match_type)


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
