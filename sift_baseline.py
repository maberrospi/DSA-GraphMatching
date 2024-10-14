import argparse
import logging
import os
import sys
from pathlib import Path
from glob import glob
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.widgets import Button
import igraph as ig
import cv2
from skimage.segmentation import watershed
from skimage.color import label2rgb
from skimage.transform import resize
from shapely.geometry import Polygon
import shapely

import SIFTTransform as sift
import Skeletonization as sklt
import graph_processing as GProc
from graph_feature_extraction import loop_segment
from evaluate import bootstrap

# Add Segmentation package path to sys path to fix importing unet
sys.path.insert(1, os.path.join(sys.path[0], "Segmentation"))
from Segmentation import predict

logger = logging.getLogger(__name__)


def create_graph(skeleton_img, skeleton_pts, dst_transform, g_name=None, verbose=True):
    if not g_name:
        g_name = "Temp"

    logger.info(f"Creating graph: {g_name} to extract segments.")

    sk_graph = ig.Graph()
    sk_graph["name"] = g_name
    sk_graph.add_vertices(len(skeleton_pts))

    sk_graph.vs["coords"] = skeleton_pts
    sk_graph.vs["y"] = skeleton_pts[:, 0]
    sk_graph.vs["x"] = skeleton_pts[:, 1]
    # Can extend the radius to the mEDT introduced in VesselVio if needed
    sk_graph.vs["radius"] = dst_transform[skeleton_img != 0]

    # Find and add the edges
    # Create vertex index Lookup table
    vertex_LUT = GProc.construct_vID_LUT(skeleton_pts, skeleton_img.shape)
    # Find edges
    edges = GProc.edge_detection(skeleton_pts, vertex_LUT)
    # Add detected edges
    sk_graph.add_edges(edges)
    # Remove loops and multiedges
    sk_graph.simplify()

    return sk_graph


def get_ordered_segment(graph, segment, segm_ids) -> list:
    # Only consider large segment paths
    if len(segment) == 1:
        return []
    else:
        segm_subg = graph.subgraph(segm_ids)
        degrees = segm_subg.degree(segment)
        endpoints = [segment[loc] for loc, deg in enumerate(degrees) if deg == 1]

        if len(endpoints) == 2:
            # Find the ordered path of the given segment
            # The indices do not correspond to the subgraph indices yet
            ordered_path = segm_subg.get_shortest_path(
                endpoints[0], endpoints[1], output="vpath"
            )

            # Add the corresponding subgraph indices to the point list
            point_list = [segm_ids[p].index for p in ordered_path]

            # Add the additional endpoint neighbors of the original graph to the list
            # if there exist any
            endpoint_neighbors = [point_list[1]] + [point_list[-2]]
            for i in range(2):
                # FYI. graph.vs[point_list[-i]].neighbors(): returns a node object
                for nb in graph.neighbors(point_list[-i]):
                    if nb not in endpoint_neighbors:
                        if i == 0:
                            point_list.insert(0, nb)
                        else:
                            point_list.append(nb)
        if len(endpoints) != 2:
            point_list = loop_segment(segm_subg, segment, segm_ids)

        return point_list


def extract_segments(
    skeleton_img, skeleton_pts, dst_transform, g_name=None, vis=False, verbose=True
):

    # Construct graph
    sk_graph = create_graph(
        skeleton_img, skeleton_pts, dst_transform, g_name=g_name, verbose=verbose
    )

    # Simplify the graph and filter unwanted edges and nodes
    GProc.filter_graph(sk_graph)

    labeled_segments = np.zeros(skeleton_img.shape)

    # Extract info only from the large segments or main vasculature
    logger.info("Extracting Segments: Analyzing large segments/main vasculature")
    segment_ids = sk_graph.vs.select(_degree_lt=3)
    subg_segments = sk_graph.subgraph(segment_ids)
    # Get list of connected components that is initially in a vertexCluster class
    segments = list(subg_segments.components())
    label_idx = 1
    for segment in segments:
        ordered_segment = get_ordered_segment(sk_graph, segment, segment_ids)
        point_list = sk_graph.vs[ordered_segment]
        x_coords = list(map(int, point_list["x"]))  # floor
        y_coords = list(map(int, point_list["y"]))
        for pt in zip(x_coords, y_coords):
            # Add its index label to the pt positions
            labeled_segments[pt[1], pt[0]] = label_idx

        label_idx += 1

    if vis:
        labels_rgb = label2rgb(labeled_segments, bg_label=0, bg_color=[0, 0, 0])
        fig, axs = plt.subplots(1, 2)
        lbl_img = axs[0].imshow(labels_rgb)
        sklt_img = axs[1].imshow(skeleton_img, cmap="gray")

        plt.show()

    return sk_graph, labeled_segments


def map_labels_to_segmentations(segm, dtf, labeled_segm, vis=False):
    # Apply the watershed algorithm using the labeled skeleton as markers
    # Int cast for issue with watershed
    labeled_segm = labeled_segm.astype(int)
    labels = watershed(-dtf, markers=labeled_segm, mask=segm)
    labels_rgb = label2rgb(labels, bg_label=0, bg_color=[0, 0, 0])

    if vis:
        fig, axs = plt.subplots(1, 2)
        axs[0].imshow(labels_rgb)
        axs[1].imshow(segm, cmap="gray")
        for a in axs:
            a.axis("off")
        plt.tight_layout()
        plt.show()

    return labels


def pixel_wise_method(segm_pre_post, vis=False) -> np.ndarray:
    pre_one_idxs = np.nonzero(segm_pre_post[0] == 1)
    final_results = np.zeros(segm_pre_post[0].shape)
    for x, y in zip(pre_one_idxs[0], pre_one_idxs[1]):
        # First check if the pixel itself corresponds
        if segm_pre_post[1][x, y] == 1:
            final_results[x, y] = 1
            continue
        # Get the neighbors
        nb_distances = {}
        for nb in GProc.candidate_neighbors((x, y), hood_size=48):
            # If the neighboor is outside the bounds of the image skip
            if (nb[0] < 0 or nb[0] >= segm_pre_post[0].shape[0]) or (
                nb[1] < 0 or nb[1] >= segm_pre_post[0].shape[1]
            ):
                continue
            # Find the nearest white pixel in the post segmentation
            if segm_pre_post[1][nb[0], nb[1]] == 0:
                nb_distances[nb[0], nb[1]] = np.inf
                continue
            # Calculate Euclidean Distance
            eucd = np.sqrt((x - nb[0]) ** 2 + (y - nb[1]) ** 2)
            nb_distances[nb[0], nb[1]] = eucd

        min_distance_idx: tuple = min(nb_distances, key=nb_distances.get)
        if nb_distances[min_distance_idx] == np.inf:
            continue
        final_results[min_distance_idx[0], min_distance_idx[1]] = 1

    if vis:
        # Plot matched segmentations
        fig, axs = plt.subplots(1, 3)
        axs[0].imshow(segm_pre_post[0], cmap="gray")
        axs[1].imshow(segm_pre_post[1], cmap="gray")
        axs[2].imshow(final_results, cmap="gray")
        axs[0].set_title("Pre-EVT Segmentation")
        axs[1].set_title("Post-EVT Segmentation")
        axs[2].set_title("Matched Segmentation")
        for a in axs:
            a.axis("off")
        fig.tight_layout()

        fig, axs = plt.subplots(1, 3)
        axs[0].imshow(segm_pre_post[0] - final_results, cmap="gray")
        axs[1].imshow(segm_pre_post[1] - final_results, cmap="gray")
        axs[2].imshow(final_results, cmap="gray")
        axs[0].set_title("Pre-EVT minus Matched")
        axs[1].set_title("Post-EVT minus Matched")
        axs[2].set_title("Matched Segmentation")
        for a in axs:
            a.axis("off")
        fig.tight_layout()

        plt.show()

    return final_results


def run_eval(json_files, pixel_wise, load_segs=True):
    # Initialise metrics
    TP = FP = FN = 0
    total_time = 0
    total_time_loading = 0
    total_time_gr = 0
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

            if load_segs:
                # If True we assume that the feature maps will also be loaded
                IMG_MIN_DIR_PATH = (
                    "C:/Users/mab03/Desktop/RuSegm/TemporalUNet/Outputs/Minip/" + pat_id
                )
                IMG_SEQ_DIR_PATH = (
                    "C:/Users/mab03/Desktop/ThesisCode/Segms/Sequence/"
                    + pat_id
                    + "/"
                    + pat_ori
                )
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
                # Feature maps not needed for the sift-baseline method. This still saves segmentations.
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
                # The list contains two maps, where pre-post are first-second
                # The maps have a torch tensor format
                IMG_SEQ_DIR_PATH = segm_output_folder

            # segm_images = load_images(IMG_MIN_DIR_PATH)
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

            if pixel_wise:
                matched_pixels = pixel_wise_method(segm_pre_post)

            # 3. Skeletonization

            # Perform skeletonization
            skeleton_images, distance_transform = sklt.get_skeletons(
                segm_pre_post,
                method="lee",
            )
            if not skeleton_images:
                return

            # 4. Create Graphs and Extract labeled segments
            skeleton_points = sklt.find_centerlines(skeleton_images[0])
            pre_graph, pre_graph_labeled_skltns = extract_segments(
                skeleton_images[0],
                skeleton_points,
                distance_transform[0],
                g_name="pre_graph",
                verbose=True,
            )
            skeleton_points = sklt.find_centerlines(skeleton_images[1])
            post_graph, post_graph_labeled_skltns = extract_segments(
                skeleton_images[1],
                skeleton_points,
                distance_transform[1],
                g_name="post_graph",
                verbose=True,
            )

            # 5. Map the labeled segments to the complete segmentations
            pre_graph_labeled_segs = map_labels_to_segmentations(
                segm_pre_post[0], distance_transform[0], pre_graph_labeled_skltns
            )
            post_graph_labeled_segs = map_labels_to_segmentations(
                segm_pre_post[1], distance_transform[1], post_graph_labeled_skltns
            )

            labels_rgb1 = label2rgb(
                pre_graph_labeled_segs, bg_label=0, bg_color=[0, 0, 0]
            )
            labels_rgb2 = label2rgb(
                post_graph_labeled_segs, bg_label=0, bg_color=[0, 0, 0]
            )
            # fig, axs = plt.subplots(1, 3)
            # axs[0].imshow(labels_rgb1)
            # axs[1].imshow(labels_rgb2)
            # axs[2].imshow(labels_rgb1)
            # axs[2].imshow(labels_rgb2, alpha=0.5)
            # for a in axs:
            #     a.axis("off")
            # plt.tight_layout()
            # plt.show()

            # 6. Iteratively compare segments and identify correspondence
            logger.info("Finding segment correspondences.")
            pre_graph_unique_segs = np.unique(pre_graph_labeled_segs)
            post_graph_unique_segs = np.unique(post_graph_labeled_segs)

            pre_matched_segs = np.zeros(pre_graph_labeled_segs.shape)
            post_matched_segs = np.zeros(post_graph_labeled_segs.shape)
            match_id = 1

            if not pixel_wise:
                for pre_segm in pre_graph_unique_segs:
                    if pre_segm == 0:  # Skip id == 0
                        continue
                    pre_temp_mask = np.where(pre_graph_labeled_segs == pre_segm, 1, 0)
                    # Convert binary images to uint8 format for OpenCV
                    pre_temp_mask = (pre_temp_mask * 255).astype(np.uint8)
                    # Find contours
                    pre_contours, _ = cv2.findContours(
                        pre_temp_mask,
                        cv2.RETR_EXTERNAL,
                        cv2.CHAIN_APPROX_SIMPLE,  # NONE
                    )
                    # FOR NOW: Skip contours with less than 4 points
                    if len(pre_contours[0].squeeze()) < 4:
                        continue
                    # Create rectangle on contour
                    x1, y1, w1, h1 = cv2.boundingRect(pre_contours[0])
                    # Convert contour to polygon
                    pre_polygon = Polygon(pre_contours[0].squeeze())
                    for post_segm in post_graph_unique_segs:
                        if post_segm == 0:
                            continue
                        post_temp_mask = np.where(
                            post_graph_labeled_segs == post_segm, 1, 0
                        )
                        post_temp_mask = (post_temp_mask * 255).astype(np.uint8)
                        post_contours, _ = cv2.findContours(
                            post_temp_mask,
                            cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE,  # NONE
                        )
                        if len(post_contours[0].squeeze()) < 4:
                            continue

                        # Create rectangle on contour
                        x2, y2, w2, h2 = cv2.boundingRect(post_contours[0])
                        # Skip if there is no overlap on the bounding boxes
                        if (x1 + w1 < x2 or x2 + w2 < x1) or (
                            y1 + h1 < y2 or y2 + h2 < y1
                        ):
                            continue

                        post_polygon = Polygon(post_contours[0].squeeze())

                        # Debugging:
                        # xv, yv = pre_polygon.exterior.xy
                        # xv2, yv2 = post_polygon.exterior.xy
                        # fig, ax = plt.subplots()
                        # ax.plot(xv, yv)
                        # ax.plot(xv2, yv2)
                        # plt.show()

                        # Calculate the intersection and union
                        intersection = pre_polygon.intersection(post_polygon)
                        while True:  # Still have to fix this it doesnt work yet.
                            try:
                                union = pre_polygon.union(post_polygon)
                                break
                            except shapely.errors.GEOSException:
                                if not shapely.is_valid(pre_polygon):
                                    pre_polygon = shapely.make_valid(pre_polygon)
                                elif not shapely.is_valid(post_polygon):
                                    post_polygon = shapely.make_valid(post_polygon)
                                xv, yv = pre_polygon.exterior.xy
                                xv2, yv2 = post_polygon.exterior.xy
                                fig, ax = plt.subplots()
                                ax.plot(xv, yv)
                                ax.plot(xv2, yv2)
                                plt.show()

                        # Calculate areas
                        intersection_area = intersection.area
                        union_area = union.area

                        # Calculate IoU
                        iou = intersection_area / union_area if union_area != 0 else 0

                        iou_threshold = 0.3

                        if iou > iou_threshold:
                            pre_matched_segs[pre_temp_mask == 255] = match_id
                            post_matched_segs[post_temp_mask == 255] = match_id
                            match_id += 1
                            break
                logger.info("Segments matched successfully!")

                pre_labels_rgb = label2rgb(
                    pre_matched_segs, bg_label=0, bg_color=[0, 0, 0]
                )
                post_labels_rgb = label2rgb(
                    post_matched_segs, bg_label=0, bg_color=[0, 0, 0]
                )
            else:
                # Use the calculated pixel-wise mask and the labeled segments to
                # identify matching segments based on the mask coverage of each segment
                for pre_segm in pre_graph_unique_segs:
                    if pre_segm == 0:  # Skip id == 0
                        continue
                    temp_mask = np.where(pre_graph_labeled_segs == pre_segm, 1, 0)
                    temp_one_idxs = np.nonzero(temp_mask == 1)
                    match_diff = (
                        temp_mask[temp_one_idxs] - matched_pixels[temp_one_idxs]
                    )
                    # If the number of 0's is larger than 50% of the difference we assume its matched
                    # If the number of 1's is larger than 50% we assume its missed
                    if np.count_nonzero(match_diff == 0) / len(match_diff) >= 0.5:
                        pre_matched_segs[temp_mask == 1] = 1
                    elif np.count_nonzero(match_diff == 1) / len(match_diff) > 0.5:
                        pre_matched_segs[temp_mask == 1] = 2

                for post_segm in post_graph_unique_segs:
                    if post_segm == 0:
                        continue
                    temp_mask = np.where(post_graph_labeled_segs == post_segm, 1, 0)
                    temp_one_idxs = np.nonzero(temp_mask == 1)
                    match_diff = (
                        temp_mask[temp_one_idxs] - matched_pixels[temp_one_idxs]
                    )
                    # If the number of 0's is larger than 50% of the difference we assume its matched
                    if np.count_nonzero(match_diff == 0) / len(match_diff) >= 0.5:
                        post_matched_segs[temp_mask == 1] = 1
                    elif np.count_nonzero(match_diff == 1) / len(match_diff) > 0.5:
                        post_matched_segs[temp_mask == 1] = 2

                logger.info("Segments matched successfully!")

                pre_labels_rgb = label2rgb(
                    pre_matched_segs, bg_label=0, bg_color=[0, 0, 0]
                )
                post_labels_rgb = label2rgb(
                    post_matched_segs, bg_label=0, bg_color=[0, 0, 0]
                )

            # Get appropriate patch to evaluate
            cur_segm_pre_post = [0, 0]
            cur_segm_pre_post[0] = pre_matched_segs[
                patch_row_idx * 128 : (patch_row_idx + 1) * 128,
                patch_col_idx * 128 : (patch_col_idx + 1) * 128,
            ]
            cur_segm_pre_post[1] = post_matched_segs[
                patch_row_idx * 128 : (patch_row_idx + 1) * 128,
                patch_col_idx * 128 : (patch_col_idx + 1) * 128,
            ]

            for idx, points in enumerate(true_matches):
                # Load annotated points - Integers
                # Min to prevent out of bounds.
                x1 = min(round(points.get("x1")), cur_segm_pre_post[0].shape[1] - 1)
                y1 = min(round(points.get("y1")), cur_segm_pre_post[0].shape[0] - 1)
                x2 = min(round(points.get("x2")), cur_segm_pre_post[1].shape[1] - 1)
                y2 = min(round(points.get("y2")), cur_segm_pre_post[0].shape[0] - 1)

                if pixel_wise:
                    # The True matches here have a label of 1
                    if (
                        cur_segm_pre_post[0][y1, x1] == 1
                        and cur_segm_pre_post[1][y2, x2] == 1
                    ):
                        TP += 1
                    else:
                        FN += 1
                else:
                    # The True matches here have a label >0
                    if (
                        cur_segm_pre_post[0][y1, x1] > 0
                        and cur_segm_pre_post[1][y2, x2] > 0
                    ):
                        TP += 1
                    else:
                        FN += 1

            print(TP, FN)
            # fig, axs = plt.subplots(1, 2)
            # pre_labels_rgb = label2rgb(
            #     cur_segm_pre_post[0], bg_label=0, bg_color=[0, 0, 0]
            # )
            # post_labels_rgb = label2rgb(
            #     cur_segm_pre_post[1], bg_label=0, bg_color=[0, 0, 0]
            # )
            # axs[0].imshow(pre_labels_rgb)
            # axs[1].imshow(post_labels_rgb)
            # for i, points in enumerate(true_matches):
            #     # Load annotated points - Integers
            #     x1 = round(points.get("x1"))
            #     y1 = round(points.get("y1"))
            #     x2 = round(points.get("x2"))
            #     y2 = round(points.get("y2"))

            #     # print(x1, y1, x2, y2, sep=" - ")
            #     axs[0].scatter(x1, y1, color="green")
            #     axs[1].scatter(x2, y2, color="green")
            # plt.show()

    # Calculate metrics (Recall)
    return TP / (TP + FN)


def evaluate(annot_dir, pixel_wise=True, calculate_ci=False, load_segs=True):
    # Load annotations
    json__files = sorted(glob(os.path.join(annot_dir, "**", "*.json"), recursive=True))

    # Calculate Confidence Interval
    if calculate_ci:
        logger.info("Calculating Recall with Confidence Interval using Bootstrap...")
        recall_values = []
        bootstrap_json_files = bootstrap(json__files, n_splits=20)
        for json_files in bootstrap_json_files:
            recall_values.append(
                run_eval(json_files, pixel_wise=pixel_wise, load_segs=load_segs)
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
    else:
        logger.info("Calculating Recall of annotated dataset.")
        recall = run_eval(json__files, pixel_wise=pixel_wise, load_segs=load_segs)
        print(f"Recall: {recall}")


def visualize_results(segm_pre, segm_post, lbls_pre, lbls_post):
    segm_pre_rgb = np.dstack((segm_pre,) * 3)
    segm_post_rgb = np.dstack((segm_post,) * 3)

    black_pixels_mask = (lbls_pre == [0, 0, 0]).all(axis=-1)
    pre_labels_rgb_overlayed = np.where(
        black_pixels_mask[..., None], segm_pre_rgb, lbls_pre
    )
    black_pixels_mask = (lbls_post == [0, 0, 0]).all(axis=-1)
    post_labels_rgb_overlayed = np.where(
        black_pixels_mask[..., None], segm_post_rgb, lbls_post
    )

    fig, axs = plt.subplots(1, 2)
    axs[0].imshow(pre_labels_rgb_overlayed)
    axs[1].imshow(post_labels_rgb_overlayed)
    for a in axs:
        a.axis("off")
    plt.tight_layout()
    plt.show()


def get_image_pairs(in_img_path: str, in_segm_path: str):
    """
    This should return a list of pairs of pre- and post- images and their associated segmentations for both the artery and total segmentations

    Note: for now we only store the first element in each list

    """
    logger.info("Loading image pair information")
    image_path_dict = {}
    patient_images = os.listdir(in_img_path)
    patient_segms = os.listdir(in_segm_path)
    selected_patients = set(patient_images).intersection(patient_segms)

    for patid in selected_patients:
        patient_dict = get_paths_for_patient(in_img_path, in_segm_path, patid)
        if patient_dict:
            image_path_dict[patid] = patient_dict
    return image_path_dict


def get_paths_for_patient(in_img_path, in_segm_path, patid):
    niftis_ap = [
        f
        for f in glob(os.path.join(in_img_path, patid, "0", "**"), recursive=True)
        if os.path.isfile(f)
    ]
    niftis_lat = [
        f
        for f in glob(os.path.join(in_img_path, patid, "1", "**"), recursive=True)
        if os.path.isfile(f)
    ]
    niftis = {"ap": niftis_ap, "lat": niftis_lat}
    segms_ap = [
        f
        for f in glob(os.path.join(in_segm_path, patid, "0", "**"), recursive=True)
        if os.path.isfile(f)
    ]
    segms_lat = [
        f
        for f in glob(os.path.join(in_segm_path, patid, "1", "**"), recursive=True)
        if os.path.isfile(f)
    ]
    segms = {"ap": segms_ap, "lat": segms_lat}
    patient_dict = {}
    patient_segms_dict = {}
    for side in ["ap", "lat"]:
        # art_side_dict = {}
        full_side_dict = {}
        segm_art_side_dict = {}
        segm_vein_side_dict = {}
        for setting in ["pre", "post"]:
            filename_ending = f"{setting}"
            try:
                # art_side_dict[filename_ending] = [
                #     f for f in niftis[side] if filename_ending in f and "art" in f
                # ][0]
                full_side_dict[filename_ending] = [
                    f for f in niftis[side] if filename_ending in f and "art" not in f
                ][0]
                segm_art_side_dict[filename_ending] = [
                    f for f in segms[side] if filename_ending in f and "art" in f
                ][0]
                segm_vein_side_dict[filename_ending] = [
                    f for f in segms[side] if filename_ending in f and "vein" in f
                ][0]
            except IndexError as e:
                logger.warning(f"One of the files is missing, skipping case {patid}")
                logger.warning(e)
                return {}
        segm_side_dict = {}
        segm_side_dict["art"] = segm_art_side_dict
        segm_side_dict["vein"] = segm_vein_side_dict
        side_dict = {}
        # side_dict["art"] = art_side_dict
        side_dict["full"] = full_side_dict
        patient_dict[side] = side_dict
        patient_segms_dict[side] = segm_side_dict
    return {"nifti": patient_dict, "segm": patient_segms_dict}


def load_images_from_paths(data):
    """
    Recursively traverses a nested dictionary and replaces file paths with images.

    :param data: The dictionary containing file paths at the leaf nodes.
    :return: The modified dictionary with images replacing paths.
    """
    if isinstance(data, dict):
        # Traverse each key-value pair in the dictionary
        return {key: load_images_from_paths(value) for key, value in data.items()}
    elif isinstance(data, list):
        # If it's a list, apply the function to each element
        return [load_images_from_paths(item) for item in data]
    elif isinstance(data, str) and os.path.isfile(data):
        # If it's a string and a valid file path, load the image
        try:
            img = sift.load_img(data)
            return img  # Return the image instead of the path
        except Exception as e:
            print(f"Failed to load image from {data}: {e}")
            return data  # Return the original path if the image fails to load
    else:
        # If it's neither a dictionary, list, nor valid path, return it as is
        return data


def sift_matching(preEVT, postEVT):
    feature_extractor = "sift"  # choose sift or orb
    prekp, predsc, postkp, postdsc = sift.feat_kp_dscr(
        preEVT, postEVT, feat_extr=feature_extractor
    )
    matches = sift.find_feat_matches(predsc, postdsc, feat_extr=feature_extractor)
    if not matches:
        return None, None
    # sift.plot_matches(preEVT, prekp, postEVT, postkp, matches,feat_extr=feature_extractor)
    transformation = sift.calculate_transform(
        prekp, postkp, matches, feat_extr=feature_extractor
    )
    return transformation, matches


def check_transform(segmentation_pair, transformation, preEVT, postEVT):
    transform_failed = False
    # Check if the transformation quality is better than the original
    if not sift.check_transform(transformation, preEVT, postEVT, verbose=True):
        # The transformation is worse than the original
        print("Using the original post-EVT image, registering as failed transform")
        final_segm_post = segmentation_pair[1]
        tr_postEVT = postEVT
        transform_failed = True
    else:
        tr_postEVT = sift.apply_transformation_v2(transformation, postEVT)
        # # # Scale transformation matrix for segmentation image size
        scale_matrix = np.array([[0.5, 0, 0], [0, 0.5, 0], [0, 0, 1]])
        transformation = scale_matrix @ transformation @ np.linalg.inv(scale_matrix)
        # Warp and display the segmentations
        final_segm_post = sift.apply_transformation_v2(
            transformation, segmentation_pair[1]
        )

    return transform_failed, tr_postEVT, final_segm_post


def generate_skeletons(segm_pre_post):
    # Perform skeletonization
    skeleton_images, distance_transform = sklt.get_skeletons(
        segm_pre_post,
        method="lee",
    )
    if not skeleton_images:
        return

    return skeleton_images, distance_transform


def extract_labeled_segments(skeleton_images, distance_transform, segm_pre_post):
    # 4. Create Graphs and Extract labeled segments
    skeleton_points = sklt.find_centerlines(skeleton_images[0])
    pre_graph, pre_graph_labeled_skltns = extract_segments(
        skeleton_images[0],
        skeleton_points,
        distance_transform[0],
        g_name="pre_graph",
        verbose=True,
    )
    skeleton_points = sklt.find_centerlines(skeleton_images[1])
    post_graph, post_graph_labeled_skltns = extract_segments(
        skeleton_images[1],
        skeleton_points,
        distance_transform[1],
        g_name="post_graph",
        verbose=True,
    )

    # 5. Map the labeled segments to the complete segmentations
    pre_graph_labeled_segs = map_labels_to_segmentations(
        segm_pre_post[0], distance_transform[0], pre_graph_labeled_skltns
    )
    post_graph_labeled_segs = map_labels_to_segmentations(
        segm_pre_post[1], distance_transform[1], post_graph_labeled_skltns
    )

    labels_rgb_pre = label2rgb(pre_graph_labeled_segs, bg_label=0, bg_color=[0, 0, 0])
    labels_rgb2_post = label2rgb(
        post_graph_labeled_segs, bg_label=0, bg_color=[0, 0, 0]
    )

    return pre_graph, post_graph, pre_graph_labeled_segs, post_graph_labeled_segs


def contour_based_matching(
    pre_graph_labeled_segs, post_graph_labeled_segs, iou_threshold=0.3
):
    pre_graph_unique_segs = np.unique(pre_graph_labeled_segs)
    post_graph_unique_segs = np.unique(post_graph_labeled_segs)

    pre_matched_segs = np.zeros_like(pre_graph_labeled_segs)
    post_matched_segs = np.zeros_like(post_graph_labeled_segs)
    match_id = 1

    for pre_segm in pre_graph_unique_segs:
        if pre_segm == 0:
            continue
        pre_temp_mask = np.where(pre_graph_labeled_segs == pre_segm, 1, 0)
        # Convert binary images to uint8 format for OpenCV
        pre_temp_mask = (pre_temp_mask * 255).astype(np.uint8)
        # Find contours
        pre_contours, _ = cv2.findContours(
            pre_temp_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE  # NONE
        )
        # FOR NOW: Skip contours with less than 4 points
        if len(pre_contours[0].squeeze()) < 4:
            continue
        # Create rectangle on contour
        x1, y1, w1, h1 = cv2.boundingRect(pre_contours[0])
        # Convert contour to polygon
        pre_polygon = Polygon(pre_contours[0].squeeze())

        for post_segm in post_graph_unique_segs:
            if post_segm == 0:
                continue
            post_temp_mask = np.where(post_graph_labeled_segs == post_segm, 1, 0)
            post_temp_mask = (post_temp_mask * 255).astype(np.uint8)
            post_contours, _ = cv2.findContours(
                post_temp_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE  # NONE
            )
            if len(post_contours[0].squeeze()) < 4:
                continue

            # Create rectangle on contour
            x2, y2, w2, h2 = cv2.boundingRect(post_contours[0])
            # Skip if there is no overlap on the bounding boxes
            if (x1 + w1 < x2 or x2 + w2 < x1) or (y1 + h1 < y2 or y2 + h2 < y1):
                continue

            post_polygon = Polygon(post_contours[0].squeeze())

            # Debugging:
            # xv, yv = pre_polygon.exterior.xy
            # xv2, yv2 = post_polygon.exterior.xy
            # fig, ax = plt.subplots()
            # ax.plot(xv, yv)
            # ax.plot(xv2, yv2)
            # plt.show()

            # Calculate the intersection and union
            intersection = pre_polygon.intersection(post_polygon)
            while True:  # Still have to fix this it doesnt work yet.
                try:
                    union = pre_polygon.union(post_polygon)
                    break
                except shapely.errors.GEOSException:
                    if not shapely.is_valid(pre_polygon):
                        pre_polygon = shapely.make_valid(pre_polygon)
                    elif not shapely.is_valid(post_polygon):
                        post_polygon = shapely.make_valid(post_polygon)
                    xv, yv = pre_polygon.exterior.xy
                    xv2, yv2 = post_polygon.exterior.xy
                    fig, ax = plt.subplots()
                    ax.plot(xv, yv)
                    ax.plot(xv2, yv2)
                    plt.show()

            # Calculate areas
            intersection_area = intersection.area
            union_area = union.area

            # Calculate IoU
            iou = intersection_area / union_area if union_area != 0 else 0

            iou_threshold = 0.3

            if iou > iou_threshold:
                pre_matched_segs[pre_temp_mask == 255] = match_id
                post_matched_segs[post_temp_mask == 255] = match_id
                match_id += 1
                break

    logger.info("Segments matched successfully!")
    return pre_matched_segs, post_matched_segs


def pixel_wise_based_matching(
    pre_graph_labeled_segs, post_graph_labeled_segs, matched_pixels
):
    pre_graph_unique_segs = np.unique(pre_graph_labeled_segs)
    post_graph_unique_segs = np.unique(post_graph_labeled_segs)

    pre_matched_segs = np.zeros(pre_graph_labeled_segs.shape)
    post_matched_segs = np.zeros(post_graph_labeled_segs.shape)

    # Use the calculated pixel-wise mask and the labeled segments to
    # identify matching segments based on the mask coverage of each segment
    for pre_segm in pre_graph_unique_segs:
        if pre_segm == 0:  # Skip id == 0
            continue
        temp_mask = np.where(pre_graph_labeled_segs == pre_segm, 1, 0)
        temp_one_idxs = np.nonzero(temp_mask == 1)
        match_diff = temp_mask[temp_one_idxs] - matched_pixels[temp_one_idxs]
        # If the number of 0's is larger than 50% of the difference we assume its matched
        # If the number of 1's is larger than 50% we assume its missed
        if np.count_nonzero(match_diff == 0) / len(match_diff) >= 0.5:
            pre_matched_segs[temp_mask == 1] = 1
        elif np.count_nonzero(match_diff == 1) / len(match_diff) > 0.5:
            pre_matched_segs[temp_mask == 1] = 2

    for post_segm in post_graph_unique_segs:
        if post_segm == 0:
            continue
        temp_mask = np.where(post_graph_labeled_segs == post_segm, 1, 0)
        temp_one_idxs = np.nonzero(temp_mask == 1)
        match_diff = temp_mask[temp_one_idxs] - matched_pixels[temp_one_idxs]
        # If the number of 0's is larger than 50% of the difference we assume its matched
        if np.count_nonzero(match_diff == 0) / len(match_diff) >= 0.5:
            post_matched_segs[temp_mask == 1] = 1
        elif np.count_nonzero(match_diff == 1) / len(match_diff) > 0.5:
            post_matched_segs[temp_mask == 1] = 2

    logger.info("Segments matched successfully!")
    return pre_matched_segs, post_matched_segs


def find_correspondences(
    pre_graph_labeled_segs: np.ndarray,
    post_graph_labeled_segs: np.ndarray,
    matched_pixels: np.ndarray,
    pixel_wise: bool,
):
    """
    This function finds the vessel correspondences between the pre and post labeled vessel segmentations.
    It uses the pixel-wise method or the contour-based method to find the correspondences.

    Args:
        pre_graph_labeled_segs (np.ndarray): The pre-labeled vessel segmentation.
        post_graph_labeled_segs (np.ndarray): The post-labeled vessel segmentation.
        matched_pixels (np.ndarray): The matched pixels between the pre and post segmentations.
        pixel_wise (bool): Whether to use the pixel-wise method or the contour-based method.

    Returns:
        pre_labels_rgb (np.ndarray): The pre labeled matched and unmatched vessel segments in RGB format.
        post_labels_rgb (np.ndarray): The post labeled matched and unmatched vessel segments in RGB format.
        pre_matched_segs (np.ndarray): The pre labeled matched and unmatched vessel segments.
        post_matched_segs (np.ndarray): The post labeled matched and unmatched vessel segments.
    """
    # 6. Iteratively compare segments and identify correspondence
    if not pixel_wise:
        pre_matched_segs, post_matched_segs = contour_based_matching(
            pre_graph_labeled_segs, post_graph_labeled_segs
        )
    else:
        pre_matched_segs, post_matched_segs = pixel_wise_based_matching(
            pre_graph_labeled_segs, post_graph_labeled_segs, matched_pixels
        )

    # Convert the matched segments to RGB format
    pre_labels_rgb = label2rgb(pre_matched_segs, bg_label=0, bg_color=[0, 0, 0])
    post_labels_rgb = label2rgb(post_matched_segs, bg_label=0, bg_color=[0, 0, 0])

    return pre_labels_rgb, post_labels_rgb, pre_matched_segs, post_matched_segs


def save_overlayed(
    root_path: str,
    patid: str,
    side: str,
    preEVT: np.ndarray,
    warped_postEVT: np.ndarray,
):
    logger.info(
        f"Writing overlayed transformation images for {patid} {side} in the {root_path} directory"
    )
    dpi = 300
    current_folder = os.path.join(root_path, patid, side)
    if not os.path.exists(current_folder):
        os.makedirs(
            current_folder,
        )
    # Overlay transformed and pre-EVT images
    fig, axs = plt.subplots(1, 3, figsize=(12, 6))
    axs[0].imshow(preEVT, cmap="gray")
    axs[0].imshow(warped_postEVT, cmap="Purples", alpha=0.5)
    axs[1].imshow(preEVT, cmap="gray")
    axs[2].imshow(warped_postEVT, cmap="Purples")
    axs[0].set_title("Overlayed Transform")
    axs[1].set_title("Pre-EVT")
    axs[2].set_title("Transformed Post-EVT")
    for a in axs:
        a.set_xticks([])
        a.set_yticks([])

    plt.savefig(
        os.path.join(current_folder, "overlayed.png"),
        dpi=dpi,
        bbox_inches="tight",
        pad_inches=0,
    )


def save_labels(
    root_path, patid, side, preEVT, tr_postEVT, pre_labels_rgb, post_labels_rgb
):
    logger.info(f"Writing images for {patid} {side} in the {root_path} directory")
    current_folder = os.path.join(root_path, patid, side)
    if not os.path.exists(current_folder):
        os.makedirs(
            current_folder,
        )

    # preEVT = resize(preEVT, (512, 512))
    # tr_postEVT = resize(tr_postEVT, (512, 512))
    # visualize_results(preEVT, tr_postEVT, pre_labels_rgb, post_labels_rgb, os.path.join(current_folder, 'rgb.png'))

    save_image(os.path.join(current_folder, "pre_evt.png"), preEVT)

    save_image(os.path.join(current_folder, "post_evt.png"), tr_postEVT)

    save_image(
        os.path.join(current_folder, "pre_seg_evt.png"), pre_labels_rgb, cmap="plasma"
    )

    save_image(
        os.path.join(current_folder, "post_seg_evt.png"), post_labels_rgb, cmap="plasma"
    )


def save_image(
    filename: str, image_arr: np.ndarray, cmap: str = "gray", size: tuple = None
):
    """
    Utility to save an ndarray as an image, normally assumes
    """
    dpi = 300
    image_arr.astype(np.uint8)
    if not size:
        size = (image_arr.shape[0] / dpi, image_arr.shape[1] / dpi)
    plt.figure(figsize=size, dpi=dpi)
    plt.axis("off")
    plt.imshow(image_arr, cmap=cmap)
    plt.savefig(filename, dpi=dpi, bbox_inches="tight", pad_inches=0)


def process_patient(pat_id: str, paths: dict, pixel_wise: bool = True):
    # replacing the image path with the loaded images
    logger.info("Loading image paths")
    images_and_segmentations = load_images_from_paths(paths)
    images = images_and_segmentations["nifti"]
    segmentations = images_and_segmentations["segm"]
    patient_results = {"patid": pat_id, "ap": True, "lat": True}

    for side in ["ap", "lat"]:
        OrigpreEVT = images[side]["full"]["pre"]
        OrigpostEVT = images[side]["full"]["post"]

        preEVT = sift.remove_text_and_border(OrigpreEVT)
        postEVT = sift.remove_text_and_border(OrigpostEVT)

        transformation, matches = sift_matching(preEVT, postEVT)
        if not matches:
            logger.info("SIFT has no matches, continuing...")
            patient_results[side] = False
            continue

        # Check if the transformation quality is better than the original
        segm_pre_post = [
            segmentations[side]["art"]["pre"],
            segmentations[side]["art"]["post"],
        ]
        try:
            transform_failed, tr_postEVT, segm_pre_post[1] = check_transform(
                segm_pre_post, transformation, preEVT, postEVT
            )
        except ValueError:
            transform_failed = True
            tr_postEVT = postEVT
            return {"patid": pat_id, "ap": False, "lat": False}

        save_overlayed("overlayed_cases", pat_id, side, preEVT, tr_postEVT)

        matched_pixels = []
        if pixel_wise:
            matched_pixels = pixel_wise_method(segm_pre_post, vis=False)

        if transform_failed:
            patient_results[side] = False

        # 3. Skeletonization

        logger.info(f"Skeletonizing {side}")

        skeleton_images, distance_transform = generate_skeletons(segm_pre_post)

        # 4. Create Graphs and Extract labeled segments

        logger.info(f"Generating Graphs and Extracting Labeled Segments")

        pre_graph, post_graph, pre_graph_labeled_segs, post_graph_labeled_segs = (
            extract_labeled_segments(skeleton_images, distance_transform, segm_pre_post)
        )

        # 6. Iteratively compare segments and identify correspondence
        logger.info("Performing segment correspondence matching.")
        pre_labels_rgb, post_labels_rgb, pre_matched_segs, post_matched_segs = (
            find_correspondences(
                pre_graph_labeled_segs,
                post_graph_labeled_segs,
                matched_pixels,
                pixel_wise,
            )
        )

        save_labels(
            "cases", pat_id, side, preEVT, tr_postEVT, pre_labels_rgb, post_labels_rgb
        )
        print(f"{side} completed")
    return patient_results


def setup_logging():
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


def main(
    in_img_path,
    in_segm_path,
    in_pre_path,
    in_post_path,
    input_format="nifti",
    load_segs=False,
    pixel_wise=False,
    eval=False,
):
    setup_logging()

    # Temporary only run this
    # Load all input images into a dictionary for processing
    image_path_dictionary = get_image_pairs(
        in_img_path=in_img_path, in_segm_path=in_segm_path
    )
    df_successes = pd.DataFrame(columns=["patid", "ap", "lat"])
    processed_cases = os.listdir("cases")
    image_path_dictionary = {
        key: image_path_dictionary[key]
        for key in image_path_dictionary
        if key not in processed_cases
    }
    for patid, paths in image_path_dictionary.items():
        logger.info(f"{patid:=^20}")
        patient_success = process_patient(patid, paths, pixel_wise)
        df_successes = pd.concat([df_successes, pd.DataFrame([patient_success])])
    df_successes.to_csv("successes.csv")
    return

    if eval:
        ANNOT_DIR_PATH = "C:/Users/mab03/Desktop/AnnotationTool/Output"
        evaluate(ANNOT_DIR_PATH, pixel_wise=True, calculate_ci=True)
        return

    df_successes = pd.DataFrame(columns=["patid", "ap", "lat"])

    # pat_id = "R0002"
    # pat_ori = "1"
    # IMG_DIR_PATH = "Niftisv2/" + pat_id + "/" + pat_ori
    if in_img_path != None:
        IMG_DIR_PATH = in_img_path
        pat_ori = str(Path(IMG_DIR_PATH).stem)
        pat_id = str(Path(IMG_DIR_PATH).parent.stem)
        images_path = sift.load_img_dir(IMG_DIR_PATH, img_type="nifti")
        # Check if list is empty
        if not images_path:
            return

        # Load images from paths
        images = sift.load_pre_post_imgs(images_path)

    elif in_pre_path and in_post_path != None:
        images = []
        images.append(sift.load_img(in_pre_path))
        images.append(sift.load_img(in_post_path))
    else:
        raise "One of two possible inputs must be provided."

    # patient_results = {"patid": pat_id, "ap": True, "lat": True}

    OrigpreEVT = images[0]
    OrigpostEVT = images[1]

    preEVT = sift.remove_text_and_border(OrigpreEVT)
    postEVT = sift.remove_text_and_border(OrigpostEVT)

    transformation, matches = sift_matching(preEVT, postEVT)
    if not matches:
        logger.info("SIFT has no matches, continuing...")
        # patient_results[side] = False
        # continue the loop

    # 2. Segmentation

    if load_segs and in_img_path != None:
        # If True we assume that the feature maps will also be loaded
        IMG_MIN_DIR_PATH = (
            "C:/Users/mab03/Desktop/RuSegm/TemporalUNet/Outputs/Minip/" + pat_id
        )
        IMG_SEQ_DIR_PATH = in_segm_path + "/" + pat_id + "/" + pat_ori
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
        # Feature maps not needed for the sift-baseline method. This still saves segmentations.
        # feat_map_pre_post = predict.run_predict(
        #     # in_img_path="E:/vessel_diff_first_50_patients/mrclean_part1_2_first_50/R0002",
        #     in_img_path=IMG_DIR_PATH,
        #     out_img_path=segm_output_folder,
        #     model="C:/Users/mab03/Desktop/RuSegm/TemporalUNet/models/1096-sigmoid-sequence-av.pt",
        #     input_type="sequence",
        #     input_format="nifti",
        #     label_type="av",
        #     amp=True,
        # )
        # On the fly segmentation now only works if you provide pre-post inputs.
        # Otherwise there is no way to know which is pre and post using a directory as input
        feat_map_pre_post = []
        feat_map_pre_post.append(
            predict.run_predict(
                # in_img_path="E:/vessel_diff_first_50_patients/mrclean_part1_2_first_50/R0002",
                in_img_path=in_pre_path,
                out_img_path=segm_output_folder,
                model="C:/Users/mab03/Desktop/RuSegm/TemporalUNet/models/1096-sigmoid-sequence-av.pt",
                input_type="sequence",
                input_format=input_format,
                label_type="av",
                amp=True,
            )
        )
        feat_map_pre_post.append(
            predict.run_predict(
                # in_img_path="E:/vessel_diff_first_50_patients/mrclean_part1_2_first_50/R0002",
                in_img_path=in_post_path,
                out_img_path=segm_output_folder,
                model="C:/Users/mab03/Desktop/RuSegm/TemporalUNet/models/1096-sigmoid-sequence-av.pt",
                input_type="sequence",
                input_format=input_format,
                label_type="av",
                amp=True,
            )
        )

        # Plot the feature maps of one seq for reference (Optional)
        # predict.display_feature_maps(feat_map_pre_post)
        # The list contains two maps, where pre-post are first-second
        # The maps have a torch tensor format
        IMG_SEQ_DIR_PATH = segm_output_folder

    # segm_images = load_images(IMG_MIN_DIR_PATH)
    segm_images = sklt.load_images(IMG_SEQ_DIR_PATH)

    # Find corresponding pre and post images from the segmentations and feat maps
    segm_pre_post = []
    # If data was prepared we check for this
    if in_img_path != None:
        for segm in segm_images:
            if Path(segm).stem.rsplit("_", 1)[1] == "artery":
                if Path(segm).stem.rsplit("_", 2)[1] == "pre":
                    segm_pre_post.insert(0, sift.load_img(segm))
                else:
                    segm_pre_post.append(sift.load_img(segm))
    else:
        for segm in segm_images:
            if Path(segm).stem.rsplit("_", 1)[1] == "artery":
                if Path(segm).stem.rsplit("_", 1)[0] == Path(in_pre_path).stem:
                    segm_pre_post.insert(0, sift.load_img(segm))
                else:
                    segm_pre_post.append(sift.load_img(segm))

    # Check if the transformation quality is better than the original

    try:
        transform_failed, tr_postEVT, segm_pre_post[1] = check_transform(
            segm_pre_post, transformation, preEVT, postEVT
        )
    except ValueError:
        transform_failed = True
        tr_postEVT = postEVT
        return {"patid": pat_id, "ap": False, "lat": False}

    # Careful with this since arrays are mutable
    # Used final_segm_post in order to plot original and transformed post
    # After the plotting this variable is not necessary anymore.
    # segm_pre_post[1] = final_segm_post

    if pixel_wise:
        matched_pixels = pixel_wise_method(segm_pre_post, vis=True)
        # return

    # 3. Skeletonization

    logger.info(f"Skeletonizing")

    skeleton_images, distance_transform = generate_skeletons(segm_pre_post)

    # 4. Create Graphs and Extract labeled segments

    logger.info(f"Generating Graphs and Extracting Labeled Segments")

    pre_graph, post_graph, pre_graph_labeled_segs, post_graph_labeled_segs = (
        extract_labeled_segments(skeleton_images, distance_transform, segm_pre_post)
    )
    # fig, axs = plt.subplots(1, 3)
    # axs[0].imshow(labels_rgb1)
    # axs[1].imshow(labels_rgb2)
    # axs[2].imshow(labels_rgb1)
    # axs[2].imshow(labels_rgb2, alpha=0.5)
    # for a in axs:
    #     a.axis("off")
    # plt.tight_layout()
    # plt.show()

    # 6. Iteratively compare segments and identify correspondence
    logger.info("Performing segment correspondence matching.")
    pre_labels_rgb, post_labels_rgb, pre_matched_segs, post_matched_segs = (
        find_correspondences(
            pre_graph_labeled_segs, post_graph_labeled_segs, matched_pixels, pixel_wise
        )
    )

    visualize_results(
        segm_pre_post[0], segm_pre_post[1], pre_labels_rgb, post_labels_rgb
    )
    preEVT = resize(preEVT, (512, 512))
    tr_postEVT = resize(tr_postEVT, (512, 512))
    visualize_results(preEVT, tr_postEVT, pre_labels_rgb, post_labels_rgb)


# fmt: off
def get_args():
    parser = argparse.ArgumentParser(description="Find correspondeces using SIFT on a set of pre/post-EVT DSA images")
    parser.add_argument('--in_img_path','-i',default=None, help='Directory of pre-post DSA sequences if data was prepared.')
    parser.add_argument('--in_segm_path','-is',default=None, help='Directory of pre-post DSA segmentations if data was prepared.')
    parser.add_argument('--in_pre_path','-pre',default=None, help='Path of pre-DSA sequence.')
    parser.add_argument('--in_post_path','-post',default=None, help='Path of post-DSA sequence.')
    parser.add_argument('--input-format','-f',default='dicom',help='Input format - dicom or nifti')
    parser.add_argument("--load-segs",action="store_true",default=False,help="Load the segmentations.")
    parser.add_argument("--pixel-wise",action="store_true",default=False,help="Use the pixel wise method for matching.")
    parser.add_argument("--eval",action="store_true",default=False,help="Evaluate the method.")
#fmt:on

    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    main(**vars(args))
