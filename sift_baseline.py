import argparse
import logging
import os
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.widgets import Button
import igraph as ig
import cv2
from skimage.segmentation import watershed
from skimage.color import label2rgb
from shapely.geometry import Polygon

import SIFTTransform as sift
import Skeletonization as sklt
import graph_processing as GProc

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
            # print(endpoint_neighbors)
            for i in range(2):
                # FYI. graph.vs[point_list[-i]].neighbors(): returns a node object
                for nb in graph.neighbors(point_list[-i]):
                    if nb not in endpoint_neighbors:
                        if i == 0:
                            point_list.insert(0, nb)
                        else:
                            point_list.append(nb)

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
            # print(pt)
            # Add its index label to the pt positions
            labeled_segments[pt[1], pt[0]] = label_idx

        label_idx += 1

    # Visualize
    labels_rgb = label2rgb(labeled_segments, bg_label=0, bg_color=[0, 0, 0])
    fig, axs = plt.subplots(1, 2)
    lbl_img = axs[0].imshow(labels_rgb)
    sklt_img = axs[1].imshow(skeleton_img, cmap="gray")

    plt.show()

    return sk_graph, labeled_segments


def map_labels_to_segmentations(segm, dtf, labeled_segm):
    # Apply the watershed algorithm using the labeled skeleton as markers
    # Int cast for issue with watershed
    labeled_segm = labeled_segm.astype(int)
    labels = watershed(-dtf, markers=labeled_segm, mask=segm)
    labels_rgb = label2rgb(labels, bg_label=0, bg_color=[0, 0, 0])

    fig, axs = plt.subplots(1, 2)
    axs[0].imshow(labels_rgb)
    axs[1].imshow(segm, cmap="gray")
    for a in axs:
        a.axis("off")
    plt.tight_layout()
    plt.show()

    return labels


def main(load_segs=False):
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

    pat_id = "R0002"
    pat_ori = "1"
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
        # The list contains two maps, where pre-post are first-second
        # The maps have a torch tensor format
        IMG_SEQ_DIR_PATH = segm_output_folder

    # segm_images = load_images(IMG_MIN_DIR_PATH)
    segm_images1 = sklt.load_images(IMG_SEQ_DIR_PATH)

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

    # Careful with this since arrays are mutable
    # Used final_segm_post in order to plot original and transformed post
    # After the plotting this variable is not necessary anymore.
    segm_pre_post[1] = final_segm_post

    # 3. Skeletonization

    # Perform skeletonization
    skeleton_images, distance_transform = sklt.get_skeletons(
        segm_pre_post,
        method="lee",
    )
    if not skeleton_images:
        return

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

    labels_rgb1 = label2rgb(pre_graph_labeled_segs, bg_label=0, bg_color=[0, 0, 0])
    labels_rgb2 = label2rgb(post_graph_labeled_segs, bg_label=0, bg_color=[0, 0, 0])
    fig, axs = plt.subplots(1, 3)
    axs[0].imshow(labels_rgb1)
    axs[1].imshow(labels_rgb2)
    axs[2].imshow(labels_rgb1)
    axs[2].imshow(labels_rgb2, alpha=0.5)
    for a in axs:
        a.axis("off")
    plt.tight_layout()
    plt.show()

    # 6. Iteratively compare segments and identify correspondence
    logger.info("Finding segment correspondences.")
    pre_graph_unique_segs = np.unique(pre_graph_labeled_segs)
    post_graph_unique_segs = np.unique(post_graph_labeled_segs)

    pre_matched_segs = np.zeros(pre_graph_labeled_segs.shape)
    post_matched_segs = np.zeros(post_graph_labeled_segs.shape)
    match_id = 1

    for pre_segm in pre_graph_unique_segs:
        if pre_segm == 0:  # Skip id == 0
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
            # x1, y1 = pre_polygon.exterior.xy
            # x2, y2 = post_polygon.exterior.xy
            # fig, ax = plt.subplots()
            # ax.plot(x1, y1)
            # ax.plot(x2, y2)
            # plt.show()

            # Calculate the intersection and union
            intersection = pre_polygon.intersection(post_polygon)
            union = pre_polygon.union(post_polygon)

            # Calculate areas
            intersection_area = intersection.area
            union_area = union.area

            # Calculate IoU
            iou = intersection_area / union_area if union_area != 0 else 0

            iou_threshold = 0.5

            if iou > iou_threshold:
                pre_matched_segs[pre_temp_mask == 255] = match_id
                post_matched_segs[post_temp_mask == 255] = match_id
                match_id += 1
                break
    logger.info("Segments matched successfully!")

    pre_labels_rgb = label2rgb(pre_matched_segs, bg_label=0, bg_color=[0, 0, 0])
    post_labels_rgb = label2rgb(post_matched_segs, bg_label=0, bg_color=[0, 0, 0])

    # fig, axs = plt.subplots(1, 2)
    # axs[0].imshow(pre_labels_rgb)
    # axs[1].imshow(post_labels_rgb)
    # for a in axs:
    #     a.axis("off")
    # plt.tight_layout()
    # plt.show()

    segm_pre_rgb = np.dstack((segm_pre_post[0],) * 3)
    segm_post_rgb = np.dstack((segm_pre_post[1],) * 3)

    # print(pre_labels_rgb[:5, :5, :])
    black_pixels_mask = (pre_labels_rgb == [0, 0, 0]).all(axis=-1)
    pre_labels_rgb_overlayed = np.where(
        black_pixels_mask[..., None], segm_pre_rgb, pre_labels_rgb
    )
    black_pixels_mask = (pre_labels_rgb == [0, 0, 0]).all(axis=-1)
    post_labels_rgb_overlayed = np.where(
        black_pixels_mask[..., None], segm_post_rgb, post_labels_rgb
    )

    fig, axs = plt.subplots(1, 2)
    axs[0].imshow(pre_labels_rgb_overlayed)
    axs[1].imshow(post_labels_rgb_overlayed)
    for a in axs:
        a.axis("off")
    plt.tight_layout()
    plt.show()


# fmt: off
def get_args():
    parser = argparse.ArgumentParser(description="Find correspondeces using SIFT on a set of pre/post-EVT DSA images")
    parser.add_argument("--load-segs",action="store_true",default=True,help="Load the segmentations.")
#fmt:on

    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    main(**vars(args))
