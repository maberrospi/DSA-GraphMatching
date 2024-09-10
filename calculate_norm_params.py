from sklearn.preprocessing import StandardScaler, MinMaxScaler
import numpy as np
from pathlib import Path
import os
from glob import glob
import json

import filter_banks as FilterBanks
import SIFTTransform as sift
import Skeletonization as sklt
import graph_processing as GProc
import graph_matching as GMatch


def calc_params():
    # Calculates mean, std, min and max from data included in the annotations
    ANNOT_DIR_PATH = "C:/Users/mab03/Desktop/AnnotationTool/Output"

    # Create Leung-Malik filter bank
    LM_filter_banks = FilterBanks.makeLMfilters()

    # Load annotations
    json__files = sorted(
        glob(os.path.join(ANNOT_DIR_PATH, "**", "*.json"), recursive=True)
    )
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

            # Use Leung-Malik features
            pre_feat_map = FilterBanks.create_ft_map(LM_filter_banks, preEVT)
            post_feat_map = FilterBanks.create_ft_map(LM_filter_banks, tr_postEVT)
            feat_map_pre_post = [pre_feat_map, post_feat_map]

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

            # Test averaging the neighborhood (3x3) or (5x5) or (7x7) values for the features
            # Pre-EVT graph
            GProc.concat_extracted_features_v2(
                pre_graph, pre_feat_map, nb_size=24, inplace=True
            )
            # Post-EVT graph
            GProc.concat_extracted_features_v2(
                post_graph, post_feat_map, nb_size=24, inplace=True
            )

            # 6. Preprocess features if needed

            # Delete the coords attribute before calculating similarity
            del pre_graph.vs["coords"]
            del post_graph.vs["coords"]

            # Delete the node radius info since I think it is misleading
            # del cur_pre_graph.vs["radius"]
            # del cur_post_graph.vs["radius"]
            # We can also delete x,y features if the graph visualization is not used.
            # del pre_graph.vs["x"]
            # del pre_graph.vs["y"]
            # del post_graph.vs["y"]
            # del post_graph.vs["x"]

            print(pre_graph.vs[0].attributes())

            # 7. Calculate graph node feature matrices and perform matching

            # Create a feature matrices from all the node attributes
            pre_feat_matrix = GMatch.create_feat_matrix(pre_graph)
            post_feat_matrix = GMatch.create_feat_matrix(post_graph)

            if idx == 0:
                feat_matrices = np.concatenate((pre_feat_matrix, post_feat_matrix))
            else:
                feat_matrices = np.concatenate(
                    (feat_matrices, pre_feat_matrix, post_feat_matrix)
                )

            print(feat_matrices.shape)

    z_score = StandardScaler()
    z_score.fit(feat_matrices)
    print(f"Mean: {z_score.mean_} - Std:{z_score.scale_}")

    min_max = MinMaxScaler()
    min_max.fit(feat_matrices)
    print(f"Min: {min_max.data_min_} - Max:{min_max.data_max_}")


def main() -> None:
    calc_params()


if __name__ == "__main__":
    main()
