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
                    if Path(segm).stem.rsplit("_")[1] == "pre":
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


def evaluate_patches(annot_dir):

    # Load annotations
    json__files = sorted(glob(os.path.join(annot_dir, "**", "*.json"), recursive=True))
    for json_path in json__files:
        with open(json_path, "r") as json_file:
            data = json.load(json_file)
            matches = data["matches"]
            print(matches)


# fmt: off
def get_args():
    parser = argparse.ArgumentParser(description='Prepare and/or evaluate method using annotations')
    parser.add_argument('--prepare', action='store_true', default=False, help='Prepare the patched images for annotation.')
    parser.add_argument('--evaluate', action='store_true', default=True, help='Evaluate method using the annotated patched images.')
    
    return parser.parse_args()
# fmt: on


def main(prepare=False, evaluate=False):
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
            IMG_DIR_PATH = "Niftisv2/"
            prepare_patches(IMG_DIR_PATH)
        else:
            print("Canceling preparation...")

    if evaluate:
        ANNOT_DIR_PATH = "C:/Users/mab03/Desktop/AnnotationTool/Output"
        evaluate_patches(ANNOT_DIR_PATH)


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
