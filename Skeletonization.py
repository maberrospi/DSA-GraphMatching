import glob
import logging
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.widgets import Button
from skimage.morphology import skeletonize, medial_axis, thin
from scipy.ndimage import distance_transform_edt

logger = logging.getLogger(__name__)


def load_images(img_dir_path):
    if not os.path.isdir(img_dir_path):
        logger.warning("Path directory {} does not exist".format(img_dir_path))
        return
    logger.info("Loading images from {}".format(img_dir_path))
    images = sorted(
        glob.glob(os.path.join(img_dir_path, "**", "*.png"), recursive=True)
    )

    return images


def vis_image(images, idx=0):
    img = cv2.imread(images[idx], cv2.IMREAD_GRAYSCALE)
    # cv2.imshow("Segmentation", img)
    filename = Path(images[idx]).stem
    logger.info("Visualizing image with filename {}".format(filename))
    art_or_vein = filename.split("_")[-1]
    sn_num = filename.split("_")[0]
    if art_or_vein == "artery" or art_or_vein == "vein":
        plt.title(f"{sn_num}-{art_or_vein} segmentation")
    else:
        plt.title(f"{sn_num} segmentation")
    plt.imshow(img, cmap="gray")
    plt.show()


def vis_skeleton(sklt_image):
    fig, ax = plt.subplots()
    ax[0].set_title("Skeleton")
    ax[0].imshow(sklt_image, cmap="gray")


def vis_skeleton_and_segm(sklt_image, segmentation):
    fig, ax = plt.subplots(1, 2, figsize=(10, 6))
    ax[0].set_title("Skeleton")
    ax[1].set_title("Skeleton on Segmentation")
    ax[0].imshow(sklt_image, cmap="gray")
    ax[1].imshow(segmentation, cmap="gray")
    # skel_col = colors.ListedColormap(["white", "red"])  # Not used after all.
    ax[1].imshow(sklt_image, cmap="Purples", alpha=0.5)


def get_skeletons(segm_images, method="zhang"):
    # A thinning algorithm can also be added here
    if not segm_images:
        logger.warning("The segmentation images list provided is empty")
        return
    if method not in ["zhang", "lee", "medial", "thin"]:
        logger.warning("Method provided should be 'zhang', 'lee', 'medial' or 'thin'")
        return

    skeleton_images = []
    distance = []
    for image in segm_images:
        # If image is a path read image, otherwise image is already read
        if isinstance(image, str):
            img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
        else:
            img = image
        if method in ["zhang", "lee"]:
            distance.append(distance_transform_edt(img))
            skeleton_images.append(skeletonize(img, method=method))
        elif method == "medial":
            # Compute the medial axis (skeleton) and the distance transform
            skel, dtf = medial_axis(img, return_distance=True)
            skeleton_images.append(skel)
            distance.append(dtf)
        else:
            # Compute skeletonization using thinning algorithm
            distance.append(distance_transform_edt(img))
            skeleton_images.append(thin(img))

    return skeleton_images, distance


def find_centerlines(skeleton):
    # Returns the coordinates of all the white pixels with x and y as columns
    # In other words the shape is (n,2) where n is the number of white pxls
    points = np.vstack(np.nonzero(skeleton)).T
    return points


class Visualize:
    ind = 1
    fig = None
    ax = None
    cur_img = None
    axprev = None
    axnext = None

    def __init__(self, images):
        self.images = images
        self.upper_bound = len(images)
        self.lower_bound = 1

    def vis_images(self):
        self.fig, self.ax = plt.subplots()
        self.fig.subplots_adjust(bottom=0.2)
        self.axprev = self.fig.add_axes([0.7, 0.05, 0.1, 0.075])
        self.axnext = self.fig.add_axes([0.81, 0.05, 0.1, 0.075])
        self.text = self.ax.text(
            0.5,  # 256 without transAxes
            -0.15,  # 590
            self.ind,
            ha="center",
            va="center",
            fontweight="bold",
            bbox={
                "capstyle": "round",
                "facecolor": "cyan",
                "alpha": 0.5,
                "pad": 7,
            },
            transform=self.ax.transAxes,
        )
        # for a in self.ax:
        self.ax.axis("off")
        # self.fig.tight_layout()
        self.vis_img(1)

    def vis_img(self, idx):
        # self.ax.clear()
        img = cv2.imread(self.images[idx - 1], cv2.IMREAD_GRAYSCALE)
        filename = Path(self.images[idx - 1]).stem
        art_or_vein = filename.split("_")[-1]
        sn_num = filename.split("_")[0]
        if art_or_vein == "artery" or art_or_vein == "vein":
            self.ax.set_title(f"{sn_num}-{art_or_vein} segmentation")
        else:
            self.ax.set_title(f"{sn_num} segmentation")
        self.cur_img = self.ax.imshow(img, cmap="gray")
        self.text.set_text(self.ind)
        logger.info("Visualizing image with filename {}".format(filename))

    def next(self, event):
        if self.ind == self.upper_bound:
            return
        self.ind += 1
        self.vis_img(self.ind)
        plt.draw()

    def prev(self, event):
        if self.ind == self.lower_bound:
            return
        self.ind -= 1
        self.vis_img(self.ind)
        plt.draw()


class VisualizeSkeletons:
    ind = 1
    fig = None
    ax = None
    cur_img = None
    axprev = None
    axnext = None
    first_pass = True

    def __init__(self, skeletons, segms):
        self.skeletons = skeletons
        self.segms = segms
        self.upper_bound = len(skeletons)
        self.lower_bound = 1

    def vis_images(self):
        self.fig, self.ax = plt.subplots(1, 2, figsize=(10, 6))
        self.fig.subplots_adjust(bottom=0.2)
        self.ax[0].set_title("Skeleton")
        self.ax[1].set_title("Skeleton on Segmentation")
        self.axprev = self.fig.add_axes([0.7, 0.05, 0.1, 0.075])
        self.axnext = self.fig.add_axes([0.81, 0.05, 0.1, 0.075])
        self.text = self.fig.text(
            0.5,
            0.1,
            self.ind,
            ha="center",
            va="center",
            fontweight="bold",
            bbox={
                "capstyle": "round",
                "facecolor": "cyan",
                "alpha": 0.5,
                "pad": 7,
            },
        )
        self.vis_img(1)

    def vis_img(self, idx):
        # How to set attrs of imshow returned class
        # https://matplotlib.org/stable/api/image_api.html#matplotlib.image.AxesImage
        # It seems to get slow if you go back and forth multiple times - Fixed
        # Tried to fix it by clearing axes but for some reason it does not
        # Re-draw properly and gets stuck on the first image.
        # NOTE: The for loop does not work but the single lines do. weird.
        # for idx in range(len(self.ax)):
        #     self.ax[idx].cla()
        #     print(idx)
        self.ax[1].cla()
        self.ax[0].cla()
        skeleton = self.skeletons[idx - 1]
        # If image is a path read image, otherwise image is already read
        if isinstance(self.segms[idx - 1], str):
            segm = cv2.imread(self.segms[idx - 1], cv2.IMREAD_GRAYSCALE)
        else:
            segm = self.segms[idx - 1]
        segm_w_skeleton = np.where(skeleton == 1, 0.5, segm)
        segm_w_skeleton = np.dstack([segm_w_skeleton, segm_w_skeleton, segm_w_skeleton])
        segm_w_skeleton = np.where(
            segm_w_skeleton == [0.5, 0.5, 0.5], [1, 0, 0], segm_w_skeleton
        )
        segm_w_skeleton *= 255
        if self.first_pass:
            self.cur_img = self.ax[0].imshow(skeleton, cmap="gray")
            self.cur_overl = self.ax[1].imshow(segm_w_skeleton)
            # self.testimg = self.ax[1].imshow(skeleton, cmap="Purples", alpha=0.5)
        else:
            self.cur_img.set(data=skeleton)
            self.cur_overl.set(data=segm_w_skeleton)
            # self.testimg.set(data=skeleton, alpha=0.5)
        self.text.set_text(self.ind)
        self.fig.canvas.draw_idle()
        # plt.draw()

    def next(self, event):
        if self.ind == self.upper_bound:
            return
        self.ind += 1
        self.vis_img(self.ind)

    def prev(self, event):
        if self.ind == self.lower_bound:
            return
        self.ind -= 1
        self.vis_img(self.ind)


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

    # IMG_MIN_DIR_PATH = "C:/Users/mab03/Desktop/RuSegm/TemporalUNet/Outputs/Minip/R0001"
    # IMG_SEQ_DIR_PATH = (
    #     "C:/Users/mab03/Desktop/RuSegm/TemporalUNet/Outputs/Sequence/R0001"
    # )

    # V2
    # IMG_MIN_DIR_PATH = "C:/Users/mab03/Desktop/RuSegm/TemporalUNet/Outputs/Minip/R0002"
    IMG_MIN_DIR_PATH = "C:/Users/mab03/Desktop/ThesisCode/Outputs/test"
    IMG_SEQ_DIR_PATH = "C:/Users/mab03/Desktop/ThesisCode/Segms/Sequence/R0002"

    segm_images = load_images(IMG_MIN_DIR_PATH)
    # vis_image(segm_images, 0)
    vis = Visualize(segm_images)
    vis.vis_images()
    bnext = Button(vis.axnext, "Next")
    bnext.on_clicked(vis.next)
    bprev = Button(vis.axprev, "Previous")
    bprev.on_clicked(vis.prev)

    segm_images1 = load_images(IMG_SEQ_DIR_PATH)
    vis1 = Visualize(segm_images1)
    vis1.vis_images()
    bnext1 = Button(vis1.axnext, "Next")
    bnext1.on_clicked(vis1.next)
    bprev1 = Button(vis1.axprev, "Previous")
    bprev1.on_clicked(vis1.prev)

    # Perform skeletonization
    skeleton_images, distance_transform = get_skeletons(segm_images1, method="lee")
    if not skeleton_images:
        return

    # Multiply skeleton images with the distance transform (optional)
    # skeleton_images = [
    #     sklt * dst for sklt, dst in zip(skeleton_images, distance_transform)
    # ]

    vis_skeletons = VisualizeSkeletons(skeleton_images, segm_images1)
    vis_skeletons.vis_images()
    bnext2 = Button(vis_skeletons.axnext, "Next")
    bnext2.on_clicked(vis_skeletons.next)
    bprev2 = Button(vis_skeletons.axprev, "Previous")
    bprev2.on_clicked(vis_skeletons.prev)

    # fig, axs = plt.subplots()
    # axs.imshow()
    # axs.imshow(skeleton_images[0], cmap="gray")

    plt.show()


if __name__ == "__main__":
    main()
