import numpy as np
import pygmtools as pygm
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter
from functools import partial
import cv2
from skimage.transform import resize
import skimage as ski


def calc_similarity(feat_matrix_1, feat_matrix_2):
    similarity_matrix = np.inner(feat_matrix_1, feat_matrix_2)
    # print(similarity_matrix.shape)
    return similarity_matrix


def create_feat_matrix(graph):
    feat_matrix = np.zeros((len(graph.vs), len(graph.vs[0].attributes())))
    # print(feat_matrix.shape)
    for idx, vertex in enumerate(graph.vs):
        feat_matrix[idx, :] = [value for value in vertex.attributes().values()]

    return feat_matrix


def calc_feat_matrix_info(feat_matrix):
    f_avg = np.mean(feat_matrix)
    f_min = np.min(feat_matrix)
    f_max = np.max(feat_matrix)
    f_std = np.std(feat_matrix)
    # Plot histogram to see the distribution
    # counts, bins = np.histogram(feat_matrix)
    # plt.stairs(counts, bins)
    # plt.xlabel("feature values")
    # plt.ylabel("frequency")
    # plt.show()

    return f_avg, f_max, f_min, f_std


def calc_graph_radius_info(graph):
    radii = graph.vs["radius"]
    r_avg = np.mean(radii)
    r_min = np.min(radii)
    r_max = np.max(radii)
    r_std = np.std(radii)
    # Plot histogram to see the distribution
    # counts, bins = np.histogram(radii)
    # plt.stairs(counts, bins)
    # plt.xlabel("radii")
    # plt.ylabel("frequency")
    # plt.show()

    return r_avg, r_max, r_min, r_std


def calc_assignment_matrix(similarity_mat, tau=1, iter=10):
    # This function uses the Sinkhorn algorithm to calculate a doubly-stochastic matrix
    # Can add unmatch arguments to use the 'dustbin' logic as in SuperGlue
    unm1 = np.zeros(similarity_mat.shape[0])
    unm2 = np.zeros(similarity_mat.shape[1])
    S = pygm.sinkhorn(
        similarity_mat,
        dummy_row=True,
        max_iter=iter,
        tau=tau,
        unmatch1=unm1,
        unmatch2=unm2,
    )
    print("row_sum:", S.sum(1), "col_sum:", S.sum(0))
    print(S.shape)
    return S


def draw_keypoints(pre_img, post_img, pre_kpts, post_kpts):
    # Resize images to fit segmentation shape
    pre_img = resize(pre_img, (pre_img.shape[0] // 2, pre_img.shape[1] // 2))
    post_img = resize(post_img, (post_img.shape[0] // 2, post_img.shape[1] // 2))
    pre_img = ski.util.img_as_ubyte(pre_img)
    post_img = ski.util.img_as_ubyte(post_img)
    pre_img = cv2.cvtColor(pre_img, cv2.COLOR_GRAY2RGB)
    post_img = cv2.cvtColor(post_img, cv2.COLOR_GRAY2RGB)

    pre_img_w_kps = np.copy(pre_img)
    post_img_w_kps = np.copy(post_img)

    cv2.drawKeypoints(pre_img, pre_kpts, pre_img_w_kps)
    cv2.drawKeypoints(post_img, post_kpts, post_img_w_kps)

    # Visualise Keypoints on both images
    fig, axs = plt.subplots(1, 2, figsize=(10, 6))

    axs[0].set_title("Pre-EVT Graph keypoints")
    axs[0].imshow(pre_img_w_kps, cmap="gray")

    axs[1].set_title("Post-EVT Graph keypoints")
    axs[1].imshow(post_img_w_kps, cmap="gray")
    plt.show()


def draw_matches(pre_img, post_img, pre_kpts, post_kpts, matches):
    # Resize images to fit segmentation shape
    pre_img = resize(pre_img, (pre_img.shape[0] // 2, pre_img.shape[1] // 2))
    post_img = resize(post_img, (post_img.shape[0] // 2, post_img.shape[1] // 2))
    pre_img = ski.util.img_as_ubyte(pre_img)
    post_img = ski.util.img_as_ubyte(post_img)
    pre_img = cv2.cvtColor(pre_img, cv2.COLOR_GRAY2RGB)
    post_img = cv2.cvtColor(post_img, cv2.COLOR_GRAY2RGB)

    # Prepare keypoints for cv2
    pre_kpts = [cv2.KeyPoint(pt[1], pt[0], 1) for pt in pre_kpts]
    post_kpts = [cv2.KeyPoint(pt[1], pt[0], 1) for pt in post_kpts]

    # Prepare matches for cv2
    matches_list = []
    # Unpacking the range into a list
    pre_graph_node_idx = [*range(len(matches))]
    # Distance is set to 1 for all as a placeholder
    distance = 1  # np.zeros(len(matches))
    for i, j in zip(pre_graph_node_idx, matches):
        # Create cv2 DMatch object
        matches_list.append(cv2.DMatch(i, j, distance))

    matches_img = cv2.drawMatches(
        pre_img,
        pre_kpts,
        post_img,
        post_kpts,
        matches_list[0:50],
        outImg=None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )

    fig, axs = plt.subplots(figsize=(10, 6))
    axs.imshow(matches_img)
    axs.set_title("Sinkhorn Matches")
    # plt.figure(figsize=(10, 6))
    # plt.imshow(matches_img)
    # plt.title("Sinkhorn Matches")
    plt.show()


def animate(idx, ax, interval, pre_img, post_img, pre_kpts, post_kpts, matches):
    ax.clear()

    plot_from = idx - interval
    plot_to = idx
    if idx == 0:
        plot_from = 0
        plot_to = interval
    if idx == len(matches):
        plot_from = len(matches) - (len(matches) % interval)
        plot_to = len(matches)

    matches_img = cv2.drawMatches(
        pre_img,
        pre_kpts,
        post_img,
        post_kpts,
        matches[plot_from:plot_to],
        outImg=None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )
    ax.imshow(matches_img)


def draw_matches_animated(
    pre_img, post_img, pre_kpts, post_kpts, matches, save_fig=False
):
    # Resize images to fit segmentation shape
    pre_img = resize(pre_img, (pre_img.shape[0] // 2, pre_img.shape[1] // 2))
    post_img = resize(post_img, (post_img.shape[0] // 2, post_img.shape[1] // 2))
    pre_img = ski.util.img_as_ubyte(pre_img)
    post_img = ski.util.img_as_ubyte(post_img)
    pre_img = cv2.cvtColor(pre_img, cv2.COLOR_GRAY2RGB)
    post_img = cv2.cvtColor(post_img, cv2.COLOR_GRAY2RGB)

    # Prepare keypoints for cv2
    pre_kpts = [cv2.KeyPoint(pt[1], pt[0], 1) for pt in pre_kpts]
    post_kpts = [cv2.KeyPoint(pt[1], pt[0], 1) for pt in post_kpts]

    # Prepare matches for cv2
    matches_list = []
    # Unpacking the range into a list
    pre_graph_node_idx = [*range(len(matches))]
    # Distance is set to 1 for all as a placeholder
    distance = 1  # np.zeros(len(matches))
    for i, j in zip(pre_graph_node_idx, matches):
        # Create cv2 DMatch object
        matches_list.append(cv2.DMatch(i, j, distance))

    interval = 30
    frames_list = [*range(0 + interval, len(matches_list), interval)]
    frames_list.append(len(matches_list))

    fig, axs = plt.subplots(figsize=(10, 6))
    axs.set_title("Sinkhorn Matches")
    # Plot the first frame
    matches_img = cv2.drawMatches(
        pre_img,
        pre_kpts,
        post_img,
        post_kpts,
        matches_list[0:interval],
        outImg=None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )
    axs.imshow(matches_img)
    axs.get_xaxis().set_visible(False)
    axs.get_yaxis().set_visible(False)

    animation = FuncAnimation(
        fig,
        partial(
            animate,
            ax=axs,
            interval=interval,
            pre_img=pre_img,
            pre_kpts=pre_kpts,
            post_img=post_img,
            post_kpts=post_kpts,
            matches=matches_list,
        ),
        frames=frames_list,
        interval=1000,
    )
    plt.show()

    if save_fig:
        animation.save(
            "Outputs/AnimatedMatches.gif", dpi=300, writer=PillowWriter(fps=1)
        )

        animation.save(
            "Outputs/AnimatedMatches.mp4", dpi=300, writer=FFMpegWriter(fps=1)
        )
