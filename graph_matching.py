import numpy as np
import pygmtools as pygm
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter
from functools import partial
import cv2
from skimage.transform import resize
import skimage as ski
from scipy.spatial import distance
import igraph as ig
from matplotlib.widgets import Slider
from random import shuffle


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


def create_edge_feat_matrix(graph):
    feat_matrix = np.zeros((len(graph.es), len(graph.es[0].attributes())))
    # print(feat_matrix.shape)
    for idx, edge in enumerate(graph.es):
        feat_matrix[idx, :] = [value for value in edge.attributes().values()]

    return feat_matrix


def calc_feat_matrix_info(feat_matrix, vis=False):
    f_avg = np.mean(feat_matrix)
    f_min = np.min(feat_matrix)
    f_max = np.max(feat_matrix)
    f_std = np.std(feat_matrix)
    if vis:
        # Plot histogram to see the distribution
        counts, bins = np.histogram(feat_matrix)
        plt.stairs(counts, bins)
        plt.xlabel("feature values")
        plt.ylabel("frequency")
        plt.show()

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


def calc_assignment_matrix(similarity_mat, tau=1, iter=10, method="sinkhorn"):
    # This function uses the Sinkhorn algorithm to calculate a doubly-stochastic matrix
    assert method in [
        "sinkhorn",
        "hungarian",
    ], "Invalid method - Choose sinkhorn or hungarian"
    if method == "sinkhorn":
        # Can add unmatch arguments to use the 'dustbin' logic as in SuperGlue
        # unm1 = np.zeros(similarity_mat.shape[0])
        # unm2 = np.zeros(similarity_mat.shape[1])
        S = pygm.sinkhorn(
            similarity_mat,
            dummy_row=True,
            max_iter=iter,
            tau=tau,
            # unmatch1=unm1,
            # unmatch2=unm2,
        )
    else:
        # Add dummy rows for the Hungarian
        orig_shape = similarity_mat.shape
        if similarity_mat.shape[0] < similarity_mat.shape[1]:
            # add dummy rows
            dummy_shape = list(similarity_mat.shape)
            dummy_shape[0] = similarity_mat.shape[1] - similarity_mat.shape[0]
            similarity_mat = np.concatenate(
                (similarity_mat, np.full(dummy_shape, 0)), axis=0
            )
        elif similarity_mat.shape[0] > similarity_mat.shape[1]:
            # add dummy cols
            dummy_shape = list(similarity_mat.shape)
            dummy_shape[1] = similarity_mat.shape[0] - similarity_mat.shape[1]
            similarity_mat = np.concatenate(
                (similarity_mat, np.full(dummy_shape, 0)), axis=1
            )
        # print(sim_matrix.shape)
        S = pygm.hungarian(similarity_mat)
        # Return the original shape of the matrix
        S = S[: orig_shape[0], : orig_shape[1]]

    print("row_sum:", S.sum(1), "col_sum:", S.sum(0))
    print(S.shape)

    return S


def calc_affinity_matrix(pre_g, post_g):
    # Create a feature matrices from all the node attributes
    pre_feat_matrix = create_feat_matrix(pre_g)
    post_feat_matrix = create_feat_matrix(post_g)

    # Create the feature matrices for all the edge attributes
    pre_e_feat_matrix = create_edge_feat_matrix(pre_g)
    post_e_feat_matrix = create_edge_feat_matrix(post_g)

    # Get adjacency matrix for both graphs
    A1 = np.array(pre_g.get_adjacency().data)
    A2 = np.array(post_g.get_adjacency().data)

    # Transform to sparse connectivity matrix for pygm
    A1, _ = pygm.utils.dense_to_sparse(A1)  # Don't need the edge weight tensor
    A2, _ = pygm.utils.dense_to_sparse(A2)

    # Change data types to conserve memory
    pre_feat_matrix = pre_feat_matrix.astype("float16")
    post_feat_matrix = post_feat_matrix.astype("float16")
    pre_e_feat_matrix = pre_e_feat_matrix.astype("float16")
    post_e_feat_matrix = post_e_feat_matrix.astype("float16")
    A1 = A1.astype("int16")
    A2 = A2.astype("int16")

    # Build the affinity matrix
    K = pygm.utils.build_aff_mat(
        pre_feat_matrix, pre_e_feat_matrix, A1, post_feat_matrix, post_e_feat_matrix, A2
    )

    return K


def spectral_matcher(aff_mat):
    # Note that S is a normalized with a squared sum of 1
    S = pygm.sm(aff_mat)
    # Since we need a doubly-stochastic matrix we must use Sinkhorn
    S = calc_assignment_matrix(S, tau=10, iter=250)

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


def draw_matches(pre_img, post_img, pre_kpts, post_kpts, matches, mask=None):
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
        matches_list,
        outImg=None,
        matchesMask=mask,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )

    fig, axs = plt.subplots(figsize=(10, 6))
    axs.imshow(matches_img)
    axs.set_title("Sinkhorn Matches")
    axs.set_xticks([])
    axs.set_yticks([])
    plt.show()


def custom_draw_matches(img1, kp1, img2, kp2, matches, inliers=None):
    """
    Draw matches between two images.

    Parameters:
    - img1: First image.
    - kp1: Keypoints in the first image.
    - img2: Second image.
    - kp2: Keypoints in the second image.
    - matches: List of matches.
    - inliers: List of booleans indicating whether a match is an inlier.
    """

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

    # Create a new output image that concatenates the two images side by side
    height1, width1 = img1.shape[:2]
    height2, width2 = img2.shape[:2]
    # Initially the matches were drawn on the image so it needed to be RGB
    # However now they aren't so this should be changed to single channel
    # But was left behind since it wasn't a priority
    output_img = np.zeros((max(height1, height2), width1 + width2, 3), dtype="uint8")
    # only_matches_img = np.zeros((max(height1, height2), width1 + width2), dtype="uint8")

    # This represents only the stithced pre and post images
    output_img[:height1, :width1] = img1
    output_img[:height2, width1:] = img2

    lines = []
    keypoints1 = []
    keypoints2 = []

    # Draw the keypoints and matches
    for i, match in enumerate(matches):
        # Get the matching keypoints for each of the images
        kp1_idx = match.queryIdx
        kp2_idx = match.trainIdx

        # Get the coordinates of the keypoints
        x1, y1 = kp1[kp1_idx].pt
        x2, y2 = kp2[kp2_idx].pt

        # Offset the coordinates in the second image
        x2 += width1

        # Choose color based on the index
        color = colors[i % len(colors)]

        # Draw lines connecting the keypoints
        if inliers is None or inliers[i]:
            # Instead of drawing the lines and circles return them to plot after
            lines.append((x1, y1, x2, y2, color))
            keypoints1.append((x1, y1, color))
            keypoints2.append((x2, y2, color))

    return output_img, lines, keypoints1, keypoints2  # only_matches_img


def draw_matches_interactive(
    pre_g,
    post_g,
    pre_evt,
    post_evt,
    pre_kpts,
    post_kpts,
    matches,
    mask=None,
    segm=False,
) -> None:
    # logger.info("Visualizing the pre and post filtered and final simplified graphs")
    visual_style = {}
    visual_style["vertex_size"] = 5
    # visual_style["vertex_color"] = "green"

    # Plot the pre and post EVT graphs in the same axes
    fig, ax = plt.subplots(figsize=(12, 6))
    # plot the first graph on the axis
    # The multiplication by 512 is done because the x,y values were normalied previously
    layout1 = [(v["x"] * 512, v["y"] * 512) for v in pre_g.vs]
    ig.plot(pre_g, layout=layout1, target=ax, **visual_style)
    # Adjust the positions of the nodes in the second graph by adding an x offset
    layout2 = [(v["x"] * 512 + 512, v["y"] * 512) for v in post_g.vs]
    # plot the second graph with the modified layout
    ig.plot(post_g, layout=layout2, target=ax, **visual_style)
    ax.set_title("Matches")
    ax.invert_yaxis()

    fig.subplots_adjust(wspace=0, bottom=0.25)
    if segm:
        # Prepare the segmentation images
        # Multiply by 255 because it is a binary image
        pre_img = cv2.cvtColor(pre_evt, cv2.COLOR_GRAY2RGB) * 255
        post_img = cv2.cvtColor(post_evt, cv2.COLOR_GRAY2RGB) * 255
    else:
        # Prepare the original images
        pre_img = resize(pre_evt, (pre_evt.shape[0] // 2, pre_evt.shape[1] // 2))
        post_img = resize(post_evt, (post_evt.shape[0] // 2, post_evt.shape[1] // 2))
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

    # Test Ransac -> Weird results so won't use it.
    # Use the function that integrates the Ransac algorithm to get the mask
    # temp1 = np.float32([pre_kpts[m.queryIdx].pt for m in matches_list])
    # temp2 = np.float32([post_kpts[m.trainIdx].pt for m in matches_list])
    # _, mask = cv2.findHomography(temp1, temp2, cv2.RANSAC, 5.0)

    stitched_img, lines, keypoints1, keypoints2 = custom_draw_matches(
        pre_img,
        pre_kpts,
        post_img,
        post_kpts,
        matches_list,
        inliers=mask,
    )

    # Plot the stitched images
    m_img = ax.imshow(stitched_img, alpha=0.5)

    # Plot the matched kpts and lines
    for line in lines:
        x1, y1, x2, y2, color = line
        ax.plot([x1, x2], [y1, y2], color=color, linewidth=1)

    for kp in keypoints1 + keypoints2:
        x, y, color = kp
        ax.scatter(x, y, color=color, s=20, edgecolor="k")

    ax.set_xticks([])
    ax.set_yticks([])

    # Create the Slider
    slider_ax = fig.add_axes([0.25, 0.1, 0.60, 0.03])
    slider = Slider(slider_ax, "Transparency", 0, 1, valinit=0.5, valstep=0.1)

    def update(val):
        m_img.set(alpha=slider.val)
        fig.canvas.draw_idle()

    slider.on_changed(update)

    def arrow_slider_control(event):
        """
        This function takes an event from an mpl_connection
        and listens for key release events specifically from
        the keyboard arrow keys (left/right) and uses this
        input to change the threshold slider.
        """
        if event.key == "left":
            cur_th = slider.val
            if cur_th - 0.1 >= 0:
                slider.set_val(cur_th - 0.1)
        if event.key == "right":
            cur_th = slider.val
            if cur_th + 0.1 <= 1:
                slider.set_val(cur_th + 0.1)

    fig.canvas.mpl_connect("key_release_event", arrow_slider_control)

    plt.show()


def get_kpts_lines(kp1, kp2, matches, im_width, inliers=None):
    """
    Get the keypoint and line locations to plot later on.

    Parameters:
    - kp1: Keypoints in the first image.
    - kp2: Keypoints in the second image.
    - matches: List of matches.
    - im_width: Width of the second image (Post-EVT)
    - inliers: List of booleans indicating whether a match is an inlier.

    Return:
        Dict: Dictionary with the lines and keypoints
    """
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

    lines = []
    keypoints1 = []
    keypoints2 = []

    # Prepare keypoints for cv2
    kp1 = [cv2.KeyPoint(pt[1], pt[0], 1) for pt in kp1]
    kp2 = [cv2.KeyPoint(pt[1], pt[0], 1) for pt in kp2]

    # Prepare matches for cv2
    matches_list = []
    # Unpacking the range into a list
    pre_graph_node_idx = [*range(len(matches))]
    # Distance is set to 1 for all as a placeholder
    distance = 1  # np.zeros(len(matches))
    for i, j in zip(pre_graph_node_idx, matches):
        # Create cv2 DMatch object
        matches_list.append(cv2.DMatch(i, j, distance))

    # Keep track of the keypoints and matches and their color for drawing
    for i, match in enumerate(matches_list):
        # Get the matching keypoints for each of the images
        kp1_idx = match.queryIdx
        kp2_idx = match.trainIdx

        # Get the coordinates of the keypoints
        x1, y1 = kp1[kp1_idx].pt
        x2, y2 = kp2[kp2_idx].pt

        # Offset the coordinates in the second image
        x2 += im_width

        # Choose color based on the index
        color = colors[i % len(colors)]

        # Draw lines connecting the keypoints
        if inliers is None or inliers[i]:
            # Instead of drawing the lines and circles return them to plot after
            lines.append((x1, y1, x2, y2, color))
            keypoints1.append((x1, y1, color))
            keypoints2.append((x2, y2, color))

    return {"lines": lines, "kpts1": keypoints1, "kpts2": keypoints2}


def multiscale_draw_matches_interactive(
    pre_g,
    post_g,
    pre_evt,
    post_evt,
    pre_kpts,
    post_kpts,
    lines,
    segm=False,
) -> None:
    visual_style = {}
    visual_style["vertex_size"] = 5
    # visual_style["vertex_color"] = "green"

    # Plot the pre and post EVT graphs in the same axes
    fig, ax = plt.subplots(figsize=(12, 6))
    # plot the first graph on the axis
    # The multiplication by 512 is done because the x,y values were normalied previously
    # The x,y values are normalized on subgraphs now so we don't need to multiply the layout
    # Which was required in the one-scale approach.
    layout1 = [(v["x"], v["y"]) for v in pre_g.vs]
    ig.plot(pre_g, layout=layout1, target=ax, **visual_style)
    # Adjust the positions of the nodes in the second graph by adding an x offset
    layout2 = [(v["x"] + 512, v["y"]) for v in post_g.vs]
    # plot the second graph with the modified layout
    ig.plot(post_g, layout=layout2, target=ax, **visual_style)
    ax.set_title("Matches")
    ax.invert_yaxis()

    fig.subplots_adjust(wspace=0, bottom=0.25)
    if segm:
        # Prepare the segmentation images
        # Multiply by 255 because it is a binary image
        pre_img = cv2.cvtColor(pre_evt, cv2.COLOR_GRAY2RGB) * 255
        post_img = cv2.cvtColor(post_evt, cv2.COLOR_GRAY2RGB) * 255
    else:
        # Prepare the original images
        pre_img = resize(pre_evt, (pre_evt.shape[0] // 2, pre_evt.shape[1] // 2))
        post_img = resize(post_evt, (post_evt.shape[0] // 2, post_evt.shape[1] // 2))
        pre_img = ski.util.img_as_ubyte(pre_img)
        post_img = ski.util.img_as_ubyte(post_img)
        pre_img = cv2.cvtColor(pre_img, cv2.COLOR_GRAY2RGB)
        post_img = cv2.cvtColor(post_img, cv2.COLOR_GRAY2RGB)

    # Stitch the images
    # Create a new output image that concatenates the two images side by side
    height1, width1 = pre_img.shape[:2]
    height2, width2 = post_img.shape[:2]
    # Initially the matches were drawn on the image so it needed to be RGB
    # However now they aren't so this should be changed to single channel
    # But was left behind since it wasn't a priority
    stitched_img = np.zeros((max(height1, height2), width1 + width2, 3), dtype="uint8")
    # only_matches_img = np.zeros((max(height1, height2), width1 + width2), dtype="uint8")

    # This represents only the stithced pre and post images
    stitched_img[:height1, :width1] = pre_img
    stitched_img[:height2, width1:] = post_img
    # Plot the stitched images
    m_img = ax.imshow(stitched_img, alpha=0.5)

    # Plot the matched kpts and lines
    for line in lines:
        x1, y1, x2, y2, color = line
        ax.plot([x1, x2], [y1, y2], color=color, linewidth=1)

    for kp in pre_kpts + post_kpts:
        x, y, color = kp
        ax.scatter(x, y, color=color, s=20, edgecolor="k")

    ax.set_xticks([])
    ax.set_yticks([])

    # Create the Slider
    slider_ax = fig.add_axes([0.25, 0.1, 0.60, 0.03])
    slider = Slider(slider_ax, "Transparency", 0, 1, valinit=0.5, valstep=0.1)

    def update(val):
        m_img.set(alpha=slider.val)
        fig.canvas.draw_idle()

    slider.on_changed(update)

    def arrow_slider_control(event):
        """
        This function takes an event from an mpl_connection
        and listens for key release events specifically from
        the keyboard arrow keys (left/right) and uses this
        input to change the threshold slider.
        """
        if event.key == "left":
            cur_th = slider.val
            if cur_th - 0.1 >= 0:
                slider.set_val(cur_th - 0.1)
        if event.key == "right":
            cur_th = slider.val
            if cur_th + 0.1 <= 1:
                slider.set_val(cur_th + 0.1)

    fig.canvas.mpl_connect("key_release_event", arrow_slider_control)

    plt.show()


def animate(idx, ax, interval, pre_img, post_img, pre_kpts, post_kpts, matches):
    ax.clear()

    ax.set_title("Sinkhorn Matches")
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

    # # Test masking based on Euclidean distance
    # feat_dist = distance.cdist(t1, t2, "euclidean")

    interval = 30
    frames_list = [*range(0 + interval, len(matches_list), interval)]
    # Append the last node index to the list
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


def create_sift_feat_matrix(pre_img, post_img, pre_graph, post_graph):
    # The changes in the graphs happen in place here
    # This function could be useful but when trying did not produce significantly better results
    # Which was determined with visual inspection.
    # When using L2 norm for matching the threshold should be set appropriately.

    # Compute the features descriptors using SIFT
    ft_extractor = cv2.SIFT_create()
    # Prepare keypoints for cv2
    pre_kpts = [(float(pt[0]), float(pt[1])) for pt in pre_graph.vs["coords"]]
    post_kpts = [(float(pt[0]), float(pt[1])) for pt in post_graph.vs["coords"]]
    # Generally size is set to 1 but I tried several.
    pre_kpts_sift = [cv2.KeyPoint(pt[1], pt[0], 10) for pt in pre_kpts]
    post_kpts_sift = [cv2.KeyPoint(pt[1], pt[0], 10) for pt in post_kpts]

    _, pre_sift_dscr = ft_extractor.compute(pre_img, pre_kpts_sift)
    _, post_sift_dscr = ft_extractor.compute(post_img, post_kpts_sift)

    # print(pre_sift_dscr.shape)

    # Concatenate the Sift features
    for attr in range(pre_sift_dscr.shape[1]):
        attr_name = "feat_" + str(attr)
        pre_graph.vs[attr_name] = pre_sift_dscr[:, attr]

    for attr in range(post_sift_dscr.shape[1]):
        attr_name = "feat_" + str(attr)
        post_graph.vs[attr_name] = post_sift_dscr[:, attr]

    # Delete the coords attribute before creating the feat matrix
    del pre_graph.vs["coords"]
    del post_graph.vs["coords"]

    # Normalize x,y features - Similar to sift features
    pre_graph.vs["x"] = 100 * [x / pre_img.shape[1] for x in pre_graph.vs["x"]]
    pre_graph.vs["y"] = 100 * [y / pre_img.shape[0] for y in pre_graph.vs["y"]]
    post_graph.vs["x"] = 100 * [x / post_img.shape[1] for x in post_graph.vs["x"]]
    post_graph.vs["y"] = 100 * [y / post_img.shape[0] for y in post_graph.vs["y"]]

    # Create a feature matrices from all the node attributes
    pre_feat_matrix = create_feat_matrix(pre_graph)
    post_feat_matrix = create_feat_matrix(post_graph)

    pre_feat_avg, pre_feat_max, pre_feat_min, pre_feat_std = calc_feat_matrix_info(
        pre_feat_matrix, vis=True
    )
    post_feat_avg, post_feat_max, post_feat_min, post_feat_std = calc_feat_matrix_info(
        post_feat_matrix, vis=True
    )
    print(
        f"Avg: {pre_feat_avg}, Max: {pre_feat_max}, Min: {pre_feat_min}, Std: {pre_feat_std}"
    )
    print(
        f"Avg: {post_feat_avg}, Max: {post_feat_max}, Min: {post_feat_min}, Std: {post_feat_std}"
    )

    # Try Z score normalization on the features
    # Use the same mean and std to keep the relation between the pre and post values
    z_mean = max(pre_feat_avg, post_feat_avg)
    z_std = max(pre_feat_std, post_feat_std)
    pre_feat_matrix = (pre_feat_matrix - z_mean) / z_std
    post_feat_matrix = (post_feat_matrix - z_mean) / z_std

    pre_feat_avg, pre_feat_max, pre_feat_min, pre_feat_std = calc_feat_matrix_info(
        pre_feat_matrix, vis=True
    )
    post_feat_avg, post_feat_max, post_feat_min, post_feat_std = calc_feat_matrix_info(
        post_feat_matrix, vis=True
    )
    print(
        f"Avg: {pre_feat_avg}, Max: {pre_feat_max}, Min: {pre_feat_min}, Std: {pre_feat_std}"
    )
    print(
        f"Avg: {post_feat_avg}, Max: {post_feat_max}, Min: {post_feat_min}, Std: {post_feat_std}"
    )

    return pre_feat_matrix, post_feat_matrix
