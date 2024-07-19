import logging
from pathlib import Path
import sys, os
import copy

import igraph as ig
import numpy as np
import matplotlib.pyplot as plt
from Skeletonization import load_images, get_skeletons, find_centerlines
import graph_feature_extraction as GFeatExt

# Overlay original
import cv2
from skimage.transform import resize
import skimage as ski
from matplotlib.widgets import Slider

logger = logging.getLogger(__name__)


def plot_pre_post(
    pre_g, post_g, pre_evt=None, post_evt=None, overlay_orig=False, overlay_seg=False
) -> None:
    logger.info("Visualizing the pre and post filtered and final simplified graphs")
    # Visualize the final simplified graph
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    visual_style = {}
    visual_style["vertex_size"] = 5
    # visual_style["vertex_color"] = "green"
    ig.plot(pre_g, target=ax[0], **visual_style)
    ig.plot(post_g, target=ax[1], **visual_style)
    ax[0].set_title("Pre-EVT")
    ax[1].set_title("Post-EVT")
    ax[0].invert_yaxis()
    ax[1].invert_yaxis()

    if overlay_orig or overlay_seg:
        fig.subplots_adjust(bottom=0.25)
        if overlay_orig:
            # Prepare the original images
            pre_img = resize(pre_evt, (pre_evt.shape[0] // 2, pre_evt.shape[1] // 2))
            post_img = resize(
                post_evt, (post_evt.shape[0] // 2, post_evt.shape[1] // 2)
            )
            pre_img = ski.util.img_as_ubyte(pre_img)
            post_img = ski.util.img_as_ubyte(post_img)
            pre_img = cv2.cvtColor(pre_img, cv2.COLOR_GRAY2RGB)
            post_img = cv2.cvtColor(post_img, cv2.COLOR_GRAY2RGB)
            im0 = ax[0].imshow(pre_img, cmap="gray", alpha=0.5)
            im1 = ax[1].imshow(post_img, cmap="gray", alpha=0.5)
        if overlay_seg:
            # Prepare the segmentation images
            im0 = ax[0].imshow(pre_evt, cmap="gray", alpha=0.5)
            im1 = ax[1].imshow(post_evt, cmap="gray", alpha=0.5)
        # Create the Slider
        slider_ax = fig.add_axes([0.25, 0.1, 0.60, 0.03])
        slider = Slider(slider_ax, "Transparency", 0, 1, valinit=0.5, valstep=0.1)

        def update(val):
            im0.set(alpha=slider.val)
            im1.set(alpha=slider.val)
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


def multiscale_graph(graph, scale=3):
    """Extract a specific scale from the graph (e.g. edges with radius g.t 3)

    Args:
        graph (igraph Graph): Graph that we want to extract scale from
        scale (int | list): Integer of minimum edge average radius | List of lower and upper bound average radii

    Returns:
        igraph Graph: The new scaled graph with vessels of a given radius
    """

    if isinstance(scale, int):
        edge_ids = graph.es.select(radius_avg_gt=scale)
    elif isinstance(scale, (list, tuple)):
        if len(scale) != 2:
            logger.error(
                "List does not contain two values. The list must contain the lower and upper bounds.",
                stack_info=True,
            )
            raise ValueError(
                "List does not contain two values. The list must contain the lower and upper bounds."
            )
        upper = max(scale)
        lower = min(scale)
        edge_ids = graph.es.select(radius_avg_gt=lower, radius_avg_lt=upper)

    if not edge_ids:
        logger.error("The new graph does not contain any nodes.", stack_info=True)
        raise Exception("The new graph does not contain any nodes.")
    new_scale_graph = graph.subgraph_edges(edge_ids)

    return new_scale_graph


def concat_extracted_features(graph, feat_map, inplace=True):
    # Create an array to store the attributes for every node
    # The array has a shape of NxD where D is the feature dimention (64) and N is the number of nodes
    feat_attributes = np.zeros((len(graph.vs), feat_map.shape[0]))
    # print(feat_attributes.shape)

    # Loop through the vertices in the graph and populate the feature attributes
    for idx, vertex in enumerate(graph.vs):
        # t = vertex["coords"]
        coords = np.round(vertex["coords"]).astype(int)
        v_idx = vertex.index

        feat_attributes[idx, :] = feat_map[:, coords[0], coords[1]]

    # print(t)
    if inplace:
        # Add the new attributes to every node
        for attr in range(feat_attributes.shape[1]):
            attr_name = "feat_" + str(attr)
            graph.vs[attr_name] = feat_attributes[:, attr]
    else:
        new_graph = copy.deepcopy(graph)
        for attr in range(feat_attributes.shape[1]):
            attr_name = "feat_" + str(attr)
            graph.vs[attr_name] = feat_attributes[:, attr]
        return new_graph


def concat_extracted_features_v2(graph, feat_map, inplace=True):
    # The difference of version 2 is that instead of getting the features from a specific pixel
    # We will calculate each feature with respect to the neighborhood of that pixel (mean,maxpool,sum)

    # Create an array to store the attributes for every node
    # The array has a shape of NxD where D is the feature dimention (64) and N is the number of nodes
    feat_attributes = np.zeros((len(graph.vs), feat_map.shape[0]))
    # print(feat_attributes.shape)
    neighb_size = 48  # 8, 24 or 48
    calc_features = np.zeros(feat_map.shape[0])

    # Loop through the vertices in the graph and populate the feature attributes
    for idx, vertex in enumerate(graph.vs):
        coords = np.round(vertex["coords"]).astype(int)
        v_idx = vertex.index

        # Calculate the features
        for map in range(feat_map.shape[0]):
            # Get the values of the neighborhood around the coords and average them
            values = []
            for nb in candidate_neighbors(
                (coords[0], coords[1]), hood_size=neighb_size
            ):
                # If the neighboor is outside the bounds of the image skip
                if (nb[0] < 0 or nb[0] >= feat_map.shape[1]) or (
                    nb[1] < 0 or nb[1] >= feat_map.shape[2]
                ):
                    continue

                values.append(feat_map[map, nb[0], nb[1]])

            # Append the coordinates since they are not included in the neighbors
            values.append(feat_map[map, coords[0], coords[1]])

            # Average the neighbor feature map values
            calc_features[map] = np.mean(values)

        # Add the features to the specific node
        feat_attributes[idx, :] = calc_features

    if inplace:
        # Add the new attributes to every node
        for attr in range(feat_attributes.shape[1]):
            attr_name = "feat_" + str(attr)
            graph.vs[attr_name] = feat_attributes[:, attr]
    else:
        new_graph = copy.deepcopy(graph)
        for attr in range(feat_attributes.shape[1]):
            attr_name = "feat_" + str(attr)
            graph.vs[attr_name] = feat_attributes[:, attr]
        return new_graph


def calc_pixel_connectivity(skeleton, skeleton_pts):
    sklt_connect = np.zeros((skeleton.shape), dtype=np.int8)
    # i = 0
    for pt in skeleton_pts:
        cnt = 0
        for nb in candidate_neighbors(pt):
            # If the neighboor is outside the bounds of the image skip
            if (nb[0] < 0 or nb[0] >= skeleton.shape[0]) or (
                nb[1] < 0 or nb[1] >= skeleton.shape[0]
            ):
                continue
            # Check if the neighboor is also part of the skeleton
            if skeleton[nb[0], nb[1]] != 0:
                cnt += 1
        sklt_connect[pt[0], pt[1]] = cnt

    return sklt_connect


def candidate_neighbors(pt, hood_size=8):
    # Return 8-pixel neighborhood
    if hood_size == 8:
        return (
            (pt[0] - 1, pt[1] - 1),
            (pt[0] - 1, pt[1]),
            (pt[0] - 1, pt[1] + 1),
            (pt[0], pt[1] - 1),
            (pt[0], pt[1] + 1),
            (pt[0] + 1, pt[1] - 1),
            (pt[0] + 1, pt[1]),
            (pt[0] + 1, pt[1] + 1),
        )
    elif hood_size == 24:
        pt_list = []
        for i in range(-2, 3):
            for j in range(-2, 3):
                pt_list.append((pt[0] + i, pt[1] + j))
        pt_list.remove((pt[0], pt[1]))
        return pt_list
    elif hood_size == 48:
        pt_list = []
        for i in range(-3, 4):
            for j in range(-3, 4):
                pt_list.append((pt[0] + i, pt[1] + j))
        pt_list.remove((pt[0], pt[1]))
        return pt_list
    else:
        logger.info(
            "The function currently only deals with a neighborhood size of 8 or 24. Sorry!"
        )


def construct_vID_LUT(sklt_points, img_shape):
    # Construct a lookup table for the vertex IDs
    ids = np.arange(0, sklt_points.shape[0])
    v_id_LUT = np.zeros(img_shape, dtype=np.int_)
    v_id_LUT[sklt_points[:, 0], sklt_points[:, 1]] = ids
    return v_id_LUT


def edge_detection(skeleton_pts, vertex_LUT):
    edges = []

    for pt in skeleton_pts:
        for nb in candidate_neighbors(pt):
            # If the neighboor is outside the bounds of the image skip
            if (nb[0] < 0 or nb[0] >= vertex_LUT.shape[0]) or (
                nb[1] < 0 or nb[1] >= vertex_LUT.shape[0]
            ):
                continue

            if vertex_LUT[nb[0], nb[1]] > 0:
                src_id = vertex_LUT[pt[0], pt[1]]
                trgt_id = vertex_LUT[nb[0], nb[1]]
                # The idea is to create an edge from pt to nb
                edges.append((src_id, trgt_id))
    return edges


def get_cliques(graph, components=False):
    # Assign all vertices their original ID
    graph.vs["id"] = np.arange(graph.vcount())

    # Isolate potential bifurcation points i.e. degree greater than 2
    subg = graph.subgraph(graph.vs.select(_degree_gt=2))

    if components:
        # Get all the connected components
        cliques = [clique for clique in subg.components() if len(clique) > 3]
    else:
        # Eliminate 1 degree edges connected to the cliques
        while True:
            count = len(subg.vs.select(_degree_lt=2))
            if count == 0:
                break
            subg = subg.subgraph(subg.vs.select(_degree_gt=1))
        # Get the maximal cliques that have 3-4 nodes
        cliques = [clique for clique in subg.maximal_cliques(3, 5)]

    return subg, cliques


def find_new_vertex_neighbors(graph, subg_vs):
    graph_vs = graph.vs[subg_vs["id"]]

    neighbors = []
    for graph_v, subg_v in zip(graph_vs, subg_vs):
        # Find external neighbors if the degree differs
        if graph_v.degree() != subg_v.degree():
            clique_nb_ids = [nb["id"] for nb in subg_v.neighbors()]
            # We use the vertex IDs since up to here they are the same
            # as the vertex index
            neighbors += [
                nb["id"] for nb in graph_v.neighbors() if nb["id"] not in clique_nb_ids
            ]
    return neighbors


def create_new_vertex(graph, subg_vs):
    new_radius = np.mean(subg_vs["radius"])
    new_coords = np.mean(subg_vs["coords"], axis=0)

    # coordinates are y,x
    new_vertex_info = {
        "radius": new_radius,
        "coords": new_coords,
        "y": new_coords[0],
        "x": new_coords[1],
    }
    neighbors = find_new_vertex_neighbors(graph, subg_vs)
    return [new_vertex_info, neighbors]


def class2_filter(graph, subg, clique):
    # Reduce the connected components to a single node/bifurcation point
    subg_vs = subg.vs[clique]

    # Add new vertex constructes from the clique
    vertex_info, neighbors = create_new_vertex(graph, subg_vs)

    return [vertex_info, neighbors]


def class2and3_decision(subg, cliques):
    vertices_to_remove = []
    class_two = []
    class_three = []

    for clique in cliques:
        vertices_to_remove.extend(subg.vs[clique]["id"])
        if len(clique) <= 50:
            class_two.append(class2_filter(sk_graph, subg, clique))
        else:
            None
            # After observation I believe there is no need for this filter in our dataset
            # class_three.append(class3_filter(g, gbs, clique))

    return [vertices_to_remove, class_two, class_three]


def class2and3_filters(subg, cliques):
    logger.info("Simplifying bifurcation clusters of connected components (Class 2)")
    new_edges = []
    vertices_to_remove = []
    class_two = []
    class_three = []
    vertices_to_remove, class_two, class_three = class2and3_decision(subg, cliques)

    # Add new vertices from class two
    for vertex_group in class_two:
        vertex_info = vertex_group[0]
        neighbors = vertex_group[1]
        new_v = sk_graph.add_vertex(
            radius=vertex_info["radius"],
            coords=vertex_info["coords"],
            x=vertex_info["x"],
            y=vertex_info["y"],
        )

        new_edges.extend(sorted(tuple([new_v.index, nb]) for nb in neighbors))

    return new_edges, vertices_to_remove


def class1_filter(subg, cliques):
    logger.info("Simplifying bifurcation clusters of node cliques (Class 1)")
    edges_to_remove = []

    for clique in cliques:
        # Get the original vertices
        orig_graph_vs = sk_graph.vs[subg.vs[clique]["id"]]

        # Check to see if the cliques fit in our class
        if any(degree >= 5 for degree in orig_graph_vs.degree()):
            continue

        # Weight the vertices based on radius and neighbor radius
        weights = orig_graph_vs["radius"]
        for idx, v in enumerate(orig_graph_vs):
            for nb in v.neighbors():
                weights[idx] += nb["radius"]

        # Sort the vertices based on their weights, remove edge between smallest
        sorted_ids = [id for _, id in sorted(zip(weights, orig_graph_vs))]

        edge = (sorted_ids[0]["id"], sorted_ids[1]["id"])
        edges_to_remove.append(edge)

    return edges_to_remove


def segment_filter(graph, f_length=2, smoothing=True):
    n_of_filtered = 0
    vertices_to_remove = []

    # Get groups of all connected components in the graph
    segments = graph.components()

    # Keep segments that are shorter than the filter length
    segments = [segm for segm in segments if len(segm) <= f_length]

    # Iterate over the segments
    for segm in segments:
        degrees = graph.degree(segm)
        # Examine isolated segments that have only 2 endpoints
        if degrees.count(1) == 2:
            iso_segment_length = GFeatExt.get_seg_length(graph, segm, smoothing)
        # If the isolated segment length is smaller than the filter length remove nodes
        if iso_segment_length < f_length:
            vertices_to_remove.extend(segm)
            n_of_filtered += len(segm)

    graph.delete_vertices(vertices_to_remove)

    return n_of_filtered


def isolated_segm_filter(graph, filter_length, verbose=False):
    logger.info("Eliminating isolated segments")
    if filter_length > 0:
        n_of_filtered = segment_filter(graph, filter_length, smoothing=True)
    if verbose:
        print(f"Number of filtered nodes: {n_of_filtered}")
    return n_of_filtered


def filter_graph(graph):
    total_removed_vertices = 0
    total_removed_edges = 0

    # 1. Apply the class 1 filter

    # Isolate cliques comprising of potential bifurcation points
    subg, cliques = get_cliques(graph)

    edges_to_remove = class1_filter(subg, cliques)
    graph.delete_edges(edges_to_remove)
    total_removed_edges += len(edges_to_remove)
    print(f"Number of removed edges: {total_removed_edges}")

    # 2. Apply the class 2 filter

    subg, cliques = get_cliques(graph, components=True)

    edges_to_add, vertices_to_remove = class2and3_filters(subg, cliques)

    # Remove duplicate edges
    edges_to_add = [edges_to_add for edges_to_add in set(edges_to_add)]
    graph.add_edges(edges_to_add)  # Add the new edges
    graph.delete_vertices(vertices_to_remove)  # Delete the spurious points
    total_removed_vertices += len(vertices_to_remove)
    print(f"Number of removed vertices: {total_removed_vertices}")

    # Delete the id attribute since its not useful anymore
    del graph.vs["id"]

    # Eliminate isolated vertices
    graph.delete_vertices(graph.vs.select(_degree=0))

    # 3. Apply the isolated segment filter

    # Eliminate isolates segments
    n_removed_vertices = isolated_segm_filter(graph, filter_length=11, verbose=False)
    total_removed_vertices += n_removed_vertices
    print(f"Number of removed vertices: {n_removed_vertices}")
    print(f"Total number of removed vertices: {total_removed_vertices}")

    # plt.show()


def analyze_simplify_graph(graph, visualize=False):
    features, edges = GFeatExt.segment_feature_extraction(graph)

    if visualize:
        # Visualize the original graph
        fig, ax = plt.subplots()
        visual_style = {}
        visual_style["vertex_size"] = 5
        # visual_style["vertex_color"] = "green"
        ig.plot(graph, target=ax, **visual_style)
        ax.invert_yaxis()

    # Simplify the graph using the calculated edges and assign their features
    # Graph is a global variable is its simplified in place
    GFeatExt.save_feature_results(graph, features, edges, simplify_graph=True)

    # Further simplification with sclass1 and sclass2
    GFeatExt.simplify_more_sclass1(graph, vis=False, verbose=True)
    GFeatExt.simplify_more_sclass2(graph, vis=False, verbose=True)

    if visualize:
        logger.info("Visualizing the filtered and final simplified graphs")
        # Visualize the final simplified graph
        fig, ax = plt.subplots()
        visual_style = {}
        visual_style["vertex_size"] = 5
        # visual_style["vertex_color"] = "green"
        ig.plot(graph, target=ax, **visual_style)
        ax.invert_yaxis()
        plt.show()


def create_graph(
    skeleton_img, skeleton_pts, dst_transform, g_name=None, vis=False, verbose=True
):

    if not g_name:
        g_name = "Temp"

    logger.info(f"Creating graph: {g_name}")

    # Create the graph
    global sk_graph
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
    vertex_LUT = construct_vID_LUT(skeleton_pts, skeleton_img.shape)
    # Find edges
    edges = edge_detection(skeleton_pts, vertex_LUT)
    # Add detected edges
    sk_graph.add_edges(edges)
    # Remove loops and multiedges
    sk_graph.simplify()

    print("Graph summary before filtering")
    ig.summary(sk_graph)
    print(
        "Summary structure: 4-char long code, number of vertices, number of edges -- graph name"
    )

    # Simplify the graph and filter unwanted edges and nodes
    filter_graph(sk_graph)

    # print("Graph summary after filtering")
    # ig.summary(sk_graph)
    # print(
    #     "Summary structure: 4-char long code, number of vertices, number of edges -- graph name"
    # )

    # Analyze the graph segments and simplify it further maintaining only bifurcation points
    analyze_simplify_graph(sk_graph, visualize=vis)

    print("Graph summary after analysis and simplification - Final Graph")
    ig.summary(sk_graph)
    print(
        "Summary structure: 4-char long code, number of vertices, number of edges -- graph name"
    )

    if not sk_graph.vs():
        logger.error("The created graph does not contain any nodes.", stack_info=True)
        raise Exception("The created graph does not contain any nodes.")

    return sk_graph


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

    # Just testing how the graph creation works.
    # IMG_SEQ_DIR_PATH = (
    #     "C:/Users/mab03/Desktop/RuSegm/TemporalUNet/Outputs/Sequence/R0002"
    # )
    IMG_SEQ_DIR_PATH = "C:/Users/mab03/Desktop/ThesisCode/Segms/Sequence/R0002/0"
    img_ind = 0
    segm_images = load_images(IMG_SEQ_DIR_PATH)
    if not segm_images:
        return
    skeletons, distance_transform = get_skeletons(segm_images, method="lee")
    skeleton_points = find_centerlines(skeletons[img_ind])

    skeleton_points_connect = calc_pixel_connectivity(
        skeletons[img_ind], skeleton_points
    )

    # Create graph
    gr = create_graph(
        skeletons[img_ind],
        skeleton_points,
        distance_transform[img_ind],
        g_name="gr",
        vis=True,
        verbose=True,
    )
    # skeleton_points = find_centerlines(skeletons[img_ind + 1])
    # gr1 = create_graph(
    #     skeletons[img_ind + 1],
    #     skeleton_points,
    #     distance_transform[img_ind + 1],
    #     g_name="gr1",
    #     vis=False,
    #     verbose=True,
    # )
    print("G1")
    ig.summary(gr)
    # print("G2")
    # ig.summary(gr1)

    # Test to see what the linegraph looks like
    # Keep in mind that when converting the attributes are discarded
    # gr_line = gr.linegraph()
    # print("Line Graph Info")
    # ig.summary(gr_line)
    # print(
    #     "Summary structure: 4-char long code, number of vertices, number of edges -- graph name"
    # )
    # print(gr_line.es.attributes())
    # # Visualize the line graph
    # fig, ax = plt.subplots()
    # visual_style = {}
    # visual_style["vertex_size"] = 5
    # # visual_style["vertex_color"] = "green"
    # ig.plot(gr_line, target=ax, **visual_style)
    # ax.invert_yaxis()
    # plt.show()


if __name__ == "__main__":
    main()
