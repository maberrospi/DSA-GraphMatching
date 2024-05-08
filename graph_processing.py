import logging
from pathlib import Path
import sys, os

import igraph as ig
import numpy as np
import matplotlib.pyplot as plt
from Skeletonization import load_images, get_skeletons, find_centerlines
import graph_feature_extraction as GFeatExt

logger = logging.getLogger(__name__)


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
        cliques = [clique for clique in subg.maximal_cliques() if 2 < len(clique) < 5]

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


def create_graph(skeleton_img, skeleton_pts, dst_transform):
    # Create the graph
    global sk_graph
    sk_graph = ig.Graph()
    sk_graph["name"] = "Skeleton Graph"
    sk_graph.add_vertices(len(skeleton_pts))

    sk_graph.vs["coords"] = skeleton_pts
    sk_graph.vs["y"] = skeleton_pts[:, 0]
    sk_graph.vs["x"] = skeleton_pts[:, 1]
    # Can extend the radius to the mEDT introduce in VesselVio if needed
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

    print("Graph summary after filtering")
    ig.summary(sk_graph)
    print(
        "Summary structure: 4-char long code, number of vertices, number of edges -- graph name"
    )

    # Analyze the graph segments and simplify it further maintaining only bifurcation points
    analyze_simplify_graph(sk_graph, visualize=False)

    print("Graph summary after analysis and simplification - Final Graph")
    ig.summary(sk_graph)
    print(
        "Summary structure: 4-char long code, number of vertices, number of edges -- graph name"
    )

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
    IMG_SEQ_DIR_PATH = (
        "C:/Users/mab03/Desktop/RuSegm/TemporalUNet/Outputs/Sequence/R0002"
    )
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
    create_graph(skeletons[img_ind], skeleton_points, distance_transform[img_ind])


if __name__ == "__main__":
    main()
