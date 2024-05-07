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
            # print(sklt_connect[nb[0], nb[1]])
            if skeleton[nb[0], nb[1]] != 0:
                cnt += 1
        sklt_connect[pt[0], pt[1]] = cnt
        # print(pt.shape)
        # print(sklt_connect[pt])
        # i += 1
        # if i == 3:
        #    break
    # plt.imshow(sklt_connect)
    # plt.colorbar()
    # plt.show()
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
    # plt.imshow(v_id_LUT)
    # plt.colorbar()
    # plt.show()
    return v_id_LUT


def edge_detection(skeleton_pts, vertex_LUT):
    edges = []

    for pt in skeleton_pts:
        # print(f"PT: {pt}")
        for nb in candidate_neighbors(pt):
            # If the neighboor is outside the bounds of the image skip
            if (nb[0] < 0 or nb[0] >= vertex_LUT.shape[0]) or (
                nb[1] < 0 or nb[1] >= vertex_LUT.shape[0]
            ):
                continue
            # if nb in skeleton_pts:
            if vertex_LUT[nb[0], nb[1]] > 0:
                # print(f"NB: {nb}")
                src_id = vertex_LUT[pt[0], pt[1]]
                trgt_id = vertex_LUT[nb[0], nb[1]]
                # print(f"srcid: {src_id}")
                # print(f"trgtid: {trgt_id}")
                # The idea is to create an edge from pt to nb
                edges.append((src_id, trgt_id))
    return edges


def get_cliques(graph, components=False):
    # Assign all vertices their original ID
    graph.vs["id"] = np.arange(graph.vcount())

    # Isolate potential bifurcation points i.e. degree greater than 2
    subg = graph.subgraph(graph.vs.select(_degree_gt=2))
    # print(subg.vs[0]["coords"])

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
        # print(len(cliques))
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
    # print(new_coords)
    # coordinates are y,x
    # new_vertex_info = (new_radius, new_coords, new_coords[0], new_coords[1])
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
    # print(vertex)

    return [vertex_info, neighbors]


def class2and3_decision(subg, cliques):
    vertices_to_remove = []
    class_two = []
    class_three = []
    # Visualize
    # fig, ax = plt.subplots()
    # visual_style = {}
    # visual_style["vertex_color"] = "red"

    # temp_len = len(cliques)
    # print(temp_len)
    # colors = [()]
    # idx=0
    # cnt = 0
    for clique in cliques:
        vertices_to_remove.extend(subg.vs[clique]["id"])
        if len(clique) <= 50:
            class_two.append(class2_filter(sk_graph, subg, clique))
        else:
            None
            # After observation I believe there is no need for this filter in our dataset
            # cnt += 1
            # class_three.append(class3_filter(g, gbs, clique))
        # ig.plot(subg.subgraph(subg.vs[clique]), target=ax, **visual_style)
        # idx+=1
    # ax.invert_yaxis()
    # print(cnt)

    return [vertices_to_remove, class_two, class_three]


def class2and3_filters(subg, cliques):
    logger.info("Simplifying bifurcation clusters of connected components (Class 2)")
    new_edges = []
    vertices_to_remove = []
    class_two = []
    class_three = []
    vertices_to_remove, class_two, class_three = class2and3_decision(subg, cliques)

    # fig, ax = plt.subplots()
    # visual_style = {}
    # visual_style["vertex_color"] = "red"

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
        # print(sk_graph.vs[new_v.index].attributes())
        new_edges.extend(sorted(tuple([new_v.index, nb]) for nb in neighbors))

        ### Visualize the new nodes

        # for clique in cliques:
        #     ig.plot(subg.subgraph(subg.vs[clique]), target=ax, **visual_style)

    # temp_g = ig.Graph()
    # temp_g["name"] = "Temp Graph"
    # temp_g.add_vertex(name=None, **new_v.attributes())
    # # print(temp_g.vs[0].attributes())
    # visual_style["vertex_color"] = "green"
    # ig.plot(temp_g, target=ax, **visual_style)
    # ax.invert_yaxis()

    return new_edges, vertices_to_remove


def class1_filter(subg, cliques):
    logger.info("Simplifying bifurcation clusters of node cliques (Class 1)")
    edges_to_remove = []

    for clique in cliques:
        # Get the original vertices
        orig_graph_vs = sk_graph.vs[subg.vs[clique]["id"]]
        # print(orig_graph_vs["id"])
        # print(subg.vs[clique]["id"])
        # print(subg.vs[clique].select(id=8108))
        # print(subg.es[0].tuple)
        # print(graph.get_eid(8108, 8116))
        # print(graph.es[8568].tuple)
        # print(orig_graph_vs.es)

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
        # test with smaller graphs
        # t1 = subg.vs[clique].select(id=sorted_ids[0]["id"])
        # t1 = t1[0].index
        # t2 = subg.vs[clique].select(id=sorted_ids[1]["id"])
        # t2 = t2[0].index
        # edge = (t1, t2)
        edge = (sorted_ids[0]["id"], sorted_ids[1]["id"])
        edges_to_remove.append(edge)
        # print(edges_to_remove)
    return edges_to_remove


def segment_filter(graph, f_length=2, smoothing=True):
    n_of_filtered = 0
    vertices_to_remove = []

    # Get groups of all connected components in the graph
    segments = graph.components()

    # Keep segments that are shorter than the filter length
    segments = [segm for segm in segments if len(segm) <= f_length]

    # Test plots
    # fig, ax = plt.subplots()
    # visual_style = {}
    # visual_style["vertex_color"] = "red"
    # visual_style["vertex_size"] = 4

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

            # for seg in segm:
            #     ig.plot(graph.subgraph(graph.vs[seg]), target=ax, **visual_style)

    # ig.summary(graph)
    graph.delete_vertices(vertices_to_remove)

    # visual_style["vertex_size"] = 1
    # visual_style["vertex_color"] = "green"
    # ig.plot(graph, target=ax, **visual_style)
    # ax.invert_yaxis()

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
    # print(
    #     f"Initial potential bifurcations nodes: {len(graph.subgraph(graph.vs.select(_degree_gt=2)).vs)}"
    # )
    # print(f"Before filter 1 potential bifurcation nodes: {len(subg.vs)}")
    edges_to_remove = class1_filter(subg, cliques)
    graph.delete_edges(edges_to_remove)
    total_removed_edges += len(edges_to_remove)
    print(f"Number of removed edges: {total_removed_edges}")

    # 2. Apply the class 2 filter

    subg, cliques = get_cliques(graph, components=True)
    # print(f"After class 1 filtering: {len(subg.vs)}")
    # print(len(subg.components()))
    # print(len([clique for clique in subg.components()]))
    # print(len([clique for clique in subg.components() if len(clique) > 3]))

    edges_to_add, vertices_to_remove = class2and3_filters(subg, cliques)

    # Print summary to see how many nodes were added
    # ig.summary(graph)

    # Remove duplicate edges
    edges_to_add = [edges_to_add for edges_to_add in set(edges_to_add)]
    graph.add_edges(edges_to_add)  # Add the new edges
    graph.delete_vertices(vertices_to_remove)  # Delete the spurious points
    total_removed_vertices += len(vertices_to_remove)
    print(f"Number of removed vertices: {total_removed_vertices}")

    # Delete the id attribute since its not useful anymore
    del graph.vs["id"]
    # print(graph.vs.attributes())

    # Eliminate isolated vertices
    graph.delete_vertices(graph.vs.select(_degree=0))

    # 3. Apply the isolated segment filter

    # Eliminate isolates segments
    n_removed_vertices = isolated_segm_filter(graph, filter_length=11, verbose=False)
    total_removed_vertices += n_removed_vertices
    print(f"Number of removed vertices: {n_removed_vertices}")
    print(f"Total number of removed vertices: {total_removed_vertices}")

    # Test with smaller graph
    # ig.summary(subg)
    # fig, ax = plt.subplots()
    # # print(min(subg.vs["radius"]))
    # visual_style = {}
    # visual_style["vertex_size"] = [rad for rad in subg.vs["radius"]]
    # ig.plot(subg, target=ax, **visual_style)
    # ax.invert_yaxis()
    # subg.delete_edges(edges_to_remove)
    # ig.summary(subg)
    # fig, ax = plt.subplots()
    # # print(min(subg.vs["radius"]))
    # visual_style = {}
    # visual_style["vertex_size"] = [rad for rad in subg.vs["radius"]]
    # ig.plot(subg, target=ax, **visual_style)
    # ax.invert_yaxis()

    # ig.summary(graph)
    # fig, ax = plt.subplots()
    # # print(min(subg.vs["radius"]))
    # ig.plot(graph, vertex_size=4, target=ax)
    # ax.invert_yaxis()
    plt.show()


def analyze_simplify_graph(graph):
    features, edges = GFeatExt.segment_feature_extraction(graph)
    # print(features[0])
    # print(edges[0])

    # Visualize the original graph
    fig, ax = plt.subplots()
    visual_style = {}
    visual_style["vertex_size"] = 5
    # visual_style["vertex_color"] = "green"
    ig.plot(graph, target=ax, **visual_style)
    ax.invert_yaxis()
    # plt.show()

    # Simplify the graph using the calculated edges and assign their features
    GFeatExt.save_feature_results(graph, features, edges, simplify_graph=True)

    # Visualize the final simplified graph
    fig, ax = plt.subplots()
    visual_style = {}
    visual_style["vertex_size"] = 5
    # visual_style["vertex_color"] = "green"
    ig.plot(graph, target=ax, **visual_style)
    ax.invert_yaxis()
    plt.show()


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
        "C:/Users/mab03/Desktop/RuSegm/TemporalUNet/Outputs/Sequence/R0003"
    )
    img_ind = 0
    segm_images = load_images(IMG_SEQ_DIR_PATH)
    if not segm_images:
        return
    skeletons, distance_transform = get_skeletons(segm_images, method="lee")
    skeleton_points = find_centerlines(skeletons[img_ind])
    # Test 50 points
    # skeleton_points = skeleton_points[0:50]
    skeleton_points_connect = calc_pixel_connectivity(
        skeletons[img_ind], skeleton_points
    )
    # print(skeleton_points[0:5, :])
    # print(skeleton_points.shape)
    # print(distance_transform[0].shape)

    # Create the graph
    global sk_graph
    sk_graph = ig.Graph()
    sk_graph["name"] = "Skeleton Graph"
    sk_graph.add_vertices(len(skeleton_points))
    # sk_graph.vs["v_coords"] = skeleton_points
    # print(
    #     f"Before: First five: {skeleton_points[:5, 0]} - Last five: {skeleton_points[-5:,0]}"
    # )
    sk_graph.vs["coords"] = skeleton_points
    sk_graph.vs["y"] = skeleton_points[:, 0]
    sk_graph.vs["x"] = skeleton_points[:, 1]
    # Can extend the radius to the mEDT introduce in VesselVio if needed
    sk_graph.vs["radius"] = distance_transform[img_ind][skeletons[img_ind] != 0]
    # print(distance_transform[0][skeletons[0] != 0][0])
    # print(max(distance_transform[0][skeletons[0] != 0]))
    # plt.imshow(distance_transform[0][skeletons[0] != 0])
    # plt.colorbar()
    # print(
    #     f"After: First five: {sk_graph.vs[:5]["y"]} - Last five: {sk_graph.vs[-5:]["y"]}"
    # )

    # Find and add the edges
    # Create vertex index Lookup table
    vertex_LUT = construct_vID_LUT(skeleton_points, skeletons[img_ind].shape)
    # Find edges
    edges = edge_detection(skeleton_points, vertex_LUT)
    # edges = [(0, 1), (1, 2), (2, 3), (3, 4), (3, 5)]
    sk_graph.add_edges(edges)

    # Remove loops and multiedges
    sk_graph.simplify()

    # degrees = sk_graph.degree()
    # print(f"Max vertex degree: {max(degrees)}\nMin vertex degree: {min(degrees)}")

    ig.summary(sk_graph)
    print(
        "Summary structure: 4-char long code, number of vertices, number of edges -- graph name"
    )

    # Simplify the graph and filter unwanted edges and nodes
    filter_graph(sk_graph)

    ig.summary(sk_graph)
    print(
        "Summary structure: 4-char long code, number of vertices, number of edges -- graph name"
    )

    # Analyze the graph segments and simplify it further maintaining only bifurcation points
    analyze_simplify_graph(sk_graph)

    ig.summary(sk_graph)
    print(
        "Summary structure: 4-char long code, number of vertices, number of edges -- graph name"
    )

    # plt.imshow(skeleton_points_connect)
    # plt.colorbar()

    # sk_graph.layout(layout="auto")
    # ig.plot(sk_graph)
    # fig, ax = plt.subplots()
    # ig.plot(sk_graph, vertex_size=2, target=ax)
    # ax.invert_yaxis()
    # # plt.show()

    # # Testing
    # degrees = sk_graph.degree()
    # endpoints = [sk_graph.vs[loc].index for loc, deg in enumerate(degrees) if deg == 1]
    # # fig, ax = plt.subplots()
    # ig.plot(sk_graph.subgraph(sk_graph.vs[endpoints]), vertex_size=10, target=ax)
    # ax.invert_yaxis()
    # plt.show()


if __name__ == "__main__":
    main()
