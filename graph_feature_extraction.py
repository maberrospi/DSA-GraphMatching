import logging
import os, sys
from pathlib import Path

from math import ceil, log
import numpy as np
from geomdl import knotvector
from scipy import interpolate
from itertools import chain
from collections import namedtuple

# Used for debugging and visualizations
import matplotlib.pyplot as plt
import igraph as ig

logger = logging.getLogger(__name__)


def calc_radius_info(radii):
    r_avg = np.mean(radii)
    r_min = np.min(radii)
    r_max = np.max(radii)
    r_std = np.std(radii)
    return r_avg, r_max, r_min, r_std


def calc_delta(num_verts):
    delta = max(3, ceil(num_verts / log(num_verts, 2)))
    if num_verts > 100:
        delta = int(delta / 2)

    return delta


def interpolate_segm(point_coords):
    # Based on the code provided by VesselVio
    # Find Bspline (basis-spline) of the points to smooth the skeleton
    # Set appropriate degree of our BSpline
    num_verts = point_coords.shape[0]
    if num_verts > 4:
        spline_degree = 3
    else:
        spline_degree = max(1, num_verts - 1)

    # Scipy knot vector format
    knots = knotvector.generate(spline_degree, num_verts)  # Knotvector
    tck = [
        knots,
        [point_coords[:, 0], point_coords[:, 1]],
        spline_degree,
    ]

    # The optimal number of interpolated segment points
    # was determined emperically in original code
    delta = calc_delta(num_verts)
    u = np.linspace(0, 1, delta, endpoint=True)  # U

    coords_list = np.array(interpolate.splev(u, tck)).T
    return coords_list


def calc_length(coords):
    # EDT calculations for segment splines
    # Calculate the ED between all coordinates
    # and sum it up to recover an approximate length
    deltas = coords[0:-1] - coords[1:]
    # print(coords.shape)
    squares = (deltas) ** 2  # * np.array([512, 512])
    results = np.sqrt(np.sum(squares, axis=1))
    return np.sum(results)


def calc_seg_length(graph, points, smoothing):
    point_vs = graph.vs[points]

    ### Length ###
    coords_list = point_vs["coords"]
    coords = np.array(coords_list)
    # print(f"Initial pointlist shape: {coords.shape}")

    # Interpolate points to create a smooth line
    # Not sure if I have to use the resolution, in my case i think 512x512
    if smoothing:
        coords = interpolate_segm(coords)
    segment_length = calc_length(coords)

    return segment_length


def get_seg_length(graph, segm, smoothing):
    segment_v_count = len(segm)
    # print(type(segm[0]))
    degrees = graph.degree(segm)
    if segment_v_count < 4:
        # If the segment contains a maximum of 3 nodes
        random_vertex = segm[0]
        if 2 in degrees:
            # If there are 3 nodes pick the middle vertex
            random_vertex = degrees.index(2)
        # Find the points in the current segment
        point_list = graph.neighbors(random_vertex)
        point_list.insert(1, random_vertex)  # Insert into the middle or right
        # Calculate segment length
        segment_length = calc_seg_length(graph, point_list, smoothing)
        # print(f"Segm_v_count: {segment_v_count}")
        # print(f"Segm_Length: {segment_length}")
    else:
        endpoints = [segm[loc] for loc, deg in enumerate(degrees) if deg == 1]
        # print(len(endpoints))
        # We have already ensured there are only 2 endpoints so move on
        # Find the points in the current segment
        point_list = graph.get_shortest_paths(
            endpoints[0], to=endpoints[1], output="vpath"
        )[0]
        # Calculate segment length
        segment_length = calc_seg_length(graph, point_list, smoothing)
        # point_coords = np.array(graph.vs[point_list]["coords"])

    # segment_length = 0
    return segment_length


def get_single_seg_path(graph, segm, segm_ids):
    # segm_ids is a list of vertex objects
    vertex = segm_ids[segm[0]].index
    point_list = graph.neighbors(vertex)
    point_list.insert(1, vertex)
    return point_list


def get_large_seg_path(graph, segm, segm_ids):
    segm_subg = graph.subgraph(segm_ids)
    degrees = segm_subg.degree(segm)
    endpoints = [segm[loc] for loc, deg in enumerate(degrees) if deg == 1]

    if len(endpoints) == 2:
        # Find the ordered path of the given segment
        # The indices do not correspond to the subgraph indices yet
        ordered_path = segm_subg.get_shortest_path(
            endpoints[0], endpoints[1], output="vpath"
        )

        # Add the corresponding subgraph indices to the point list
        point_list = [segm_ids[p].index for p in ordered_path]
        # print(point_list)

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

        # print(point_list)

    # Visualization of what's going on if needed.
    # print(f"Segm length: {len(segm)}")
    if len(endpoints) == 0:
        print(segm_ids[endpoints[0]].index)
        print([p.index for p in segm_ids[endpoints[0]].neighbors()])
        fig, ax = plt.subplots()
        visual_style = {}
        visual_style["vertex_size"] = 20
        visual_style["vertex_color"] = "green"
        ig.plot(
            graph.subgraph(
                graph.vs[
                    [
                        segm_ids[p].index
                        for p in [
                            segm[loc] for loc, deg in enumerate(degrees) if deg == 1
                        ]
                    ]
                ]
            ),
            target=ax,
            **visual_style,
        )
        visual_style["vertex_size"] = 10
        visual_style["vertex_color"] = "red"
        ig.plot(
            graph.subgraph(graph.vs[[segm_ids[p].index for p in segm]]),
            target=ax,
            **visual_style,
        )
        visual_style["vertex_size"] = 1
        visual_style["vertex_color"] = "green"
        ig.plot(graph, target=ax, **visual_style)
        ax.invert_yaxis()
        plt.show()

    # Data exploration
    # cnt = 0
    # cnt1 = 0
    # if len(endpoints) == 2:
    #     cnt += 1
    # else:
    #     cnt1 += 1
    #     # None
    # return cnt, cnt1

    return point_list


# Pythonic way to deal with indexing tuples later on
FeatureVars = namedtuple(
    "FeatureVariables",
    "segment_length,turtuosity,radius_avg,radius_max,radius_min,radius_std",
)


def feature_extraction(graph, point_list, smoothing=True):
    # Calculate radius info
    point_vs = graph.vs[point_list]
    radii_list = point_vs["radius"]
    r_avg, r_max, r_min, r_std = calc_radius_info(radii_list)

    # Calculate segment length
    segment_length = calc_seg_length(graph, point_list, smoothing)

    # Calculate turtuosity/curvature
    # Turtuosity is a number reflecting the  bending of the vessel segment. Definition
    # from: Quantification of Tortuosity and Fractal Dimension of the Lung Vessels in Pulmonary Hypertension Patients
    # Doi: 10.1371/journal.pone.0087515
    # Formula: Ratio of the length of the curve (arc-length (C)) to the distance between its ends (L): C/L
    coords_list = point_vs["coords"]
    coords = np.array(coords_list)
    endpoints = np.array([coords[0], coords[-1]])
    cord_length = calc_length(endpoints)
    # Coord length is 0 in the case of self-loops
    if cord_length == 0:
        # Assign coord length as 0.5 since the smallest cord length
        # is minimum 1
        cord_length = 0.5
    turtuosity = segment_length / cord_length

    # Can also include lateral surface area and volume features based on radius and length
    # All these values are not real world values

    features = FeatureVars(segment_length, turtuosity, r_avg, r_max, r_min, r_std)

    return features


def vessel_segment_feature_extraction(graph, segments, segm_ids):
    # Finds features for all the segments that include nodes between bifurcations
    # These are the majority of segments
    features = []
    edges = []
    c = 0
    cnt = 0
    cnt1 = 0
    # print(type(segments[0][0]))
    # print(type(segments[0]))
    # print(type(segm_ids))
    for segment in segments:
        if len(segment) == 1:
            ordered_segment = get_single_seg_path(graph, segment, segm_ids)
            # print(ordered_segment)
        else:  # More than one node
            ordered_segment = get_large_seg_path(graph, segment, segm_ids)
            # c += 1
            # if c == 3:
            #     break
            # Data exploration
            # c, c1 = get_large_seg_path(graph, segment, segm_ids)
            # cnt += c
            # cnt1 += c1
            # if cnt == 1:
            #     break

        # Keep track of edges to add if the graph is simplified
        edges.append([ordered_segment[0], ordered_segment[-1]])

        # Perform feature extraction on the segment
        features.append(feature_extraction(graph, ordered_segment, smoothing=True))

    # print(cnt)
    # print(cnt1)

    return features, edges


def branch_segment_feature_extraction(graph, segments, segm_ids):
    # Finds features for the segments that are between bifurcations without any vertices in them
    # This is the minority and usually comprises of 2-3 vertices
    features = []
    edges = []
    for segment in segments:
        vertices = [segm_ids[segment.source].index, segm_ids[segment.target].index]
        edges.append(vertices)
        features.append(feature_extraction(graph, vertices, smoothing=False))

    return features, edges


def segment_feature_extraction(graph):
    # First analyze the large segments or main vasculature
    logger.info("Analyzing large segments/main vasculature")
    segment_ids = graph.vs.select(_degree_lt=3)
    subg_segments = graph.subgraph(segment_ids)
    # Get list of connected components that is initially in a vertexCluster class
    segments = list(subg_segments.components())
    features, edges = vessel_segment_feature_extraction(graph, segments, segment_ids)
    # print(f"# of edges to keep: {len(features)}")

    ###
    # Visualize the reached segments
    # fig, ax = plt.subplots()
    # visual_style = {}
    # visual_style["vertex_size"] = 5
    # visual_style["vertex_color"] = "green"
    # ig.plot(graph, target=ax, **visual_style)
    # visual_style["vertex_size"] = 10
    # visual_style["vertex_color"] = "red"
    # ig.plot(subg_segments, target=ax, **visual_style)
    ###

    # Then analyze the smaller segments between bifurcation points
    logger.info("Analyzing small segments/between bifurcations")
    segment_ids = graph.vs.select(_degree_gt=2)
    subg_segments = graph.subgraph(segment_ids)
    segments = subg_segments.es()  # Edges in the subgraph
    temp_features, temp_edges = branch_segment_feature_extraction(
        graph, segments, segment_ids
    )
    features.extend(temp_features)
    edges.extend(temp_edges)
    # print(f"# of edges to keep: {len(edges)}")

    ###
    # visual_style["vertex_size"] = 5
    # visual_style["vertex_color"] = "green"
    # visual_style["edge_color"] = "red"
    # ig.plot(subg_segments, target=ax, **visual_style)
    # plt.show()
    ###

    return features, edges


def save_feature_results(graph, features, edges, simplify_graph=True):
    segment_count = len(features)
    lengths = np.zeros(segment_count)
    turtuosities = np.zeros(segment_count)
    radii_avg = np.zeros(segment_count)
    radii_max = np.zeros(segment_count)
    radii_min = np.zeros(segment_count)
    radii_std = np.zeros(segment_count)

    for i, feature in enumerate(features):
        lengths[i] = feature.segment_length
        turtuosities[i] = feature.turtuosity
        radii_avg[i] = feature.radius_avg
        radii_max[i] = feature.radius_max
        radii_min[i] = feature.radius_min
        radii_std[i] = feature.radius_std

    if simplify_graph:
        reduce_graph(
            graph,
            edges,
            lengths,
            turtuosities,
            radii_avg,
            radii_max,
            radii_min,
            radii_std,
        )

    # The features can be returned here and used for further analysis
    return (
        lengths,
        turtuosities,
        radii_avg,
        radii_max,
        radii_min,
        radii_std,
    )


def reduce_graph(
    graph, edges, lengths, turtuosities, radii_avg, radii_max, radii_min, radii_std
):
    graph.delete_edges(graph.es())
    graph.add_edges(
        edges,
        {
            "length": lengths,
            "turtuosity": turtuosities,
            "radius_avg": radii_avg,
            "radius_max": radii_max,
            "radius_min": radii_min,
            "radius_std": radii_std,
        },
    )
    graph.delete_vertices(graph.vs.select(_degree=0))
    return


def simplified_branch_segment_ft_extraction(graph, new_edges):
    # Same effect as branch_segment_feature_extraction() but modified arguments
    # for further_simplified graphs
    features = []
    for edge in new_edges:
        # Every edge contains the source and target node index
        vertices = [edge[0], edge[1]]
        features.append(feature_extraction(graph, vertices, smoothing=False))

    return features


def max_distance_vs(vertices: list):
    # List of igraph.Vertex objects
    assert len(vertices) == 2, "Vertices length can only be two"

    v_xs = [v["x"] for v in vertices]
    v_ys = [v["y"] for v in vertices]
    max_dist = 0

    for v in vertices:
        max_dist = max([max(max_dist, nums) for nums in abs(v_xs - v["x"])])
        max_dist = max([max(max_dist, nums) for nums in abs(v_ys - v["y"])])

    # print(f"Max distance: {max_dist}")
    return max_dist


def max_distance(clique: ig.VertexSeq):
    # Calculate the max distance between the node pixels in a given clique
    # The distance is measure in x and y pixel difference
    max_dist = 0
    subgr_x = [x for x in clique["x"]]
    subgr_y = [y for y in clique["y"]]

    for node in clique:
        max_dist = max([max(max_dist, nums) for nums in abs(node["x"] - subgr_x)])
        max_dist = max([max(max_dist, nums) for nums in abs(node["y"] - subgr_y)])

    # print(max_dist)
    return max_dist


def find_new_vertex_neighbors(graph, subg_vs):
    graph_vs = graph.vs[subg_vs["id"]]

    neighbors = []
    e_features = []
    # print(f'Subgraph ids: {subg_vs["id"]}')
    clique_nb_ids = subg_vs["id"]
    for graph_v, subg_v in zip(graph_vs, subg_vs):

        # The vertex IDs since up to here they are the same as the vertex index
        # Maybe use extend here
        cur_nbs = [
            nb["id"] for nb in graph_v.neighbors() if nb["id"] not in clique_nb_ids
        ]

        neighbors += [
            nb["id"] for nb in graph_v.neighbors() if nb["id"] not in clique_nb_ids
        ]

        num_of_all_nbs = len(neighbors)

        for nb in cur_nbs:
            # Store original edge features
            e_id = graph.get_eid(graph_v["id"], nb)
            e_features.append(FeatureVars(*graph.es[e_id].attributes().values()))

        # print(f"Neighbors: {neighbors}")
    for neighbor in neighbors[:]:  # Iterate cloned neighbors
        if neighbor in vertices_to_remove:
            index = neighbors.index(neighbor)
            nodes_to_keep_track.append(neighbors.pop(index))
            e_features.pop(index)

    # print(len(neighbors))
    # print(e_features, len(e_features), sep=", ")

    return neighbors, e_features, num_of_all_nbs


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
    neighbors, e_features, num_of_all_nbs = find_new_vertex_neighbors(graph, subg_vs)

    # print(num_of_all_nbs, len(neighbors))
    if num_of_all_nbs != len(neighbors):
        for _ in range(num_of_all_nbs - len(neighbors)):
            # append the source clique index
            src_cl_idxs.append(src_idx)

    return [new_vertex_info, neighbors, e_features]


def get_cliques(graph, l_bound=3, u_bound=5, bifurcation_cliques=True):
    # Assign all vertices their original ID
    graph.vs["id"] = np.arange(graph.vcount())

    if bifurcation_cliques:
        # Retrieve the nodes that have a degree greater than 2
        # Naturally this will be the majority of the nodes left
        segment_ids = graph.vs.select(_degree_gt=2)
        subg_segments = graph.subgraph(segment_ids)

        # Eliminate 1 degree edges connected to the cliques
        while True:
            count = len(subg_segments.vs.select(_degree_lt=2))
            if count == 0:
                break
            subg_segments = subg_segments.subgraph(
                subg_segments.vs.select(_degree_gt=1)
            )
        # Get the maximal cliques that have 3-4 nodes
        cliques = [clique for clique in subg_segments.maximal_cliques(l_bound, u_bound)]

        # Keep the cliques that have a max given distance between their nodes (i.e max 10 pixels)
        cliques = [
            clique for clique in cliques if max_distance(subg_segments.vs[clique]) < 11
        ]
    else:
        segment_ids = graph.vs.select(_degree_gt=0)
        subg_segments = graph.subgraph(segment_ids)

        # Get the maximal cliques
        cliques = [
            clique for clique in subg_segments.maximal_cliques(l_bound, u_bound)
        ]  # if 2 < len(clique) < 5]
        # Keep the cliques that have a max given distance between their nodes (i.e max 10 pixels)
        cliques = [
            clique for clique in cliques if max_distance(subg_segments.vs[clique]) <= 4
        ]

        # Create a subgraph from those cliques
        subg_segments = subg_segments.subgraph(list(chain(*cliques)))

        # Get updated connected components
        cliques = [clique for clique in subg_segments.components()]

        # Remove the "long" edges from the subg_segments which connect neighboring cliques
        edges_to_remove = []
        for edge in subg_segments.es():
            src_node_id = edge.source
            trgt_node_id = edge.target
            src_node = subg_segments.vs[edge.source]
            trgt_node = subg_segments.vs[edge.target]

            if max_distance_vs([src_node, trgt_node]) > 4:
                edges_to_remove.append((src_node_id, trgt_node_id))

        subg_segments.delete_edges(edges_to_remove)

        # Lastly re-get the connected components cliques
        cliques = [clique for clique in subg_segments.components()]

    return subg_segments, cliques


def calc_new_edge_features(e_fts: list[FeatureVars], idxs: list[int]) -> None:
    assert e_fts, "Edge features list is empty"
    # Which edge feature to keep? Largest avg radius or longest length?
    max_length = 0
    max_idx = 0
    for i in idxs:
        if e_fts[i].segment_length > max_length:
            max_length = e_fts[i].segment_length
            max_idx = i

    for i in idxs:
        if i != max_idx:
            e_fts.pop(i)


def simplify_more(graph, subg_segments, cliques, vis=False):
    # NOTE: Using globals is not advised. The only reason they were used was to reduce
    # the complexity of the functions and the arguments passed.
    # This behaviour should probably be refactored down the line.
    global vertices_to_remove
    global nodes_to_keep_track
    global src_cl_idxs
    global src_idx
    src_idx = 0
    src_cl_idxs = []
    vertices_to_remove = []
    nodes_to_keep_track = []

    vertices_to_remove.extend(subg_segments.vs[clique]["id"] for clique in cliques)
    # Flatten the list of lists
    vertices_to_remove = list(chain.from_iterable(vertices_to_remove))
    # print("Vertices to remove: ", vertices_to_remove)

    new_vs = []
    e_features = []
    # Group these cliques of points into one point and account for the neighbors
    # in the original graph so we don't miss connections
    for clique in cliques:
        # Don't use the one from graph processing as its slightly different
        # v_info, neighbors = GProcess.create_new_vertex(graph, subg_segments.vs[clique])
        v_info, neighbors, edge_features = create_new_vertex(
            graph, subg_segments.vs[clique]
        )
        # print(neighbors)
        new_vs.append((v_info, neighbors))
        e_features.extend(edge_features)

        src_idx += 1

    # Create a list that is basically the target clique index
    trgt_cl_idxs = []
    # For every node that was kept track of earlier find its clique index
    for node in nodes_to_keep_track:
        # node is the id of the node
        for index, clique in enumerate(cliques):
            if node in subg_segments.vs[clique]["id"]:
                trgt_cl_idxs.append(index)
                break

    # Print what we have so far
    # print(src_cl_idxs, trgt_cl_idxs, nodes_to_keep_track)

    new_edges = []
    new_v_indices = []
    # Add new vertices that were created
    for vertex_group in new_vs:
        vertex_info = vertex_group[0]
        neighbors = vertex_group[1]
        new_v = graph.add_vertex(
            radius=vertex_info["radius"],
            coords=vertex_info["coords"],
            x=vertex_info["x"],
            y=vertex_info["y"],
            # For testing
            # color="yellow",
            # size=5,
        )

        new_edges.extend(sorted(tuple([new_v.index, nb]) for nb in neighbors))
        new_v_indices.append(new_v.index)

    # Prints to hopefully help with correctly dealing with edge features.
    for i, it in enumerate(new_edges):
        print(i, it, sep=" - ")
    for i, it in enumerate(e_features):
        print(i, it, sep=" - ")

    # Create dictionary of new edges
    edge_dict = {k: [] for k in new_edges}

    for idx, edge in enumerate(new_edges):
        edge_dict[edge].append(idx)

    for idx, val in enumerate(edge_dict.values()):
        if len(val) > 1:
            # Modify edge features list in place
            calc_new_edge_features(e_features, val)

    temp_new_edges = []
    # Add the edges between two new nodes added from neighboring cliques
    for new_edge_src, new_edge_trgt in zip(src_cl_idxs, trgt_cl_idxs):
        temp_new_edges.append(
            tuple(sorted([new_v_indices[new_edge_src], new_v_indices[new_edge_trgt]]))
        )
        # new_edges.append(
        #     tuple(sorted([new_v_indices[new_edge_src], new_v_indices[new_edge_trgt]]))
        # )

    # Remove duplicate edges
    temp_new_edges = [edges for edges in set(temp_new_edges)]
    new_features = simplified_branch_segment_ft_extraction(graph, temp_new_edges)
    # # TODO: concat new features with the rest of the edge features
    features_to_add = e_features
    features_to_add.extend(new_features)
    lengths, turtuosities, radii_avg, radii_max, radii_min, radii_std = (
        save_feature_results(
            graph=None, features=features_to_add, edges=None, simplify_graph=False
        )
    )

    edges_to_add = [edge for edge in edge_dict.keys()]
    edges_to_add.extend(temp_new_edges)

    for i, it in enumerate(features_to_add):
        print(i, it, sep=" - ")

    graph.add_edges(
        edges_to_add,
        {
            "length": lengths,
            "turtuosity": turtuosities,
            "radius_avg": radii_avg,
            "radius_max": radii_max,
            "radius_min": radii_min,
            "radius_std": radii_std,
        },
    )

    # graph.add_edges(edges_to_add)  # Add the new edges
    graph.delete_vertices(vertices_to_remove)  # Delete old vertices

    # Delete the id attribute since its not useful anymore - IMPORTANT
    del graph.vs["id"]

    # Cleanup globals
    del vertices_to_remove
    del nodes_to_keep_track
    del src_cl_idxs
    del src_idx

    if vis:
        # Visualize the reached segments
        fig, ax = plt.subplots()
        visual_style = {}
        visual_style["vertex_size"] = 5
        # visual_style["vertex_color"] = "green"
        ig.plot(graph, target=ax, **visual_style)
        visual_style["vertex_size"] = 10
        visual_style["vertex_color"] = "red"
        ig.plot(subg_segments, target=ax, **visual_style)
        ax.invert_yaxis()
        visual_style["vertex_size"] = 10
        visual_style["vertex_color"] = "blue"
        visual_style["edge_color"] = "red"
        for clique in cliques:
            ig.plot(subg_segments.subgraph(clique), target=ax, **visual_style)

        fig, ax = plt.subplots()
        visual_style = {}
        visual_style["vertex_size"] = 5
        # visual_style["vertex_color"] = "green"
        ig.plot(graph, target=ax, **visual_style)
        ax.invert_yaxis()

        plt.show()


def simplify_more_sclass1(graph, iters=3, vis=False):
    # Simplify the graph further to combine final 'bifurcation' points that are clustered
    # or that are simply very close to each other

    # Not really sure if this is a good approach or not
    # Might not be since it adds more change to the graph structure.

    logger.info("Simplifying the graph further - Bifurcation Clusters")

    for _ in range(iters):
        subg_segments, cliques = get_cliques(graph, l_bound=3, u_bound=5)

        simplify_more(graph, subg_segments, cliques, vis)

    return None


def simplify_more_sclass2(graph, iters=3, vis=False):

    ### Here we can extend this simplification with the idea of grouping pairs of
    ### nodes that are within a given distance from each other
    ### This must be separate from the above simplification to avoid overlaps.

    logger.info("Simplifying the graph further - Neighbooring node pairs")

    for _ in range(iters):
        subg_segments, cliques = get_cliques(
            graph, l_bound=2, u_bound=3, bifurcation_cliques=False
        )

        simplify_more(graph, subg_segments, cliques, vis)

    return None


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


if __name__ == "__main__":
    main()
