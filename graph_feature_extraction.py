import logging
import os, sys
from pathlib import Path

from math import ceil, log
import numpy as np
from geomdl import knotvector
from scipy import interpolate
from collections import namedtuple

# Remove after debugging
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
    # I believe we calculate the ED between all coordinates
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
    return


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
