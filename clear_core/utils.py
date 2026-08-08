


class GMap:
    def __init__(self, elevation_map, landcover_map, scale_xy=30.0):
        self.elevation = elevation_map
        self.type = landcover_map
        self.scale_xy = scale_xy


import numpy as np

# ---------------- Utility Functions ---------------- #
def pixel_to_coords(xs, ys, transform):
    coords = [transform * (x, y) for x, y in zip(xs, ys)]
    return np.array(coords)


# ---------------- Plotting Functions ---------------- #


import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d
def plot_voronoi_graph(convex_rb, center_graph, vor, elevation, region_path, start_coord, goal_coord):

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111)

    voronoi_plot_2d(vor, ax=ax, show_vertices=False, show_points=False, line_colors='black', line_width=0.5, line_alpha=0.3, point_size=0)
    # ax.plot(vor_points[:, 0], vor_points[:, 1], 'ro', markersize=2, alpha=0.5)
    # Overlay elevation as background
    ax.imshow(elevation, cmap='terrain', origin='lower')#, extent=(0, elevation.shape[1], elevation.shape[0], 0))


    start_region = convex_rb.find_region_containing_point(start_coord)
    goal_region = convex_rb.find_region_containing_point(goal_coord)



    # for i, j in center_graph.edges:
    #     p1, p2 = vor.points[i], vor.points[j]
    #     ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color='white', linewidth=0.8)

    # for idx, region_index in enumerate(vor.point_region):
    #     point = vor.points[idx]
    #     ax.plot(point[0], point[1], 'ro', markersize=2, alpha=0.5)
    #     if region_index == start_region:
    #         ax.text(point[0], point[1], 'Start' +str(region_index), fontsize=8, color='green')
    #     elif region_index == goal_region:
    #         ax.text(point[0], point[1], 'Goal' +str(region_index), fontsize=8, color='blue')
    #     # ax.text(point[0], point[1], str(idx), fontsize=8, color='black')
            

    for region in region_path:

        point_index = np.where(convex_rb.voronoi.point_region == region)[0][0]
        point = convex_rb.voronoi.points[point_index]
        ax.text(point[0], point[1], region, fontsize=8, color='blue')

    # plot path
    for i, j in zip(region_path[:-1], region_path[1:]):
        # p1, p2 = vor.points[i], vor.points[j]
        point_index = np.where(convex_rb.voronoi.point_region == i)[0][0]
        p1 = convex_rb.voronoi.points[point_index]
        point_index = np.where(convex_rb.voronoi.point_region == j)[0][0]
        p2 = convex_rb.voronoi.points[point_index]

        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color='yellow', linewidth=3)


    ax.plot(start_coord[0], start_coord[1], 'go', markersize=8, label='Start')
    ax.plot(goal_coord[0], goal_coord[1], 'bo', markersize=8, label='Goal')

    plt.tight_layout()
    plt.xlim(0, elevation.shape[1])
    plt.ylim(elevation.shape[0], 0)

    plt.savefig("results/region_graph.png", dpi=300)


def plot_gcs_path(vor, gcs_graph, paths, landcover_map, output_file="gcs_path.png", plot_poly=False):



    # Visualize path
    fig, ax = plt.subplots(figsize=(8, 6))

    im = ax.imshow(landcover_map, cmap='tab20b', origin='lower', alpha=0.5)

    if vor:
        voronoi_plot_2d(vor, ax=ax, show_vertices=False, show_points=False, line_colors='black', line_width=0.5, line_alpha=0.3, point_size=0)

    if plot_poly:
        for label, data in gcs_graph.nodes(data=True):
            poly = np.array(data['region'].polygon_pts)
            x_poly, y_poly = poly[:, 0], poly[:, 1]
            # print("Poly:", poly)
            ax.fill(x_poly, y_poly, alpha=0.5, ec='black')

    colors = ['g', 'b', 'c', 'm', 'y', 'k']

    decomp_to_color = {
        "Boundary": 'g',
        "Quadtree": 'b',
        "Hex": 'r',
        "Grid": 'y'
    }
    cindx = 0
    decomp_list = list(decomp_to_color.keys())


    for name, (path, cost, compute_time) in paths.items():
        decomp_name=False
        for decomp in decomp_list:
            if name.startswith(decomp):
                color = decomp_to_color[decomp]
                decomp_name = True
                break
        if not decomp_name:
            color = colors[cindx]
            cindx = (cindx + 1) % len(colors)
        name_only = name.split("_")[0]
        if len(path) == 0:
            continue
        path_coords = np.array(path)
        ax.plot(path_coords[:, 0], path_coords[:, 1], color + '', linewidth=1.0, label=name_only+": "+str(round(cost,2)) +"("+str(round(compute_time,2))+" sec)", alpha=0.7)

    # plot start and goal
    ax.scatter([path_coords[0][0], path_coords[-1][0]], [path_coords[0][1], path_coords[-1][1]], color='red', marker='x', s=100, label='Start/Goal')

    # ax.plot(*zip(*path_coords), 'r-o', linewidth=2, label='GCS Path')
    # ax.set_title("Shortest Path on GCS Graph")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_xlim(0, landcover_map.shape[1])
    ax.set_ylim(0,landcover_map.shape[0])

    plt.colorbar(im, ax=ax, label='Landcover', fraction=0.026, pad=0.04) 
    
    if True:
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=3)
    else:

        # Dynamically scale figure height based on number of legend items
        num_items = len(ax.get_legend_handles_labels()[1])
        rows = ((num_items + 2) // 3)  + 3 # assuming ncol=3
        extra_space = 0.2 * rows     # vertical space for legend rows
        fig.set_figheight(fig.get_figheight() + extra_space)

        # Place legend outside the plot
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1 + 0.15 * rows),  ncol=3)

    plt.tight_layout()
    plt.savefig(output_file)



def plot_gcs_path_elevation(vor, gcs_graph, paths, elevation_map, output_file="gcs_path.png", plot_poly=False):

    # Visualize path
    fig, ax = plt.subplots(figsize=(8, 6))

    im = ax.imshow(elevation_map, cmap='terrain', origin='lower', alpha=0.5)

    if vor:
        voronoi_plot_2d(vor, ax=ax, show_vertices=False, show_points=False, line_colors='black', line_width=0.5, line_alpha=0.3, point_size=0)

    if plot_poly:
        for label, data in gcs_graph.nodes(data=True):
            poly = np.array(data['region'].polygon_pts)
            x_poly, y_poly = poly[:, 0], poly[:, 1]
            # print("Poly:", poly)
            ax.fill(x_poly, y_poly, alpha=0.5, ec='black')

    # colors = ['g', 'b', 'c', 'm', 'y', 'k']

    decomp_to_color = {
        "Boundary": 'g',
        "Quadtree": 'b',
        "Hex": 'r',
        "Grid": 'y'
    }

    decomp_to_color_smooth = {
        "Boundary": 'g',
        "Quadtree": 'b',
        "Hex": 'r',
        "Grid": 'y'
    }
    decomp_list = list(decomp_to_color.keys())

    for name, (path, cost, compute_time) in paths.items():
        for decomp in decomp_list:
            if name.startswith(decomp):
                if name.endswith("smooth"):
                    color = decomp_to_color_smooth[decomp]
                else:
                    color = decomp_to_color[decomp]
                break

        if len(path) == 0:
            continue

        name_only = name.split("_")[0]
        path_coords = np.array(path)
        ax.plot(path_coords[:, 0], path_coords[:, 1], color + '', linewidth=1.0, label=name_only+": "+str(round(cost,2)) +"("+str(round(compute_time,2))+" sec)", alpha=0.7)

    # plot start and goal
    ax.scatter([path_coords[0][0]], [path_coords[0][1]], color='red', marker='o', s=100, label='Start')
    ax.scatter([path_coords[-1][0]], [path_coords[-1][1]], color='blue', marker='o', s=100, label='Goal')


    label_fontsize = 12

    # ax.plot(*zip(*path_coords), 'r-o', linewidth=2, label='GCS Path')
    # ax.set_title("Shortest Path on GCS Graph")
    ax.set_xlabel("X", fontsize=label_fontsize)
    ax.set_ylabel("Y", fontsize=label_fontsize)
    ax.set_xlim(0, elevation_map.shape[1])
    ax.set_ylim(0,elevation_map.shape[0])
    
    plt.colorbar(im, ax=ax, label='Elevation', fraction=0.026, pad=0.04) 

    if True:
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=3, fontsize=label_fontsize)
    else:

        # Dynamically scale figure height based on number of legend items
        num_items = len(ax.get_legend_handles_labels()[1])
        rows = ((num_items + 2) // 3)  + 3 # assuming ncol=3
        extra_space = 0.2 * rows     # vertical space for legend rows
        fig.set_figheight(fig.get_figheight() + extra_space)

        # Place legend outside the plot
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1 + 0.15 * rows),  ncol=3)


    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')





def plot_satellite_path(paths, satellite_image, output_file="satellite_path.png"):



    # Visualize path
    fig, ax = plt.subplots(figsize=(8, 6))

    im = ax.imshow(satellite_image, origin='lower', alpha=0.5)

    # if vor:
    #     voronoi_plot_2d(vor, ax=ax, show_vertices=False, show_points=False, line_colors='black', line_width=0.5, line_alpha=0.3, point_size=0)

    # colors = ['g', 'b', 'c', 'm', 'y', 'k']

    decomp_to_color = {
        "Boundary": 'g',
        "Quadtree": 'b',
        "Hex": 'r',
        "Grid": 'y'
    }

    decomp_to_color_smooth = {
        "Boundary": 'g',
        "Quadtree": 'b',
        "Hex": 'r',
        "Grid": 'y'
    }
    decomp_list = list(decomp_to_color.keys())

    for name, (path, cost, compute_time) in paths.items():
        for decomp in decomp_list:
            if name.startswith(decomp):
                if name.endswith("smooth"):
                    color = decomp_to_color_smooth[decomp]
                else:
                    color = decomp_to_color[decomp]
                break

        if len(path) == 0:
            continue
        path_coords = np.array(path)
        ax.plot(path_coords[:, 0], path_coords[:, 1], color + '', linewidth=1.0, label=name+": "+str(round(cost,2)) +"("+str(round(compute_time,2))+" sec)", alpha=0.7)

    # plot start and goal
    ax.scatter([path_coords[0][0], path_coords[-1][0]], [path_coords[0][1], path_coords[-1][1]], color='red', marker='x', s=100, label='Start/Goal')

    # ax.plot(*zip(*path_coords), 'r-o', linewidth=2, label='GCS Path')
    # ax.set_title("Shortest Path on GCS Graph")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_xlim(0, satellite_image.shape[1])
    ax.set_ylim(0,satellite_image.shape[0])
    

    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=3)


    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')





# # -*- coding: utf-8 -*-
# """
# A NetworkX based implementation of Yen's algorithm for computing K-shortest paths.   
# Yen's algorithm computes single-source K-shortest loopless paths for a 
# graph with non-negative edge cost. For more details, see: 
# http://en.m.wikipedia.org/wiki/Yen%27s_algorithm
# """
# __author__ = 'Guilherme Maia <guilhermemm@gmail.com>'

# __all__ = ['k_shortest_paths']

from heapq import heappush, heappop
from itertools import count

import networkx as nx

def get_path_length_old(G, path, weight='weight'):
    return sum(G[u][v].get(weight, 1) for u, v in zip(path[:-1], path[1:]))

def k_shortest_paths_old(G, source, target, k=1, weight='weight'):
    if source == target:
        return ([0], [[source]])

    length, path = nx.single_source_dijkstra(G, source, weight=weight)
    if target not in length:
        raise nx.NetworkXNoPath(f"node {target} not reachable from {source}")

    lengths = [length[target]]
    paths = [path[target]]
    c = count()
    B = []
    G_original = G.copy()

    for _ in range(1, k):
        for j in range(len(paths[-1]) - 1):
            spur_node = paths[-1][j]
            root_path = paths[-1][:j + 1]

            G_temp = G.copy()
            edges_removed = []

            for p in paths:
                if len(p) > j and root_path == p[:j + 1]:
                    u, v = p[j], p[j + 1]
                    if G_temp.has_edge(u, v):
                        G_temp.remove_edge(u, v)
                        edges_removed.append((u, v))

            for node in root_path[:-1]:
                for u, v, _ in list(G_temp.edges(node, data=True)):
                    G_temp.remove_edge(u, v)
                    edges_removed.append((u, v))
                if G_temp.is_directed():
                    for u, v, _ in list(G_temp.in_edges(node, data=True)):
                        G_temp.remove_edge(u, v)
                        edges_removed.append((u, v))

            try:
                spur_length, spur_path = nx.single_source_dijkstra(G_temp, spur_node, weight=weight)
                if target in spur_path:
                    total_path = root_path[:-1] + spur_path[target]
                    total_length = get_path_length(G_original, total_path, weight)
                    heappush(B, (total_length, next(c), total_path))
            except nx.NetworkXNoPath:
                continue

        if B:
            l, _, p = heappop(B)
            lengths.append(l)
            paths.append(p)
        else:
            break

    return lengths, paths 
    




def get_path_length(G, path, weight='weight'):
    return sum(G[u][v].get(weight, 1) for u, v in zip(path[:-1], path[1:]))

def k_shortest_paths(G, source, target, k=1, weight='weight', T=30):
    if source == target:
        return ([0], [[source]])

    length, path = nx.single_source_dijkstra(G, source, weight=weight)
    if target not in length:
        raise nx.NetworkXNoPath(f"node {target} not reachable from {source}")

    lengths = [length[target]]
    paths = [path[target]]
    c = count()
    B = []
    G_original = G.copy()

    for _ in range(1, k):
        for j in range(0, len(paths[-1]) - 1, T):  # Only every T-th node
            spur_node = paths[-1][j]
            root_path = paths[-1][:j + 1]

            G_temp = G.copy()
            edges_removed = []

            # Collect up to T conflicting edges from prior paths
            conflicting_edges = set()
            for p in paths:
                if len(p) > j and root_path == p[:j + 1]:
                    u, v = p[j], p[j + 1]
                    conflicting_edges.add((u, v))

            for u, v in list(conflicting_edges)[:T]:  # Remove up to T
                if G_temp.has_edge(u, v):
                    G_temp.remove_edge(u, v)
                    edges_removed.append((u, v))

            # Remove all outgoing and incoming edges for root_path[:-1]
            for node in root_path[:-1]:
                for u, v, _ in list(G_temp.edges(node, data=True)):
                    G_temp.remove_edge(u, v)
                    edges_removed.append((u, v))
                if G_temp.is_directed():
                    for u, v, _ in list(G_temp.in_edges(node, data=True)):
                        G_temp.remove_edge(u, v)
                        edges_removed.append((u, v))

            # Try spur path
            try:
                spur_length, spur_path = nx.single_source_dijkstra(G_temp, spur_node, weight=weight)
                if target in spur_path:
                    total_path = root_path[:-1] + spur_path[target]
                    total_length = get_path_length(G_original, total_path, weight)
                    heappush(B, (total_length, next(c), total_path))
            except nx.NetworkXNoPath:
                continue

        if B:
            l, _, p = heappop(B)
            lengths.append(l)
            paths.append(p)
        else:
            break

    return lengths, paths







def get_path_length(G, path, weight='weight'):
    return sum(G[u][v].get(weight, 1) for u, v in zip(path[:-1], path[1:]))

def k_shortest_paths(G, source, target, k=1, weight='weight', T=10):
    if source == target:
        return ([0], [[source]])

    length, path = nx.single_source_dijkstra(G, source, weight=weight)
    if target not in length:
        raise nx.NetworkXNoPath(f"node {target} not reachable from {source}")

    lengths = [length[target]]
    paths = [path[target]]
    c = count()
    B = []
    G_original = G.copy()
    seen_paths = {tuple(path[target])}

    while len(paths) < k:
        for j in range(0, len(paths[-1]) - 1, T):  # every Tth spur node
            spur_node = paths[-1][j]
            root_path = paths[-1][:j + 1]

            G_temp = G.copy()
            edges_removed = []

            # Remove up to T conflicting edges
            conflicting_edges = set()
            for p in paths:
                if len(p) > j and root_path == p[:j + 1]:
                    u, v = p[j], p[j + 1]
                    conflicting_edges.add((u, v))

            for u, v in list(conflicting_edges)[:T]:
                if G_temp.has_edge(u, v):
                    G_temp.remove_edge(u, v)
                    edges_removed.append((u, v))

            # Remove outgoing/incoming edges for root_path except spur node
            for node in root_path[:-1]:
                for u, v, _ in list(G_temp.edges(node, data=True)):
                    G_temp.remove_edge(u, v)
                    edges_removed.append((u, v))
                if G_temp.is_directed():
                    for u, v, _ in list(G_temp.in_edges(node, data=True)):
                        G_temp.remove_edge(u, v)
                        edges_removed.append((u, v))

            try:
                spur_length, spur_path = nx.single_source_dijkstra(G_temp, spur_node, weight=weight)
                if target in spur_path:
                    total_path = root_path[:-1] + spur_path[target]
                    if tuple(total_path) in seen_paths:
                        continue
                    total_length = get_path_length(G_original, total_path, weight)
                    heappush(B, (total_length, next(c), total_path))
                    seen_paths.add(tuple(total_path))
            except nx.NetworkXNoPath:
                continue

        if B:
            l, _, p = heappop(B)
            lengths.append(l)
            paths.append(p)
        else:
            break

    return lengths, paths






def get_path_length(G, path, weight='weight'):
    return sum(G[u][v].get(weight, 1) for u, v in zip(path[:-1], path[1:]))

def k_diverse_paths(G, source, target, k=1, weight='weight', T=10, min_pct_increase=5):
    if source == target:
        return ([0], [[source]])

    length, path = nx.single_source_dijkstra(G, source, weight=weight)
    if target not in length:
        raise nx.NetworkXNoPath(f"node {target} not reachable from {source}")

    lengths = [length[target]]
    paths = [path[target]]
    last_cost = length[target]

    c = count()
    B = []
    G_original = G.copy()
    seen_paths = {tuple(path[target])}

    while len(paths) < k:
        for j in range(0, len(paths[-1]) - 1, T):  # every Tth spur node
            spur_node = paths[-1][j]
            root_path = paths[-1][:j + 1]

            G_temp = G.copy()
            edges_removed = []

            conflicting_edges = set()
            for p in paths:
                if len(p) > j and root_path == p[:j + 1]:
                    u, v = p[j], p[j + 1]
                    conflicting_edges.add((u, v))

            for u, v in list(conflicting_edges)[:T]:
                if G_temp.has_edge(u, v):
                    G_temp.remove_edge(u, v)
                    edges_removed.append((u, v))

            for node in root_path[:-1]:
                for u, v, _ in list(G_temp.edges(node, data=True)):
                    G_temp.remove_edge(u, v)
                    edges_removed.append((u, v))
                if G_temp.is_directed():
                    for u, v, _ in list(G_temp.in_edges(node, data=True)):
                        G_temp.remove_edge(u, v)
                        edges_removed.append((u, v))

            try:
                spur_length, spur_path = nx.single_source_dijkstra(G_temp, spur_node, weight=weight)
                if target in spur_path:
                    total_path = root_path[:-1] + spur_path[target]
                    total_length = get_path_length(G_original, total_path, weight)

                    if tuple(total_path) in seen_paths:
                        continue
                    if total_length <= last_cost * (1 + min_pct_increase / 100):
                        continue

                    heappush(B, (total_length, next(c), total_path))
                    seen_paths.add(tuple(total_path))
            except nx.NetworkXNoPath:
                continue

        if B:
            l, _, p = heappop(B)
            if l > last_cost * (1 + min_pct_increase / 100):
                lengths.append(l)
                paths.append(p)
                last_cost = l
        else:
            break

    return lengths, paths
