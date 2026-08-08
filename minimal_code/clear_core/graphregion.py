
import networkx as nx
from shapely.geometry import Polygon as ShapelyPolygon
import numpy as np


from info_loss_surface import  RegionBuilderPatches, RegionBuilder, PlaneRegion, SurfaceRegion

from cost_models import AgentModelObjective
from scipy.spatial import cKDTree



from abc import ABC, abstractmethod

class GraphRegionBuilderBase(ABC):
    def __init__(self, rb: RegionBuilder, model: AgentModelObjective):
        # super().__init__(rb.landcover_map, rb.elevation_map, rb.transform, rb.map_name, region_count=rb.region_count)
        self.graph = None 
        self.graph_building_time = 0.0

        self.model = model
        self.rb = rb


    @abstractmethod
    def build_graph(self):
        """
        Build the graph based on the Voronoi regions and their attributes.
        """
        raise NotImplementedError("This method should be implemented in subclasses.")
    

    def get_graph_build_time(self):
        """
        Return the time taken to build the graph.
        """
        if self.graph is None:
            raise ValueError("Graph has not been built yet.")
        return self.graph_building_time
    
    @abstractmethod
    def compute_nx_path(self, source, target):
        """
        Compute the shortest path between source and target points using the graph.
        """
        raise NotImplementedError("This method should be implemented in subclasses.")
    

    @abstractmethod
    def find_region_containing_point(self, xy):
        """
        Given a point (x, y), return the index of the region containing the point.
        """
        raise NotImplementedError("This method should be implemented in subclasses.")



class GraphPatchRegionBuilder(GraphRegionBuilderBase):
    def __init__(self, rb: RegionBuilderPatches, model: AgentModelObjective):
        super().__init__(rb, model)
        
        # self.model = model
        # self.graph = None 

        self.voronoi = rb.voronoi
        self.regions = rb.regions
        self.all_output = rb.all_output

        self.label_to_vertex = None


        points = []
        for i in self.regions:
            current_region = self.regions[i]
            points.append(current_region.centroid)
            if len(current_region.centroid) != 2:
                print("=============Centroid:", current_region.centroid)
                raise ValueError("Centroid is not 2D")
        # print(points)
        self.points = np.array(points)

        self.polygons = rb.polygons
        self.patches = rb.patches
        self.map_scale = rb.map_scale

        self.neighbors = rb.neighbors
        print("ConvexPatchRegionBuilder initialized with points:", len(self.points))


    def find_region_containing_point(self, xy):
        """
        Given a point (x, y) and a scipy.spatial.Voronoi object,
        return the index of the region containing the point.
        """
        ##TODO: FIX THIS
        print("Points" , len(self.points), xy)
        tree = cKDTree(self.points)
        _, i = tree.query(xy)
        region_index = i
        return region_index



    def compute_nx_n_paths(self, source, target, n=5, T=10):
        """
        Given source and target points, compute n suboptimal paths using GCS.
        """

        # Find the Voronoi regions containing the start and goal points
        start_region = self.find_region_containing_point(source)
        goal_region = self.find_region_containing_point(target)

        from utils import k_shortest_paths, k_diverse_paths

        import itertools

        try:

            # _, k_paths = k_shortest_paths(self.graph, source=start_region, target=goal_region, weight="weight", k=n, T=T)
            
            _ , k_paths = k_diverse_paths(self.graph, source=start_region, target=goal_region, weight="weight", k=n, T=T, min_pct_increase=1)
            # G: weighted graph, u: start, v: goal
            # k_paths = shortest_simple_paths(self.graph, source=start_region, target=goal_region, weight="weight")
        except nx.NetworkXNoPath:
            return [], []
        # Get top-k suboptimal paths
        paths = list(itertools.islice(k_paths, n))

        path_coords_list = []
        for path in paths:
            path_coords = []
            # for region in path:

            #     point_index = np.where(self.voronoi.point_region == region)[0][0]
            #     region_coords = self.voronoi.points[point_index]
            #     path_coords.append([region_coords[0], region_coords[1]])
            # path_coords_list.append(path_coords)


            path_coords = []
            for region_ind in path:
                region = self.regions[region_ind]

                if region is None:
                    raise ValueError("Region is None")
                
                region_coords = region.centroid

                path_coords.append([region_coords[0], region_coords[1]])

            first_coord = path_coords[0]
            last_coord = path_coords[-1]

            # if first_coord != source then add source
            if first_coord != source:
                path_coords.insert(0, source)
            # if last_coord != target then add target
            if last_coord != target:
                path_coords.append(target)

            path_coords_list.append(path_coords)


        return path_coords_list, paths


    def compute_nx_path(self, source, target):

        # Find the Voronoi regions containing the start and goal points
        start_region = self.find_region_containing_point(source)
        goal_region = self.find_region_containing_point(target)

        # print("Start Region:", start_region)
        # print("Goal Region:", goal_region)

        attributes_start  = self.regions[start_region].attributes
        attributes_goal  = self.regions[goal_region].attributes
        # print("Start Region Attributes:", attributes_start)
        # print("Goal Region Attributes:", attributes_goal)

        try:
            region_path = nx.shortest_path(self.graph, source=start_region, target=goal_region, weight='weight')
            path_cost = nx.shortest_path_length(self.graph, source=start_region, target=goal_region, weight='weight')

        except Exception as e:
            print("No path found between the start and goal regions.")
            return [], []
        # print("Region Path:", region_path)
        path_coords = []
        for region_ind in region_path:
            region = self.regions[region_ind]

            if region is None:
                raise ValueError("Region is None")
            
            region_coords = region.centroid

            path_coords.append([region_coords[0], region_coords[1]])

        first_coord = path_coords[0]
        last_coord = path_coords[-1]

        # if first_coord != source then add source
        if first_coord != source:
            path_coords.insert(0, source)
        # if last_coord != target then add target
        if last_coord != target:
            path_coords.append(target)

        # print("Path Coords:", path_coords)
        return path_coords, (region_path, path_cost)



    def get_neighboring_regions(self, region_path, N, include_self=False):
        n_hop_set = set()
        for node in region_path:
            if node not in self.graph:
                continue
            lengths = nx.single_source_shortest_path_length(self.graph, source=node, cutoff=N)
            for n, dist in lengths.items():
                if dist == N:
                    n_hop_set.add(n)
        if not include_self:
            n_hop_set -= set(region_path)
        return list(n_hop_set)



    def build_graph(self):
        # Compute convex sets based on the voronoi regions
        model = self.model

        center_graph = nx.DiGraph()

        for i in self.regions:
            current_region = self.regions[i]

            centroid = current_region.centroid
            if len(centroid) != 2:
                print("=============Centroid:", centroid)
                raise ValueError("Centroid is not 2D")
            center_graph.add_node(i, region=current_region, pos=centroid)

        print("Graph Build --- Neighbors:", len(self.neighbors), len(self.polygons))

        edges_map = {}

        for i in self.regions:

            neighbor_indices = self.neighbors[i]
            for j in neighbor_indices:
                if i == j:
                    continue
                region_i = self.regions[i]
                region_j = self.regions[j]

                p1 = np.floor(region_i.centroid).astype(int)
                p2 = np.floor(region_j.centroid).astype(int)

                # print("Region i:", region_i, "Region j:", region_j)
                # print("P1:", p1, "P2:", p2)

                # cost_ji = model.compute_segment_cost(p2, p1)
                # cost_ij = model.compute_segment_cost(p1, p2)

                cost_ij = model.compute_segment_cost_step(p1, p2)
                cost_ji = model.compute_segment_cost_step(p2, p1)



                if (i, j)  in edges_map:
                    continue

                edges_map[(i, j)] = cost_ij
                edges_map[(j, i)] = cost_ji

                center_graph.add_edge(i, j, weight=cost_ij)
                center_graph.add_edge(j, i, weight=cost_ji)

        print("Region Graph Edges:", len(center_graph.edges))
        self.graph = center_graph

        

    def get_polygon_coords(self, region_idx):
        vor = self.voronoi

        region = vor.regions[region_idx]
        # print("Region:", vor.regions, region_idx)
        if -1 in region or len(region) == 0:
            return None
        polygon_coords = [vor.vertices[i] for i in region]

        polygon_coords_xy = polygon_coords
        polygon = ShapelyPolygon(polygon_coords_xy)
        return  polygon    






class GraphRegionBuilder(GraphRegionBuilderBase):
    def __init__(self, rb: RegionBuilder, model: AgentModelObjective):
        super().__init__(rb, model)
        
        # self.model = model
        # self.graph = None 
        # self.graph_building_time =0.0

        self.voronoi = rb.voronoi
        self.regions = rb.regions
        self.all_output = rb.all_output
        self.use_ridge_points = True # True #use_ridge_points



    def find_region_containing_point(self, xy):
        """
        Given a point (x, y) and a scipy.spatial.Voronoi object,
        return the index of the region containing the point.
        """
        vor = self.voronoi
        tree = cKDTree(vor.points)
        _, i = tree.query(xy)
        region_index = vor.point_region[i]

        return region_index




    def compute_nx_n_paths(self, source, target, n=5, T=10):
        """
        Given source and target points, compute n suboptimal paths using GCS.
        """

        # Find the Voronoi regions containing the start and goal points
        start_region = self.find_region_containing_point(source)
        goal_region = self.find_region_containing_point(target)


        from utils import k_shortest_paths, k_diverse_paths

        import itertools

        try:

            # _, k_paths = k_shortest_paths(self.graph, source=start_region, target=goal_region, weight="weight", k=n, T=T)

            _ , k_paths = k_diverse_paths(self.graph, source=start_region, target=goal_region, weight="weight", k=n, T=T, min_pct_increase=1)

            # G: weighted graph, u: start, v: goal
            # k_paths = shortest_simple_paths(self.graph, source=start_region, target=goal_region, weight="weight")
        except nx.NetworkXNoPath:
            return [], []
        # Get top-k suboptimal paths
        paths = list(itertools.islice(k_paths, n))

        path_coords_list = []
        for path in paths:
            path_coords = []
            for region in path:

                point_index = np.where(self.voronoi.point_region == region)[0][0]
                region_coords = self.voronoi.points[point_index]
                path_coords.append([region_coords[0], region_coords[1]])
            path_coords_list.append(path_coords)
        return path_coords_list, paths


    def compute_nx_path(self, source, target):

        # Find the Voronoi regions containing the start and goal points
        start_region = self.find_region_containing_point(source)
        goal_region = self.find_region_containing_point(target)

        # print("Start Region:", start_region)
        # print("Goal Region:", goal_region)

        attributes_start  = self.regions[start_region].attributes
        attributes_goal  = self.regions[goal_region].attributes
        # print("Start Region Attributes:", attributes_start)
        # print("Goal Region Attributes:", attributes_goal)

        try:
            region_path = nx.shortest_path(self.graph, source=start_region, target=goal_region, weight='weight')
            path_cost = nx.shortest_path_length(self.graph, source=start_region, target=goal_region, weight='weight')

        except Exception as e:
            print("No path found between the start and goal regions.")
            return [], []
        path_coords = []
        for region in region_path:

            point_index = np.where(self.voronoi.point_region == region)[0][0]
            region_coords = self.voronoi.points[point_index]
            path_coords.append([region_coords[0], region_coords[1]])

        first_coord = path_coords[0]
        last_coord = path_coords[-1]

        # if first_coord != source then add source
        if first_coord != source:
            path_coords.insert(0, source)
        # if last_coord != target then add target
        if last_coord != target:
            path_coords.append(target)
        return path_coords, (region_path, path_cost)


    def get_neighboring_regions(self, region_path, N, include_self=False):
        n_hop_set = set()
        for node in region_path:
            if node not in self.graph:
                continue
            lengths = nx.single_source_shortest_path_length(self.graph, source=node, cutoff=N)
            for n, dist in lengths.items():
                if dist == N:
                    n_hop_set.add(n)
        if not include_self:
            n_hop_set -= set(region_path)
        return list(n_hop_set)




    def get_region_point_and_landcover(self, region_idx, elevation_type='max'):
        """
        Given a region index, return the centroid and landcover type.
        """
        region = self.regions[region_idx]
        if region is None:
            return None, None
        centroid = region.centroid
        landcover = region.attributes['landcover']

        #{'mean_elevation': 316.0, 'min_elevation': (316, array([3000, 7800])), 'max_elevation': (316, array([3000, 7800])), 'landcover': 5, 'max_distance': 0.0, 'max_z_distance': 0.0, 'region_size': 1, 'grade': 0.0, 'direction': 180.0}

        #min point elevation
        if elevation_type == 'min':
            elevation, _ = region.attributes['min_elevation']
        elif elevation_type == 'max':
            elevation, _ = region.attributes['max_elevation']
        else:
            elevation = region.attributes['mean_elevation']
            # If elevation_type is not specified, use the mean elevation
            # elevation = region.attributes['mean_elevation']
            # If elevation_type is not specified, use the mean elevation

        # mean elevation
        # elevation = region.attributes['mean_elevation']
        point3d = np.array([centroid[0], centroid[1], elevation])

        return point3d, landcover
    
    def get_raster_points_elevation_landcover_between_regions(self, region_i, region_j):

        region1 = self.regions[region_i]
        region2 = self.regions[region_j]
        if region1 is None or region2 is None:
            return []

        pt13d, lc1 = self.get_region_point_and_landcover(region_i, elevation_type='max')
        pt23d, lc2 = self.get_region_point_and_landcover(region_j, elevation_type='max')
        pt1 = np.array(pt13d)[:2]
        pt2 = np.array(pt23d)[:2]
        row1, col1 = np.round(pt1).astype(int)
        row2, col2 = np.round(pt2).astype(int)


        from skimage.draw import line

        def get_line_pixels(p1, p2):
            """
            p1, p2: tuples of (row, col) or (y, x)
            Returns list of pixel coordinates along the line.
            """
            rr, cc = line(p1[0], p1[1], p2[0], p2[1])
            # rr, cc, val = line_aa(p1[0], p1[1], p2[0], p2[1])

            return list(zip(rr, cc))
        
        pixels = get_line_pixels((row1, col1), (row2, col2))

        # print("Pixels between regions {} and {}: {}".format(region_i, region_j, len(pixels)))
    
        all_pts = []
        for i in range(len(pixels)):
            x1, y1 = pixels[i]
            res = region1.get_z_value_and_lc(x1, y1)
            if res is None:
                res = region2.get_z_value_and_lc(x1, y1)
            if res is None:
                return []
                raise ValueError("No elevation data found for point ({}, {})".format(x1, y1))
            
            all_pts.append((x1, y1, res[0], res[1]))  # (x, y, elevation, landcover)

        return all_pts




    def build_graph(self):

        from collections import defaultdict
        import itertools

        import time
        start_time = time.time()
        # Compute convex sets based on the voronoi regions
        model = self.model
        vor = self.voronoi

        center_graph = nx.DiGraph()

        vert_list= []

        for i in self.regions:
            current_region = self.regions[i]

            if current_region is None:
                continue

            verts_i = vor.regions[i]
            if -1 in verts_i:
                continue

            center_graph.add_node(i, region=current_region)


        # self.use_ridge_points = False
        if self.use_ridge_points:
            shared_vertex_pairs = vor.ridge_points
        else:
            # Build vertex-to-region map
            vertex_to_regions = defaultdict(set)
            for pt_idx, reg_idx in enumerate(vor.point_region):
                region = vor.regions[reg_idx]
                if not region or -1 in region:
                    continue
                for v in region:
                    vertex_to_regions[v].add(pt_idx)

            # Get region pairs sharing a vertex
            shared_vertex_pairs = set()
            for regions in vertex_to_regions.values():
                for a, b in itertools.combinations(sorted(regions), 2):
                    shared_vertex_pairs.add((a, b))

        # if self.use_ridge_points:
        #     for i, j in vor.ridge_points:
        #         shared_vertex_pairs.add((i, j))
        #         shared_vertex_pairs.add((j, i))




        # for i, j in vor.ridge_points:

        for i, j in shared_vertex_pairs:

            region_i = vor.point_region[i]
            region_j = vor.point_region[j]

            if region_i not in center_graph.nodes or region_j not in center_graph.nodes:
                continue

            verts_i = vor.regions[region_i]
            verts_j = vor.regions[region_j]

            # Skip infinite regions
            if -1 in verts_i or -1 in verts_j:
                continue

            if False:

                raster_points = self.get_raster_points_elevation_landcover_between_regions(region_i, region_j)

                if len(raster_points) == 0:
                    continue
                    raise ValueError("No raster points found between regions {} and {}".format(region_i, region_j))
                
                cost_ij = 0.0
                cost_ji = 0.0
                for i in range(len(raster_points)-1):
                    pt1_lc = raster_points[i]
                    pt2_lc = raster_points[i+1]


                    pt3d_i, lc_i = pt1_lc[:3], pt1_lc[3]
                    pt3d_j, lc_j = pt2_lc[:3], pt2_lc[3]

                    cost_ij += model.compute_cost(pt3d_i, pt3d_j, lc_i)
                    cost_ji += model.compute_cost(pt3d_j, pt3d_i, lc_j)

                center_graph.add_edge(region_i, region_j, weight=cost_ij)
                center_graph.add_edge(region_j, region_i, weight=cost_ji)


                # pt3d_i, lc_i = self.get_region_point_and_landcover(region_i, elevation_type='max')
                # pt3d_j, lc_j = self.get_region_point_and_landcover(region_j, elevation_type='max')


                # cost_ij = model.compute_cost(pt3d_i, pt3d_j, lc_i)
                # cost_ji = model.compute_cost(pt3d_j, pt3d_i, lc_j)

            use_landcover_points = False
            if use_landcover_points:


                p1 = np.round(vor.points[i]).astype(int)
                p2 = np.round(vor.points[j]).astype(int)


                best_ij = model.compute_segment_cost_step(p1, p2)#, check_obstacle=True)
                best_ji = model.compute_segment_cost_step(p2, p1)#, check_obstacle=True)


                region_i_lc_points = self.regions[region_i].attributes['landcover_points']
                region_j_lc_points =  self.regions[region_j].attributes['landcover_points']


                # print("region_i_lc_points", region_i_lc_points)
                # print("region_j_lc_points", region_j_lc_points)
                for lc_i, lc_j in itertools.product(region_i_lc_points, region_j_lc_points):
                    # print("lc_i:", lc_i, "lc_j:", lc_j)
                    # print("region_i_lc_points[lc_i]:", region_i_lc_points[lc_i], "region_j_lc_points[lc_j]:", region_j_lc_points[lc_j])
                    # p1 = np.round(region_i_lc_points[lc_i][:2]).astype(int)
                    # p2 = np.round(region_j_lc_points[lc_j][:2]).astype(int)

                    cost_ij = model.compute_segment_cost_step(p1, p2)#, check_obstacle=True)
                    cost_ji = model.compute_segment_cost_step(p2, p1)#, check_obstacle=True)

                    # cost_ij = model.compute_cost(p1, p2, lc_i)
                    # cost_ji = model.compute_cost(p2, p1, lc_j)

                    if cost_ij < best_ij:
                        best_ij = cost_ij
                    if cost_ji < best_ji:
                        best_ji = cost_ji

                cost_ij = best_ij
                cost_ji = best_ji

                print(" - Cost: {} | {}".format( cost_ij, cost_ji))

            else:
                p1 = np.round(vor.points[i]).astype(int)
                p2 = np.round(vor.points[j]).astype(int)

                # print("vor.points[i]:", vor.points[i], "vor.points[j]:", vor.points[j])

                # cost_ij = model.compute_segment_cost(p1, p2)#, check_obstacle=True)
                # cost_ji = model.compute_segment_cost(p2, p1)

                cost_ij = model.compute_segment_cost_step(p1, p2)#, check_obstacle=True)
                cost_ji = model.compute_segment_cost_step(p2, p1)#, check_obstacle=True)

            center_graph.add_edge(region_i, region_j, weight=cost_ij)
            center_graph.add_edge(region_j, region_i, weight=cost_ji)


        self.graph_building_time = time.time() - start_time

        print("Graph Build - Region Graph Edges:", len(center_graph.edges))
        self.graph = center_graph




