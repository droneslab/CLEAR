
import os
import pickle



import numpy as np


from utils import *

from cost_models import *

from dataloader import load_data_region_count_older


from graphregion import GraphRegionBuilder, GraphPatchRegionBuilder, GraphRegionBuilderBase


import os
os.environ["MOSEKLM_LICENSE_FILE"] = "../mosek.lic"


def get_start_goal_coords(example):
    if example == 1:
        start_coord =   (100, 260) 
        goal_coord =  (100,10)
    elif example == 2:
        # start_coord =   (1836, 1077) 
        # goal_coord =  (1339, 435)

        start_coord = (2109, 1691)
        goal_coord = (1339, 435)
    elif example == 3:
        start_coord = (1073, 177)
        goal_coord = (1000, 1200)
    elif example == 4:
        # start_coord = (2500, 250)
        # goal_coord = (2000, 2000)

        start_coord = (2696, 1801)
        goal_coord = (2051, 60)


    return start_coord, goal_coord



import os
import pickle
import time
import numpy as np
from abc import ABC, abstractmethod

class BasePathPlanner(ABC):
    def __init__(self, example=1, region_count=60000, decomposition="voronoi", method="plane", cost_model=HumanModelObjective, base_path="data/data_patch", region_count_to_save=None):
        self.example = example
        self.region_count = region_count
        self.decomposition = decomposition
        self.method = method

        self.cost_model = cost_model

        self.output_file_root = base_path #"data/data_patch"
        os.makedirs(self.output_file_root, exist_ok=True)
        file_region_count = region_count
        if region_count_to_save is not None:
            file_region_count = region_count_to_save

        self.output_file = f"{self.output_file_root}/region_builder_ex{example}_rc{file_region_count}_{decomposition}_{method}.pkl"
        self.rb = None
        self.model = None
        self.convex_rb = None
        self.paths = {}

        self.output_graph_file = f"{self.output_file_root}/graph_builder_ex{example}_rc{file_region_count}_{decomposition}_{method}.pkl"

        self.use_new_loader = False




    def load_region_builder(self, regenerate=False, landcover_map=None, elevation_map=None, use_new_loader=False, min_area=1, flatness_ratio=0.7):

        
        if use_new_loader:
            min_area = self.region_count
        if regenerate:
            self.rb = load_data_region_count(example=self.example, region_count=self.region_count, decomposition=self.decomposition, landcover_data=landcover_map, elevation_data=elevation_map)

            self.rb.build_regions(decomposition=self.decomposition,
                                  elevation_abstraction_method=self.method,
                                  elevation_bins=5, max_planes=10, error_thresh=10.0, min_area=min_area, flatness_ratio=flatness_ratio) #, min_area=self.region_count
            return

        self.use_new_loader = use_new_loader
        if os.path.exists(self.output_file):
            with open(self.output_file, "rb") as f:
                self.rb = pickle.load(f)
        else:

            if use_new_loader:
                self.rb = load_data_region_count(example=self.example, region_count=self.region_count, decomposition=self.decomposition)
            else:
                self.rb = load_data_region_count_older(example=self.example, region_count=self.region_count, decomposition=self.decomposition, landcover_data=landcover_map, elevation_data=elevation_map)
            # self.rb = load_data_region_count(example=self.example, region_count=self.region_count, decomposition=self.decomposition)
                
            self.rb.build_regions(decomposition=self.decomposition,
                                  elevation_abstraction_method=self.method,
                                  elevation_bins=5, max_planes=10, error_thresh=10.0, min_area=min_area, flatness_ratio=flatness_ratio) #, min_area=self.region_count
            with open(self.output_file, "wb") as f:
                pickle.dump(self.rb, f)
            print(f"Output saved to {self.output_file}")
        print(f"Region Actual Count: {len(self.rb.regions)}")



    def load_graph(self, min_area=1, flatness_ratio=0.7):

        if os.path.exists(self.output_graph_file):
            with open(self.output_graph_file, "rb") as f:
                self.convex_rb = pickle.load(f)

            self.rb = self.convex_rb.rb
            print(f"Graph loaded from {self.output_graph_file}")
        else:
            print(f"Graph file {self.output_graph_file} does not exist. Building graph.")
            self._build_model_and_graph(min_area=min_area, flatness_ratio=flatness_ratio)
            with open(self.output_graph_file, "wb") as f:
                pickle.dump(self.convex_rb, f)
            print(f"Graph saved to {self.output_graph_file}")




    @abstractmethod
    def _build_model_and_graph(self, min_area=1, flatness_ratio=0.7):
        pass

    @abstractmethod
    def compute_path(self, start_coord, goal_coord):
        pass

    @abstractmethod
    def run(self, plot=False, use_new_loader=False, use_step_cost=True, min_area=1, flatness_ratio=0.7):
        pass

    @abstractmethod
    def refine_path(self, region_path, num_samples=20, step_size=2.0):
        pass

    @abstractmethod
    def run_with_multiple_goals(self, start_coords, goal_coords, use_step_cost=True):
        pass




from tqdm import tqdm

class GraphPathPlanningPipeline(BasePathPlanner):
    def __init__(self, example=1, region_count=60000, decomposition="voronoi", method="plane", cost_model=HumanModelObjective, base_path="data/data_patch", region_count_to_save=None):
        super().__init__(example, region_count, decomposition, method, cost_model, base_path, region_count_to_save)
        self.graph_building_time = 0
        # self.convex_rb = None



    def _build_model_and_graph(self, min_area=1, flatness_ratio=0.7):

        if self.rb is None:
            self.load_region_builder(use_new_loader=False, min_area=min_area, flatness_ratio=flatness_ratio)

        gmap = GMap(self.rb.elevation_map, self.rb.landcover_map)
        self.model = self.cost_model(gmap)

        # if self.use_new_loader:
        #     self.convex_rb = GraphPatchRegionBuilder(self.rb, self.model)
        # else:
        #     use_ridge_points = self.decomposition != "voronoi"
        #     self.convex_rb = GraphRegionBuilder(self.rb, self.model, use_ridge_points=use_ridge_points)

        # print(GraphPatchRegionBuilder)

        self.convex_rb = GraphRegionBuilder(self.rb, self.model)

        st = time.time()
        self.convex_rb.build_graph()
        self.graph_building_time = time.time() - st

    def compute_nx_path(self, start_coord, goal_coord, use_step_cost=True):
        st = time.time()
        path_nx, region_path = self.convex_rb.compute_nx_path(start_coord, goal_coord)
        graph_path_time = time.time() - st

        cost = self.convex_rb.model.compute_path_cost(path_nx, use_step_cost=use_step_cost)

        print(f"Path: {len(path_nx)}, Cost: {cost}, Time: {graph_path_time:.2f}s")
        # total_time = self.graph_building_time + graph_path_time
        total_time = graph_path_time
        self.paths["Graph Path"] = (np.array(path_nx), cost, total_time)
        return region_path

    def compute_nx_n_paths(self, start_coord, goal_coord, n=5, T=10):
        st = time.time()
        path_nx_list, region_paths = self.convex_rb.compute_nx_n_paths(start_coord, goal_coord, n=n, T=T)
        graph_path_time = time.time() - st


        for i, path_nx in enumerate(path_nx_list):
            cost = self.model.compute_path_cost(path_nx, use_step_cost=True)

            print(f"Path: {len(path_nx)}, Cost: {cost}, Time: {graph_path_time:.2f}s")
            # total_time = self.graph_building_time + graph_path_time
            total_time = graph_path_time
            self.paths["Graph Path_"+str(i)] = (np.array(path_nx), cost, total_time)
        return region_paths
    
    def compute_path(self, start_coord, goal_coord, use_step_cost=True):
        region_path = self.compute_nx_path(start_coord, goal_coord, use_step_cost=use_step_cost)
        if False:
            length_of_path = len(region_path)
            percent_path_block = 0.7
            num_block = int(length_of_path * percent_path_block)
            region_path = self.compute_nx_n_paths(start_coord, goal_coord, n=2, T=num_block)
        
        return region_path


    def refine_path(self, region_path, num_samples=20, step_size=2.0):
        from shapely.geometry import Point, Polygon, LineString
        import networkx as nx

        def closest_node(G: nx.Graph, query_pt: Point, pos_attr='pos'):
            # pts = [Point(G.nodes[n][pos_attr]) for n in G.nodes]
            # closest_pt= min(pts, key=lambda pt: query_pt.distance(pt))
            # # closest_n =  min(
            # #     G.nodes,
            # #     key=lambda n: query_pt.distance(Point(G.nodes[n][pos_attr]))
            # # )
            # closest_n = pts.index(closest_pt)

            import heapq
            from shapely.geometry import Point

            def get_k_closest_nodes(G, query_pt, pos_attr='pos', k=4):
                query_pt_mid = Point(query_pt.x, query_pt.y)
                
                return heapq.nsmallest(
                    k,
                    G.nodes,
                    key=lambda n: query_pt_mid.distance(Point(G.nodes[n][pos_attr]))
                )
            

            closest_nodes = get_k_closest_nodes(G, query_pt, pos_attr=pos_attr, k=4)

            for closest in closest_nodes:
                reg = region_graph.nodes[closest]['region']
                if reg.get_z_value_and_lc(query_pt.x, query_pt.y):
                    return closest

            raise ValueError(f"No valid region found for point {query_pt} in the graph.")

            # query_pt_mid = Point(query_pt.x+0.5, query_pt.y+0.5)


            query_pt_mid = Point(query_pt.x, query_pt.y)

            closest_n =  min(
                G.nodes,
                key=lambda n: query_pt_mid.distance(Point(G.nodes[n][pos_attr]))
            )


            return closest_n

        region_graph = self.convex_rb.graph
        path = region_path

        # def rewire_path_by_node_replacement(path, region_graph, num_samples=20, step_size=2.0):
        def get_z_value(x, y):
            # print(f"Getting z value for point ({x}, {y})")
            closest = closest_node(region_graph, Point(x, y), pos_attr='pos')
            if closest is None:
                print(f"Closest node not found for point ({x}, {y})")
                return None
            reg = region_graph.nodes[closest]['region']
            # print(f"Closest node: {closest}, Region: {reg}")



            return reg.get_z_value_and_lc(x, y)


        def sample_between(p0, p1):
            closest = closest_node(region_graph, Point(p0[0], p0[1]), pos_attr='pos')
            reg1_pts = region_graph.nodes[closest]['region'].polygon_pts
            closest = closest_node(region_graph, Point(p1[0], p1[1]), pos_attr='pos')
            reg2_pts = region_graph.nodes[closest]['region'].polygon_pts

            def sample_from_two_polygons(poly1_pts, poly2_pts):
                import random

                # Randomly pick one of the two polygons
                chosen_pts = random.choice([poly1_pts, poly2_pts])
                
                if len(chosen_pts) < 4:
                    return Point(random.choice(chosen_pts))
                
                polygon = Polygon(chosen_pts)
                if not polygon.is_valid or polygon.area == 0:
                    return Point(random.choice(chosen_pts))
                
                # Uniform sampling inside polygon
                minx, miny, maxx, maxy = map(int, polygon.bounds)
                
                for _ in range(1000):
                    x = random.randint(minx, maxx)
                    y = random.randint(miny, maxy)
                    pt = Point(x, y)
                    if polygon.contains(pt):
                        return pt
                    
                # Fallback if all else fails
                idx = random.randint(0, chosen_pts.shape[0] - 1)
                return Point(tuple(chosen_pts[idx]))

                print(poly1_pts, poly2_pts)
                raise RuntimeError("Failed to sample point inside selected polygon.")



            for _ in range(num_samples):
                pt = sample_from_two_polygons(reg1_pts, reg2_pts)
                tx, ty = pt.x, pt.y

                # tx = np.random.uniform(p0[0], p1[0])
                # ty = np.random.uniform(p0[1], p1[1])
                tz_lc = get_z_value(tx, ty)
                if tz_lc is None:
                    continue
                tz, lc = tz_lc
                p = (tx, ty, tz)
                # print(f"Sampled point: {p}, Landcover: {lc}")
                # print(f"p0: {p0}, p1: {p1}")
                p_2d = (tx, ty)
                if (
                    np.linalg.norm(np.subtract(p_2d, p0)) < step_size and
                    np.linalg.norm(np.subtract(p_2d, p1)) < step_size
                ):
                    yield (p, lc)



        from concurrent.futures import ThreadPoolExecutor
        from tqdm import tqdm

        def refine_segment(i, path, get_z_value, model):
            p_prev, p_curr, p_next = path[i - 1], path[i], path[i + 1]
            prev_z, prev_lc = get_z_value(p_prev[0], p_prev[1])
            curr_z, curr_lc = get_z_value(p_curr[0], p_curr[1])
            next_z, next_lc = get_z_value(p_next[0], p_next[1])
            
            pt_prev = (p_prev[0], p_prev[1], prev_z)
            pt_curr = (p_curr[0], p_curr[1], curr_z)
            pt_next = (p_next[0], p_next[1], next_z)
            
            original_cost = model.compute_cost(pt_prev, pt_curr, prev_lc) + \
                            model.compute_cost(pt_curr, pt_next, curr_lc)

            best_point = p_curr
            best_cost = original_cost

            for p_sample in sample_between(p_prev, p_next):
                pt_sample, p_sample_lc = p_sample
                new_cost = model.compute_cost(pt_prev, pt_sample, prev_lc) + \
                        model.compute_cost(pt_sample, pt_next, p_sample_lc)

                if new_cost + 1e-3 < best_cost:
                    best_cost = new_cost
                    best_point = p_sample

            return i, best_point

        def refine_path_parallel(path, get_z_value, model):
            refined = [path[0]]
            futures = []

            with ThreadPoolExecutor() as executor:
                for i in range(1, len(path) - 1):
                    futures.append(executor.submit(refine_segment, i, path, get_z_value, model))

                results = [None] * (len(path) - 2)
                for f in tqdm(futures, desc="Refining path segments", unit="segment"):
                    i, pt = f.result()
                    results[i - 1] = pt  # offset by 1 since index i started at 1

            refined.extend(results)
            refined.append(path[-1])
            return refined

        return refine_path_parallel(region_path, get_z_value, self.model)



    def run(self, plot=False, use_new_loader=False, use_step_cost=True, min_area=1, flatness_ratio=0.7):
        print(f"Example: {self.example}\nRegion Count: {self.region_count}\nDecomposition: {self.decomposition}\nMethod: {self.method}")

        self.load_graph(min_area=min_area, flatness_ratio=flatness_ratio)

        #NOTE: Revert after debugging as quadtree processing is stoped by this
        # if use_new_loader:
        #     print("Using new loader, skipping path computation.")
        #     return

        start_coord, goal_coord = get_start_goal_coords(self.example)

        region_path = self.compute_path(start_coord, goal_coord, use_step_cost=use_step_cost)
        # self.compute_gcs_path(start_coord, goal_coord, region_path)
        if plot:
            output_file = f"results/graph_path_ex{self.example}_rc{self.region_count}_{self.decomposition}_{self.method}.png"
            plot_gcs_path(self.convex_rb.voronoi, self.convex_rb.graph, self.paths,
                        self.convex_rb.landcover_map, output_file=output_file)


    def run_with_multiple_goals(self, start_coords, goal_coords, use_step_cost=True):
        results = {}
        path_index = 0
        for start_coord, goal_coord in zip(start_coords, goal_coords):
            print(f"Start: {start_coord}, Goal: {goal_coord}")

            st = time.time()
            path_nx, region_path = self.convex_rb.compute_nx_path(start_coord, goal_coord)

            path_smooth = shortcut_smooth(path_nx, energy_fn=self.convex_rb.model.compute_path_cost, max_window=4, min_window=2, use_step_cost=use_step_cost)

            graph_path_time = time.time() - st

            cost = self.convex_rb.model.compute_path_cost(path_nx, use_step_cost=use_step_cost)
            best_cost = self.convex_rb.model.compute_path_cost(path_smooth, use_step_cost=use_step_cost)

            print(f"Path: {len(path_nx)}, Cost: {cost}, Time: {graph_path_time:.2f}s")
            results["Path_"+str(path_index)] = (np.array(path_nx), cost, graph_path_time)

            print(f"Path: {len(path_smooth)}, Cost: {best_cost}, Time: {graph_path_time:.2f}s")
            results["Path_smooth_"+str(path_index)] = (np.array(path_smooth), best_cost, graph_path_time)

            path_index+=1
        return results



class GCSPathPlanningPipeline(BasePathPlanner):
    def _build_model_and_graph(self, min_area=1, flatness_ratio=0.7):
        from convexregion import ConvexRegionBuilder

        gmap = GMap(self.rb.elevation_map, self.rb.landcover_map)
        self.model = self.cost_model(gmap)
        use_ridge_points = self.decomposition != "voronoi"

        self.convex_rb = ConvexRegionBuilder(self.rb, self.model, use_ridge_points=use_ridge_points)

        print("Region Actual Count: ", len(self.convex_rb.regions))
        st = time.time()
        self.convex_rb.build_graph()
        self.graph_building_time = time.time() - st

    def refine_path(self, region_path, num_samples=20, step_size=2.0):
        raise NotImplementedError("Refinement not implemented for GCSPathPlanningPipeline.")


    def compute_nx_path(self, start_coord, goal_coord):
        st = time.time()
        path_nx, region_path = self.convex_rb.compute_nx_path(start_coord, goal_coord)
        graph_path_time = time.time() - st

        cost = self.model.compute_path_cost(path_nx, use_step_cost=True)
        total_time = self.graph_building_time + graph_path_time
        # self.paths["Graph Path"] = (np.array(path_nx), cost, total_time)
        return region_path

    def compute_gcs_path(self, start_coord, goal_coord, region_path):
        st = time.time()
        self.convex_rb.build_gcs_from_graph(region_path=region_path)
        gcs_building_time = time.time() - st

        st = time.time()
        path = self.convex_rb.compute_path(start_coord, goal_coord, convex_relaxation=False)
        gcs_path_time = time.time() - st

        cost = self.model.compute_path_cost(path, use_step_cost=True)
        total_time = self.graph_building_time + gcs_building_time + gcs_path_time
        self.paths["GCS Path"] = (path, cost, total_time)
        

    

    def compute_path(self, start_coord, goal_coord):
        region_path = self.compute_nx_path(start_coord, goal_coord)

        # print(f"Region Path: {region_path}")
        #Pranay: N>0 can be used to include neighboring regions
        region_path = self.convex_rb.get_neighboring_regions(region_path, N=0, include_self=True)
        # print(f"Region Path: {region_path}")
        self.compute_gcs_path(start_coord, goal_coord, region_path=region_path)
        return region_path

    def run(self, plot=False, use_new_loader=False, use_step_cost=True, min_area=1, flatness_ratio=0.7):
        print(f"Example: {self.example}\nRegion Count: {self.region_count}\nDecomposition: {self.decomposition}\nMethod: {self.method}")

        self.load_graph(min_area=min_area, flatness_ratio=flatness_ratio)
        start_coord, goal_coord = get_start_goal_coords(self.example)
        self.compute_path(start_coord, goal_coord)
        # region_path = self.compute_path(start_coord, goal_coord)
        # self.compute_gcs_path(start_coord, goal_coord, region_path)
        if plot:
            output_file = f"results/gcs_path_ex{self.example}_rc{self.region_count}_{self.decomposition}_{self.method}.png"
            plot_gcs_path(self.convex_rb.voronoi, self.convex_rb.graph, self.paths,
                        self.convex_rb.landcover_map, output_file=output_file)
            
    




def shortcut_smooth(path_in, energy_fn, max_window=10, min_window=2, use_step_cost=False):
    # return path  # No smoothing applied, return original path
    # use_step_cost= True
    path = list(path_in)
    N = len(path)
    i = 0

    while i < len(path) - min_window:
        improved = False
        for w in range(max_window, min_window - 1, -1):
            j = i + w
            if j >= len(path):
                continue

            segment = path[i:j+1]
            straight = [path[i], path[j]]

            e_orig = energy_fn(segment, use_step_cost=use_step_cost)
            e_straight = energy_fn(straight, use_step_cost=use_step_cost)

            if e_straight < e_orig:
                path = path[:i+1] + path[j:]
                improved = True
                break  # re-check from same i

        if not improved:
            i += 1
    return path




class RegionLoaderWarpper():
    def __init__(self, example, region_count, method="plane", decomposition_list=["quadtree", "voronoi", "grid", "hex"], is_ablation=False):
        self.example = example
        self.region_count = region_count
        self.method = method

        self.decomposition_list = decomposition_list

        self.is_ablation = is_ablation
        
        # self.rb_cache_path_root = "./data/data_patch"  #REVERT to this

        if is_ablation:

            self.rb_cache_path_root = "./data/data_patch_ablation"
        else:
            self.rb_cache_path_root = "./data/data_patch"

        self.result_cache_path = "./data/results/planner"


        region_count_current = region_count
        if not os.path.exists(self.rb_cache_path_root):
            os.makedirs(self.rb_cache_path_root)


        example_to_map_name = {
            1: "Wharton",
            3: "Humphreys",
            4: "Mount Rainier",
            5: "Twenty Nine Palms",
            }
        self.map_name = example_to_map_name[self.example]


        self.decomposition_to_planner = {}



    def run_planner(self, planner_class="Graph", save_path=True, cost_model_fn=VehicleObjective, min_area=1, flatness_ratio=0.7):


        all_paths = {}
        last_convex_rb = None

        decomp_map = {
            "grid": "Grid",
            "hex": "Hex",
            "voronoi": "Boundary",
            "quadtree": "Quadtree",
            # Add more decompositions as needed
        }

        if planner_class == "GCS":
            PlannerCls = GCSPathPlanningPipeline
        elif planner_class == "Graph":
            PlannerCls = GraphPathPlanningPipeline

        decompose_to_rb = {}#decomp: None for decomp in self.decomposition_list}
        decompose_to_planner = {}

        region_count = self.region_count
        example = self.example

        last_convex_rb =None


        region_count_current = region_count

        print("Running planner for decompositions:",self.decomposition_list)

        for decomposition in self.decomposition_list:
            print(f"Running planner for decomposition: {decomposition}")
            planner = PlannerCls(example=self.example, region_count=region_count_current, decomposition=decomposition, method=self.method, cost_model=cost_model_fn, base_path=self.rb_cache_path_root, region_count_to_save = region_count)

            planner.run(min_area=min_area, flatness_ratio=flatness_ratio)

            if region_count_current == region_count:
                region_count_current = len(planner.rb.regions)

            use_step_cost = True
            decompose_to_planner[decomp_map[decomposition]] = planner
            self.decomposition_to_planner [decomposition] = planner

            for k, (path, cost, time_o) in planner.paths.items():


                key = f"{decomp_map[decomposition]}_{k}"
                cost_new = planner.model.compute_path_cost(path, use_step_cost=use_step_cost)
                # all_paths[key] = (path, cost_new, time_o)

                key = f"{decomp_map[decomposition]}_{k}_smooth"
                # Smooth the path using the total energy function
                time_s = time.time()
                best_cost = cost_new
                best_smoothing = 0.0
                path_smooth = path


                path_smooth = shortcut_smooth(path, energy_fn=planner.model.compute_path_cost, max_window=4, min_window=2, use_step_cost=use_step_cost)
                best_cost = planner.model.compute_path_cost(path_smooth, use_step_cost=use_step_cost)
                time_s = time.time() - time_s

                print(f"Decomposition: {decomp_map[decomposition]}, Smoothing: {best_smoothing}, Cost: {best_cost}, Time: {time_s}")

                all_paths[key] = (path_smooth, best_cost, time_o+time_s)


                if save_path:
                    base_path = "results/planner/paths"
                    if not os.path.exists(base_path):
                        os.makedirs(base_path)
                    np.save(f"results/planner/paths/{PlannerCls.__name__}_{decomp_map[decomposition]}_ex{example}_rc{region_count}_{k}.npy", path)
                    np.save(f"results/planner/paths/{PlannerCls.__name__}_{decomp_map[decomposition]}_ex{example}_rc{region_count}_{k}_smooth.npy", path_smooth)

            
            decompose_to_rb[decomp_map[decomposition]] = planner.convex_rb
            last_convex_rb = planner.convex_rb  # store for plotting
        decomposition_list_str = [decomp_map[decomp] for decomp in decomposition_list]
        str_decomposition_list = "_".join(decomposition_list_str)

        output_file = f"results/planner/{PlannerCls.__name__}_{str_decomposition_list}_ex{example}_rc{region_count}.pdf"


        # load json and update the results 
        import json
        if self.is_ablation:
            json_file = f"results/planner/planning_results_ablation.json"
        else:
            json_file = f"results/planner/planning_results.json"


        print(f"Loading results from {json_file}")


        if not os.path.exists(json_file):
            results = {}
        else:
            with open(json_file, 'r') as f:
                results = json.load(f)


        for name_full, (path, cost, compute_time) in all_paths.items():
            name = name_full.split("_")[0]  # Get the decomposition name
            if name not in decompose_to_rb:
                print(f"{name} decomposition not found in decompose_to_rb")
                continue
            if decompose_to_rb[name] is None:
                print(f"{name} decomposition is None")
                continue 
            region_count_curr = len(decompose_to_rb[name].regions)
            print(f"{name}:  Cost: {cost}, Compute Time: {compute_time}, Path Length: {len(path)}, Region Count: {region_count_curr}")
                    
            rb_current = decompose_to_rb[name]
            all_output = rb_current.all_output

            
            total_decomposition_time = all_output["decomposition_time"] + all_output["elevation_abstraction_time"] + decompose_to_planner[name].graph_building_time
            print(f"Total Decomposition Time: {total_decomposition_time:.2f}s")

            path_length_euclidean = 0
            if len(path) > 1:
                for i in range(len(path) - 1):
                    path_length_euclidean += np.linalg.norm(np.array(path[i+1]) - np.array(path[i]))  
            else:
                path_length_euclidean = np.inf    
            # results should be [example][decomposition][region_count] = {"planner": planner_class
            # ""cost": cost, "compute_time": compute_time, "path_length": len(path), "path_length_euclidean": path_length_euclidean}
            map_name = self.map_name
            if map_name not in results:
                results[map_name] = {}
            if name not in results[map_name]:
                results[map_name][name] = {}
            if region_count_curr not in results[map_name][name]:
                results[map_name][name][region_count_curr] = {}
            results[map_name][name][region_count_curr][planner_class] = {
                "cost": cost,
                "compute_time": compute_time,
                "path_length": len(path),
                "path_length_euclidean": path_length_euclidean,
                "region_count": region_count_curr,
                "abstraction_time": total_decomposition_time,
                "map": self.map_name

            }
            print(f"Saving results to {json_file}")
            print(results[map_name][name][region_count_curr][planner_class])
            with open(json_file, 'w') as f:
                json.dump(results, f, indent=4)

        # if "Boundary" not in decompose_to_rb:
        #     boundary_convex_rb = last_convex_rb
        # else:
        #     boundary_convex_rb = decompose_to_rb["Boundary"]

        output_file = f"results/planner/{planner_class}_{cost_model_fn.__name__}_all_ex{example}_rc{region_count}.pdf"



        plot_gcs_path_elevation(
            last_convex_rb.voronoi, last_convex_rb.graph, all_paths,
            last_convex_rb.elevation_map, output_file=output_file.replace(".pdf", "_elevation.pdf")
        )




    def run_planner_to_coordinates(self, goal_coords, planner_class="Graph", cost_model_fn=VehicleObjective, save_path=True, use_step_cost=True, min_area=1, flatness_ratio=0.7):


        start_coords = goal_coords[:-1] # all but last coordinate
        end_coords = goal_coords[1:]  # all but first coordinate

        all_paths = {}

        decomp_map = {
            "grid": "Grid",
            "hex": "Hex",
            "voronoi": "Boundary",
            "quadtree": "Quadtree",
            # Add more decompositions as needed
        }

        if planner_class == "GCS":
            PlannerCls = GCSPathPlanningPipeline
        elif planner_class == "Graph":
            PlannerCls = GraphPathPlanningPipeline

        region_count = self.region_count
        example = self.example


        region_count_current = region_count



        print("Running planner for decompositions:",self.decomposition_list)

        for decomposition in self.decomposition_list:
            print(f"Running planner for decomposition: {decomposition}")
            planner = PlannerCls(example=self.example, region_count=region_count_current, decomposition=decomposition, method=self.method, cost_model=cost_model_fn, base_path=self.rb_cache_path_root, region_count_to_save = region_count)

            planner.load_graph(min_area=min_area, flatness_ratio=flatness_ratio)

            if region_count_current == region_count:
                region_count_current = len(planner.rb.regions)

            self.decomposition_to_planner[decomp_map[decomposition]] = planner

            for i in range(len(goal_coords)-1):
                start_coord = goal_coords[i]
                goal_coord = goal_coords[i+1]
                print(f"Start: {start_coord}, Goal: {goal_coord}")

            all_paths[decomp_map[decomposition]] = planner.run_with_multiple_goals(start_coords=start_coords, goal_coords=end_coords, use_step_cost=use_step_cost)
            

            if save_path:
                base_path = "results/planner/paths"
                if not os.path.exists(base_path):
                    os.makedirs(base_path)

                for k in all_paths[decomp_map[decomposition]]:
                    (path_smooth, best_cost, graph_path_time) = all_paths[decomp_map[decomposition]][k]
                    np.save(f"results/planner/paths/{PlannerCls.__name__}_{decomp_map[decomposition]}_ex{example}_rc{region_count}_{k}.npy", path_smooth)




        
        # for path_index in range(len(start_coords)):
        #     path_results = {}
        #     current_path_key = "Path_"+str(path_index)
        #     output_file = f"results/planner/{planner_class}_{cost_model_fn.__name__}_all_ex{example}_rc{region_count}_{current_path_key}.pdf"

        #     for decomposition in self.decomposition_list:
        #         path_results[decomp_map[decomposition]] = all_paths[decomp_map[decomposition]][current_path_key]
            
        #     last_planner = self.decomposition_to_planner["Boundary"]


        #     plot_gcs_path_elevation(
        #         last_planner.rb.voronoi, last_planner.convex_rb.graph, path_results,
        #         last_planner.rb.elevation_map, output_file=output_file.replace(".pdf", "_elevation.pdf")
        #     )



        return all_paths






example_to_goal_coords = {
    1: [[99, 262],
            [98, 9],
            [327, 93],
            [26, 192],
            [140, 22],
            [99, 260],
            [304, 106],
            [92, 11],
            [63, 202],
            [318, 100],
            # [59, 204]
            [59, 200]
        ],
    3: [[1408, 1319],
        [898, 99],
        [188, 113],
        [1440, 398],
        [1553, 1202],
        [1354, 239],
        [1237, 1459],
        [834, 375],
        [53, 131],
        [1630, 1541],
        [1594, 398]
        ],
    4: [
        [2404, 170],
        [693, 1400],
        [2255, 2015],
        [2029, 432],
        # [2489, 1902], #INF
        # [403, 1803], #INF
        [3090, 1860],
        [2178, 319],
        [495, 234],
        [778, 1839],
        [2305, 92],
        [608, 2397],
        [608, 403],
        ]
}




def ablation_path_different_regions_count():
    decomposition_list =["quadtree", "voronoi"]

    RegionLoaderWarpper(example=1, region_count=1, method="plane", decomposition_list=decomposition_list).run_planner()
    RegionLoaderWarpper(example=1, region_count=2, method="plane", decomposition_list=decomposition_list).run_planner()
    RegionLoaderWarpper(example=1, region_count=4, method="plane", decomposition_list=decomposition_list).run_planner()
    RegionLoaderWarpper(example=1, region_count=8, method="plane", decomposition_list=decomposition_list).run_planner()
    RegionLoaderWarpper(example=1, region_count=16, method="plane", decomposition_list=decomposition_list).run_planner()



def run_planner_default_path(decomposition_list =["quadtree", "voronoi", "grid", "hex"]):
    

    RegionLoaderWarpper(example=1, region_count=1, method="plane", decomposition_list=decomposition_list).run_planner()
    RegionLoaderWarpper(example=3, region_count=4, method="plane", decomposition_list=decomposition_list).run_planner()
    RegionLoaderWarpper(example=4, region_count=8, method="plane", decomposition_list=decomposition_list).run_planner()
    




if __name__ == "__main__":
    print("Testing Path Planning Pipelines")


    # ablation_path_different_regions_count()

    decomposition_list =["quadtree", "voronoi", "grid", "hex"]

    # run_planner_default_path(decomposition_list=decomposition_list)


    example_to_region_count = {
        1: 1,
        3: 4,
        4: 8,
    }

    # example_to_region_count = {
    #     1: 1,
    #     3: 8,
    #     4: 16,
    # }

    example_to_min_area = {
        1: 2,
        3: 4,
        4: 4,
    }


    # example = 4

    example_list = [1]# [1,3, 4]


    decomposition_list =["quadtree", "voronoi", "grid", "hex"]

    # for example in [1,3,4]:

    example_to_planner_wrapper = {}
    example_to_results = {}

    for example in example_list:
        region_count = example_to_region_count[example]

        reg_loader_wrapper = RegionLoaderWarpper(example=example, region_count=region_count, method="plane", decomposition_list=decomposition_list)

        goal_coords = example_to_goal_coords[example]


        goal_coords_forward = goal_coords
        goal_coords_reverse = goal_coords_forward[::-1]
        goal_coords_all = goal_coords_forward + goal_coords_reverse[1:]  # all but first coordinate of reverse

        min_area = example_to_min_area[example]

        all_results = reg_loader_wrapper.run_planner_to_coordinates(goal_coords_all, use_step_cost=True, min_area=min_area)


        # with open(f"./results/planner/all_results_{example}_20.txt", "w") as f:
        #     for decomp, paths in all_results.items():
        #         f.write(f"Decomposition: {decomp}\n")
        #         for path_name, (path, cost, time_complete) in paths.items():
        #             f.write(f"  {path_name}: Cost: {cost}, Time: {time_complete}, Path Length: {len(path)}\n")
        #         f.write("\n")


        example_to_planner_wrapper[example] = reg_loader_wrapper
        example_to_results[example] = all_results
        import json

        output = {}
        for decomp, paths in all_results.items():
            output[decomp] = {}
            for path_name, (path, cost, time_complete) in paths.items():

                if "_smooth" in path_name:
                    continue  # Skip smoothed paths for this output
                planner = reg_loader_wrapper.decomposition_to_planner[decomp]
                all_output = planner.convex_rb.all_output

                region_count = len(planner.convex_rb.regions)
                total_decomposition_time = all_output["decomposition_time"] + all_output["elevation_abstraction_time"] + planner.convex_rb.graph_building_time

                

                path_length_euclidean = 0
                if len(path) > 1:
                    for i in range(len(path) - 1):
                        path_length_euclidean += np.linalg.norm(np.array(path[i+1]) - np.array(path[i]))  
                else:
                    path_length_euclidean = np.inf    


                output[decomp][path_name] = {
                    "cost": cost,
                    "time": time_complete,
                    "path_length": len(path),
                    "path_length_euclidean": path_length_euclidean,
                    "abstraction_time": total_decomposition_time,
                    "region_count": region_count,
                }

        with open(f"./results/planner/all_results_{example}_20.json", "w") as f:
            json.dump(output, f, indent=2)


        import csv

        with open(f"./results/planner/all_results_{example}_20.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Decomposition", "Path Name", "Cost", "Time", "Path Length", "Path Length Euclidean", "Abstraction Time", "Region Count"])
            
            for decomp, paths in all_results.items():

                for path_name, (path, cost, time_complete) in paths.items():

                    if "_smooth" in path_name:
                        continue  # Skip smoothed paths for this output

                    planner = reg_loader_wrapper.decomposition_to_planner[decomp]
                    all_output = planner.convex_rb.all_output
                    total_decomposition_time = all_output["decomposition_time"] + all_output["elevation_abstraction_time"] + planner.convex_rb.graph_building_time

                    region_count = len(planner.convex_rb.regions)

                    path_length_euclidean = 0
                    if len(path) > 1:
                        for i in range(len(path) - 1):
                            path_length_euclidean += np.linalg.norm(np.array(path[i+1]) - np.array(path[i]))  
                    else:
                        path_length_euclidean = np.inf    

                    writer.writerow([decomp, path_name, cost, time_complete, len(path), path_length_euclidean, total_decomposition_time, region_count])

    exit(0)
    print("Generating plots for all examples...")

    decomp_map = {
        "grid": "Grid",
        "hex": "Hex",
        "voronoi": "Boundary",
        "quadtree": "Quadtree",
        # Add more decompositions as needed
    }
    cost_model_fn = VehicleObjective
    planner_class = "Graph"
    for example in example_list:

        reg_loader_wrapper = example_to_planner_wrapper[example]
        all_results = example_to_results[example]
        for path_index in range(20):


            path_results = {}
            current_path_key = "Path_"+str(path_index)
            output_file = f"results/planner/{planner_class}_{cost_model_fn.__name__}_all_ex{example}_{current_path_key}.pdf"

            for decomposition in reg_loader_wrapper.decomposition_list:
                path_results[decomp_map[decomposition]] = all_results[decomp_map[decomposition]][current_path_key]
            
            last_planner = reg_loader_wrapper.decomposition_to_planner["Boundary"]


            plot_gcs_path_elevation(
                last_planner.rb.voronoi, last_planner.convex_rb.graph, path_results,
                last_planner.rb.elevation_map, output_file=output_file.replace(".pdf", "_elevation.pdf")
            )




    # pipeline = GCSPathPlanningPipeline(example=1, region_count=60000, decomposition="grid", method="plane")
    # pipeline.run()

