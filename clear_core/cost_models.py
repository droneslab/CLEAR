

import math 
import cvxpy as cp

import numpy as np


class StaticEnergyCost:
    def __init__(self, Body=80, Load=5, Speed=1.4):
        self.BodyKG = Body
        self.LoadKG = Load
        self.SpeedMPS = Speed  # Meters/second

    def power_estimate(
        self, Tf, Grade
    ):  # Rise,Run): # Pandolf-Santee 5 parameter model
        # Grade=100*Rise/Run
        power = (
            (1.5 * self.BodyKG)
            + (2 * (self.BodyKG + self.LoadKG) * ((self.LoadKG / self.BodyKG) ** 2))
            + (
                Tf
                * (self.BodyKG + self.LoadKG)
                * ((1.5 * self.SpeedMPS**2) + 0.35 * self.SpeedMPS * Grade)
            )
            - (
                (Grade < 0)
                * Tf
                * (
                    (self.SpeedMPS * Grade * (self.BodyKG + self.LoadKG) / 3.5)
                    - ((self.BodyKG + self.LoadKG) * ((Grade + 6) ** 2) / self.BodyKG)
                    + 25 * self.SpeedMPS**2
                )
            )
        )
        return power / 4


class StaticPerformanceModel:

    def __init__(self, body=85, load=8, speed=1.8):
        self.model = StaticEnergyCost(Body=body, Load=load, Speed=speed)

    def cost(self, grade, type, length=1):
        return self.model.power_estimate(Tf=type, Grade=grade)



class AgentModelObjective():
    def __init__(self, gmap, obstacle_threshold=1000):
        self.map = gmap
        self.scale_xy = gmap.scale_xy
        self.obstacle_threshold = obstacle_threshold


    def compute_segment_cost(self, pt1, pt2, check_obstacle=False):
        raise NotImplementedError("This method should be implemented in subclasses.")

    def compute_segment_cost_step(self, pt1, pt2, step_size=1, check_obstacle=False):
        raise NotImplementedError("This method should be implemented in subclasses.")



    def compute_path_cost(self, path, use_step_cost=False):
        """
        Computes the total cost of a given path by summing the costs of each segment.

        Parameters:
        path: List of tuples representing the path segments.
        Each tuple should contain two points (pt1, pt2).
        returns:
        - Total accumulated cost along the path.
        """
        total_cost = 0
        # print("Computing path cost for:", path)
        for pt1, pt2 in zip(path[:-1], path[1:]):
            # print("Computing cost for segment:", pt1, pt2)
            pt1 = np.round(pt1).astype(int)
            pt2 = np.round(pt2).astype(int)
            if use_step_cost:
                total_cost += self.compute_segment_cost_step(pt1, pt2, step_size=1, check_obstacle=False)
            else:
                total_cost += self.compute_segment_cost(pt1, pt2, check_obstacle=False)

        return total_cost
    
    def compute_segment_cost_drake(self, edge, pt1, pt2, region_lc_i):
        """
        Computes the cost of a segment between two points in 3D space.
        This function is a placeholder and should be implemented in subclasses.
        """
        raise NotImplementedError("This method should be implemented in subclasses.")


    def compute_cost(self, pt1_3d, pt2_3d, region_lc_i):
        """
        Computes the cost of a segment between two points in 3D space.
        This function is a placeholder and should be implemented in subclasses.
        """
        raise NotImplementedError("This method should be implemented in subclasses.")




    def check_obstacle(self, pt1, pt2):
        # return False
        step_size=1

        row1, col1 = np.round(pt1).astype(int) #self.geo_to_pixel(*pt1) 
        row2, col2 = np.round(pt2).astype(int)

        # Generate discrete steps along the line (Bresenham’s-like approach)
        num_steps = max(abs(row2 - row1), abs(col2 - col1)) // step_size
        if num_steps == 0:
            num_steps = 1  # Avoid division by zero

        # Generate intermediate points
        rows = np.linspace(row1, row2, num_steps, dtype=int)
        cols = np.linspace(col1, col2, num_steps, dtype=int)


        # Compute step-wise cost
        for i in range(len(rows)):
            x1, y1 = rows[i], cols[i]
            if self.map.elevation[y1, x1] > self.obstacle_threshold:
                return True

        return False
    

class HumanModelObjective(AgentModelObjective):

    def __init__(self, gmap, body=85, load=8, speed=0.5, obstacle_threshold=400):
        super().__init__(gmap, obstacle_threshold)
        self.model = StaticPerformanceModel(body=body, load=load, speed=speed)
        self.terrain_table = [
            3,
            3,
            3,
            3,
            3,
            3,
            1.2,
            1.5,
            1.1,
            1.1,
            1.2,
            1.2,
            1.2,
            1.8,
            1.8,
            1,
            10,
            5,
        ]
        self.body = body
        self.load = load
        self.speed = speed

        self.pandolf_quad_bound = self.compute_pandolf_grade_approximation_quadratic_bound(d=30.0)


        # self.model_for_tf = {}
        # for tf in self.terrain_table:
        #     current_values = fit_pandolf_quadratic_model(W=self.body , L=self.load, v=self.speed, eta=tf)
        #     # print(f"a_dz, a_d2, intercept for {tf}:", current_values)
        #     self.model_for_tf[tf] = current_values


        self.model_for_tf = {}
        for tf in self.terrain_table:
            current_values = self.compute_pandolf_quadratic_fit(d=30, eta=tf, W=self.body , L=self.load, v=self.speed)
            # print(f"a_dz, a_d2, intercept for {tf}:", current_values)
            self.model_for_tf[tf] = current_values


        self.max_grade = 10


    def compute_pandolf_quadratic_fit(self, d=30.0, eta=3.0, W=85, L=8, v=0.5):
        import numpy as np
        from sklearn.linear_model import LinearRegression
        from sklearn.preprocessing import PolynomialFeatures

        def pandolf_exact_from_dz(dz, d, eta):
            mass = W + L
            grade = (dz / d) * 100
            c0 = 1.5 * W + 2 * mass * (L / W) ** 2 + eta * mass * (1.5 * v ** 2)
            return c0 + eta * mass * (0.35 * v * grade)

        dz_vals = np.linspace(-10, 10, 100)
        P_vals = [pandolf_exact_from_dz(dz, d, eta) for dz in dz_vals]

        poly = PolynomialFeatures(degree=2)
        X = poly.fit_transform(dz_vals.reshape(-1, 1))
        model = LinearRegression().fit(X, P_vals)
        a = model.coef_[2]
        b = model.coef_[1]
        c = model.intercept_
        return a, b, c


    def compute_pandolf_grade_approximation_quadratic_bound(self, d=30.0):

        import numpy as np
        from sklearn.linear_model import LinearRegression
        from sklearn.preprocessing import PolynomialFeatures
        def pandolf_exact_from_dz(dz, d=30.0, W=85, L=8, v=0.5, eta=3.0):
            mass = W + L
            grade = (dz / d) * 100  # percent grade
            c0 = 1.5 * W + 2 * mass * (L / W) ** 2 + eta * mass * (1.5 * v ** 2)
            return c0 + eta * mass * (0.35 * v * grade)

        dz_vals = np.linspace(-10, 10, 100)  # elevation change in meters
        P_vals = [pandolf_exact_from_dz(dz, d=d) for dz in dz_vals]
        # Fit quadratic
        poly = PolynomialFeatures(degree=2)
        X = poly.fit_transform(dz_vals.reshape(-1,1))
        model = LinearRegression().fit(X, P_vals)
        a, b, c = model.coef_[2], model.coef_[1], model.intercept_
        return a, b, c


    def _compute_grade(self, x1, y1, x2, y2):
        #scale x,y  by self.scale_xy
        x1s, y1s = int(x1 * self.scale_xy) , int(y1 * self.scale_xy)
        x2s , y2s = int(x2 * self.scale_xy), int(y2 * self.scale_xy)
        d = math.dist((x1s, y1s), (x2s, y2s))

        g1, g2 = self.map.elevation[y1, x1], self.map.elevation[y2, x2]
        gd = g2 - g1
        grade = (gd / (d + 0.01)) * 100
        return grade

    def _compute_terrain_factor(self, x1, y1):
        idx = self.map.type[y1, x1]
        return self.terrain_table[idx - 1]

    
    def motionCost_float(self, s1, s2):
        x1, y1, x2, y2 = int(s1.getX()), int(s1.getY()), int(s2.getX()), int(s2.getY())


        grade = self._compute_grade(x1, y1, x2, y2)
        ttype = self._compute_terrain_factor(x1, y1)
        cost = self.model.cost(grade, ttype)
        return cost
    


    def compute_cost(self, pt1_3d, pt2_3d, region_lc_i):
        """
        Computes the cost of a segment between two points in 3D space.
        This function is a placeholder and should be implemented in subclasses.
        """
        x1, y1, g1 = pt1_3d
        x2, y2, g2 = pt2_3d
        x1s, y1s = int(x1 * self.scale_xy) , int(y1 * self.scale_xy)
        x2s , y2s = int(x2 * self.scale_xy), int(y2 * self.scale_xy)
        d = math.dist((x1s, y1s), (x2s, y2s))
        gd = g2 - g1
        grade = (gd / (d + 0.01)) * 100
        
        ttype = self.terrain_table[region_lc_i-1]
        cost_value = self.model.cost(grade, ttype)
        return cost_value



    def compute_segment_cost(self, pt1, pt2, check_obstacle=False):
        

        x1,y1 = np.round(pt1).astype(int)
        x2,y2 = np.round(pt2).astype(int)

        grade = self._compute_grade(x1, y1, x2, y2)

        if  check_obstacle and (self.check_obstacle(pt1, pt2) or self.max_grade < abs(grade)):
            # print("Grade too steep, skipping segment")
            return float('inf')

        terrain_factor = self._compute_terrain_factor(x1, y1)

        step_cost = self.model.cost(grade, terrain_factor) 
        return step_cost  


    def compute_segment_cost_step(self, pt1, pt2, step_size=1, check_obstacle=False):
        """
        Computes the total cost from pt1 to pt2 by summing per-unit step costs.

        Parameters:
        - pt1, pt2: Tuple (row, col) start and end points.
        - step_size: int (distance between discrete steps in pixels)

        Returns:
        - Total accumulated cost along the path.
        """

        row1, col1 = np.round(pt1).astype(int) #self.geo_to_pixel(*pt1)
        row2, col2 = np.round(pt2).astype(int) #self.geo_to_pixel(*pt2)

        # # change x,y to be within the map
        # row1 = min(max(row1, 0), self.map.elevation.shape[1] - 1)
        # row2 = min(max(row2, 0), self.map.elevation.shape[1] - 1)
        # col1 = min(max(col1, 0), self.map.elevation.shape[0] - 1)
        # col2 = min(max(col2, 0), self.map.elevation.shape[0] - 1)


        from skimage.draw import line

        def get_line_pixels(p1, p2):
            """
            p1, p2: tuples of (row, col) or (y, x)
            Returns list of pixel coordinates along the line.
            """
            rr, cc = line(p1[0], p1[1], p2[0], p2[1])
            return list(zip(rr, cc))
        
        pixels = get_line_pixels((row1, col1), (row2, col2))

        total_cost = 0
        for i in range(len(pixels) - 1):
            x1, y1 = pixels[i]
            x2, y2 = pixels[i + 1]

            grade = self._compute_grade(x1, y1, x2, y2)

            if  check_obstacle and (self.check_obstacle(pt1, pt2) or self.max_grade < abs(grade)):
                # print("Grade too steep, skipping segment")
                return float('inf')

            terrain_factor = self._compute_terrain_factor(x1, y1)

            step_cost = self.model.cost(grade, terrain_factor) 

            total_cost += step_cost  # Accumulate cost

        return total_cost
    

    def compute_segment_cost_cp(self, pt1, pt2, region_elevations_i, region_elevations_j, region_tf_i):

        p_i, p_j = pt1, pt2
        delta = p_j - p_i
        pt_dist = cp.norm(delta, 2)
        delta_h = region_elevations_j - region_elevations_i

        # Grade-based energy model (Tf = 3)
        alpha_up = 0.35 * region_tf_i * 1.4  # ~1.47
        alpha_down = 0.35 * region_tf_i * 1.4 / 3.5  # ~0.42


        cost = pt_dist \
            + alpha_up * cp.pos(delta_h) \
            + alpha_down * cp.pos(-delta_h)
        
        return cost



    def compute_segment_cost_drake_inprogress(self, edge, pt1, pt2, region_lc_i):
        terrain_factor = float(self.terrain_table[region_lc_i - 1])
        a_dz, a_d2, intercept = self.model_for_tf[terrain_factor]

        x_src = pt1.x()  # should be 3D [x, y, z]
        x_dst = pt2.x()
        dz = x_dst[2] - x_src[2]
        dxy2 = (x_dst[0] - x_src[0])**2 + (x_dst[1] - x_src[1])**2

        # cost_expr = pandolf_cost_expr_from_fit(dz, dxy2, a_dz, a_d2, intercept)
        # edge.AddCost(cost_expr)



        try:
            # TODO: Pranay - Check why using different tf quadratic bound model fails optimization, while single model works
            a_dz, a_d2, intercept = self.model_for_tf[terrain_factor]
            cost = (a_dz * dz + intercept) 
            edge.AddCost(cost)
        except Exception as e:
            print(f"Skipping region with eta={terrain_factor} due to error: {e}")




    def compute_segment_cost_drake(self, edge, pt1, pt2, region_lc_i):
        """
        Computes the cost of a segment between two points in 3D space.
        This function is a placeholder and should be implemented in subclasses.
        """

        terrain_factor = float(self.terrain_table[region_lc_i - 1])
        x_src = pt1.x()  # should be 3D [x, y, z]
        x_dst = pt2.x()
        dx = x_dst[0] - x_src[0]
        dy = x_dst[1] - x_src[1]
        dz = x_dst[2] - x_src[2]  # elevation

        # a,b,c = 0.0, 48.82499999999996, 233.7726124567475
        # a,b,c = 0.0, 162.74999999999994, 233.77261245674737  #d_approx = 30.0


        # a,b,c =self.model_for_tf[terrain_factor]
        a,b,c =  self.pandolf_quad_bound

        #pandolf approximation for d_approx=30.0 
        p_approx = a*dz**2 + b*dz + c


        cost_expr = terrain_factor* self.scale_xy * (dx**2 + dy**2) + p_approx

        edge.AddCost(cost_expr)

        dx = self.scale_xy * (x_dst[0] - x_src[0])
        dy = self.scale_xy * (x_dst[1] - x_src[1])
        dz = x_dst[2] - x_src[2]

        return dx, dy, dz


        a,b,c =  self.pandolf_quad_bound

        #pandolf approximation for d_approx=30.0 
        p_approx = a*dz**2 + b*dz + c

        cost_expr = terrain_factor* self.scale_xy * (dx**2 + dy**2) + p_approx

        edge.AddCost(cost_expr)






class EuclideanObjective(AgentModelObjective):

    def __init__(self, gmap, body=85, load=8, speed=0.5, obstacle_threshold=400):
        super().__init__(gmap, obstacle_threshold)
        self.elevation = gmap.elevation
        # self.landcover = landcover
        self.obstacle_threshold = obstacle_threshold
        self.terrain_table = [
            3,
            3,
            3,
            3,
            3,
            3,
            1.2,
            1.5,
            1.1,
            1.1,
            1.2,
            1.2,
            1.2,
            1.8,
            1.8,
            1,
            10,
            5,
        ]


    def compute_segment_cost(self, pt1, pt2, check_obstacle=False):
        """
        Computes the total cost from pt1 to pt2 by summing per-unit step costs.

        Parameters:
        - pt1, pt2: Tuple (row, col) start and end points.
        - step_size: int (distance between discrete steps in pixels)

        Returns:
        - Total accumulated cost along the path.
        """

        x1, y1 = pt1 
        x2, y2 = pt2 


        if check_obstacle and self.check_obstacle(pt1, pt2):
            return float('inf')

        g1, g2 = self.map.elevation[y1, x1], self.map.elevation[y2, x2]
        pt13d = np.array([x1*self.scale_xy, y1*self.scale_xy, g1])
        pt23d = np.array([x2*self.scale_xy, y2*self.scale_xy, g2])

        # total_cost = (x2 - x1)**2 + (y2 - y1)**2 + (g2 - g1)**2
        total_cost = np.linalg.norm(pt23d - pt13d)
        # total_cost =  math.hypot(x2 - x1, y2 - y1)

        return total_cost

    def compute_segment_cost_cp(self, pt1, pt2, region_elevations_i, region_elevations_j, region_tf_i):

        pt1_3d = cp.hstack([pt1, region_elevations_i])
        pt2_3d = cp.hstack([pt2, region_elevations_j])
        delta = pt2_3d - pt1_3d
        pt_dist = cp.norm(delta, 3)
        return pt_dist


    def compute_segment_cost_drake(self, edge, pt1, pt2, region_lc_i):
        """
        Computes the cost of a segment between two points in 3D space.
        This function is a placeholder and should be implemented in subclasses.
        """
        # vertex_landcover_map = self.vertex_landcover_map
        # model = self.model
        x_src = pt1.x()  # should be 3D [x, y, z]
        x_dst = pt2.x()

        scale = 1.0 / max(self.terrain_table)
        terrain_factor = float(self.terrain_table[region_lc_i - 1]) * scale

        diff = x_dst - x_src

        cost_expr =  self.scale_xy*diff[0]**2 + self.scale_xy*diff[1]**2 + diff[2]**2
        # cost_expr = float(terrain_factor) * (diff[0]**2 + diff[1]**2) + diff[2]**2
        edge.AddCost(cost_expr)

        dx = self.scale_xy * (x_dst[0] - x_src[0])
        dy = self.scale_xy * (x_dst[1] - x_src[1])
        dz = x_dst[2] - x_src[2]

        return dx, dy, dz


    def compute_segment_cost_step(self, pt1, pt2, step_size=1, check_obstacle=False):
        """
        Computes the total cost from pt1 to pt2 by summing per-unit step costs.

        Parameters:
        - pt1, pt2: Tuple (row, col) start and end points.
        - step_size: int (distance between discrete steps in pixels)

        Returns:
        - Total accumulated cost along the path.
        """

        row1, col1 = np.round(pt1).astype(int) #self.geo_to_pixel(*pt1)
        row2, col2 = np.round(pt2).astype(int) #self.geo_to_pixel(*pt2)


        from skimage.draw import line

        def get_line_pixels(p1, p2):
            """
            p1, p2: tuples of (row, col) or (y, x)
            Returns list of pixel coordinates along the line.
            """
            rr, cc = line(p1[0], p1[1], p2[0], p2[1])
            return list(zip(rr, cc))
        
        pixels = get_line_pixels((row1, col1), (row2, col2))

        total_cost = 0
        for i in range(len(pixels) - 1):
            x1, y1 = pixels[i]
            x2, y2 = pixels[i + 1]

            cost_current = self.compute_segment_cost(pixels[i], pixels[i + 1], check_obstacle= check_obstacle)

            if cost_current == float('inf'):
                # print("Skipping segment due to obstacle")
                return float('inf')
            total_cost += cost_current

        return total_cost





class VehicleObjective(AgentModelObjective):

    def __init__(self, gmap, body=85, load=8, speed=0.5, obstacle_threshold=400):
        super().__init__(gmap, obstacle_threshold)
        self.elevation = gmap.elevation
        # self.landcover = landcover
        self.obstacle_threshold = obstacle_threshold
        self.terrain_table = [
            3,
            3,
            3,
            3,
            3,
            3,
            1.2,
            1.5,
            1.1,
            1.1,
            1.2,
            1.2,
            1.2,
            1.8,
            1.8,
            1,
            10,
            5,
        ]

        self.friction_coefficients = [
            0.0,     # 0  None
            0.504,   # 1  Temperate or sub-polar needleleaf forest
            0.446,   # 2  Sub-polar taiga needleleaf forest
            0.395,   # 3  Tropical or sub-tropical broadleaf evergreen forest
            0.453,   # 4  Tropical or sub-tropical broadleaf deciduous forest
            0.498,   # 5  Temperate or sub-polar broadleaf deciduous forest
            0.506,   # 6  Mixed forest
            0.547,   # 7  Tropical or sub-tropical shrubland
            0.595,   # 8  Temperate or sub-polar shrubland
            0.404,   # 9  Tropical or sub-tropical grassland
            0.447,   # 10 Temperate or sub-polar grassland
            0.356,   # 11 Sub-polar or polar shrubland-lichen-moss
            0.297,   # 12 Sub-polar or polar grassland-lichen-moss
            0.344,   # 13 Sub-polar or polar barren-lichen-moss
            0.254,   # 14 Wetland
            0.495,   # 15 Cropland
            0.705,   # 16 Barren land
            0.796,   # 17 Urban and built-up
            0.156,   # 18 Water
            0.147    # 19 Snow and ice
        ]

        self.roughness_values = [
            0.0,      # 0  None
            66.5142,  # 1  Temperate or sub-polar needleleaf forest
            40.3429,  # 2  Sub-polar taiga needleleaf forest  
            109.6633, # 3  Tropical or sub-tropical broadleaf evergreen forest
            66.5142,  # 4  Tropical or sub-tropical broadleaf deciduous forest
            40.3429,  # 5  Temperate or sub-polar broadleaf deciduous forest
            66.5142,  # 6  Mixed forest
            24.4692,  # 7  Tropical or sub-tropical shrubland
            14.8413,  # 8  Temperate or sub-polar shrubland
            5.4598,   # 9  Tropical or sub-tropical grassland
            3.3115,   # 10 Temperate or sub-polar grassland
            9.0017,   # 11 Sub-polar or polar shrubland-lichen-moss
            2.0086,   # 12 Sub-polar or polar grassland-lichen-moss
            109.6633, # 13 Sub-polar or polar barren-lichen-moss
            1.2182,   # 14 Wetland
            0.7389,   # 15 Cropland
            298.0958, # 16 Barren land
            0.4482,   # 17 Urban and built-up
            0.1649,   # 18 Water
            0.2718    # 19 Snow and ice
        ]



    def compute_segment_cost(self, pt1, pt2, check_obstacle=False):
        """
        Computes the total cost from pt1 to pt2 by summing per-unit step costs.

        Parameters:
        - pt1, pt2: Tuple (row, col) start and end points.
        - step_size: int (distance between discrete steps in pixels)

        Returns:
        - Total accumulated cost along the path.
        """

        x1, y1 = pt1 
        x2, y2 = pt2 


        if check_obstacle and self.check_obstacle(pt1, pt2):
            return float('inf')

        g1, g2 = self.map.elevation[y1, x1], self.map.elevation[y2, x2]
        pt1s = np.array([x1*self.scale_xy, y1*self.scale_xy])
        pt2s = np.array([x2*self.scale_xy, y2*self.scale_xy])

        lc = self.map.type[y1, x1]

        total_cost = self.transfer_cost(pt1s, pt2s, g1, g2, lc)
        # total_cost = (x2 - x1)**2 + (y2 - y1)**2 + (g2 - g1)**2
        # total_cost = np.linalg.norm(pt23d - pt13d)
        # total_cost =  math.hypot(x2 - x1, y2 - y1)

        return total_cost

    def compute_segment_cost_cp(self, pt1, pt2, region_elevations_i, region_elevations_j, region_tf_i):
        
        raise NotImplementedError("This method should be implemented in subclasses.")
        pt1_3d = cp.hstack([pt1, region_elevations_i])
        pt2_3d = cp.hstack([pt2, region_elevations_j])
        delta = pt2_3d - pt1_3d
        pt_dist = cp.norm(delta, 3)
        return pt_dist


    def compute_segment_cost_drake(self, edge, pt1, pt2, region_lc_i):
        """
        Computes the cost of a segment between two points in 3D space.
        This function is a placeholder and should be implemented in subclasses.
        """

        # vertex_landcover_map = self.vertex_landcover_map
        # model = self.model
        x_src = pt1.x()  # should be 3D [x, y, z]
        x_dst = pt2.x()


        dx = self.scale_xy * (x_dst[0] - x_src[0])
        dy = self.scale_xy * (x_dst[1] - x_src[1])
        dz = x_dst[2] - x_src[2]


        diff = x_dst - x_src

        d = dx**2 + dy**2 
        elevation_diff = dz  # elevation

        friction_coeff = self.friction_coefficients[region_lc_i]
        roughness_val = self.roughness_values[region_lc_i]

        w_dist = 1.0      # distance cost
        w_slope = 5.0     # elevation change cost
        w_friction = 20.0 # (1 - μ)
        w_rough = 1.0     # terrain roughness


        edge_cost = (
            w_dist * d +
            w_slope * elevation_diff +
            w_friction * (1 - friction_coeff) +
            w_rough * roughness_val
        )


        # cost_expr = float(terrain_factor) * (diff[0]**2 + diff[1]**2) + diff[2]**2
        edge.AddCost(edge_cost)


        return dx, dy, dz


    def compute_segment_cost_step_non_vectorized(self, pt1, pt2, step_size=1, check_obstacle=False):
        """
        Computes the total cost from pt1 to pt2 by summing per-unit step costs.

        Parameters:
        - pt1, pt2: Tuple (row, col) start and end points.
        - step_size: int (distance between discrete steps in pixels)

        Returns:
        - Total accumulated cost along the path.
        """

        row1, col1 = np.round(pt1).astype(int) #self.geo_to_pixel(*pt1)
        row2, col2 = np.round(pt2).astype(int) #self.geo_to_pixel(*pt2)


        from skimage.draw import line

        def get_line_pixels(p1, p2):
            """
            p1, p2: tuples of (row, col) or (y, x)
            Returns list of pixel coordinates along the line.
            """
            rr, cc = line(p1[0], p1[1], p2[0], p2[1])
            return list(zip(rr, cc))
        
        pixels = get_line_pixels((row1, col1), (row2, col2))

        total_cost = 0
        for i in range(len(pixels) - 1):
            # x1, y1 = pixels[i]
            # x2, y2 = pixels[i + 1]

            cost_current = self.compute_segment_cost(pixels[i], pixels[i + 1], check_obstacle= check_obstacle)

            if cost_current == float('inf'):
                # print("Skipping segment due to obstacle")
                return float('inf')
            total_cost += cost_current

        return total_cost


    def compute_segment_cost_step(self, pt1, pt2, step_size=1, check_obstacle=False):
        """
        Vectorized cost computation from pt1 to pt2 along a straight line.
        Accumulates cost for each pixel-to-pixel transition.
        """

        row1, col1 = np.round(pt1).astype(int)
        row2, col2 = np.round(pt2).astype(int)

        from skimage.draw import line
        rr, cc = line(row1, col1, row2, col2)

        # Apply step size (sample every step_size pixels)
        rr = rr[::step_size]
        cc = cc[::step_size]

        # Start and end points for each small segment
        start_pts = np.stack([rr[:-1], cc[:-1]], axis=1)
        end_pts   = np.stack([rr[1:],  cc[1:]],  axis=1)

        # Vectorized cost computation
        # Assumes compute_segment_cost can accept (N,2) arrays for start and end
        # Otherwise, replace with list comprehension and np.sum
        costs = np.array([
            self.compute_segment_cost(start_pts[i], end_pts[i], check_obstacle=check_obstacle)
            for i in range(len(start_pts))
        ])

        # If any inf, path blocked
        if np.isinf(costs).any():
            return float('inf')

        return np.sum(costs)





    def compute_cost(self, pt1_3d, pt2_3d, region_lc_i):
        """
        Computes the cost of a segment between two points in 3D space.
        This function is a placeholder and should be implemented in subclasses.
        """
        x1, y1, g1 = pt1_3d
        x2, y2, g2 = pt2_3d


        p0 = np.array([x1 * self.scale_xy, y1 * self.scale_xy])
        p1 = np.array([x2 * self.scale_xy, y2 * self.scale_xy])
        h0 = g1
        h1 = g2
        # check_obstacle=True

        # if check_obstacle and h1 > self.obstacle_threshold:
        #     return float('inf')


        l = region_lc_i
        cost_value = self.transfer_cost(p0, p1, h0, h1, l)

    
        return cost_value





    def transfer_cost_original(self, p0, p1, h0, h1, l):
        distance = np.linalg.norm(p1 - p0)
        if distance == 0:
            return 0
        height_change = abs(h1 - h0)
        mu = self.friction_coefficients[l]
        rough = self.roughness_values[l]
        climb_cost = (math.exp(5 * height_change / distance / max(mu, 0.1)) - 1) / 10
        energy_cost = distance
        friction_cost = (1 / max(mu, 0.1) - 1) / 2
        roughness_cost = rough
        obstacle_cost = 1e9 if height_change / distance > 0.6 else 0
        return climb_cost + energy_cost + friction_cost + roughness_cost + obstacle_cost



    def transfer_cost(self, p0, p1, h0, h1, landcover_class):
        distance = np.linalg.norm(p1 - p0)
        if distance == 0:
            return 0

        # --- Geometry ---
        dz = h1 - h0
        dx, dy = p1[0] - p0[0], p1[1] - p0[1]
        slope = dz / distance
        grade_percent = slope * 100

        # --- Heading (vehicle travel direction in degrees) ---
        heading = (math.degrees(math.atan2(dy, dx)) + 360) % 360

        # --- Slope Direction (steepest ascent) ---
        # Assume slope direction is aligned with elevation change
        slope_direction = (math.degrees(math.atan2(dz, distance)) + 360) % 360

        # --- Landcover Properties ---
        mu = self.friction_coefficients[landcover_class] #max(, 0.1)
        rough = self.roughness_values[landcover_class]

        if mu <=0 or rough <= 0:
            return float('inf')  # Avoid division by zero

        # --- Cost Terms ---

        # # 1. Grade threshold
        # if grade_percent > 35:
        #     return 1e9
        # print("Grade percent:", grade_percent)

        max_grade = 35#20 #35


        if abs(grade_percent) > max_grade:
            return 1e6 #float('inf')  # Untraversable
        
        # else:
        #     # Penalty increases quadratically with grade
        #     # grade_penalty = distance * (1 + (grade_percent / max_grade)**2)
        #     grade_penalty = distance * np.exp(abs(grade_percent) / max_grade)


        # 2. Slope alignment penalty
        delta_theta = abs(heading - slope_direction) % 360
        delta_theta = min(delta_theta, 360 - delta_theta)  # shortest angle
        slope_align_cost = 1 - math.cos(math.radians(delta_theta))  # [0, 2]

        # 3. Climb cost (only for positive slope)
        climb_cost = (math.exp(5 * abs(slope) / mu) - 1) / 10 if slope > 0 else 0

        # 4. Baseline energy
        energy_cost = distance

        # 5. Friction and terrain

        friction_cost = (1 / mu - 1) / 2
        roughness_cost = 0.1* rough

        # 6. Obstacle barrier
        # obstacle_cost = float('inf')  # 1e9 if abs(slope) > 0.6 else 0

        # --- Final Total Cost ---
        return (
            (climb_cost +
            friction_cost )+
            energy_cost +
            slope_align_cost +
            roughness_cost 
            # + obstacle_cost
            # +grade_penalty
        )



class EuclideanObjective2D(AgentModelObjective):

    def __init__(self, gmap, body=85, load=8, speed=0.5, obstacle_threshold=400):
        super().__init__(gmap, obstacle_threshold)
        self.elevation = gmap.elevation
        # self.landcover = landcover
        self.obstacle_threshold = obstacle_threshold
        self.terrain_table = [
            3,
            3,
            3,
            3,
            3,
            3,
            1.2,
            1.5,
            1.1,
            1.1,
            1.2,
            1.2,
            1.2,
            1.8,
            1.8,
            1,
            10,
            5,
        ]
    def check_obstacle(self, pt1, pt2):
        # return False
        step_size=1

        row1, col1 = np.round(pt1).astype(int) #self.geo_to_pixel(*pt1) 
        row2, col2 = np.round(pt2).astype(int)

        # Generate discrete steps along the line (Bresenham’s-like approach)
        num_steps = max(abs(row2 - row1), abs(col2 - col1)) // step_size
        if num_steps == 0:
            num_steps = 1  # Avoid division by zero

        # Generate intermediate points
        rows = np.linspace(row1, row2, num_steps, dtype=int)
        cols = np.linspace(col1, col2, num_steps, dtype=int)

        total_cost = 0

        # Compute step-wise cost
        for i in range(len(rows)):
            x1, y1 = rows[i], cols[i]
            if self.elevation[y1, x1] > self.obstacle_threshold:
                return True

        return False

    def compute_segment_cost(self, pt1, pt2, check_obstacle=False):
        """
        Computes the total cost from pt1 to pt2 by summing per-unit step costs.

        Parameters:
        - pt1, pt2: Tuple (row, col) start and end points.
        - step_size: int (distance between discrete steps in pixels)

        Returns:
        - Total accumulated cost along the path.
        """

        x1, y1 = pt1 
        x2, y2 = pt2 


        # if check_obstacle and self.check_obstacle(pt1, pt2):
        #     return float('inf')

        total_cost =  math.hypot(self.scale_xy*x2 - self.scale_xy*x1, self.scale_xy*y2 - self.scale_xy*y1)

        return total_cost

    def compute_segment_cost_cp(self, pt1, pt2, region_elevations_i, region_elevations_j, region_tf_i):

        pt1_3d = cp.hstack([pt1, region_elevations_i])
        pt2_3d = cp.hstack([pt2, region_elevations_j])
        delta = pt2_3d - pt1_3d
        pt_dist = cp.norm(delta, 3)
        return pt_dist

    def compute_segment_cost_drake(self, edge,  pt1, pt2, region_lc_i):
        """
        Computes the cost of a segment between two points in 3D space.
        This function is a placeholder and should be implemented in subclasses.
        """
        
        # vertex_landcover_map = self.vertex_landcover_map
        # model = self.model
        x_src = pt1.x()  # should be 3D [x, y, z]
        x_dst = pt2.x()

        dx = x_dst[0] - x_src[0]
        dy = x_dst[1] - x_src[1]
        # dz = x_dst[2] - x_src[2]
        
        terrain_factor =  1.0#self.terrain_table[region_lc_i-1]  # assumed stored earlier

        # print("Terrain Factor:", terrain_factor)

        # Squared 3D distance (convex)
        dist2 = self.scale_xy* dx**2 + self.scale_xy*dy**2 #+ dz**2

        # Linear terrain cost multiplier
        # cost_expr = terrain_factor * dist2
        edge.AddCost( dist2)

        dx = self.scale_xy * (x_dst[0] - x_src[0])
        dy = self.scale_xy * (x_dst[1] - x_src[1])
        dz = x_dst[2] - x_src[2]
        return dx, dy, dz
        # return cost_expr


    def compute_segment_cost(self, pt1, pt2, check_obstacle=False):
        

        x1,y1 = np.round(pt1).astype(int)
        x2,y2 = np.round(pt2).astype(int)

        if  check_obstacle and self.check_obstacle(pt1, pt2):
            # print("Grade too steep, skipping segment")
            return float('inf')

        pt12d = np.array([x1*self.scale_xy, y1*self.scale_xy])
        pt22d = np.array([x2*self.scale_xy, y2*self.scale_xy])

        # total_cost = (x2 - x1)**2 + (y2 - y1)**2 + (g2 - g1)**2
        total_cost = np.linalg.norm(pt22d - pt12d)
        return total_cost  

    def compute_segment_cost_step(self, pt1, pt2, step_size=1, check_obstacle=False):
        """
        Computes the total cost from pt1 to pt2 by summing per-unit step costs.

        Parameters:
        - pt1, pt2: Tuple (row, col) start and end points.
        - step_size: int (distance between discrete steps in pixels)

        Returns:
        - Total accumulated cost along the path.
        """

        row1, col1 = np.round(pt1).astype(int) #self.geo_to_pixel(*pt1)
        row2, col2 = np.round(pt2).astype(int) #self.geo_to_pixel(*pt2)


        from skimage.draw import line

        def get_line_pixels(p1, p2):
            """
            p1, p2: tuples of (row, col) or (y, x)
            Returns list of pixel coordinates along the line.
            """
            rr, cc = line(p1[0], p1[1], p2[0], p2[1])
            return list(zip(rr, cc))
        
        pixels = get_line_pixels((row1, col1), (row2, col2))

        total_cost = 0
        for i in range(len(pixels) - 1):
            x1, y1 = pixels[i]
            x2, y2 = pixels[i + 1]

            cost_current = self.compute_segment_cost(pixels[i], pixels[i + 1], check_obstacle= check_obstacle)

            if cost_current == float('inf'):
                # print("Skipping segment due to obstacle")
                return float('inf')
            total_cost += cost_current

        return total_cost







class LinearEnergyFunction(AgentModelObjective):

    def __init__(self, gmap, body=85, load=8, speed=0.5, obstacle_threshold=400):
        super().__init__(gmap, obstacle_threshold)
        self.elevation = gmap.elevation
        # self.landcover = landcover
        self.obstacle_threshold = obstacle_threshold
        self.terrain_table = [
            3,
            3,
            3,
            3,
            3,
            3,
            1.2,
            1.5,
            1.1,
            1.1,
            1.2,
            1.2,
            1.2,
            1.8,
            1.8,
            1,
            10,
            5,
        ]
        self.terrain_table = np.array(self.terrain_table)
    def check_obstacle(self, pt1, pt2):
        # return False
        step_size=1

        row1, col1 = np.round(pt1).astype(int) #self.geo_to_pixel(*pt1) 
        row2, col2 = np.round(pt2).astype(int)

        # Generate discrete steps along the line (Bresenham’s-like approach)
        num_steps = max(abs(row2 - row1), abs(col2 - col1)) // step_size
        if num_steps == 0:
            num_steps = 1  # Avoid division by zero

        # Generate intermediate points
        rows = np.linspace(row1, row2, num_steps, dtype=int)
        cols = np.linspace(col1, col2, num_steps, dtype=int)

        total_cost = 0

        # Compute step-wise cost
        for i in range(len(rows)):
            x1, y1 = rows[i], cols[i]
            if self.elevation[y1, x1] > self.obstacle_threshold:
                return True

        return False
    

    def compute_segment_cost(self, pt1, pt2, check_obstacle=False):
        """
        Compute cost between pt1 and pt2 using slope and terrain weights.
        Uses terrain table and checks obstacle and max grade threshold.

        Args:
            pt1, pt2: 2D coordinates (x, y) in the map
            check_obstacle: if True, blocks segments with obstacles or steep grades

        Returns:
            step_cost (float): cost value or inf if segment is invalid
        """

        x1, y1 = np.round(pt1).astype(int)
        x2, y2 = np.round(pt2).astype(int)




        if check_obstacle and self.check_obstacle(pt1, pt2) :
            return float('inf')
        

        dx = (x2 - x1) * self.scale_xy
        dy = (y2 - y1) * self.scale_xy
        dz = self.elevation[y2, x2] - self.elevation[y1, x1]

        dx2 = dx**2
        dy2 = dy**2
        dz2 = dz**2

        terrain_factor = float(self.terrain_table[self.map.type[y1, x1] - 1])





        # dz_linear = max(0, dz)   # uphill only
        # dz_quad = dz**2          # always ≥ 0åå
        # dx2 = dx**2
        # dy2 = dy**2

        # cost = (
        #     10.0 * dz_linear +        # uphill bias
        #     0.05 * dz_quad +          # quadratic slope
        #     0.00005 * (dx2 + dy2) +   # horizontal effort
        #     1.0 * terrain_factor +    # terrain
        #     10                        # base
        # )


        # terms = np.array([
        #     1,               # intercept
        #     dz,              # dz
        #     # d,               # d
        #     # dz**2,           # dz^2
        #     # dz * d,          # dz*d
        #     # d**2             # d^2
        # ])
        # coeffs = np.array([
        #     -1.620791e-13,   # Intercept
        #     3.611025e-01,   # dz
        #     #  3.386931e-15,   # d
        #     #  1.849024e-17,   # dz^2
        #     #  0.0,            # dz*d
        #     #  0.0             # d^2
        # ])
        a = 3.611025e-01
        b =-1.620791e-13 
        grade = (dz*a +b )#*100.0

        d = np.sqrt(dx2 + dy2)

        grade = (dz / (d +0.01))*100.0

        if abs(grade) > 10.0:
            # print("Grade too steep, skipping segment")
            return float('inf')



        # Compute average terrain factor over the segment
        from skimage.draw import line

        rr, cc = line(y1, x1, y2, x2)

        y1s, x1s = rr[:-1], cc[:-1]
        y2s, x2s = rr[1:], cc[1:]
        # Elevation difference
        z1s = self.elevation[y1s, x1s]
        z2s = self.elevation[y2s, x2s]
        dzs = z2s - z1s

        dxs = (x2s - x1s) * self.scale_xy
        dys = (y2s - y1s) * self.scale_xy
        dists = np.sqrt(dxs**2 + dys**2)

        grade = (dzs / (dists + 0.01))*100.0



        cost = 105.765 + terrain_factor * 75 * (2.16 + 0.42 * grade)

        cost = np.sum(cost)
        print("Cost:", cost)
        return cost



        # Compute average terrain factor over the segment
        from skimage.draw import line

        rr, cc = line(y1, x1, y2, x2)

        if len(rr) < 2:
            return 0.0

        # rr,cc = np.array([y1, y2]), np.array([x1, x2])
        def get_weight(lc):
            return self.terrain_table[lc-1] if 0 <= lc < len(self.terrain_table) else 5.0
        
        y1s, x1s = rr[:-1], cc[:-1]
        y2s, x2s = rr[1:], cc[1:]
        # Elevation difference
        z1s = self.elevation[y1s, x1s]
        z2s = self.elevation[y2s, x2s]
        dzs = z2s - z1s

        dxs = (x2s - x1s) * self.scale_xy
        dys = (y2s - y1s) * self.scale_xy
        dists = np.sqrt(dxs**2 + dys**2)
        # Slope calculation
        slope = dzs / (dists + 1e-5)  # Avoid division by zero

        # slope_norm =slope # np.min(np.abs(slope), 1.0)  # or slope / threshold if needed

        grade_percent = slope * 100 
        # print("Grade Percent:", grade_percent)
        if np.any(abs(grade_percent)) > 30.0:
            # print("Grade too steep, skipping segment")
            return float('inf')

        uphill_weight = 1.2
        downhill_weight = 0.8

        # Penalize uphill more than downhill
        for i in range(len(slope)):
            if slope[i] > 0:
                slope[i] = slope[i] * uphill_weight   # e.g., 1.0 or higher
            elif slope[i]< 0:
                slope[i] = -slope[i] * downhill_weight  # e.g., 0.2 or 0.5
        slope_norm = slope

        lc_values = self.map.type[y1s, x1s]

        terrain_cost = np.vectorize(get_weight)(lc_values)

        terrain_norm = terrain_cost / max(self.terrain_table)

        # Combine cost (equal weight)
        segment_cost = np.sum(0.5 * slope_norm + 0.5 * terrain_norm)
        # print("Segment Cost:", segment_cost)
        return segment_cost


        x1, y1 = pt1#.astype(int)
        x2, y2 = pt2#.astype(int)

        x1i, y1i = np.round(x1).astype(int), np.round(y1).astype(int)
        x2i, y2i = np.round(x2).astype(int), np.round(y2).astype(int)

        if check_obstacle and self.check_obstacle(pt1, pt2) :
            return float('inf')
        

        # Horizontal penalty weight
        offset = 10#1030#3000
        # Coefficients from previous fit
        a1 = 0.1#3.0
        a2 = 0#0.07778
        b = 0#3.0
        beta = 0.00009#0.9#0.001  # weight for dx^2 + dy^2


        a1 = 5.0       # lower uphill penalty
        a2 = 0.03      # gentler quadratic slope penalty
        beta = 0.00001 # even smaller horizontal weight
        offset = 0     # no need for bias

        dxs = (x2 - x1) * self.scale_xy
        dys = (y2 - y1) * self.scale_xy


        z1s = self.elevation[y1i, x1i]
        z2s = self.elevation[y2i, x2i]
        dzs = z2s - z1s

        DX2 = (dxs)**2 
        DY2 = (dys)**2


        DZ = dzs

        # Updated energy function with dx^2 + dy^2 penalty and constant offset
        def full_energy(dx2, dy2, dz, a1, a2, b, beta, offset):
            return a1 * dz + a2 * dz**2 + b + offset + beta * (dx2 + dy2)
        
        E_full = full_energy(DX2, DY2, DZ, a1, a2, b, beta, offset)
        # E_full = np.sum(E_full)  # Sum over all segments
        print("E_full:", E_full)        
        
        return E_full





        # Compute average terrain factor over the segment
        from skimage.draw import line

        rr, cc = line(y1, x1, y2, x2)

        if len(rr) < 2:
            return 0.0

        # rr,cc = np.array([y1, y2]), np.array([x1, x2])
        def get_weight(lc):
            return self.terrain_table[lc-1] if 0 <= lc < len(self.terrain_table) else 5.0
        
        y1s, x1s = rr[:-1], cc[:-1]
        y2s, x2s = rr[1:], cc[1:]
        # Elevation difference
        z1s = self.elevation[y1s, x1s]
        z2s = self.elevation[y2s, x2s]
        dzs = z2s - z1s

        dxs = (x2s - x1s) * self.scale_xy
        dys = (y2s - y1s) * self.scale_xy
        dists = np.sqrt(dxs**2 + dys**2)


        # Horizontal penalty weight
        offset = 30#3000
        # Coefficients from previous fit
        a1 = 3.0
        a2 = 0.07778
        b = 30.0
        beta = 0#0.001  # weight for dx^2 + dy^2

        DX2 = dxs**2
        DY2 = dys**2

        DZ = dzs

        # Updated energy function with dx^2 + dy^2 penalty and constant offset
        def full_energy(dx2, dy2, dz, a1, a2, b, beta, offset):
            return a1 * dz + a2 * dz**2 + b + offset #+ beta * (dx2 + dy2)

        E_full = full_energy(DX2, DY2, DZ, a1, a2, b, beta, offset)
        E_full = np.sum(E_full)  # Sum over all segments
        print("E_full:", E_full)
        return E_full



    def compute_segment_cost_step(self, pt1, pt2, step_size=1, check_obstacle=False):
        """
        Computes the total cost from pt1 to pt2 by summing per-unit step costs.

        Parameters:
        - pt1, pt2: Tuple (row, col) start and end points.
        - step_size: int (distance between discrete steps in pixels)

        Returns:
        - Total accumulated cost along the path.
        """

        return self.compute_segment_cost(pt1, pt2, check_obstacle=check_obstacle)
    


    def compute_segment_cost_drake(self, edge, pt1, pt2, region_lc_i):
        """
        Computes the cost of a segment between two points in 3D space.
        This function is a placeholder and should be implemented in subclasses.
        """

        terrain_factor = float(self.terrain_table[region_lc_i - 1])
        x_src = pt1.x()  # should be 3D [x, y, z]
        x_dst = pt2.x()
        dx = (x_dst[0] - x_src[0])* self.scale_xy 
        dy = (x_dst[1] - x_src[1])* self.scale_xy 
        dz = x_dst[2] - x_src[2]  # elevation


        # Approximate 2D distance
        base_distance = dx + dy            # linearized form
        slope_term = 0.1 * dz              # elevation effect (scaled)

        # cost_expr = terrain_factor * (base_distance + slope_term)

        cost_expr = 0.5 * ((dx**2 + dy**2) +(slope_term**2)) + 0.5 * terrain_factor
        
        edge.AddCost(cost_expr)

        edge.AddConstraint(dz <=10)
        edge.AddConstraint(dz >=-10.0)

 
        return dx, dy, dz
    




class HumanModelSimplifiedObjective_TODO(AgentModelObjective):

    def __init__(self, gmap, body=85, load=8, speed=0.5, obstacle_threshold=400):
        super().__init__(gmap, obstacle_threshold)
        self.model = StaticPerformanceModel(body=body, load=load, speed=speed)
        self.terrain_table = [
            3,
            3,
            3,
            3,
            3,
            3,
            1.2,
            1.5,
            1.1,
            1.1,
            1.2,
            1.2,
            1.2,
            1.8,
            1.8,
            1,
            10,
            5,
        ]
        self.body = body
        self.load = load
        self.speed = speed

        self.pandolf_quad_bound = self.compute_pandolf_grade_approximation_quadratic_bound(d=30.0)


        # self.model_for_tf = {}
        # for tf in self.terrain_table:
        #     current_values = fit_pandolf_quadratic_model(W=self.body , L=self.load, v=self.speed, eta=tf)
        #     # print(f"a_dz, a_d2, intercept for {tf}:", current_values)
        #     self.model_for_tf[tf] = current_values


        self.model_for_tf = {}
        for tf in self.terrain_table:
            current_values = self.compute_pandolf_quadratic_fit(d=30, eta=tf, W=self.body , L=self.load, v=self.speed)
            # print(f"a_dz, a_d2, intercept for {tf}:", current_values)
            self.model_for_tf[tf] = current_values


        self.max_grade = 10


    def compute_pandolf_quadratic_fit(self, d=30.0, eta=3.0, W=85, L=8, v=0.5):
        import numpy as np
        from sklearn.linear_model import LinearRegression
        from sklearn.preprocessing import PolynomialFeatures

        def pandolf_exact_from_dz(dz, d, eta):
            mass = W + L
            grade = (dz / d) * 100
            c0 = 1.5 * W + 2 * mass * (L / W) ** 2 + eta * mass * (1.5 * v ** 2)
            return c0 + eta * mass * (0.35 * v * grade)

        dz_vals = np.linspace(-10, 10, 100)
        P_vals = [pandolf_exact_from_dz(dz, d, eta) for dz in dz_vals]

        poly = PolynomialFeatures(degree=2)
        X = poly.fit_transform(dz_vals.reshape(-1, 1))
        model = LinearRegression().fit(X, P_vals)
        a = model.coef_[2]
        b = model.coef_[1]
        c = model.intercept_
        return a, b, c


    def compute_pandolf_grade_approximation_quadratic_bound(self, d=30.0):

        import numpy as np
        from sklearn.linear_model import LinearRegression
        from sklearn.preprocessing import PolynomialFeatures
        def pandolf_exact_from_dz(dz, d=30.0, W=85, L=8, v=0.5, eta=3.0):
            mass = W + L
            grade = (dz / d) * 100  # percent grade
            c0 = 1.5 * W + 2 * mass * (L / W) ** 2 + eta * mass * (1.5 * v ** 2)
            return c0 + eta * mass * (0.35 * v * grade)

        dz_vals = np.linspace(-10, 10, 100)  # elevation change in meters
        P_vals = [pandolf_exact_from_dz(dz, d=d) for dz in dz_vals]
        # Fit quadratic
        poly = PolynomialFeatures(degree=2)
        X = poly.fit_transform(dz_vals.reshape(-1,1))
        model = LinearRegression().fit(X, P_vals)
        a, b, c = model.coef_[2], model.coef_[1], model.intercept_
        return a, b, c


    def _compute_grade(self, x1, y1, x2, y2):
        #scale x,y  by self.scale_xy
        x1s, y1s = int(x1 * self.scale_xy) , int(y1 * self.scale_xy)
        x2s , y2s = int(x2 * self.scale_xy), int(y2 * self.scale_xy)
        d = math.dist((x1s, y1s), (x2s, y2s))
        g1, g2 = self.map.elevation[y1, x1], self.map.elevation[y2, x2]
        gd = g2 - g1
        grade = (gd / (d + 0.01)) * 100
        return grade

    def _compute_terrain_factor(self, x1, y1):
        idx = self.map.type[y1, x1]
        return self.terrain_table[idx - 1]

    
    def motionCost_float(self, s1, s2):
        x1, y1, x2, y2 = int(s1.getX()), int(s1.getY()), int(s2.getX()), int(s2.getY())
        grade = self._compute_grade(x1, y1, x2, y2)
        ttype = self._compute_terrain_factor(x1, y1)
        cost = self.model.cost(grade, ttype)
        return cost
    

    def compute_segment_cost(self, pt1, pt2, check_obstacle=False):
        

        x1,y1 = np.round(pt1).astype(int)
        x2,y2 = np.round(pt2).astype(int)

        grade = self._compute_grade(x1, y1, x2, y2)

        if  check_obstacle and (self.check_obstacle(pt1, pt2) or self.max_grade < abs(grade)):
            # print("Grade too steep, skipping segment")
            return float('inf')

        terrain_factor = self._compute_terrain_factor(x1, y1)

        step_cost = self.model.cost(grade, terrain_factor) 
        return step_cost  


    def compute_segment_cost_step(self, pt1, pt2, step_size=1, check_obstacle=False):
        """
        Computes the total cost from pt1 to pt2 by summing per-unit step costs.

        Parameters:
        - pt1, pt2: Tuple (row, col) start and end points.
        - step_size: int (distance between discrete steps in pixels)

        Returns:
        - Total accumulated cost along the path.
        """

        row1, col1 = np.round(pt1).astype(int) #self.geo_to_pixel(*pt1)
        row2, col2 = np.round(pt2).astype(int) #self.geo_to_pixel(*pt2)


        from skimage.draw import line

        def get_line_pixels(p1, p2):
            """
            p1, p2: tuples of (row, col) or (y, x)
            Returns list of pixel coordinates along the line.
            """
            rr, cc = line(p1[0], p1[1], p2[0], p2[1])
            return list(zip(rr, cc))
        
        pixels = get_line_pixels((row1, col1), (row2, col2))

        total_cost = 0
        for i in range(len(pixels) - 1):
            x1, y1 = pixels[i]
            x2, y2 = pixels[i + 1]

            grade = self._compute_grade(x1, y1, x2, y2)

            if  check_obstacle and (self.check_obstacle(pt1, pt2) or self.max_grade < abs(grade)):
                # print("Grade too steep, skipping segment")
                return float('inf')

            terrain_factor = self._compute_terrain_factor(x1, y1)

            step_cost = self.model.cost(grade, terrain_factor) 

            total_cost += step_cost  # Accumulate cost

        return total_cost
    

    def compute_segment_cost_cp(self, pt1, pt2, region_elevations_i, region_elevations_j, region_tf_i):

        p_i, p_j = pt1, pt2
        delta = p_j - p_i
        pt_dist = cp.norm(delta, 2)
        delta_h = region_elevations_j - region_elevations_i

        # Grade-based energy model (Tf = 3)
        alpha_up = 0.35 * region_tf_i * 1.4  # ~1.47
        alpha_down = 0.35 * region_tf_i * 1.4 / 3.5  # ~0.42


        cost = pt_dist \
            + alpha_up * cp.pos(delta_h) \
            + alpha_down * cp.pos(-delta_h)
        
        return cost




    def compute_segment_cost_drake(self, edge, pt1, pt2, region_lc_i):
        """
        Computes the cost of a segment between two points in 3D space.
        This function is a placeholder and should be implemented in subclasses.
        """

    def power_cost_quadratic(dx, dy, dz, terrain_factor, alpha=3.5, beta=0.001):
        """
        Approximate power cost:
            power ≈ α * Tf * dz^2 + β * (dx^2 + dy^2) + 162 * Tf + 105.765

        Args:
            dx, dy, dz: Differences (in meters)
            terrain_factor: Tf from terrain table
            alpha: weight for dz^2 term (default: 3.5)
            beta: weight for (dx^2 + dy^2) term (default: 0.001)

        Returns:
            Symbolic power cost expression
        """
        dx2 = dx**2
        dy2 = dy**2
        dz2 = dz**2

        vertical_cost = alpha * terrain_factor * dz2
        horizontal_cost = beta * (dx2 + dy2)
        base_cost = 162 * terrain_factor + 105.765

        return vertical_cost + horizontal_cost + base_cost


        terrain_factor = float(self.terrain_table[region_lc_i - 1])
        x_src = pt1.x()  # should be 3D [x, y, z]
        x_dst = pt2.x()
        dx = (x_dst[0] - x_src[0]) * self.scale_xy
        dy = (x_dst[1] - x_src[1]) * self.scale_xy
        dz = x_dst[2] - x_src[2]

        terrain_factor = float(self.terrain_table[region_lc_i - 1])

        cost_expr = power_cost_quadratic(dx, dy, dz, terrain_factor)
        edge.AddCost(cost_expr)






        # a,b,c =self.model_for_tf[terrain_factor]
        a,b,c =  self.pandolf_quad_bound

        #pandolf approximation for d_approx=30.0 
        p_approx = a*dz**2 + b*dz + c


        cost_expr = terrain_factor* self.scale_xy * (dx**2 + dy**2) + p_approx

        edge.AddCost(cost_expr)

        dx = self.scale_xy * (x_dst[0] - x_src[0])
        dy = self.scale_xy * (x_dst[1] - x_src[1])
        dz = x_dst[2] - x_src[2]

        return dx, dy, dz


        a,b,c =  self.pandolf_quad_bound

        #pandolf approximation for d_approx=30.0 
        p_approx = a*dz**2 + b*dz + c

        cost_expr = terrain_factor* self.scale_xy * (dx**2 + dy**2) + p_approx

        edge.AddCost(cost_expr)
##========UNUSED Functions==========




from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import numpy as np

def fit_pandolf_quadratic_model(W=85, L=8, v=0.5, eta=3.0):
    """Fit Pandolf equation in terms of dz and d^2 using quadratic approximation."""
    mass = W + L

    def pandolf_energy(dz, d):
        grade = (dz / d) * 100
        c0 = 1.5 * W + 2 * mass * (L / W)**2 + eta * mass * (1.5 * v**2)
        return c0 + eta * mass * (0.35 * v * grade)

    # Create training data
    dz_vals = np.linspace(-10, 10, 50)
    d_vals = np.linspace(5, 60, 50)
    DZ, D = np.meshgrid(dz_vals, d_vals)
    X_data = np.column_stack([DZ.ravel(), D.ravel()**2])
    Y_data = np.array([pandolf_energy(dz, d) for dz, d in zip(DZ.ravel(), D.ravel())])

    # Fit linear model
    model = LinearRegression().fit(X_data, Y_data)
    a_dz, a_d2 = model.coef_
    intercept = model.intercept_

    return a_dz, a_d2, intercept

def pandolf_cost_expr_from_fit(dz_expr, d2_expr, a_dz, a_d2, intercept):
    a_dz = 581.23
    a_d2 = 0.0  # or use -6.07e-16 if needed
    intercept = 870.36

    cost_expr = intercept + a_dz * dz_expr
    # if d2_expr is not None:
    cost_expr += a_d2 * d2_expr 
    # cost_expr = d2_expr+ dz_expr
    return cost_expr


