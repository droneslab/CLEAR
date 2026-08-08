



import rasterio


def example_1():
    # ---------------- Load Data ---------------- #
    elevation_path = "../sample/nasadem.tif"
    landcover_path = "../sample/landcover.tif"

    with rasterio.open(elevation_path) as elev_src:
        elevation_data = elev_src.read(1)
        elev_transform = elev_src.transform

    with rasterio.open(landcover_path) as lc_src:
        landcover_data = lc_src.read(1)
        lc_transform = lc_src.transform


    return elevation_data, elev_transform, landcover_data, lc_transform, "example_1"



def example_1_512_512():
    # ---------------- Load Data ---------------- #
    elevation_path = "../sample/nasadem.tif"
    landcover_path = "../sample/landcover.tif"

    with rasterio.open(elevation_path) as elev_src:
        elevation_data = elev_src.read(1)
        elev_transform = elev_src.transform

    with rasterio.open(landcover_path) as lc_src:
        landcover_data = lc_src.read(1)
        lc_transform = lc_src.transform


    filename = "../sample/data_wharton.pkl"
    pmap = PickleMap(filename)

    elevation_data = pmap.elevation
    landcover_data = pmap.type


    return elevation_data, elev_transform, landcover_data, lc_transform, "example_1"


import numpy as np
from dataclasses import dataclass, field

import pickle as pkl
@dataclass
class PickleMap:
    filename: str = field(init=True)
    texture: np.ndarray = field(init=False)
    elevation: np.ndarray = field(init=False)
    type: np.ndarray = field(init=False)
    obstacles: np.ndarray = field(init=False)

    def __post_init__(self):
        with open(self.filename, "rb") as fh:
            data = pkl.load(fh)
            self.texture = data["texture"]
            self.elevation = data["elevation"]
            self.type = data["landcover"]
            self._compute_obstacles()

    def _compute_obstacles(self):
        self.obstacles = self.elevation > 400

def example_2():

    elevation_path = "../sample/nasadem.tif"

    with rasterio.open(elevation_path) as elev_src:
        elev_transform = elev_src.transform


    filename = "../sample/multipath_map.pkl"
    pmap = PickleMap(filename)

    elevation_data = pmap.elevation
    landcover_data = pmap.type

    return elevation_data, elev_transform, landcover_data, elev_transform, "example_2"


def example_3():
    # ---------------- Load Data ---------------- #
    elevation_path = "../sample/nasadem.tif"

    with rasterio.open(elevation_path) as elev_src:
        elev_transform = elev_src.transform


    filename = "../sample/data_humphreys.pkl"
    pmap = PickleMap(filename)

    elevation_data = pmap.elevation
    landcover_data = pmap.type

    return elevation_data, elev_transform, landcover_data, elev_transform, "example_3"


def example_4():
    # ---------------- Load Data ---------------- #
    elevation_path = "../sample/nasadem.tif"

    with rasterio.open(elevation_path) as elev_src:
        elev_transform = elev_src.transform


    filename = "../sample/data_Mount Rainier.pkl"
    pmap = PickleMap(filename)

    elevation_data = pmap.elevation
    landcover_data = pmap.type

    return elevation_data, elev_transform, landcover_data, elev_transform, "example_4"







from info_loss_surface import RegionBuilder


def load_data_region_count(example, region_count, decomposition="voronoi", landcover_data=None, elevation_data=None):


    # ---------------- Load Data ---------------- #
    if example == 1:
        # Load example 1 data
        elevation_data_1, elev_transform, landcover_data_1, lc_transform, example_name = example_1()
    elif example == 2:
        elevation_data_1, elev_transform, landcover_data_1, lc_transform, example_name = example_2()
    elif example == 3:
        elevation_data_1, elev_transform, landcover_data_1, lc_transform, example_name = example_3()
    elif example == 4:
        elevation_data_1, elev_transform, landcover_data_1, lc_transform, example_name = example_4()
    else:
        raise ValueError("Invalid example number. Choose 1 or 2.")
    # ---------------- Execute Pipeline ---------------- #


    if landcover_data is None:
        landcover_data = landcover_data_1
    if elevation_data is None:
        elevation_data = elevation_data_1
        
    # print("Voronoi generation took: ", time.time() - st_time_total)
    print("Decomposition: ", decomposition)
    if decomposition =="quadtree":
        rb = RegionBuilderQuadtree(landcover_data, elevation_data, elev_transform, example_name, region_count=region_count)
        print("Quadtree decomposition")
    else:
        rb = RegionBuilderPatches(landcover_data, elevation_data, elev_transform, example_name, region_count=region_count)
    return rb



def load_data_region_count_older(example, region_count, decomposition="voronoi", landcover_data=None, elevation_data=None):


    from info_loss_surface import  get_landcover_boundary_and_elevation_bin_region_count_v7

    # get_landcover_boundary_and_elevation_bin_region_count_v4 : working best for map 1 and 4, not good for map 3

    # ---------------- Load Data ---------------- #
    if landcover_data is not None and elevation_data is not None:
        elevation_data_1 = elevation_data
        landcover_data_1 = landcover_data
        elev_transform = None
        lc_transform = None
        example_name = "external_map"
    elif example == 1:
        # Load example 1 data
        elevation_data_1, elev_transform, landcover_data_1, lc_transform, example_name = example_1()
    elif example == 2:
        elevation_data_1, elev_transform, landcover_data_1, lc_transform, example_name = example_2()
    elif example == 3:
        elevation_data_1, elev_transform, landcover_data_1, lc_transform, example_name = example_3()
    elif example == 4:
        elevation_data_1, elev_transform, landcover_data_1, lc_transform, example_name = example_4()
    else:
        raise ValueError("Invalid example number. Choose 1 or 2.")
    # ---------------- Execute Pipeline ---------------- #


    if landcover_data is None:
        landcover_data = landcover_data_1
    if elevation_data is None:
        elevation_data = elevation_data_1
        
    print("Map Size: ", elevation_data.shape)
    # print("Voronoi generation took: ", time.time() - st_time_total)
    print("Decomposition: ", decomposition)
    if decomposition != "voronoi":
        raise ValueError("The standalone package supports CLEAR only.")
    rb = RegionBuilder(
        landcover_data, elevation_data, elev_transform, example_name,
        region_count=region_count,
        decomposition_function=get_landcover_boundary_and_elevation_bin_region_count_v7,
    )
    return rb
