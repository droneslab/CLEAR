


from scipy.spatial import Voronoi

def sample_hex_grid_from_region_count(landcover_map, sample_count=1000):
    """
    Sample approximately `sample_count` points in a hexagonal grid across the landcover map.
    """
    rows, cols = landcover_map.shape[:2]
    area = rows * cols
    target_density = sample_count / area  # points per pixel

    # Compute hex step size for given density
    step = np.sqrt(2 / (np.sqrt(3) * target_density))
    dy = step * np.sqrt(3) / 2

    sampled_points = []

    num_rows = int(np.ceil(rows / dy))
    for i in range(num_rows):
        y = i * dy
        if y >= rows:
            break
        offset = step / 2 if i % 2 else 0
        num_cols = int(np.ceil((cols - offset) / step))
        for j in range(num_cols):
            x = j * step + offset
            if x < cols:
                sampled_points.append((int(round(x)), int(round(y))))

    sampled_points = np.unique(sampled_points, axis=0)
    return sampled_points

def get_landcover_hex_samples(landcover_map, region_count=1000):
    sampled_points = sample_hex_grid_from_region_count(landcover_map, sample_count=region_count)

    vor = Voronoi(sampled_points)
    return vor


def sample_grid_from_region_count(landcover_map, sample_count=1000):
    """
    Sample approximately `sample_count` points in a grid across the landcover map.
    """
    height, width = landcover_map.shape
    total_points = height * width
    step = round(np.sqrt(total_points / sample_count))
    step = max(1, step)  # Ensure step is at least 1

    sampled_points = np.array([
        (j, i)
        for i in range(0, height, step)
        for j in range(0, width, step)
    ])
    return sampled_points

def get_landcover_grid_samples(landcover_map, region_count=1000):
    sampled_points = sample_grid_from_region_count(landcover_map, sample_count=region_count)

    vor = Voronoi(sampled_points)
    return vor


import numpy as np
from skimage.segmentation import find_boundaries

def get_landcover_boundary_samples(landcover_map, min_samples=1000):
    """
    Uniformly sample at least `min_samples` from landcover boundaries.
    Returns:
        landcover_boundaries: Boolean array of boundary pixels.
        sampled_points: (N, 2) array of (x, y) sampled coordinates.
    """

    h, w = landcover_map.shape

    # Compute all label-wise boundaries
    landcover_boundaries = np.zeros_like(landcover_map, dtype=bool)
    for label_val in np.unique(landcover_map):
        class_mask = (landcover_map == label_val)
        class_boundary = find_boundaries(class_mask, mode='thick', connectivity=4)
        landcover_boundaries |= class_boundary

    # Get all boundary pixel coordinates
    boundary_coords = np.argwhere(landcover_boundaries)  # (y, x)

    # Uniformly subsample
    if len(boundary_coords) > min_samples:
        step = len(boundary_coords) // min_samples
        boundary_coords = boundary_coords[::step]

    # Convert to (x, y)
    sampled_points = np.array([(x, y) for y, x in boundary_coords])
    return sampled_points



def get_landcover_boundary_samples_with_elevation(landcover_map, elevation_map, min_samples=1000, elevation_bins=10):
    """
    Samples at least `min_samples` from the combined landcover + elevation bins boundaries.

    Args:
        landcover_map: 2D array of landcover labels.
        elevation_map: 2D array of elevation values (same size).
        min_samples: Minimum number of boundary samples.
        elevation_bins: Number of bins to discretize elevation.

    Returns:
        sampled_points: (N, 2) array of (x, y) sampled coordinates.
    """
    assert landcover_map.shape == elevation_map.shape
    h, w = landcover_map.shape

    # Bin elevation values
    elev_min, elev_max = elevation_map.min(), elevation_map.max()
    elev_binned = np.digitize(elevation_map, np.linspace(elev_min, elev_max, elevation_bins + 1), right=True)

    # Form mixed label: unique code per landcover-elevation_bin combo
    mixed_label = landcover_map.astype(np.int32) * (elevation_bins + 1) + elev_binned

    # Compute boundaries across mixed regions
    landcover_boundaries = np.zeros_like(mixed_label, dtype=bool)
    for label_val in np.unique(mixed_label):
        mask = (mixed_label == label_val)
        boundary = find_boundaries(mask, mode='thick', connectivity=2)
        landcover_boundaries |= boundary

    # Get boundary coordinates
    boundary_coords = np.argwhere(landcover_boundaries)  # (y, x)

    # Uniform subsampling
    if len(boundary_coords) > min_samples:
        step = len(boundary_coords) // min_samples
        boundary_coords = boundary_coords[::step]

    # Return (x, y) format
    sampled_points = np.array([(x, y) for y, x in boundary_coords])
    return sampled_points



def get_landcover_boundary_samples_with_elevation_v2(landcover_map, elevation_map, min_samples=1000, elevation_bins=5):
    """
    Stepwise boundary sampling:
    1. Landcover-only boundaries (coarse).
    2. Elevation-binned subregions within landcover (fine), in descending area.
    
    Returns:
        sampled_points: (N, 2) array of (x, y) sampled coordinates.
    """
    assert landcover_map.shape == elevation_map.shape
    sampled_coords = []

    h, w = landcover_map.shape
    total_mask = np.zeros_like(landcover_map, dtype=bool)

    # --- Step 1: Boundaries from pure landcover map
    # Compute all label-wise boundaries
    landcover_boundaries = np.zeros_like(landcover_map, dtype=bool)
    for label_val in np.unique(landcover_map):
        class_mask = (landcover_map == label_val)
        class_boundary = find_boundaries(class_mask, mode='thick', connectivity=2)
        landcover_boundaries |= class_boundary

    # Get all boundary pixel coordinates
    boundary_coords = np.argwhere(landcover_boundaries)  # (y, x)

    sampled_coords = [(x, y) for y, x in boundary_coords]

    total_mask = landcover_boundaries

    # Uniformly subsample
    if len(sampled_coords) > min_samples:
        step = len(sampled_coords) // min_samples
        sampled_coords = sampled_coords[::step]



    # --- Step 2: Sort landcover by size
    remaining = min_samples - len(sampled_coords)
    lc_sizes = [(lc, np.sum(landcover_map == lc)) for lc in np.unique(landcover_map)]
    lc_sorted = sorted(lc_sizes, key=lambda x: -x[1])  # descending

    for lc, _ in lc_sorted:
        if remaining <= 0:
            break

        lc_mask = (landcover_map == lc)
        elev_vals = elevation_map[lc_mask]
        if elev_vals.size == 0:
            continue

        bins = np.linspace(elev_vals.min(), elev_vals.max(), elevation_bins + 1)
        elev_binned = np.digitize(elevation_map, bins, right=True)

        for b in range(1, elevation_bins + 1):
            sub_mask = lc_mask & (elev_binned == b)
            if np.count_nonzero(sub_mask) == 0:
                continue
            boundary = find_boundaries(sub_mask, mode='thick', connectivity=2)
            coords = np.argwhere(boundary)
            coords = [tuple(p[::-1]) for p in coords if not total_mask[p[0], p[1]]]  # (x, y)
            total_mask |= boundary
            sampled_coords.extend(coords)
            remaining = min_samples - len(sampled_coords)
            if remaining <= 0:
                break

    return np.array(sampled_coords)




# Try again with fully relaxed parameters and fallback to boundary point if offset goes outside
def sample_boundary_aware_voronoi_seeds_v2(landcover_map, min_samples=1000, stride=1, margin=1, distance_thresh=0.0):
    from skimage.segmentation import find_boundaries
    from scipy.ndimage import distance_transform_edt, gaussian_filter

    H, W = landcover_map.shape
    boundary_mask = find_boundaries(landcover_map, mode='thick', connectivity=2)
    inverted = ~boundary_mask
    dist_map = distance_transform_edt(inverted)

    # Gradient-based direction into class
    grad_dy, grad_dx = np.gradient(gaussian_filter(dist_map, sigma=1.0))
    grad_mag = np.sqrt(grad_dx**2 + grad_dy**2)
    grad_dx[grad_mag > 0] /= grad_mag[grad_mag > 0]
    grad_dy[grad_mag > 0] /= grad_mag[grad_mag > 0]

    seeds = []
    for y in range(margin, H - margin, stride):
        for x in range(margin, W - margin, stride):
            if boundary_mask[y, x]:# and dist_map[y, x] >= distance_thresh:
                dx = int(round(grad_dx[y, x]))
                dy = int(round(grad_dy[y, x]))
                nx, ny = x + dx * margin, y + dy * margin
                if 0 <= nx < W and 0 <= ny < H:
                    seeds.append((nx, ny))
                else:
                    seeds.append((x, y))  # fallback to boundary point

    # Deduplicate and clip
    seeds = list(dict.fromkeys(seeds))
    if len(seeds) > min_samples:
        step = len(seeds) // min_samples
        seeds = seeds[::step]

    return np.array(seeds)


def get_landcover_boundary_samples_with_elevation_v3(landcover_map, elevation_map, min_samples=1000, elevation_bins=5):
    """
    Stepwise boundary sampling:
    1. Landcover-only boundaries (coarse).
    2. Elevation-binned subregions within landcover (fine), in descending area.
    
    Returns:
        sampled_points: (N, 2) array of (x, y) sampled coordinates.
    """
    assert landcover_map.shape == elevation_map.shape
    sampled_coords = []

    h, w = landcover_map.shape
    total_mask = np.zeros_like(landcover_map, dtype=bool)

    # --- Step 1: Boundaries from pure landcover map
    # Compute all label-wise boundaries
    landcover_boundaries = np.zeros_like(landcover_map, dtype=bool)
    for label_val in np.unique(landcover_map):
        class_mask = (landcover_map == label_val)
        class_boundary = find_boundaries(class_mask, mode='thick', connectivity=2)
        landcover_boundaries |= class_boundary

    # Get all boundary pixel coordinates
    boundary_coords = np.argwhere(landcover_boundaries)  # (y, x)

    sampled_coords = [(x, y) for y, x in boundary_coords]

    total_mask = landcover_boundaries

    # Uniformly subsample
    if len(sampled_coords) > min_samples:


        sampled_coords = sample_boundary_aware_voronoi_seeds_v2(landcover_map, min_samples=min_samples, stride=1, margin=1, distance_thresh=0.0)


        # step = len(sampled_coords) // min_samples
        # sampled_coords = sampled_coords[::step]



        



    # --- Step 2: Sort landcover by size
    remaining = min_samples - len(sampled_coords)
    lc_sizes = [(lc, np.sum(landcover_map == lc)) for lc in np.unique(landcover_map)]
    lc_sorted = sorted(lc_sizes, key=lambda x: -x[1])  # descending

    for lc, _ in lc_sorted:
        if remaining <= 0:
            break

        lc_mask = (landcover_map == lc)
        elev_vals = elevation_map[lc_mask]
        if elev_vals.size == 0:
            continue

        bins = np.linspace(elev_vals.min(), elev_vals.max(), elevation_bins + 1)
        elev_binned = np.digitize(elevation_map, bins, right=True)

        for b in range(1, elevation_bins + 1):
            sub_mask = lc_mask & (elev_binned == b)
            if np.count_nonzero(sub_mask) == 0:
                continue
            boundary = find_boundaries(sub_mask, mode='thick', connectivity=2)
            coords = np.argwhere(boundary)
            coords = [tuple(p[::-1]) for p in coords if not total_mask[p[0], p[1]]]  # (x, y)
            total_mask |= boundary
            sampled_coords.extend(coords)
            remaining = min_samples - len(sampled_coords)
            if remaining <= 0:
                break

    return np.array(sampled_coords)







from scipy.stats import entropy
from skimage.util.shape import view_as_windows

def compute_local_entropy_windowed(landcover_map, coords, window_size=9):
    """
    Compute local entropy in a square window around each (x, y) coordinate.
    Returns list of (entropy, (x, y)) tuples.
    """
    H, W = landcover_map.shape
    pad = window_size // 2
    padded = np.pad(landcover_map, pad_width=pad, mode='edge')

    entropy_points = []
    for x, y in coords:
        x_pad, y_pad = x + pad, y + pad
        window = padded[y_pad - pad:y_pad + pad + 1, x_pad - pad:x_pad + pad + 1]
        hist = np.bincount(window.ravel())
        probs = hist / hist.sum()
        e = entropy(probs, base=2)
        entropy_points.append((e, (x, y)))

    return entropy_points

def entropy_windowed_downsample(coords, landcover_map, min_samples, window_size=9, seed=0):
    np.random.seed(seed)
    entropy_points = compute_local_entropy_windowed(landcover_map, coords, window_size)
    entropy_points.sort(reverse=True, key=lambda x: x[0])  # sort by entropy
    selected = [pt for (_, pt) in entropy_points[:min_samples]]
    return selected


def get_landcover_boundary_samples_with_elevation_entropy_old(landcover_map, elevation_map, min_samples=1000, elevation_bins=5):
    assert landcover_map.shape == elevation_map.shape
    H, W = landcover_map.shape
    sampled_coords = []
    total_mask = np.zeros((H, W), dtype=bool)

    # --- Step 1: Coarse boundaries from full landcover map
    coarse_boundaries = np.zeros((H, W), dtype=bool)
    for label in np.unique(landcover_map):
        mask = (landcover_map == label)
        boundary = find_boundaries(mask, mode='thick', connectivity=2)
        coarse_boundaries |= boundary

    boundary_coords = np.argwhere(coarse_boundaries)
    coords = [(x, y) for y, x in boundary_coords]

    if len(coords) > min_samples:
        selected_coords = entropy_windowed_downsample(coords, landcover_map, min_samples, window_size=2, seed=0)
    else:
        selected_coords = coords

    for x, y in selected_coords:
        sampled_coords.append((x, y))
        total_mask[y, x] = True

    remaining = min_samples - len(sampled_coords)
    if remaining <= 0:
        return np.array(sampled_coords)

    # --- Step 2: Refined sampling by elevation bins per landcover
    lc_sorted = sorted(
        [(lc, np.sum(landcover_map == lc)) for lc in np.unique(landcover_map)],
        key=lambda x: -x[1]
    )

    for lc, _ in lc_sorted:
        if remaining <= 0:
            break

        lc_mask = (landcover_map == lc)
        elev_vals = elevation_map[lc_mask]
        if elev_vals.size == 0:
            continue

        bins = np.linspace(elev_vals.min(), elev_vals.max(), elevation_bins + 1)
        elev_binned = np.digitize(elevation_map, bins, right=True)

        for b in range(1, elevation_bins + 1):
            sub_mask = lc_mask & (elev_binned == b)
            boundary = find_boundaries(sub_mask, mode='thick', connectivity=2)
            coords = np.argwhere(boundary & (~total_mask))
            np.random.shuffle(coords)

            for y, x in coords:
                sampled_coords.append((x, y))
                total_mask[y, x] = True
                remaining -= 1
                if remaining <= 0:
                    break
            if remaining <= 0:
                break

    return np.array(sampled_coords)





from scipy.stats import entropy
from skimage.util.shape import view_as_windows

def compute_local_entropy_windowed(landcover_map, coords, window_size=9):
    """
    Compute local entropy in a square window around each (x, y) coordinate.
    Returns list of (entropy, (x, y)) tuples.
    """
    H, W = landcover_map.shape
    pad = window_size // 2
    padded = np.pad(landcover_map, pad_width=pad, mode='edge')

    entropy_points = []
    for x, y in coords:
        x_pad, y_pad = x + pad, y + pad
        window = padded[y_pad - pad:y_pad + pad + 1, x_pad - pad:x_pad + pad + 1]
        hist = np.bincount(window.ravel())
        probs = hist / hist.sum()
        e = entropy(probs, base=2)
        entropy_points.append((e, (x, y)))

    return entropy_points

def entropy_windowed_downsample(coords, landcover_map, min_samples, window_size=9, seed=0):
    np.random.seed(seed)
    entropy_points = compute_local_entropy_windowed(landcover_map, coords, window_size)
    entropy_points.sort(reverse=True, key=lambda x: x[0])  # sort by entropy
    selected = [pt for (_, pt) in entropy_points[:min_samples]]
    return selected



from tqdm import tqdm
def boundary_preserving_removal(coords, landcover_map, min_samples=1000, window=9, lam=0.5):
    from scipy.stats import entropy
    from scipy.ndimage import distance_transform_edt
    from skimage.segmentation import find_boundaries

    H, W = landcover_map.shape
    pad = window // 2
    padded = np.pad(landcover_map, pad, mode='edge')

    boundaries = find_boundaries(landcover_map, mode='thick')
    dist = distance_transform_edt(~boundaries)
    dmax = dist.max()

    scored = []
    for x, y in tqdm(coords):
        x_pad, y_pad = x + pad, y + pad
        patch = padded[y_pad-pad:y_pad+pad+1, x_pad-pad:x_pad+pad+1]
        hist = np.bincount(patch.ravel(), minlength=landcover_map.max()+1)
        p = hist / hist.sum()
        Hval = entropy(p, base=2)
        Dval = dist[y, x]
        score = lam * Hval + (1 - lam) * (1 - Dval / dmax)
        scored.append((score, (x, y)))

    # Keep top N points with highest boundary scores
    scored.sort(reverse=True)
    N = min_samples
    return [pt for (_, pt) in scored[:N]]







from scipy.stats import entropy
from scipy.ndimage import distance_transform_edt
from skimage.segmentation import find_boundaries
from skimage.util import view_as_windows
from tqdm import tqdm
import numpy as np
from shapely.geometry import Point
from shapely.strtree import STRtree

def poisson_disk_filter(points, min_dist, size):
    """
    Greedily filter points to enforce minimum separation using STRtree.
    """
    accepted = []
    tree = None

    for pt in points:
        p = Point(pt)
        # if tree is None or not tree.query(p.buffer(min_dist)):
        if tree is None or len(tree.query(p.buffer(min_dist))) == 0:

            accepted.append(pt)
            if len(accepted) % 20 == 0:
                tree = STRtree([Point(xy) for xy in accepted])
            if len(accepted) >= size:
                break
    return accepted

def boundary_preserving_removal_poisson(coords, landcover_map, min_samples=1000, window=9, lam=0.5, min_dist=1):
    H, W = landcover_map.shape
    pad = window // 2
    padded = np.pad(landcover_map, pad, mode='edge')
    patches = view_as_windows(padded, (window, window))

    boundaries = find_boundaries(landcover_map, mode='thick')
    dist = distance_transform_edt(~boundaries)
    dmax = dist.max()

    max_label = landcover_map.max() + 1
    scored = []
    for x, y in coords:
        if y >= H or x >= W:
            continue
        patch = patches[y, x]
        hist = np.bincount(patch.ravel(), minlength=max_label)
        p = hist / hist.sum()
        Hval = entropy(p, base=2)
        Dval = dist[y, x]
        score = lam * Hval + (1 - lam) * (1 - Dval / dmax)
        scored.append((score, (x, y)))

    scored.sort(reverse=True)  # high score = better candidate
    sorted_pts = [pt for _, pt in scored]

    # Poisson-disk like suppression
    return poisson_disk_filter(sorted_pts, min_dist=min_dist, size=min_samples)





from skimage.util import view_as_windows

def boundary_preserving_removal_fast(coords, landcover_map, min_samples=1000, window=9, lam=0.5):
    from scipy.stats import entropy
    from scipy.ndimage import distance_transform_edt
    from skimage.segmentation import find_boundaries

    H, W = landcover_map.shape
    pad = window // 2
    padded = np.pad(landcover_map, pad, mode='edge')
    patches = view_as_windows(padded, (window, window))

    boundaries = find_boundaries(landcover_map, mode='thick')
    dist = distance_transform_edt(~boundaries)
    dmax = dist.max()

    max_label = landcover_map.max() + 1
    scored = []
    for x, y in tqdm(coords):
        patch = patches[y, x]
        hist = np.bincount(patch.ravel(), minlength=max_label)
        p = hist / hist.sum()
        Hval = entropy(p, base=2)
        Dval = dist[y, x]
        score = lam * Hval + (1 - lam) * (1 - Dval / dmax)
        scored.append((score, (x, y)))

    scored.sort(reverse=True)
    return [pt for (_, pt) in scored[:min_samples]]




def get_landcover_boundary_samples_with_elevation_entropy(landcover_map, elevation_map, min_samples=1000, elevation_bins=5):
    assert landcover_map.shape == elevation_map.shape
    H, W = landcover_map.shape
    sampled_coords = []
    total_mask = np.zeros((H, W), dtype=bool)

    # --- Step 1: Coarse boundaries from full landcover map
    coarse_boundaries = np.zeros((H, W), dtype=bool)
    for label in np.unique(landcover_map):
        mask = (landcover_map == label)
        boundary = find_boundaries(mask, mode='thick', connectivity=2)
        coarse_boundaries |= boundary

    boundary_coords = np.argwhere(coarse_boundaries)
    coords = [(x, y) for y, x in boundary_coords]

    if len(coords) > min_samples:
        selected_coords = boundary_preserving_removal_fast(coords, landcover_map, min_samples, window=3)
        # selected_coords = boundary_preserving_removal(coords, landcover_map, min_samples, window=3)

    else:
        selected_coords = coords

    for x, y in selected_coords:
        sampled_coords.append((x, y))
        total_mask[y, x] = True

    remaining = min_samples - len(sampled_coords)
    if remaining <= 0:
        return np.array(sampled_coords)

    # --- Step 2: Refined sampling by elevation bins per landcover
    lc_sorted = sorted(
        [(lc, np.sum(landcover_map == lc)) for lc in np.unique(landcover_map)],
        key=lambda x: -x[1]
    )

    for lc, _ in lc_sorted:
        if remaining <= 0:
            break

        lc_mask = (landcover_map == lc)
        elev_vals = elevation_map[lc_mask]
        if elev_vals.size == 0:
            continue

        bins = np.linspace(elev_vals.min(), elev_vals.max(), elevation_bins + 1)
        elev_binned = np.digitize(elevation_map, bins, right=True)

        for b in range(1, elevation_bins + 1):
            sub_mask = lc_mask & (elev_binned == b)
            boundary = find_boundaries(sub_mask, mode='thick', connectivity=2)
            coords = np.argwhere(boundary & (~total_mask))
            np.random.shuffle(coords)

            for y, x in coords:
                sampled_coords.append((x, y))
                total_mask[y, x] = True
                remaining -= 1
                if remaining <= 0:
                    break
            if remaining <= 0:
                break

    return np.array(sampled_coords)


def get_uniform_boundary_samples_inside_outside_elevation(landcover_map, elevation_map, min_samples=1000, elevation_bins=5, stride=2):
    """
    Uniformly sample boundary pixels on both inner and outer sides within elevation bins.
    """
    from skimage.segmentation import find_boundaries

    H, W = landcover_map.shape
    sampled_coords = []

    # Compute elevation bins
    elev_min, elev_max = elevation_map.min(), elevation_map.max()
    bins = np.linspace(elev_min, elev_max, elevation_bins + 1)
    elev_binned = np.digitize(elevation_map, bins, right=True)

    # Compute boundaries and inner/outer masks
    landcover_boundaries = np.zeros_like(landcover_map, dtype=bool)
    for label_val in np.unique(landcover_map):
        mask = landcover_map == label_val
        landcover_boundaries |= find_boundaries(mask, mode='thick', connectivity=2)

    inner_mask = np.zeros_like(landcover_map, dtype=bool)
    outer_mask = np.zeros_like(landcover_map, dtype=bool)

    for y in range(1, H - 1):
        for x in range(1, W - 1):
            if landcover_boundaries[y, x]:
                label = landcover_map[y, x]
                for dy in [-1, 0, 1]:
                    for dx in [-1, 0, 1]:
                        ny, nx = y + dy, x + dx
                        if landcover_map[ny, nx] != label:
                            outer_mask[ny, nx] = True
                        else:
                            inner_mask[ny, nx] = True

    # Subsample within each elevation bin
    def subsample(mask, bin_id):
        return [(x, y) for y in range(0, H, stride) for x in range(0, W, stride)
                if mask[y, x] and elev_binned[y, x] == bin_id]

    all_pts = []
    for b in range(1, elevation_bins + 1):
        inner_pts = subsample(inner_mask, b)
        outer_pts = subsample(outer_mask, b)
        all_pts.extend(inner_pts + outer_pts)

    # Limit to min_samples
    if len(all_pts) > min_samples:
        step = len(all_pts) // min_samples
        all_pts = all_pts[::step]

    return np.array(all_pts)



def get_landcover_boundary_region_count(landcover_map, region_count=1000):
    
    sampled_points = get_landcover_boundary_samples(landcover_map, min_samples=region_count)

    vor = Voronoi(sampled_points)
    return vor


def get_landcover_boundary_and_elevation_bin_region_count(landcover_map, elevation_map, region_count=1000, elevation_bins=10):
    
    sampled_points = get_landcover_boundary_samples_with_elevation(landcover_map, elevation_map, min_samples=region_count, elevation_bins=elevation_bins)

    vor = Voronoi(sampled_points)
    return vor








from skimage.morphology import dilation, square
from skimage.feature import peak_local_max

from scipy.ndimage import distance_transform_edt
from skimage.filters import sobel
def get_uniform_fill_samples_elevation_v3(elevation_map, sampled_coords, min_distance=10, num_points=1000, grad_threshold=0.2):
    """
    Uniformly sample from uncovered flat elevation regions, ensuring:
    - Points are min_distance apart
    - No overlap with existing samples
    - Flat regions only
    - Target number of points is met if possible

    Args:
        elevation_map: 2D elevation map
        sampled_coords: (N, 2) array of existing (x, y) samples
        min_distance: minimum spacing between points
        num_points: desired number of new points
        grad_threshold: gradient threshold for flatness

    Returns:
        uniform_coords: (K, 2) array of new (x, y) sample points
    """
    H, W = elevation_map.shape
    grad_mag = np.sqrt(sobel(elevation_map, axis=0)**2 + sobel(elevation_map, axis=1)**2)
    flat_mask = grad_mag < grad_threshold

    # Initialize mask from existing points
    init_mask = np.zeros_like(elevation_map, dtype=bool)
    for x, y in sampled_coords:
        if 0 <= x < W and 0 <= y < H:
            init_mask[y, x] = True

    blocked = dilation(init_mask, square(min_distance))
    candidate_mask = flat_mask & (~blocked)

    # Iteratively sample using distance transform and suppress nearby points
    dist_map = distance_transform_edt(candidate_mask)
    selected_points = []

    for _ in range(num_points):
        if np.max(dist_map) < min_distance:
            break  # no more well-separated points available
        y, x = np.unravel_index(np.argmax(dist_map), dist_map.shape)
        selected_points.append((x, y))

        # Block out region around selected point
        mask = np.zeros_like(dist_map, dtype=bool)
        mask[max(0, y - min_distance):y + min_distance + 1,
             max(0, x - min_distance):x + min_distance + 1] = True
        dist_map[mask] = 0
        dist_map = distance_transform_edt(dist_map > 0)

    return np.array(selected_points)



def sample_landcover_boundaries(landcover_map, lc_sample_count):
    """
    Samples boundary points from the landcover map.

    Args:
        landcover_map: 2D array
        lc_sample_count: number of points to sample

    Returns:
        lc_coords: (N, 2) array of (x, y) sampled points
        landcover_boundaries: full boundary mask for reference
    """
    from skimage.segmentation import find_boundaries

    lc_coords = []
    landcover_boundaries = np.zeros_like(landcover_map, dtype=bool)
    for label_val in np.unique(landcover_map):
        class_mask = (landcover_map == label_val)
        class_boundary = find_boundaries(class_mask, mode='thick', connectivity=2)
        landcover_boundaries |= class_boundary

    boundary_coords = np.argwhere(landcover_boundaries)
    coords = [(x, y) for y, x in boundary_coords]

    print("Initial Landcover points:", len(coords))

    if len(coords) > lc_sample_count:
        step = len(coords) // lc_sample_count
        coords = coords[::step]

    lc_coords.extend(coords)
    return np.array(lc_coords), landcover_boundaries



from scipy.spatial import cKDTree

# Define fast uniform fill sampler again
def get_uniform_fill_samples_elevation_fast(elevation_map, sampled_coords, min_distance=10, num_points=1000, grad_threshold=0.2):
    H, W = elevation_map.shape
    grad_mag = np.sqrt(sobel(elevation_map, axis=0)**2 + sobel(elevation_map, axis=1)**2)
    flat_mask = grad_mag < grad_threshold

    mask = flat_mask.copy()
    for x, y in sampled_coords:
        if 0 <= x < W and 0 <= y < H:
            mask[max(0, y - min_distance):min(H, y + min_distance + 1),
                 max(0, x - min_distance):min(W, x + min_distance + 1)] = False

    candidate_coords = np.argwhere(mask)
    if len(candidate_coords) == 0:
        return np.empty((0, 2), dtype=int)

    np.random.shuffle(candidate_coords)
    tree = cKDTree(candidate_coords)
    selected = []
    occupied = np.zeros(len(candidate_coords), dtype=bool)

    for i, pt in enumerate(candidate_coords):
        if occupied[i]:
            continue
        selected.append((pt[1], pt[0]))  # (x, y)
        if len(selected) >= num_points:
            break
        idxs = tree.query_ball_point(pt, r=min_distance)
        occupied[idxs] = True

    return np.array(selected)


# Update main function to use the above
def get_landcover_and_elevation_boundary_samples_v2(
    landcover_map,
    elevation_map,
    min_samples=1000,
    elevation_ratio=0.2,
    uniform_fill_distance=10,
    grad_threshold=0.2
):
    """
    Combined sampler using:
    1. Landcover boundaries
    2. Elevation-aware uniform fill in uncovered flat areas

    Returns:
        final_samples: (N, 2) array of (x, y) coordinates
    """
    assert landcover_map.shape == elevation_map.shape

    elev_sample_count = int(min_samples * elevation_ratio)
    lc_sample_count = min_samples - elev_sample_count

    if lc_sample_count >0:
        # Step 1: Sample from landcover boundaries
        lc_coords, _ = sample_landcover_boundaries(landcover_map, lc_sample_count)

        print("======Landcover points:", len(lc_coords))
        if len(lc_coords) > min_samples:
            lc_coords = boundary_preserving_removal_fast(lc_coords, landcover_map, min_samples, window=3)

            return np.array(lc_coords), lc_coords, []
    else:
        lc_coords = np.empty((0, 2), dtype=int)
        print("======Landcover points: 0")
    #Newly added not tested
    elev_sample_count = min_samples - len(lc_coords)

    # Step 2: Elevation-aware uniform fill
    fill_coords = get_uniform_fill_samples_elevation_fast(
        elevation_map, lc_coords, min_distance=uniform_fill_distance,
        num_points=elev_sample_count, grad_threshold=grad_threshold
    )

    final_coords = np.vstack([lc_coords, fill_coords])
    return final_coords, lc_coords, fill_coords




def get_landcover_boundary_and_elevation_bin_region_count_v2(landcover_map, elevation_map, region_count=1000, elevation_bins=10, min_area=1, flatness_ratio=0.7):
    # COMMENTED out is the best decomposition method, but causing the plane fitting to get stuck
    #  ( result dir : data/dataset_results)
    ## BEST TILL NOW for decomposition only
    # V3 
    sampled_points = get_landcover_boundary_samples_with_elevation_entropy(landcover_map, elevation_map, min_samples=region_count, elevation_bins=elevation_bins)

    ##NOTE: Following is tested to be BEST so far, revert back if other methods are not giving good results for decomposition and planning
    ## Following is BEST so far for decomposition only ( result dir : data/dataset_results_v3)
    #V1

    # sampled_points =  get_landcover_boundary_samples_with_elevation_v2(landcover_map, elevation_map, min_samples=region_count, elevation_bins=elevation_bins)

    # # # only tested for path planning 
    # # #V2 
    # # results in "./data/dataset_results_samples_v2"
    # sampled_points, _, _ = get_landcover_and_elevation_boundary_samples_v2(
    #     landcover_map,
    #     elevation_map,
    #     min_samples=region_count,
    #     elevation_ratio=0.2,
    #     uniform_fill_distance=2,
    #     grad_threshold=0.2
    # )

    # print("======Landcover total points:", sampled_points.shape, landcover_map.shape, elevation_map.shape, region_count)
    # TESTing this : results bad for 12 Ranked entropy samples ( result dir : data/dataset_results_v2)
    # sampled_points = get_landcover_boundary_samples_with_elevation_v3(landcover_map, elevation_map, min_samples=region_count, elevation_bins=elevation_bins)

    vor = Voronoi(sampled_points)
    return vor




def get_landcover_boundary_and_elevation_bin_region_count_v3(landcover_map, elevation_map, region_count=1000, elevation_bins=10, min_area=1):

    # # only tested for path planning 
    # #V2 
    # results in "./data/dataset_results_samples_v2"

    # traied so far: elevation_ratio=0.2


    sampled_points, _, _ = get_landcover_and_elevation_boundary_samples_v2(
        landcover_map,
        elevation_map,
        min_samples=region_count,
        elevation_ratio=1.0,
        uniform_fill_distance=2,
        grad_threshold=0.05
    )

    # print("======Landcover total points:", sampled_points.shape, landcover_map.shape, elevation_map.shape, region_count)
    # TESTing this : results bad for 12 Ranked entropy samples ( result dir : data/dataset_results_v2)
    # sampled_points = get_landcover_boundary_samples_with_elevation_v3(landcover_map, elevation_map, min_samples=region_count, elevation_bins=elevation_bins)

    vor = Voronoi(sampled_points)
    return vor







def get_uniform_then_boundary_samples(
    landcover_map, elevation_map, min_samples=20000, min_distance=10):

    # Ensure same shape
    assert landcover_map.shape == elevation_map.shape, \
        f"Shape mismatch: {landcover_map.shape} vs {elevation_map.shape}"

    H, W = landcover_map.shape
    total_mask = np.zeros((H, W), dtype=bool)
    sampled_coords = []

    # Compute local flatness
    local_std = generic_filter(elevation_map, np.std, size=5)

    # Step 1: uniform Poisson-like sampling biased to flatness
    flatness_sorted = np.argsort(local_std.ravel())
    flat_coords = np.column_stack(np.unravel_index(flatness_sorted, (H, W)))

    r = min_distance
    for y, x in flat_coords:
        if len(sampled_coords) >= min_samples:
            break
        if total_mask[y, x]:
            continue
        sampled_coords.append([x, y])
        y0, y1 = max(0, y-r), min(H, y+r+1)
        x0, x1 = max(0, x-r), min(W, x+r+1)
        total_mask[y0:y1, x0:x1] = True

    sampled_coords = np.array(sampled_coords)

    # Step 2: boundary-based replacement
    coarse_boundaries = np.zeros((H, W), dtype=bool)
    for label in np.unique(landcover_map):
        coarse_boundaries |= find_boundaries(landcover_map == label, mode='thick', connectivity=2)

    boundary_coords = np.argwhere(coarse_boundaries)

    n_boundary = min(min_samples - len(sampled_coords), len(boundary_coords))

    print(f"Number of boundary coordinates found: {len(boundary_coords)}")
    print("n_boundary:", n_boundary)
    if n_boundary > 0 and len(boundary_coords) > 0:
        std_vals = local_std[boundary_coords[:, 0], boundary_coords[:, 1]]
        sorted_idx = np.argsort(std_vals)
        boundary_coords = boundary_coords[sorted_idx]

        selected = boundary_coords[:n_boundary]
        print(f"Number of boundary coordinates selected: {len(selected)}")
        replacement = np.array([[x, y] for y, x in selected])
        # sampled_coords[:n_boundary] = replacement
        # sampled_coords.extend(replacement)  # Ensure we return the full set of sampled coordinates
        sampled_coords = np.concatenate((sampled_coords, replacement), axis=0)

    return sampled_coords





from skimage.measure import label, regionprops

from scipy.ndimage import generic_filter
from skimage.segmentation import find_boundaries
import numpy as np

def get_uniform_then_blob_then_boundary_samples(
    landcover_map, elevation_map, min_samples=20000, min_distance=10):
    """
    1. Uniform Poisson sampling (min_distance constraint).
    2. Add seeds for flat landcover blobs (only duplicate check).
    3. Add seeds for boundaries (only duplicate check).
    """
    assert landcover_map.shape == elevation_map.shape, \
        f"Shape mismatch: {landcover_map.shape} vs {elevation_map.shape}"

    H, W = landcover_map.shape

    # Two masks:
    spacing_mask = np.zeros((H, W), dtype=bool)  # for Step 1 spacing
    point_mask = np.zeros((H, W), dtype=bool)    # for duplicate checking
    sampled_coords = []

    local_std = generic_filter(elevation_map, np.std, size=5)

    # --- Step 1: Poisson-like uniform sampling with min_distance ---
    flatness_sorted = np.argsort(local_std.ravel())
    flat_coords = np.column_stack(np.unravel_index(flatness_sorted, (H, W)))
    r = min_distance
    for y, x in flat_coords:
        if len(sampled_coords) >= min_samples:
            break
        if spacing_mask[y, x]:
            continue
        sampled_coords.append([x, y])
        # Apply blocking only to spacing_mask
        y0, y1 = max(0, y-r), min(H, y+r+1)
        x0, x1 = max(0, x-r), min(W, x+r+1)
        spacing_mask[y0:y1, x0:x1] = True
        point_mask[y, x] = True

    # --- Step 2: Add seeds for flat landcover blobs (no min_distance) ---
    std_thresh = np.percentile(local_std, 25)
    flat_mask = local_std <= std_thresh
    for label_val in np.unique(landcover_map):
        mask = (landcover_map == label_val) & flat_mask
        if not np.any(mask):
            continue
        labeled = label(mask)
        for region in regionprops(labeled):
            cy, cx = np.round(region.centroid).astype(int)
            if not point_mask[cy, cx]:
                sampled_coords.append([cx, cy])
                point_mask[cy, cx] = True
            if len(sampled_coords) >= min_samples:
                break
        if len(sampled_coords) >= min_samples:
            break

    # return np.array(sampled_coords)

    # --- Step 3: Boundary-based seeds (no min_distance) ---
    needed = max(0, min_samples - len(sampled_coords))
    if needed > 0:
        coarse_boundaries = np.zeros((H, W), dtype=bool)
        for label_val in np.unique(landcover_map):
            coarse_boundaries |= find_boundaries(
                landcover_map == label_val, mode='thick', connectivity=2
            )

        boundary_coords = np.argwhere(coarse_boundaries)
        if len(boundary_coords) > 0:
            std_vals = local_std[boundary_coords[:, 0], boundary_coords[:, 1]]
            sorted_idx = np.argsort(std_vals)
            boundary_coords = boundary_coords[sorted_idx]

            added = 0
            for y, x in boundary_coords:
                if added >= needed:
                    break
                if point_mask[y, x]:
                    continue
                sampled_coords.append([x, y])
                point_mask[y, x] = True  # only duplicate check
                added += 1

    return np.array(sampled_coords)




def get_landcover_boundary_and_elevation_bin_region_count_v5(landcover_map, elevation_map, region_count=1000, elevation_bins=10, min_area=1):

    # sampled_points = get_landcover_boundary_samples_with_elevation_entropy_sampled_flatter(landcover_map, elevation_map, min_samples=region_count, elevation_bins=elevation_bins)

    # min_area =2 # best for Wharton

    # min_area = 4
    sampled_points = get_uniform_then_blob_then_boundary_samples(landcover_map, elevation_map, min_samples=region_count, min_distance=min_area)
    vor = Voronoi(sampled_points)
    return vor





import numpy as np
from skimage.measure import label, regionprops
from skimage.segmentation import find_boundaries
from skimage.util import view_as_windows
from scipy.ndimage import generic_filter, distance_transform_edt
from scipy.stats import entropy

def boundary_preserving_removal_fast(coords, landcover_map, min_samples=1000,
                                     window=9, lam=0.5):
    H, W = landcover_map.shape
    pad = window // 2
    padded = np.pad(landcover_map, pad, mode='edge')
    patches = view_as_windows(padded, (window, window))

    boundaries = find_boundaries(landcover_map, mode='thick')
    dist = distance_transform_edt(~boundaries)
    dmax = dist.max()
    max_label = landcover_map.max() + 1

    scored = []
    for x, y in coords:
        patch = patches[y, x]
        hist = np.bincount(patch.ravel(), minlength=max_label)
        p = hist / hist.sum()
        Hval = entropy(p, base=2)
        Dval = dist[y, x]
        score = lam * Hval + (1 - lam) * (1 - Dval / dmax)
        scored.append((score, (x, y)))

    scored.sort(reverse=True)
    return [pt for (_, pt) in scored[:min_samples]]


def unified_hybrid_sampling(
    landcover_map, elevation_map, N_total=2000,
    alpha=0.5, min_distance=10, elevation_bins=5
):
    """
    Unified hybrid sampler.
    alpha = 0   -> flatness-driven
    alpha = 1   -> boundary-aware
    """

    assert landcover_map.shape == elevation_map.shape

    H, W = landcover_map.shape
    local_std = generic_filter(elevation_map, np.std, size=5)

    # Quotas
    N_boundary = int(round(alpha * N_total))
    N_flat = N_total - N_boundary

    sampled_coords = []
    point_mask = np.zeros((H, W), dtype=bool)

    # ---- Flatness-driven seeds ----
    if N_flat > 0:
        spacing_mask = np.zeros((H, W), dtype=bool)
        flatness_sorted = np.argsort(local_std.ravel())
        flat_coords = np.column_stack(np.unravel_index(flatness_sorted, (H, W)))
        r = min_distance
        for y, x in flat_coords:
            if len(sampled_coords) >= N_flat:
                break
            if spacing_mask[y, x]:
                continue
            sampled_coords.append([x, y])
            point_mask[y, x] = True
            y0, y1 = max(0, y-r), min(H, y+r+1)
            x0, x1 = max(0, x-r), min(W, x+r+1)
            spacing_mask[y0:y1, x0:x1] = True

        # Add flat blob centroids
        std_thresh = np.percentile(local_std, 25)
        flat_mask = local_std <= std_thresh
        for label_val in np.unique(landcover_map):
            mask = (landcover_map == label_val) & flat_mask
            if not np.any(mask):
                continue
            labeled = label(mask)
            for region in regionprops(labeled):
                if len(sampled_coords) >= N_flat:
                    break
                cy, cx = np.round(region.centroid).astype(int)
                if not point_mask[cy, cx]:
                    sampled_coords.append([cx, cy])
                    point_mask[cy, cx] = True
            if len(sampled_coords) >= N_flat:
                break

    # ---- Boundary-aware seeds ----
    if N_boundary > 0:
        coarse_boundaries = np.zeros((H, W), dtype=bool)
        for lbl in np.unique(landcover_map):
            coarse_boundaries |= find_boundaries(
                landcover_map == lbl, mode='thick', connectivity=2
            )

        boundary_coords = np.argwhere(coarse_boundaries)
        coords = [(x, y) for y, x in boundary_coords]

        if len(coords) > N_boundary:
            selected_coords = boundary_preserving_removal_fast(
                coords, landcover_map, N_boundary, window=3
            )
        else:
            selected_coords = coords

        # Add boundary-aware seeds
        for x, y in selected_coords:
            if len(sampled_coords) >= N_total:
                break
            if not point_mask[y, x]:
                sampled_coords.append([x, y])
                point_mask[y, x] = True

        # Fill remaining quota by elevation-bin refinement
        remaining = N_total - len(sampled_coords)
        if remaining > 0:
            lc_sorted = sorted(
                [(lc, np.sum(landcover_map == lc)) for lc in np.unique(landcover_map)],
                key=lambda x: -x[1]
            )
            for lc, _ in lc_sorted:
                if remaining <= 0:
                    break
                lc_mask = (landcover_map == lc)
                elev_vals = elevation_map[lc_mask]
                if elev_vals.size == 0:
                    continue
                bins = np.linspace(elev_vals.min(), elev_vals.max(), elevation_bins + 1)
                elev_binned = np.digitize(elevation_map, bins, right=True)

                for b in range(1, elevation_bins + 1):
                    sub_mask = lc_mask & (elev_binned == b)
                    boundary = find_boundaries(sub_mask, mode='thick', connectivity=2)
                    coords = np.argwhere(boundary & (~point_mask))
                    np.random.shuffle(coords)
                    for y, x in coords:
                        if remaining <= 0:
                            break
                        sampled_coords.append([x, y])
                        point_mask[y, x] = True
                        remaining -= 1
                    if remaining <= 0:
                        break

    return np.array(sampled_coords)






def get_landcover_boundary_and_elevation_bin_region_count_v6(landcover_map, elevation_map, region_count=1000, elevation_bins=10, min_area=1):

    # sampled_points = get_landcover_boundary_samples_with_elevation_entropy_sampled_flatter(landcover_map, elevation_map, min_samples=region_count, elevation_bins=elevation_bins)

    # min_area =2 # best for Wharton

    # alpha 0.0 : Flatness-driven
    # alpha 1.0 : Boundary-aware
    # alpha 0.5 : Hybrid


    # alpha = 0.0 # Flatness-driven
    alpha = 1.0 # Boundary-aware
    # alpha = 0.5 # Hybrid

    # min_area = 4
    sampled_points = unified_hybrid_sampling(landcover_map, elevation_map, 
                                             N_total=region_count, alpha=alpha, min_distance=min_area, elevation_bins=elevation_bins)
                                             
                                             
                                             #, min_samples=region_count, min_distance=min_area)
    vor = Voronoi(sampled_points)
    return vor




import numpy as np
from skimage.measure import label, regionprops
from skimage.segmentation import find_boundaries
from scipy.ndimage import generic_filter
from scipy.stats import entropy

def flatness_sampler_core(landcover_map, local_std, min_samples, min_distance, point_mask):
    H, W = landcover_map.shape
    sampled = []
    spacing_mask = np.zeros((H, W), dtype=bool)

    flatness_sorted = np.argsort(local_std.ravel())
    flat_coords = np.column_stack(np.unravel_index(flatness_sorted, (H, W)))

    # Poisson-like sampling
    r = min_distance
    for y, x in flat_coords:
        if len(sampled) >= min_samples:
            break
        if spacing_mask[y, x]:
            continue
        sampled.append([x, y])
        point_mask[y, x] = True
        y0, y1 = max(0, y-r), min(H, y+r+1)
        x0, x1 = max(0, x-r), min(W, x+r+1)
        spacing_mask[y0:y1, x0:x1] = True

    # Add blob centroids
    std_thresh = np.percentile(local_std, 25)
    flat_mask = local_std <= std_thresh
    for label_val in np.unique(landcover_map):
        if len(sampled) >= min_samples:
            break
        mask = (landcover_map == label_val) & flat_mask
        if not np.any(mask):
            continue
        labeled = label(mask)
        for region in regionprops(labeled):
            if len(sampled) >= min_samples:
                break
            cy, cx = np.round(region.centroid).astype(int)
            if not point_mask[cy, cx]:
                sampled.append([cx, cy])
                point_mask[cy, cx] = True

    return sampled


def boundary_sampler_core(landcover_map, elevation_map, coarse_boundaries,
                          min_samples, elevation_bins, point_mask):
    
    lam = 0.5  # Weight for entropy vs distance
    H, W = landcover_map.shape

    # Score boundary points
    boundary_coords = np.argwhere(coarse_boundaries)
    coords = [(x, y) for y, x in boundary_coords]

    from skimage.util import view_as_windows
    from scipy.ndimage import distance_transform_edt

    # Precompute for scoring
    patches = view_as_windows(np.pad(landcover_map, 1, mode='edge'), (3, 3))
    dist = distance_transform_edt(~coarse_boundaries)
    dmax = dist.max()
    max_label = landcover_map.max() + 1

    scored = []
    for x, y in coords:
        patch = patches[y, x]
        hist = np.bincount(patch.ravel(), minlength=max_label)
        p = hist / hist.sum()
        Hval = entropy(p, base=2)
        Dval = dist[y, x]
        score = lam * Hval + (1 - lam) * (1 - Dval / dmax)
        scored.append((score, (x, y)))

    scored.sort(reverse=True)
    selected = [pt for _, pt in scored[:min_samples]]

    sampled = []
    for x, y in selected:
        if len(sampled) >= min_samples:
            break
        if not point_mask[y, x]:
            sampled.append([x, y])
            point_mask[y, x] = True

    # Fill shortfall using elevation-binned boundaries
    remaining = min_samples - len(sampled)
    if remaining > 0:
        lc_sorted = sorted(
            [(lc, np.sum(landcover_map == lc)) for lc in np.unique(landcover_map)],
            key=lambda x: -x[1]
        )
        for lc, _ in lc_sorted:
            if remaining <= 0:
                break
            lc_mask = (landcover_map == lc)
            elev_vals = elevation_map[lc_mask]
            if elev_vals.size == 0:
                continue
            bins = np.linspace(elev_vals.min(), elev_vals.max(), elevation_bins + 1)
            elev_binned = np.digitize(elevation_map, bins, right=True)
            for b in range(1, elevation_bins + 1):
                sub_mask = lc_mask & (elev_binned == b)
                boundary = find_boundaries(sub_mask, mode='thick', connectivity=2)
                coords_bin = np.argwhere(boundary & (~point_mask))
                np.random.shuffle(coords_bin)
                for y, x in coords_bin:
                    if remaining <= 0:
                        break
                    sampled.append([x, y])
                    point_mask[y, x] = True
                    remaining -= 1
                if remaining <= 0:
                    break
    return sampled


def unified_sampler(landcover_map, elevation_map, N_total=2000,
                    alpha=0.5, min_distance=10, lam=0.5, elevation_bins=5):
    assert landcover_map.shape == elevation_map.shape
    H, W = landcover_map.shape
    point_mask = np.zeros((H, W), dtype=bool)

    # Precompute shared quantities
    local_std = generic_filter(elevation_map, np.std, size=5)
    coarse_boundaries = np.zeros((H, W), dtype=bool)
    for lbl in np.unique(landcover_map):
        coarse_boundaries |= find_boundaries(
            landcover_map == lbl, mode='thick', connectivity=2
        )

    # Allocate
    N_flat = int((1 - alpha) * N_total)
    N_boundary = N_total - N_flat

    # Sample
    seeds_flat = flatness_sampler_core(landcover_map, local_std, N_flat, min_distance, point_mask)
    seeds_boundary = boundary_sampler_core(landcover_map, elevation_map, coarse_boundaries,
                                           N_boundary, elevation_bins, point_mask)

    all_seeds = np.vstack([seeds_flat, seeds_boundary])
    all_seeds = np.unique(all_seeds, axis=0)

    # Fill shortfall
    if len(all_seeds) < N_total:
        extra = N_total - len(all_seeds)
        extra_seeds = boundary_sampler_core(landcover_map, elevation_map, coarse_boundaries,
                                            extra, elevation_bins, point_mask)
        all_seeds = np.vstack([all_seeds, extra_seeds])
        all_seeds = np.unique(all_seeds, axis=0)

    return all_seeds





def hybrid_sampling(landcover_map, elevation_map, total_samples=2000, alpha=0.5, elevation_bins=5, min_distance=10, **kwargs):
    n_boundary = int(total_samples * alpha)
    n_flat = total_samples - n_boundary

    seeds_flat = get_uniform_then_blob_then_boundary_samples(
        landcover_map, elevation_map, min_samples=n_flat, min_distance=min_distance
    )

    n_boundary = total_samples - len(seeds_flat)

    # print(f"Flat samples: {len(seeds_flat)}, Boundary samples needed: {n_boundary}, Total samples: {total_samples}")
    if n_boundary <= 0:
        return seeds_flat[:total_samples]
    seeds_boundary = get_landcover_boundary_samples_with_elevation_entropy(
        landcover_map, elevation_map, min_samples=n_boundary, elevation_bins=elevation_bins
    )

    # print(f"Boundary samples: {len(seeds_boundary)}")
    if len(seeds_boundary) > n_boundary:
        seeds_boundary = seeds_boundary[:n_boundary]

    # print(f"Final boundary samples: {len(seeds_boundary)}")

    # Merge and deduplicate
    all_seeds = np.vstack([seeds_flat, seeds_boundary])
    all_seeds = np.unique(all_seeds, axis=0)

    # print(f"Total unique seeds after merging: {len(all_seeds)}")

    # If after deduplication we have fewer points, fill from boundary again
    if len(all_seeds) < total_samples:
        extra = total_samples - len(all_seeds)
        extra_seeds = get_landcover_boundary_samples_with_elevation_entropy(
            landcover_map, elevation_map, min_samples=extra, elevation_bins=elevation_bins
        )
        all_seeds = np.vstack([all_seeds, extra_seeds])
        all_seeds = np.unique(all_seeds, axis=0)

    return all_seeds



def hybrid_sampling_corrected(landcover_map, elevation_map, total_samples=2000, alpha=0.5, elevation_bins=5, min_distance=10, **kwargs):

    if alpha ==1.0:
        return get_landcover_boundary_samples_with_elevation_entropy(
        landcover_map, elevation_map, min_samples=total_samples, elevation_bins=elevation_bins
        )
    
    n_boundary = int(total_samples * alpha)
    n_flat = total_samples - n_boundary

    seeds_flat = get_uniform_then_blob_then_boundary_samples(
        landcover_map, elevation_map, min_samples=n_flat, min_distance=min_distance
    )

    n_boundary = total_samples - len(seeds_flat)

    # print(f"Flat samples: {len(seeds_flat)}, Boundary samples needed: {n_boundary}, Total samples: {total_samples}")
    if n_boundary <= 0:
        return seeds_flat[:total_samples]
    seeds_boundary = get_landcover_boundary_samples_with_elevation_entropy(
        landcover_map, elevation_map, min_samples=n_boundary, elevation_bins=elevation_bins
    )

    # print(f"Boundary samples: {len(seeds_boundary)}")
    if len(seeds_boundary) > n_boundary:
        seeds_boundary = seeds_boundary[:n_boundary]

    # print(f"Final boundary samples: {len(seeds_boundary)}")

    # Merge and deduplicate
    all_seeds = np.vstack([seeds_flat, seeds_boundary])
    all_seeds = np.unique(all_seeds, axis=0)

    # print(f"Total unique seeds after merging: {len(all_seeds)}")

    # If after deduplication we have fewer points, fill from boundary again
    if len(all_seeds) < total_samples:
        extra = total_samples - len(all_seeds)
        extra_seeds = get_landcover_boundary_samples_with_elevation_entropy(
            landcover_map, elevation_map, min_samples=extra, elevation_bins=elevation_bins
        )
        all_seeds = np.vstack([all_seeds, extra_seeds])
        all_seeds = np.unique(all_seeds, axis=0)

    return all_seeds




def get_landcover_boundary_and_elevation_bin_region_count_v7(landcover_map, elevation_map, region_count=1000, elevation_bins=10, min_area=1, flatness_ratio=0.7):
    
    # sampled_points = get_landcover_boundary_samples_with_elevation_entropy_sampled_flatter(landcover_map, elevation_map, min_samples=region_count, elevation_bins=elevation_bins)

    # min_area =2 # best for Wharton

    # alpha 0.0 : Flatness-driven
    # alpha 1.0 : Boundary-aware
    # alpha 0.5 : Hybrid


    # alpha = 0.0 # Flatness-driven
    # alpha = 1.0 # Boundary-aware
    # alpha = 0.7 # Hybrid 

    #results map 1:
    # 1.0 : good for Planning
    # 0.0 : good for Decomposition
    # # 0.7, 0.8 better than 1.0
    # 0.5, 0.6 is bad(discarded)


    # alpha =  0.0
    # alpha = 1.0



    # sampled_points = unified_hybrid_sampling(landcover_map, elevation_map, 
    #                                          N_total=region_count, alpha=alpha, min_distance=min_area, elevation_bins=elevation_bins)
                                             

    # alpha = 0.0 , Flatness-driven, Good for Planning
    # alpha = 1.0 , Boundary-aware, Good for Decomposition
    print("======Landcover total points:",region_count)

    alpha = flatness_ratio
    # alpha = 1.0 # Currently being evaluated
    # alpha = 0.0 # to try next
    sampled_points = hybrid_sampling_corrected(landcover_map, elevation_map, 
                                             total_samples=region_count, alpha=alpha, min_distance=min_area, elevation_bins=elevation_bins)
                              

    print("======Landcover total points:", sampled_points.shape, landcover_map.shape, elevation_map.shape, region_count)

    vor = Voronoi(sampled_points)
    return vor




from tqdm import tqdm
import pandas as pd
from shapely.geometry import Polygon
from rasterio.features import geometry_mask

from scipy.spatial import Delaunay, ConvexHull

import trimesh

from scipy.spatial import ConvexHull, QhullError




import numpy as np

from scipy.spatial import Delaunay, ConvexHull

from sklearn.metrics import accuracy_score, confusion_matrix


from matplotlib.path import Path

def check_inside_polygon_old(polygon, points, radius=1e-10):
    polygon = np.asarray(polygon)
    points = np.asarray(points)

    if len(polygon) < 3:
        # Degenerate cases: point or line
        if len(polygon) == 1:
            return np.linalg.norm(points - polygon[0], axis=1) <= radius
        elif len(polygon) == 2:
            # Check point-to-segment distance
            a, b = polygon
            ap = points - a
            ab = b - a
            ab_norm_sq = np.dot(ab, ab)
            t = np.clip(np.dot(ap, ab) / ab_norm_sq, 0, 1)
            closest = a + t[:, np.newaxis] * ab
            return np.linalg.norm(points - closest, axis=1) <= radius
        else:
            return np.zeros(len(points), dtype=bool)

    path = Path(polygon)
    return path.contains_points(points, radius=radius)


# from shapely.geometry import Polygon, Point

# def check_inside_polygon(polygon, points, radius=0.5):
#     polygon = np.asarray(polygon)
#     points = np.asarray(points)

#     if len(polygon) < 3:
#         # Degenerate: point or line
#         if len(polygon) == 1:
#             return np.linalg.norm(points - polygon[0], axis=1) <= radius
#         elif len(polygon) == 2:
#             a, b = polygon
#             ap = points - a
#             ab = b - a
#             ab_norm_sq = np.dot(ab, ab)
#             t = np.clip(np.dot(ap, ab) / ab_norm_sq, 0, 1)
#             closest = a + t[:, np.newaxis] * ab
#             return np.linalg.norm(points - closest, axis=1) <= radius
#         else:
#             return np.zeros(len(points), dtype=bool)

#     poly = Polygon(polygon)
#     return np.array([poly.buffer(radius).covers(Point(p)) for p in points])



from shapely.geometry import Polygon, Point
import numpy as np

def check_inside_polygon_bug_with_line(polygon, points, radius=0.71): # was radius=0.5
    polygon = np.asarray(polygon)
    points = np.asarray(points)

    if len(polygon) < 3:
        # Degenerate: point or line
        if len(polygon) == 1:
            center = polygon[0]
            dists = np.linalg.norm(points - center, axis=1)
            return np.where(dists <= radius)[0]
            # return np.linalg.norm(points - polygon[0], axis=1) <= radius
        elif len(polygon) == 2:
            a, b = polygon
            ap = points - a
            ab = b - a
            ab_norm_sq = np.dot(ab, ab)
            t = np.clip(np.sum(ap * ab, axis=1) / ab_norm_sq, 0, 1)
            closest = a + t[:, np.newaxis] * ab
            indices = np.where(np.linalg.norm(points - closest, axis=1) <= radius)[0]
            if len(indices) > 0:
                return indices
            else:
                return np.array([])
            # return np.linalg.norm(points - closest, axis=1) <= radius
        else:
            center = np.mean(polygon, axis=0)
            dists = np.linalg.norm(points - center, axis=1)
            return np.where(dists <= radius)[0]


            # return np.zeros(len(points), dtype=bool)

    poly = Polygon(polygon)
    return np.array([i  for i in range(len(points)) if poly.buffer(radius).covers(Point(points[i])) ])








from shapely.geometry import Polygon, Point, LineString
import numpy as np

def is_colinear(polygon):
    if len(polygon) < 3:
        return True
    a, b, c = polygon[:3]
    return np.isclose(np.linalg.det(np.array([
        [a[0], a[1], 1],
        [b[0], b[1], 1],
        [c[0], c[1], 1]
    ])), 0)

def check_inside_polygon(polygon, points, radius=0.71):
    polygon = np.asarray(polygon)
    points = np.asarray(points)

    if len(polygon) < 3 or is_colinear(polygon):
        if len(polygon) == 1:
            center = polygon[0]

            points = np.array(points, dtype=float)
            center = np.array(center, dtype=float)
            dists = np.linalg.norm(points - center, axis=1)
            # print("Points:", points.shape, "Center:", center.shape, "Dists:", dists.shape, points, center, dists)

            # if dists[0] <= radius:
            #     return np.array([True])
            # return []
            # return np.where(np.isclose(dists, 0.0, atol=1e-6) | (dists <= radius))[0]
                        
            dists = np.linalg.norm(points - center, axis=1)
            
            return np.where(dists <= radius)[0]
        elif len(polygon) == 2 or is_colinear(polygon):
            line = LineString(polygon)
            return np.array([
                i for i, pt in enumerate(points)
                if line.buffer(radius).contains(Point(pt))
            ])
        else:
            center = np.mean(polygon, axis=0)
            dists = np.linalg.norm(points - center, axis=1)
            # print("Points:", points.shape, "Center:", center.shape, "Dists:", dists.shape, points, center, dists)
            return np.where(dists <= radius)[0]

    poly = Polygon(polygon)
    return np.array([
        i for i in range(len(points))
        if poly.buffer(radius).covers(Point(points[i]))
    ])



class EvaluationMetrics:

    def elevation_info_loss(self, z, z_pred):
        residuals = z - z_pred
        rmse = np.sqrt(np.mean(residuals ** 2))
        var = np.var(residuals)
        max_residual = np.max(np.abs(residuals))
        return rmse, var, max_residual
    

    def landcover_info_loss_not_used(self, landcover_region, total_size=None):
        unique, counts = np.unique(landcover_region, return_counts=True)
        probs = counts / counts.sum()
        if np.any(probs == 0) or not np.isclose(np.sum(probs), 1.0):
            return 0.0
        entropy = -np.sum(probs * np.log(probs))
        if total_size is None:
            return entropy
        region_weight = len(landcover_region) / total_size
        return entropy * region_weight



    def evaluate_landcover_classification(self, true_labels, predicted_labels):
        """
        Evaluate classification accuracy, skipping invalid (-1) predictions.

        Parameters:
            true_labels: np.ndarray of ground truth labels
            predicted_labels: np.ndarray of predicted labels

        Returns:
            dict with accuracy and confusion matrix
        """
        mask = predicted_labels != -1
        true_valid = true_labels[mask]
        pred_valid = predicted_labels[mask]

        accuracy = accuracy_score(true_valid, pred_valid)
        cm = confusion_matrix(true_valid, pred_valid)

        return {
            "accuracy": accuracy,
            "confusion_matrix": cm
        }


    def mean_iou(self, target: np.ndarray, pred: np.ndarray, num_classes: int = None) -> float:
        """
        Compute mean Intersection over Union (mIoU) between two landcover maps.

        Args:
            pred: HxW predicted landcover (integers).
            target: HxW ground truth landcover (integers).
            num_classes: total number of landcover classes. If None, inferred from union of inputs.

        Returns:
            mIoU: float, mean intersection over union.
        """
        assert pred.shape == target.shape, "Input shapes must match"

        if num_classes is None:
            num_classes = int(max(pred.max(), target.max()) + 1)

        iou_per_class = []
        for cls in range(num_classes):
            pred_mask = (pred == cls)
            target_mask = (target == cls)
            intersection = np.logical_and(pred_mask, target_mask).sum()
            union = np.logical_or(pred_mask, target_mask).sum()
            if union == 0:
                continue  # ignore classes not present in either
            iou_per_class.append(intersection / union)

        if not iou_per_class:
            return 0.0  # no valid classes
        return np.mean(iou_per_class)








# Re-import required packages after environment reset
import numpy as np
from typing import List, Tuple, Dict
from sklearn.linear_model import LinearRegression
from scipy.spatial import ConvexHull
from shapely.geometry import Point, Polygon
import os
# os.environ["OMP_NUM_THREADS"] = "1"
# os.environ["OPENBLAS_NUM_THREADS"] = "1"
# os.environ["MKL_NUM_THREADS"] = "1"
# os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
# os.environ["NUMEXPR_NUM_THREADS"] = "1"


import numpy as np
from scipy.spatial import ConvexHull
from typing import List, Tuple, Dict

class PlaneFitter:
    def __init__(self, map_scale=30.0):
        self.map_scale = map_scale

    def compute_rmse(self, Z_true, Z_pred):
        return np.sqrt(np.mean((Z_true - Z_pred) ** 2))

    def fit_plane(self, points_subset: np.ndarray) -> Tuple[Dict[str, float], float]:
        X = points_subset[:, 0] * self.map_scale
        Y = points_subset[:, 1] * self.map_scale
        Z = points_subset[:, 2]
        A = np.column_stack((X, Y, np.ones_like(X)))
        coeffs, _, _, _ = np.linalg.lstsq(A, Z, rcond=None)
        a, b, d = coeffs
        c = -1.0
        Z_pred = A @ coeffs
        rmse = self.compute_rmse(Z, Z_pred)
        return {"a": a, "b": b, "c": c, "d": d}, rmse

    def get_convex_polygon(self, points_subset: np.ndarray) -> np.ndarray:
        points_2d = np.unique(points_subset[:, :2], axis=0)
        if len(points_2d) >= 3:
            try:
                hull = ConvexHull(points_2d)
                return points_2d[hull.vertices]
            except:
                return points_2d
        return points_2d

    def is_small_enough(self, points, min_size=1e-2):
        x_range = points[:, 0].max() - points[:, 0].min()
        y_range = points[:, 1].max() - points[:, 1].min()
        return x_range < min_size and y_range < min_size

    
    def split_points_with_attrs(self, points_subset: np.ndarray, attrs_subset: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
        x = points_subset[:, 0]
        y = points_subset[:, 1]
        x_mid = np.median(x)
        y_mid = np.median(y)

        # masks = [
        #     (x < x_mid) & (y < y_mid),
        #     (x >= x_mid) & (y < y_mid),
        #     (x < x_mid) & (y >= y_mid),
        #     (x >= x_mid) & (y >= y_mid),
        # ]

        margin = 0.08
        masks = [
            (x <= x_mid + margin) & (y <= y_mid + margin),
            (x >= x_mid - margin) & (y <= y_mid + margin),
            (x <= x_mid + margin) & (y >= y_mid - margin),
            (x >= x_mid - margin) & (y >= y_mid - margin),
        ]


        subsets = [(points_subset[m], attrs_subset[m]) for m in masks]
        return subsets #[s for s in subsets if len(s[0]) >= 3]


    def define_z_up_plane(self, pts, attr):

        all_results = []
        for pt, pt_attr in zip(pts, attr):
            z = pt[2]
            plane = {"a": 0.0, "b": 0.0, "c": -1.0, "d": z}
            polygon = [[pt[0], pt[1]]]

            all_results.append((plane, polygon, int(pt_attr)))

        return  all_results 

    def fit_planes_with_landcover(self, points, attributes, rmse_thresh=0.01, min_points=10) -> List[Tuple[Dict[str, float], np.ndarray, int]]:
        queue = [(points, attributes)]
        results = []
        covered_pts = []

        # rmse_thresh=5.0

        while queue:
            pts, attr = queue.pop()
            if len(pts) < 3:
                results.extend(self.define_z_up_plane(pts, attr))
                covered_pts.append(pts)

                continue

            plane, rmse = self.fit_plane(pts)

            if rmse <= rmse_thresh or len(pts) <= min_points or self.is_small_enough(pts, min_size=3):
                try:
                    polygon = self.get_convex_polygon(pts)
                except:
                    results.extend(self.define_z_up_plane(pts, attr))
                    continue
                dominant_attr = np.bincount(attr).argmax()
                results.append((plane, polygon, int(dominant_attr)))
                covered_pts.append(pts)
            else:
                try:
                    subsets = self.split_points_with_attrs(pts, attr)
                    queue.extend(subsets)
                except:
                    results.extend(self.define_z_up_plane(pts, attr))
                    covered_pts.append(pts)

                    continue
        
        if False:
            # Optional check for full coverage
            all_input = set(map(tuple, points[:, :2]))
            all_output = set(map(tuple, np.vstack(covered_pts)[:, :2]))
            missed = all_input - all_output
            # if missed:
            #     print(f"⚠️ Warning: {len(missed)}/ {len(all_input)} points not covered in final output.")
            #     raise ValueError("Not all points were covered by the fitted planes.")
            if missed:
                missed_mask = np.array([tuple(pt[:2]) in missed for pt in points])
                missed_points = points[missed_mask]
                missed_attrs = attributes[missed_mask]
                print(f"⚠️ Recovering {len(missed_points)}/ {len(all_input)} missed points as flat patches. Fitted Planes: {len(results)}")
                results.extend(self.define_z_up_plane(missed_points, missed_attrs))


        return results






class ElevationAbstractor:

    def __init__(self, abstraction_method="surface",  error_thresh=10.0, max_planes=100, map_scale=30):
        self.error_thresh = error_thresh
        self.max_planes = max_planes
        #abstraction_method can be "surface" or "plane"
        self.abstraction_method = abstraction_method
        self.map_scale = map_scale

    def build_abstraction(self, points, landcover_array=None):

        if self.abstraction_method  =="plane":
            return "plane", self.fit_planes_with_landcover(points, landcover_array=landcover_array)

        elif self.abstraction_method == "surface":
            
            if len(points) < 3:
                return  "plane", self.fit_planes_with_landcover(points, landcover_array=landcover_array)
            
            use_planes_for_mixed_regions = False
            if use_planes_for_mixed_regions:
                unique_classes = np.unique(landcover_array)
                if len(unique_classes) > 1:
                    return  "plane", self.fit_planes_with_landcover(points, landcover_array=landcover_array)
            try:
                coeffs, mse = self.fit_quadratic_surface(points)
                hull = ConvexHull(points[:, :2])
            except QhullError:
                return "plane", self.fit_planes_with_landcover(points, landcover_array=landcover_array)

            polygon = [(points[i, 0], points[i, 1]) for i in hull.vertices]
            surface_dict = coeffs
            if landcover_array is not None:
                # landcover = landcover_array[0]
                landcover =  np.bincount(landcover_array).argmax()
            
            surface_region = (surface_dict, polygon, landcover)
            return "surface", surface_region

        else:
            raise ValueError("Invalid abstraction method. Choose 'surface' or 'plane'.")



    def fit_quadratic_surface(self, points):
        """
        Fit z = ax^2 + by^2 + cxy + dx + ey + f to input point cloud.
        
        Args:
            points (np.ndarray): shape (N, 3), point cloud as [x, y, z]
        
        Returns:
            coeffs (np.ndarray): shape (6,), [a, b, c, d, e, f]
            mse (float): mean squared error
        """
        x, y, z = points[:, 0], points[:, 1], points[:, 2]
        A = np.column_stack((x**2, y**2, x*y, x, y, np.ones_like(x)))
        coeffs, _, _, _ = np.linalg.lstsq(A, z, rcond=None)
        z_pred = A @ coeffs
        mse = np.mean((z - z_pred) ** 2)
        return coeffs, mse


    def reconstruct_surface(self, coeffs, x, y):
        """
        Reconstruct z values from given x, y and surface coefficients.
        
        Args:
            coeffs (np.ndarray): [a, b, c, d, e, f]
            x, y (np.ndarray): shape (N,), input grid or points
        
        Returns:
            z (np.ndarray): shape (N,), reconstructed z values
        """
        a, b, c, d, e, f = coeffs
        z = a*x**2 + b*y**2 + c*x*y + d*x + e*y + f
        return z




    def define_z_up_plane(self, points, landcover_array):
        """
        Define a Z-up plane from a single 3D point.
        Returns a dict with plane normal (a, b, c) and offset d.
        """

        output = []
        for i in range(len(points)):
            point = points[i]
            x0, y0, z0 = point
            plane_dict = {"a": 0.0, "b": 0.0, "c": 1.0, "d": -z0}
            polygon = [(x0, y0)]
            label = landcover_array[i] if landcover_array is not None else None

            output.append((plane_dict, polygon, int(label)))


        return output


    def compute_rmse(self, Z_true, Z_pred):
        return np.sqrt(np.mean((Z_true - Z_pred) ** 2))



    def fit_planes_with_landcover_best(self,
        points: np.ndarray,
        landcover_array: np.ndarray,
        overlap: int = 1
    ) -> List[Tuple[Dict[str, float], np.ndarray, np.ndarray, float]]:
        ##PRANAY: This algoritm is overfitting the planes to preserve RMSE and Landcover attributes, 
        rmse_thresh = self.error_thresh
        attributes = landcover_array
        if len(points) < 3:
            planes = self.define_z_up_plane(points, landcover_array)
            return planes
        try:
            tri = Delaunay(points[:, :2])
        except Exception as e:
            planes = self.define_z_up_plane(points, landcover_array)
            return planes




        import numpy as np
        import matplotlib.pyplot as plt
        from scipy.spatial import Delaunay, ConvexHull
        from sklearn.linear_model import LinearRegression
        from sklearn.cluster import KMeans
        import trimesh

        # Plane fitting with RMSE calculation in ax + by - z + d = 0 form
        def fit_plane_with_error(verts):
            X = verts[:, :2] * self.map_scale  # Scale to match map scale
            Z = verts[:, 2]
            reg = LinearRegression().fit(X, Z)
            a, b = reg.coef_
            c = -1.0
            d = reg.intercept_
            rmse = np.sqrt(np.mean((Z - reg.predict(X))**2))
            return (a, b, c, d), rmse

        # Recursive region fitting with landcover attributes
        def recursive_plane_fit_split_4(mesh, face_indices, error_thresh, attr, depth=0, max_depth=6):
            verts = np.vstack([mesh.vertices[mesh.faces[f]] for f in face_indices])
            face_attrs = np.concatenate([attr[mesh.faces[f]] for f in face_indices])
            plane, err = fit_plane_with_error(verts)
            xi, yi = verts[:, 0], verts[:, 1]

            # try:
            #     hull = ConvexHull(np.column_stack((xi, yi)))
            #     polygon = [(xi[i], yi[i]) for i in hull.vertices]
            # except:
            #     polygon = list(set(tuple(p) for p in zip(xi, yi)))
            # dominant_attr = np.bincount(face_attrs).argmax()
            # plane_dict = {"a": plane[0], "b": plane[1], "c": plane[2], "d": plane[3]}
            # return [(plane_dict, polygon, int(dominant_attr))]



            if err <= error_thresh or depth >= max_depth or len(face_indices) < 4:
                try:
                    hull = ConvexHull(np.column_stack((xi, yi)))
                    polygon = [(xi[i], yi[i]) for i in hull.vertices]
                except:
                    polygon = list(set(tuple(p) for p in zip(xi, yi)))
                dominant_attr = np.bincount(face_attrs).argmax()
                plane_dict = {"a": plane[0], "b": plane[1], "c": plane[2], "d": plane[3]}
                return [(plane_dict, polygon, int(dominant_attr))]
            else:
                face_centers = mesh.triangles_center[face_indices]
                kmeans = KMeans(n_clusters=4, n_init=10).fit(face_centers)
                planes = []
                for cluster_id in range(4):
                    group = [face_indices[i] for i in range(len(face_indices)) if kmeans.labels_[i] == cluster_id]
                    if len(group) >= 3:
                        planes += recursive_plane_fit_split_4(mesh, group, error_thresh, attr, depth + 1, max_depth)
                    else:
                        # Fallback for small group
                        small_verts = np.vstack([mesh.vertices[mesh.faces[f]] for f in group])
                        small_attrs = np.concatenate([attr[mesh.faces[f]] for f in group])
                        xi, yi = small_verts[:, 0], small_verts[:, 1]
                        try:
                            hull = ConvexHull(np.column_stack((xi, yi)))
                            polygon = [(xi[i], yi[i]) for i in hull.vertices]
                        except:
                            polygon = list(set(tuple(p) for p in zip(xi, yi)))
                        small_plane, _ = fit_plane_with_error(small_verts)
                        dominant_attr = np.bincount(small_attrs).argmax()
                        plane_dict = {"a": small_plane[0], "b": small_plane[1], "c": small_plane[2], "d": small_plane[3]}
                        planes.append((plane_dict, polygon, int(dominant_attr)))
                return planes

        # Wrapper to initialize mesh and call recursion
        def recursive_plane_fit_wrapper(points, attributes, error_thresh=0.01, max_depth=3):
            tri = Delaunay(points[:, :2])
            mesh = trimesh.Trimesh(vertices=points, faces=tri.simplices)
            face_indices = np.arange(len(mesh.faces))
            return recursive_plane_fit_split_4(mesh, face_indices, error_thresh, attributes, max_depth=max_depth)



        return  recursive_plane_fit_wrapper(points, attributes, error_thresh=rmse_thresh, max_depth=0)






    def fit_planes_with_landcover(self,
        points: np.ndarray,
        landcover_array: np.ndarray,
        overlap: int = 1
    ) -> List[Tuple[Dict[str, float], np.ndarray, np.ndarray, float]]:
        """
        Fit planes to the given points with landcover attributes.
        Returns a list of tuples containing plane parameters, polygon vertices, and dominant landcover label.
        """
        rmse_thresh = self.error_thresh
        attributes = landcover_array

        if len(points) < 3:
            planes = self.define_z_up_plane(points, landcover_array)
            return planes

        try:
            tri = Delaunay(points[:, :2])
        except Exception as e:
            planes = self.define_z_up_plane(points, landcover_array)
            return planes

        plane_fitter = PlaneFitter(map_scale=self.map_scale)

        return plane_fitter.fit_planes_with_landcover(points, attributes, rmse_thresh, min_points=3)






    def fit_planes_with_landcover_older_version(self, 
        points: np.ndarray,
        landcover_array: np.ndarray,
        overlap: int = 1
    ) -> List[Tuple[Dict[str, float], np.ndarray, np.ndarray, float]]:
        rmse_thresh = self.error_thresh
        attributes = landcover_array

        if len(points) < 3:
            planes = self.define_z_up_plane(points, landcover_array)
            return planes

        try:
            tri = Delaunay(points[:, :2])
        except Exception as e:
            planes = self.define_z_up_plane(points, landcover_array)
            return planes


        def fit_plane(points_subset: np.ndarray) -> Tuple[Dict[str, float], float]:
            X = points_subset[:, 0]*self.map_scale
            Y = points_subset[:, 1]*self.map_scale
            Z = points_subset[:, 2]

            A = np.column_stack((X, Y, np.ones_like(X)))
            coeffs, _, _, _ = np.linalg.lstsq(A, Z, rcond=None)
            a, b, d = coeffs
            Z_pred = A @ coeffs
            c = -1

            rmse = self.compute_rmse(Z, Z_pred)
            plane = {"a": a, "b": b, "c": c, "d": d}
            return plane, rmse


        def get_convex_polygon(points_subset: np.ndarray) -> np.ndarray:
            points_2d = points_subset[:, :2]

            if len(points_2d) >= 3:

                hull = ConvexHull(points_2d)
                return points_2d[hull.vertices]

            else:
                return points_2d



        def is_small_enough(points, min_size=1e-2):
            x_range = points[:, 0].max() - points[:, 0].min()
            y_range = points[:, 1].max() - points[:, 1].min()
            return x_range < min_size and y_range < min_size


        def split_points_with_attrs(points_subset: np.ndarray, attrs_subset: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
            x = points_subset[:, 0]
            y = points_subset[:, 1]
            x_mid = np.median(x)
            y_mid = np.median(y)

            q1 = (x <= x_mid) & (y <= y_mid)
            q2 = (x >  x_mid) & (y <= y_mid)
            q3 = (x <= x_mid) & (y >  y_mid)
            q4 = (x >  x_mid) & (y >  y_mid)

            masks = [q1, q2, q3, q4]
            subsets = [(points_subset[m], attrs_subset[m]) for m in masks]
            return [s for s in subsets if len(s[0]) >= 1]


        # Recursive processing
        queue = [(points, attributes)]
        results = []

        while queue:
            pts, attr = queue.pop()

            plane, rmse = fit_plane(pts)
            
            if rmse <= rmse_thresh or len(pts) <= 3 or is_small_enough(pts):
                try:
                    polygon = get_convex_polygon(pts)
                except Exception as e:
                    # curr_result =  self.fit_planes_with_landcover_old(pts, landcover_array=attr)
                    curr_result = self.define_z_up_plane(pts, attr)
                    results.extend(curr_result)
                    continue
                if len(attr) == 0 and len(pts) == 0:
                    # print(f"Warning: No attributes found for points: {pts.shape}")
                    continue

                dominant_attr = np.bincount(attr).argmax() if np.issubdtype(attr.dtype, np.integer) else np.unique(attr)[np.argmax(np.unique(attr, return_counts=True)[1])]
                results.append((plane, polygon, int(dominant_attr)))
            else:
                try:
                    splitted_points = split_points_with_attrs(pts, attr)
                    queue.extend(splitted_points)

                except Exception as e:
                    # print(f"Error splitting points: {e}")
                    # curr_result =  self.fit_planes_with_landcover_old(pts, landcover_array=attr)
                    curr_result = self.define_z_up_plane(pts, attr)
                    results.extend(curr_result)
                    continue



        #Pranay : FIX this for GCS
        debug = False

        if debug:
            all_points = points[:, :2]
            inside_points_all = []
            for plane_dict, polygon, label in results:
                inside_points = check_inside_polygon(polygon, all_points)
                inside_points_all.extend(inside_points)
            inside_points_all = np.array(inside_points_all)

      
            inside_points_all = np.unique(inside_points_all, axis=0)
            # print(f"inside_points_all: {inside_points_all}, all_points: {len(all_points)}")

            if len(inside_points_all) == len(all_points):
                return results

            # from numpy_indexed import indices
            # print(f"inside_points_all: {inside_points_all}, {all_points.shape}")
            # inside_points_all and all_points must be 2D arrays
            if len(inside_points_all) == 0:
                points_not_inside_index = np.arange(len(all_points))
            else:
                inside_set = set(map(tuple, all_points[inside_points_all]))

                points_not_inside_index = np.array([
                    i for i in range(len(all_points)) if tuple(all_points[i]) not in inside_set
                ])


            # print(f"points_not_inside_index: {points_not_inside_index}")
            for i in points_not_inside_index:

                point = points[i]
                label = landcover_array[i] if landcover_array is not None else None

                x0, y0, z0 = point
                point = np.array([x0, y0])

                plane_dict = {"a": 0.0, "b": 0.0, "c": 1.0, "d": -z0}
                polygon = [(x0, y0)]
                
                results.append((plane_dict, polygon, int(label)))


        # print(f"Final results: {results}")    
        return results







    def fit_planes_with_landcover_old(self, points, landcover_array=None):

        # Step 2: Generate mesh from points using Delaunay triangulation

        if len(points) < 3:
            planes = self.define_z_up_plane(points, landcover_array)
            return planes

        try:
            tri = Delaunay(points[:, :2])
        except Exception as e:
            planes = self.define_z_up_plane(points, landcover_array)
            return planes

        error_thresh = self.error_thresh

        mesh = trimesh.Trimesh(vertices=points, faces=tri.simplices)
        face_indices = np.arange(len(mesh.faces))
        planes = self.recursive_plane_fit_landcover(mesh, face_indices, error_thresh, landcover_array= landcover_array, max_depth=6)

        return planes

    def fit_plane_with_error_memory_issue(self, points, min_c=0.1):
        """
        Fit a plane ax + by + cz + d = 0 with a bias to avoid vertical planes.
        Enforces |c| >= min_c to ensure Z is defined.
        """
        import numpy as np
        from numpy.linalg import norm

        points = np.asarray(points)
        centroid = points.mean(axis=0)
        centered = points - centroid

        # SVD: last row of V gives the normal vector
        _, _, vh = np.linalg.svd(centered)
        normal = vh[-1]

        # Flip normal to make c positive
        if normal[2] < 0:
            normal = -normal

        # Reject nearly vertical planes
        if abs(normal[2]) < min_c:
            # Force a fallback to horizontal-like plane (0, 0, 1)
            normal = np.array([0.0, 0.0, 1.0])

        d = -np.dot(normal, centroid)
        plane = (*normal, d)

        # Compute point-to-plane error (optional)
        distances = np.abs((points @ normal + d)) / np.linalg.norm(normal)
        rmse = np.sqrt(np.mean(distances**2))
        plane = {"a": plane[0], "b": plane[1], "c": plane[2], "d": plane[3]}

        return plane, rmse



    def fit_plane_with_error(self, points, min_c=0.1):
        points = np.asarray(points)
        centroid = points.mean(axis=0)
        centered = points - centroid

        cov = np.cov(centered, rowvar=False)
        eigvals, eigvecs = np.linalg.eigh(cov)
        normal = eigvecs[:, 0]  # eigenvector with smallest eigenvalue

        if normal[2] < 0:
            normal = -normal
        if abs(normal[2]) < min_c:
            normal = np.array([0.0, 0.0, 1.0])

        d = -np.dot(normal, centroid)
        distances = np.abs(centered @ normal)
        rmse = np.sqrt(np.mean(distances**2))

        plane = {"a": normal[0], "b": normal[1], "c": normal[2], "d": d}
        return plane, rmse



    def make_qhull_safe(self, points, eps=1e-6, max_attempts=10):
        """
        Add small perturbations to avoid Qhull failures while keeping all original points.
        """
        points = np.asarray(points).copy()
        if len(points) < 3 or points.shape[1] != 2:
            return points

        attempt = 0

        while attempt < max_attempts:
            centered = points - points[0]
            if np.linalg.matrix_rank(centered) >= 2:
                try:
                    _ = ConvexHull(points)
                    return points
                except QhullError:
                    pass

            # Perturb interior points (not first or last)
            for i in range(1, len(points)):# - 1):
                direction = np.random.randn(2)
                perp = np.array([-direction[1], direction[0]])
                points[i] += perp * eps * (attempt + 1)

            attempt += 1

        return points  # Still includes all original points (perturbed)


    def recursive_plane_fit_landcover(self, mesh, face_indices, error_thresh, landcover_array=None,     depth=0, max_depth=6):
        """
        Recursively fit planes to mesh regions and assign majority landcover label.

        Parameters:
            mesh: Trimesh mesh object
            face_indices: indices of faces to fit
            error_thresh: RMSE threshold to stop recursion
            depth: current recursion depth
            max_depth: max recursion depth
            landcover_array: 1D array of landcover labels (same length as mesh.vertices)

        Returns:
            List of tuples (plane_dict, polygon, landcover label)
        """
        verts_idx = mesh.faces[face_indices].flatten()
        verts = mesh.vertices[mesh.faces[face_indices]].reshape(-1, 3)
        plane, err = self.fit_plane_with_error(verts)
        xi, yi = verts[:, 0], verts[:, 1]
        

        if err <= error_thresh or depth >= max_depth or len(face_indices) < 5:
            output = []
            plane_dict = plane #{"a": plane[0], "b": plane[1], "c": plane[2], "d": plane[3]}
            try:
                hull = ConvexHull(np.column_stack((xi, yi)))
            except:
                safe_points = self.make_qhull_safe(np.column_stack((xi, yi)))
                hull = ConvexHull(safe_points)

            polygon = [(xi[i], yi[i]) for i in hull.vertices]

            # Assign majority landcover
            if landcover_array is not None:
                region_labels = landcover_array[verts_idx]
                region_labels = region_labels[region_labels >= 0].astype(int)
                if len(region_labels) == 0:
                    landcover = -1
                else:
                    landcover = np.bincount(region_labels).argmax()
            else:
                landcover = None


            output.append((plane_dict, polygon, landcover))


            return output
        else:
            face_centers = mesh.triangles_center[face_indices]
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=2, n_init=10).fit(face_centers)
            group1 = [face_indices[i] for i in range(len(face_indices)) if kmeans.labels_[i] == 0]
            group2 = [face_indices[i] for i in range(len(face_indices)) if kmeans.labels_[i] == 1]

            # Fallback if cluster too small
            if len(group1) < 3 or len(group2) < 3:
                output = []
                plane_dict = plane # {"a": plane[0], "b": plane[1], "c": plane[2], "d": plane[3]}

                hull = ConvexHull(np.column_stack((xi, yi)))
                polygon = [(xi[i], yi[i]) for i in hull.vertices]

                # Assign majority landcover
                if landcover_array is not None:
                    region_labels = landcover_array[verts_idx]
                    region_labels = region_labels[region_labels >= 0].astype(int)
                    if len(region_labels) == 0:
                        landcover = -1
                    else:
                        landcover = np.bincount(region_labels).argmax()
                else:
                    landcover = None

                output.append((plane_dict, polygon, landcover))


                return output

            # Recurse
            planes = []
            planes += self.recursive_plane_fit_landcover(mesh, group1, error_thresh, landcover_array, depth + 1, max_depth)
            planes += self.recursive_plane_fit_landcover(mesh, group2, error_thresh, landcover_array, depth + 1, max_depth)
            return planes





import sys

class AbstractRegion:
    def __init__(self, regions, centroid, polygon_pts, map_scale=30):
        self.centroid = centroid
        self.polygon_pts = polygon_pts
        self.regions  = regions
        self.memory_MB = None
        self.plane_count = 0

        # fill in the attributes for abstraction
        self.attributes = {}

        self.map_scale = map_scale
        

    def __str__(self):
        return f"AbstractRegion(centroid={self.centroid}, polygon_pts={len(self.polygon_pts)}, regions={len(self.regions)}), memory_MB={self.memory_MB})"
    
    def reconstruct_region(self, x_vals, y_vals):
        """
        Reconstruct z-values for each (x, y) based on the plane.
        """
        return None 
    
    def evaluate_region(self, points):
        """
        Evaluate the region based on the plane.
        """
        return None
    
    def get_plane_count(self):
        """
        Get the number of planes in the region.
        """
        return self.plane_count
    
    def get_z_value_and_lc(self, x, y):
        """
        Get the z value for a given (x, y) point.
        """
        return None
    
class PlaneRegion(AbstractRegion):
    def __init__(self, regions, centroid, polygon_pts, map_scale=30):
        super().__init__(regions, centroid, polygon_pts, map_scale)

        self.memory_MB = 0
        self.plane_count = len(regions)
        for plane, polygon, landcover in regions:
            self.memory_MB+= sys.getsizeof(plane) + sys.getsizeof(polygon) + sys.getsizeof(landcover)/ 1024**2

        self.map_scale = map_scale

    def get_z_value_and_lc(self, x, y):
        """
        Get the z value for a given (x, y) point.
        """
        for plane, polygon, landcover in self.regions:

            if len(check_inside_polygon(polygon, np.array([[x, y]])))>0:
                return self.evaluate_plane_z(x, y, plane), landcover
        if False:
            print("List of polygons:")
            for plane, polygon, landcover in self.regions:
                print("Polygon:", polygon)
            print(self.regions)
            print("Warning: Point not inside any polygon.")
            print(f"Point: ({x}, {y})")
            print("Polygons:", self.polygon_pts)
            print("Polygon constains :", check_inside_polygon(self.polygon_pts, np.array([[x, y]])))
        return None

    def reconstruct_region(self, x_vals, y_vals):
        """
        Reconstruct z-values for each (x, y) based on the quadratic surface.
        """
        plane_regions = self.regions
        z_reconstructed, landcover_reconstructed = self.reconstruct_surface_from_planes_with_landcover(plane_regions, x_vals, y_vals)

        return z_reconstructed, landcover_reconstructed
        
    
    def reconstruct_surface_from_planes_with_landcover(self, plane_regions, x_vals, y_vals):
        """
        Reconstruct z-values for each (x, y) based on covering plane polygons.

        Args:
            plane_regions: list of (plane, polygon)
            x_vals, y_vals: 1D arrays of coordinates

        Returns:
            z_reconstructed: 1D array of z values
        """
        z_reconstructed = np.full_like(x_vals, np.nan, dtype=np.float64)
        # z_reconstructed = np.full_like(x_vals, 0, dtype=np.float64)

        landcover_reconstructed = np.full_like(x_vals, 0, dtype=np.int32)


        for plane, polygon, landcover in plane_regions:
            inside_points = check_inside_polygon(polygon, np.column_stack((x_vals, y_vals)))# 1e-4)
            # print("inside_points", inside_points)
            if  len(inside_points) >0:# np.any(inside_points):
                z_reconstructed[inside_points] = self.evaluate_plane_z(x_vals[inside_points]*self.map_scale, y_vals[inside_points]*self.map_scale, plane)

                landcover_reconstructed[inside_points] = landcover



        if np.any(np.isnan(z_reconstructed)):
            print("x,y")#, x_vals, y_vals)
            print("[")
            for x,y, z_r in zip(x_vals, y_vals, z_reconstructed):
                print(f"[{x}, {y}]: {z_r}", end=", ")
            print("]")
            # print(z_reconstructed)

            print("Warning: Some z values are NaN after reconstruction.")


            print(plane_regions)
            # xy = np.column_stack((x_vals, y_vals))
            # print(xy)

            # plot the points
            import matplotlib.pyplot as plt
            plt.scatter(x_vals, y_vals, marker='o')

            for plane, polygon, landcover in plane_regions:
                inside_points = check_inside_polygon(polygon, np.column_stack((x_vals, y_vals)))#1e-4)

                plt.scatter(x_vals[inside_points], y_vals[inside_points], color='red', marker='x')
                polygon = np.array(polygon)
                #randomly color the polygon
                color = np.random.rand(3,)
                plt.fill(polygon[:, 0], polygon[:, 1], color=color, alpha=0.3)

            # plot overall polygon self.polygon_pts
            polygon = np.array(self.polygon_pts)
            plt.fill(polygon[:, 0], polygon[:, 1], color='blue', alpha=0.1)
            # plt.xlim(0, 400)
            # plt.ylim(0, 400)


            plt.title("Reconstructed Points")
            plt.xlabel("X")
            plt.ylabel("Y")
            plt.savefig("results/debug/reconstructed_points.png")


            plt.close()

            exit(0)

        return z_reconstructed, landcover_reconstructed


    def evaluate_region(self, points):
        """
        Evaluate the region based on the plane.
        """
        return self.evaluate_plane_regions_with_landcover(self.regions, points)

    def evaluate_plane_regions_with_landcover(self, plane_regions, points):
        """
        Given plane_regions and original surface points, compute:
        - RMSE of reconstructed z values
        - Memory usage of raw points vs plane_regions
        """
        x_vals, y_vals, z_true = points[:, 0], points[:, 1], points[:, 2]
        z_pred = np.full_like(z_true, np.nan)

        # For each point, find the region it belongs to
        all_points = np.column_stack((x_vals, y_vals))
        for plane, polygon, label in plane_regions:
            inside_points = check_inside_polygon(polygon, all_points)

            if len(inside_points) > 0:
                xi, yi = x_vals[inside_points], y_vals[inside_points]
                z_pred[inside_points] = self.evaluate_plane_z(xi, yi, plane)

        if False:
            for i in range(len(points)):
                xi, yi = x_vals[i], y_vals[i]
                for plane, polygon, landcover in plane_regions:

                    if check_inside_polygon(polygon, np.array([[xi, yi]])):
                        z_pred[i] = self.evaluate_plane_z(xi, yi, plane)
                        break
        

        # Mask out points not covered by any region
        valid = ~np.isnan(z_pred)
        rmse = np.sqrt(np.mean((z_true[valid] - z_pred[valid]) ** 2))

        # Memory
        mem_points = points.nbytes
        mem_planes = sum(sys.getsizeof(plane) + sys.getsizeof(polygon) + sys.getsizeof(landcover)
                        for plane, polygon, landcover in plane_regions)

        # print("coverage", 100 * np.sum(valid) / len(points))
        return {
            "rmse": rmse,
            "memory_points_MB": mem_points / 1024**2,
            "memory_plane_regions_MB": mem_planes / 1024**2,
            "coverage": 100 * np.sum(valid) / len(points)
        }



    def evaluate_plane_z(self, xs, ys, plane):
        """
        Evaluate z = f(x, y) from implicit plane ax + by + cz + d = 0
        plane = (a, b, c, d)
        """
        a, b, c, d = plane["a"], plane["b"], plane["c"], plane["d"]
        if abs(c) < 1e-8:
            print(plane)
            raise ValueError("Plane is vertical; cannot solve for z.")
        return -(a * xs + b * ys + d) / c


class SurfaceRegion(AbstractRegion):
    def __init__(self, regions, centroid, polygon_pts, map_scale=30):
        super().__init__(regions, centroid, polygon_pts, map_scale)
        self.plane_count = 1 # only one surface fitted

        plane, polygon, landcover = regions
        # print("plane", plane, polygon, landcover)
        self.memory_MB =  (sys.getsizeof(plane) + sys.getsizeof(polygon) + sys.getsizeof([landcover]))/ 1024**2


    def reconstruct_region(self, x_vals, y_vals):
        """
        Reconstruct z-values for each (x, y) based on the quadratic surface.
        """
        coeffs, polygon, landcover = self.regions
        # plane_regions = (self.coeffs, self.polygon, self.landcover)
        z_reconstructed = self.reconstruct_surface(coeffs, x_vals, y_vals)

        landcover_reconstructed = np.full_like(x_vals, landcover, dtype=np.int32)

        return z_reconstructed, landcover_reconstructed

    def reconstruct_surface(self, coeffs, x, y):
        """
        Reconstruct z values from given x, y and surface coefficients.
        
        Args:
            coeffs (np.ndarray): [a, b, c, d, e, f]
            x, y (np.ndarray): shape (N,), input grid or points
        
        Returns:
            z (np.ndarray): shape (N,), reconstructed z values
        """
        a, b, c, d, e, f = coeffs
        z = a*x**2 + b*y**2 + c*x*y + d*x + e*y + f
        return z

    def evaluate_region(self, points):
        """
        Evaluate the region based on the plane.
        """
        return self.evaluate_surface(self.regions, points)
    
    def evaluate_surface(self, plane_regions, points):
        """
        Given plane_regions and original surface points, compute:
        - RMSE of reconstructed z values
        - Memory usage of raw points vs plane_regions
        """



        x_vals, y_vals, z_true = points[:, 0], points[:, 1], points[:, 2]
        # z_pred = np.full_like(z_true, np.nan)


        coeffs, polygon, landcover = self.regions
        # plane_regions = (self.coeffs, self.polygon, self.landcover)
        z_pred = self.reconstruct_surface(coeffs, x_vals, y_vals)


        # Mask out points not covered by any region
        valid = ~np.isnan(z_pred)
        rmse = np.sqrt(np.mean((z_true[valid] - z_pred[valid]) ** 2))

        # Memory
        mem_points = points.nbytes
        mem_planes = self.memory_MB

        return {
            "rmse": rmse,
            "memory_points_MB": mem_points / 1024**2,
            "memory_plane_regions_MB": mem_planes ,
            "coverage": 100 * np.sum(valid) / len(points)
        }






def get_first_indices_per_landcover(points, landcover_region):
    landcover_region = np.asarray(landcover_region)
    points = np.asarray(points)

    unique_classes, first_indices = np.unique(landcover_region, return_index=True)

    sort_order = np.argsort(first_indices)
    unique_classes = unique_classes[sort_order]
    first_indices = first_indices[sort_order]

    return unique_classes, first_indices, points[first_indices]



def get_region_attributes_static(points, landcover_region, map_scale=30.0):
    attributes = {}

        
    z = points[:, 2]
    attributes["mean_elevation"] = np.mean(z)

    # min_z and point xy 
    min_z_idx = np.argmin(z)
    min_z = z[min_z_idx]
    min_z_xy = points[min_z_idx, :2]
    attributes["min_elevation"] = (min_z, min_z_xy)
    # max_z and point xy
    max_z_idx = np.argmax(z)
    max_z = z[max_z_idx]
    max_z_xy = points[max_z_idx, :2]
    attributes["max_elevation"] = (max_z, max_z_xy)
    attributes["landcover"] = np.bincount(landcover_region).argmax() if landcover_region is not None else None
    max_dist = np.max(np.linalg.norm(points[:, :2] - points[:, :2].mean(axis=0), axis=1))
    max_z_dist = np.max(np.abs(z - np.mean(z)))
    attributes["max_distance"] = max_dist
    attributes["max_z_distance"] = max_z_dist
    attributes["region_size"] = len(points)


    unique_classes, first_indices, first_points = get_first_indices_per_landcover(points, landcover_region)

    attributes["landcover_points"] = {
        cls: pts for cls, pts in zip(unique_classes, first_points)
    }

    from sklearn.linear_model import LinearRegression

    def fit_plane(points):
        X = points[:, :2]  # x, y
        X[:,0] = X[:,0]*map_scale
        X[:,1] = X[:,1]*map_scale
        y = points[:, 2]   # z (elevation)
        model = LinearRegression().fit(X, y)
        a, b = model.coef_
        c = model.intercept_
        return a, b, c  # z = ax + by + c
    def compute_grade_and_direction(points):
        a, b, _ = fit_plane(points)
        
        # Tangent of slope angle (rise/run)
        tan_theta = np.sqrt(a**2 + b**2)
        
        # Percent grade
        percent_grade = tan_theta * 100
        
        # Grade direction (downslope)
        direction_rad = np.arctan2(-a, -b)
        direction_deg = np.degrees(direction_rad) % 360
        
        return percent_grade, direction_deg

    attributes["grade"], attributes["direction"] = compute_grade_and_direction(points)


    return attributes





from skimage.draw import polygon2mask
import numpy as np

def get_voronoi_mask_independent(region_idx, vor, shape):
    """
    Compute the mask for a given region in a Voronoi diagram.
    Independent of class state.
    """
    region = vor.regions[region_idx]
    if -1 in region or len(region) == 0:
        return None

    polygon_coords = np.array([vor.vertices[i] for i in region])
    return polygon2mask(shape, polygon_coords[:, [1, 0]])




from matplotlib.path import Path
import numpy as np

def fast_voronoi_mask(region_idx, vor, shape):
    """
    Faster mask computation using Path.contains_points instead of polygon2mask.
    """
    region = vor.regions[region_idx]
    if -1 in region or len(region) == 0:
        return None

    polygon_coords = np.array([vor.vertices[i] for i in region])

    # Create a grid of pixel coordinates (row, col)
    yy, xx = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
    points = np.column_stack((yy.ravel(), xx.ravel()))

    path = Path(polygon_coords[:, [1, 0]])  # swap to (x,y)
    mask_flat = path.contains_points(points)
    return mask_flat.reshape(shape)




def get_voronoi_data_independent(region_idx, vor, shape, xv, yv, elevation_map, landcover_map):
    """
    Independent version of get_voronoi_data for parallel processing.
    """
    st_time = time.time()
    mask = get_voronoi_mask_independent(region_idx, vor, shape)

    if mask is None:
        return None, None, mask

    x_region = xv[mask]
    y_region = yv[mask]
    z_region = elevation_map[mask]
    landcover_region = landcover_map[mask]

    if not mask.any():
        return None, None, mask
    if np.any(np.isnan(z_region)) or np.any(np.isnan(landcover_region)):
        return None, None, mask

    points = np.column_stack((x_region, y_region, z_region))
    # print(f"Mask computation for region {region_idx} took {time.time() - st_time:.4f} seconds")

    return points, landcover_region, mask




from scipy.spatial import ConvexHull
import numpy as np

def process_region(
    i,
    vor, xv, yv, elevation_map, landcover_map,
    map_scale=30.0, error_thresh=10.0, max_planes=100,
    elevation_abstraction_method="plane"
):
    """
    Standalone region processor:
      1. Extract region points/mask from Voronoi
      2. Perform elevation abstraction
      3. Build region object with attributes
    """

    # --- 1. Compute region data ---
    shape = elevation_map.shape
    points, landcover_region, _ = get_voronoi_data_independent(
        i, vor, shape, xv, yv, elevation_map, landcover_map
    )


    if points is None:
        return i, None

    vor_points, vor_point_region = vor.points, vor.point_region 

    elevation_abstractor = ElevationAbstractor(
        abstraction_method=elevation_abstraction_method,
        error_thresh=error_thresh,
        max_planes=max_planes
    )
    st_time = time.time()
    method_used, plane_regions = elevation_abstractor.build_abstraction(points, landcover_array=landcover_region)

    try:
        convex_hull = ConvexHull(points[:, :2])
        polygon_pts = points[convex_hull.vertices, :2]
    except Exception:
        polygon_pts = points[:, :2]

    point_idx = np.where(vor_point_region == i)[0][0]
    voronoi_point = vor_points[point_idx]

    if method_used == "surface":
        region_abstraction = SurfaceRegion(plane_regions, voronoi_point, polygon_pts)
    else:
        region_abstraction = PlaneRegion(plane_regions, voronoi_point, polygon_pts)

    region_abstraction.attributes = get_region_attributes_static(points=points, landcover_region=landcover_region, map_scale=map_scale)

    # print(f"Region {i} processed with method {method_used} in {time.time() - st_time:.4f} seconds and {len(plane_regions)} planes.")

    return i, region_abstraction




class RegionBuilder:
    def __init__(self, landcover_map, elevation_map, trasnform, map_name, region_count=1000, map_scale=30.0, decomposition_function=get_landcover_boundary_and_elevation_bin_region_count_v2):
       
        
        self.decomposition_function = decomposition_function

        self.landcover_map = landcover_map
        self.region_count = region_count
        self.elevation_map = elevation_map
        self.transform = trasnform
        self.map_name = map_name

        self.map_scale = map_scale

        rows, cols = np.meshgrid(np.arange(self.elevation_map.shape[0]), np.arange(self.elevation_map.shape[1]), indexing='ij')
        xv  = cols.reshape(self.elevation_map.shape)
        yv  = rows.reshape(self.elevation_map.shape)

        # coords_grid = pixel_to_coords(cols.ravel(), rows.ravel(), self.transform)
        # xv = coords_grid[:, 0].reshape(self.elevation_map.shape)
        # yv = coords_grid[:, 1].reshape(self.elevation_map.shape)
        self.xv = xv
        self.yv = yv


        self.voronoi = None
        self.regions = {}
        self.all_output = {}

        self.patches = []
        self.polygons = []

        self.neighbors = {} 


    def get_polygon_coords(self, region_idx):
        vor = self.voronoi

        region = vor.regions[region_idx]
        if -1 in region or len(region) == 0:
            return None
        polygon_coords = [vor.vertices[i] for i in region]

        polygon_coords_xy = polygon_coords
        polygon = Polygon(polygon_coords_xy)
        return  polygon        

    def get_voronoi_mask(self, region_idx):
        from skimage.draw import polygon2mask

        vor, shape = self.voronoi, self.elevation_map.shape

        region = vor.regions[region_idx]
        if -1 in region or len(region) == 0:
            return None
        polygon_coords = [vor.vertices[i] for i in region]

        polygon_coords = np.array(polygon_coords)

        mask = polygon2mask(shape, polygon_coords[:, [1, 0]])

        return mask

    def get_voronoi_data(self, region_idx):
        """
        Get Voronoi data for a specific region index.
        Returns the vertices and region indices.
        """
        mask = self.get_voronoi_mask(region_idx)

        if mask is None:
            return None, None, mask
        x_region = self.xv[mask]
        y_region = self.yv[mask]
        z_region = self.elevation_map[mask]
        landcover_region = self.landcover_map[mask]

        if np.all(mask == False):
            return None, None, mask
        if np.any(np.isnan(z_region)) or np.any(np.isnan(landcover_region)):
            return None, None, mask
    
        points = np.column_stack((x_region, y_region, z_region))
        return points, landcover_region, mask


    def get_region_attributes(self, region_idx, points=None, landcover_region=None):
        attributes = {}

        if points is None:
            points, landcover_region, mask = self.get_voronoi_data(region_idx)
            
        z = points[:, 2]
        attributes["mean_elevation"] = np.mean(z)

        # min_z and point xy 
        min_z_idx = np.argmin(z)
        min_z = z[min_z_idx]
        min_z_xy = points[min_z_idx, :2]
        attributes["min_elevation"] = (min_z, min_z_xy)
        # max_z and point xy
        max_z_idx = np.argmax(z)
        max_z = z[max_z_idx]
        max_z_xy = points[max_z_idx, :2]
        attributes["max_elevation"] = (max_z, max_z_xy)
        attributes["landcover"] = np.bincount(landcover_region).argmax() if landcover_region is not None else None
        max_dist = np.max(np.linalg.norm(points[:, :2] - points[:, :2].mean(axis=0), axis=1))
        max_z_dist = np.max(np.abs(z - np.mean(z)))
        attributes["max_distance"] = max_dist
        attributes["max_z_distance"] = max_z_dist
        attributes["region_size"] = len(points)
        # attributes["region_area"] = np.sum(mask) * (self.transform[0] * self.transform[4])
        # grade = max_z_dist / max_dist if max_dist > 0 else 0
        # attributes["grade"] = grade



        from sklearn.linear_model import LinearRegression

        def fit_plane(points):
            X = points[:, :2]  # x, y
            X[:,0] = X[:,0]*self.map_scale
            X[:,1] = X[:,1]*self.map_scale
            y = points[:, 2]   # z (elevation)
            model = LinearRegression().fit(X, y)
            a, b = model.coef_
            c = model.intercept_
            return a, b, c  # z = ax + by + c
        def compute_grade_and_direction(points):
            a, b, _ = fit_plane(points)
            
            # Tangent of slope angle (rise/run)
            tan_theta = np.sqrt(a**2 + b**2)
            
            # Percent grade
            percent_grade = tan_theta * 100
            
            # Grade direction (downslope)
            direction_rad = np.arctan2(-a, -b)
            direction_deg = np.degrees(direction_rad) % 360
            
            return percent_grade, direction_deg

        attributes["grade"], attributes["direction"] = compute_grade_and_direction(points)


        return attributes

    def decompose(self,  decomposition="voronoi" , elevation_bins=10, min_area=1, flatness_ratio=0.7):

        st_time = time.time()
        
        if decomposition == "voronoi":
            # self.voronoi = get_landcover_boundary_region_count(self.landcover_map, region_count=self.region_count)
            # self.voronoi = get_landcover_boundary_and_elevation_bin_region_count_v2(self.landcover_map, self.elevation_map, region_count=self.region_count, elevation_bins=elevation_bins)
        
            self.voronoi = self.decomposition_function(self.landcover_map, self.elevation_map, region_count=self.region_count, elevation_bins=elevation_bins, min_area=min_area, flatness_ratio=flatness_ratio)
        
        else:
            raise ValueError("The standalone package supports CLEAR only.")
        

        voronoi_time = time.time() - st_time
        self.all_output["decomposition_time"] = voronoi_time


    def build_regions(self, decomposition="voronoi", elevation_abstraction_method="plane", elevation_bins=10, min_area=1, max_planes=100, error_thresh=10.0, flatness_ratio=0.7):


        self.decompose(decomposition=decomposition, elevation_bins=elevation_bins, min_area=min_area, flatness_ratio=flatness_ratio)
        vor = self.voronoi

        st_time= time.time()




        if False:
            elevation_abstractor = ElevationAbstractor(abstraction_method=elevation_abstraction_method, error_thresh=error_thresh, max_planes=max_planes)

            for i in tqdm(range(len(vor.regions))):

                points, landcover_region, _  = self.get_voronoi_data(i)
                if points is None:
                    continue

                method_used, plane_regions = elevation_abstractor.build_abstraction(points, landcover_array=landcover_region)
                try:
                    convex_hull = ConvexHull(points[:, :2])
                    # polygon_pts = [(points[j, 0], points[j, 1]) for j in convex_hull.vertices]
                    polygon_pts = np.array(convex_hull.vertices)
                    polygon_pts = points[polygon_pts, :2]

                except Exception as e:# QhullError as e:

                    polygon_pts = np.array(points[:, :2])

                point_idx = np.where(vor.point_region == i)[0][0]
                voronoi_point = vor.points[point_idx]

                if method_used=="surface":
                    region_abstraction = SurfaceRegion(plane_regions, voronoi_point, polygon_pts)
                else:
                    region_abstraction = PlaneRegion(plane_regions, voronoi_point, polygon_pts)

                region_abstraction.attributes = self.get_region_attributes(i, points=points, landcover_region=landcover_region)

                self.regions[i]=region_abstraction

        else:
            map_scale = self.map_scale
            xv, yv = self.xv, self.yv
            elevation_map = self.elevation_map
            landcover_map = self.landcover_map


            def process_batch(batch_indices, vor, xv, yv, elevation_map, landcover_map,
                  map_scale, error_thresh, max_planes, elevation_abstraction_method):
                results = []
                for i in batch_indices:
                    results.append(
                        process_region(i, vor, xv, yv, elevation_map, landcover_map,
                                    map_scale, error_thresh, max_planes,
                                    elevation_abstraction_method)
                    )
                return results



            from concurrent.futures import ProcessPoolExecutor, as_completed, ThreadPoolExecutor
            from tqdm import tqdm
            import math

            indices = list(range(len(vor.regions)))
            batch_size = 5000  # tune this (20–100) based on task cost
            batches = [indices[i:i+batch_size] for i in range(0, len(indices), batch_size)]

            # with ProcessPoolExecutor(max_workers=2) as ex:
            #     futures = [
            #         ex.submit(process_batch, batch, vor, xv, yv, elevation_map, landcover_map,
            #                 map_scale, error_thresh, max_planes, elevation_abstraction_method)
            #         for batch in batches
            #     ]

            #     for f in tqdm(as_completed(futures), total=len(futures)):
            #         for i, region_abstraction in f.result():
            #             if region_abstraction is not None:
            #                 self.regions[i] = region_abstraction

            with ThreadPoolExecutor(max_workers=4) as ex:
                futures = [ex.submit(process_batch,
                                    batch, vor, xv, yv, elevation_map, landcover_map,
                                    map_scale, error_thresh, max_planes,
                                    elevation_abstraction_method)
                        for batch in batches]

                for f in tqdm(as_completed(futures), total=len(futures)):
                    for i, region_abstraction in f.result():
                        if region_abstraction is not None:
                            self.regions[i] = region_abstraction




            # #########################################################


            # from concurrent.futures import ThreadPoolExecutor, as_completed

            
            # indices = list(range(len(vor.regions)))


            # with ThreadPoolExecutor(max_workers=16) as ex:
            #     futures = [ex.submit(process_region,
            #                         i, vor, xv, yv, elevation_map, landcover_map,
            #                         map_scale, error_thresh, max_planes,
            #                         elevation_abstraction_method)
            #             for i in indices]

            #     for f in tqdm(as_completed(futures), total=len(futures)):
            #         i, region_abstraction = f.result()
            #         if region_abstraction is not None:
            #             self.regions[i] = region_abstraction


            # #########################################################

            
            # map_scale = self.map_scale


            # from concurrent.futures import ProcessPoolExecutor, as_completed

            # from scipy.spatial import ConvexHull, QhullError

            # xv, yv = self.xv, self.yv
            # elevation_map = self.elevation_map
            # landcover_map = self.landcover_map

            # print("Processing regions in parallel..."   )
            # with ProcessPoolExecutor() as executor:
            #     futures = {}
            #     start_iter_time = time.time()

            #     for i in range(len(vor.regions)):
            #         futures[executor.submit(
            #                 process_region,
            #                 i,
            #                 vor, xv, yv, elevation_map, landcover_map,
            #                 map_scale, error_thresh, max_planes,
            #                 elevation_abstraction_method
            #         )] = i
            #     print(f"Region processing started in {time.time() - start_iter_time:.2f} seconds.")

            # for future in tqdm(as_completed(futures), total=len(futures)):
            #     i, region_abstraction = future.result()
            #     if region_abstraction is not None:
            #         self.regions[i] = region_abstraction


        self.all_output["elevation_abstraction_time"] = time.time() - st_time
        print(f"Region building took {self.all_output['elevation_abstraction_time']:.2f} seconds for {len(self.regions)} regions.")




    def reconstruction_evaluation(self, debug = False):

        st_time= time.time()
        elevation_data = self.elevation_map
        landcover_data = self.landcover_map
        
        elev_rmse = []
        elev_std = []
        elev_max = []

        plane_counts = []
        region_indices = []

        vor = self.voronoi

        elevation_fitted = np.full(elevation_data.shape, np.nan)

        landcover_data = landcover_data.astype(np.int32)
        landcover_fitted = np.full(landcover_data.shape, 0).astype(np.int32)

        total_memory_original = []
        total_memory_planes = []
        em = EvaluationMetrics()

        for i in tqdm(range(len(vor.regions))):

            points, _, mask = self.get_voronoi_data(i)
            if points is None:
                continue

            abst_region = self.regions[i]

            x_region, y_region, z_region = points[:, 0], points[:, 1], points[:, 2]

            points = np.column_stack((x_region, y_region, z_region))

            # print("points", points)
            reconstructed_z, landcover_reconstructed = abst_region.reconstruct_region(x_region, y_region)

            
            metrics = abst_region.evaluate_region(points)
            
            memory_points_MB = metrics["memory_points_MB"]
            memory_planes_MB = metrics["memory_plane_regions_MB"]
            # coverage = metrics["coverage"]
            total_memory_original.append(memory_points_MB)
            total_memory_planes.append(memory_planes_MB)


            plane_count = abst_region.get_plane_count()

            elevation_fitted[mask] = reconstructed_z

            landcover_fitted[mask] = landcover_reconstructed

            rmse, var, max_res = em.elevation_info_loss(z_region, reconstructed_z)
        
            elev_rmse.append(rmse)
            elev_std.append(np.sqrt(var))
            elev_max.append(max_res)
            plane_counts.append(plane_count)
            region_indices.append(i)


        print("Plane fitting took: ", time.time() - st_time)

        loss_df = pd.DataFrame(
            { "Region Index": region_indices,
            "Elevation RMSE": elev_rmse,
            "Elevation (Std Dev)": elev_std,
            "Elevation Max Residual": elev_max,
            "Plane Count": plane_counts,

        })

        values1 = loss_df["Plane Count"].dropna()

        # elevation statistics
        values2 = loss_df["Elevation RMSE"].dropna()
        values3 = loss_df["Elevation (Std Dev)"].dropna()
        values4 = loss_df["Elevation Max Residual"].dropna()

        # memory statistics
        values5 = np.array(total_memory_original)
        original_memory = values5.sum()

        values6 = np.array(total_memory_planes)
        plane_memory = values6.sum()

        
        # classification error 
        result = em.evaluate_landcover_classification(landcover_data.flatten(), landcover_fitted.flatten())

        if debug:
            print(f"Plane Count: min={values1.min()}, max={values1.max()}, mean={values1.mean():.2f}")

            print(f"Elevation RMSE: min={values2.min():.4f}, max={values2.max():.4f}, mean={values2.mean():.4f}")
            print(f"Elevation Std Dev: min={values3.min():.4f}, max={values3.max():.4f}, mean={values3.mean():.4f}")
            print(f"Elevation Max Residual: min={values4.min():.4f}, max={values4.max():.4f}, mean={values4.mean():.4f}")
            print(f"Original Memory: min={values5.min():.4f} MB, max={values5.max():.4f} MB, mean={values5.mean():.4f} MB, total={values5.sum():.4f} MB")
            print(f"Plane Memory: min={values6.min():.4f} MB, max={values6.max():.4f} MB, mean={values6.mean():.4f} MB, total={values6.sum():.4f} MB")
            print(f"Landcover Classification Accuracy: {result['accuracy']:.4f}")

        all_output = {
            "loss_df": loss_df,
            "elevation_fitted": elevation_fitted,
            "landcover_fitted": landcover_fitted,
            "original_memory": original_memory,
            "plane_memory": plane_memory,
            "landcover_accuracy": result["accuracy"],
        }
        for key, value in all_output.items():
            self.all_output[key] = value

        return all_output

    def get_patches(self):
        """
        Get the patches for the regions.
        """
        return self.patches

    def set_patches(self, patches):
        """
        Set the patches for the regions.
        """
        self.patches = patches


    def get_neighbors(self, region_idx):
        """
        Get the neighbors for the regions.
        """
        return self.neighbors[region_idx]

        # vor = self.voronoi
        # neighbors = vor.neighbors[region_idx]
        # return neighbors


from shapely.geometry import Polygon, LineString, Point, box
from scipy.spatial import Voronoi
import numpy as np

def get_bounded_voronoi_polygons_old(vor: Voronoi, elevation_map_shape):
    """
    Returns list of shapely.Polygon objects clipped to image extent,
    including finite and infinite Voronoi regions.
    """
    height, width = elevation_map_shape
    bbox = box(0, 0, width, height)
    center = vor.points.mean(axis=0)
    radius = max(width, height) * 2  # large enough to close open regions

    regions = []

    for point_idx, region_idx in enumerate(vor.point_region):
        vertices = vor.regions[region_idx]
        if len(vertices) == 0:
            continue

        if -1 in vertices:
            # Infinite region: reconstruct using ridges
            region_lines = []
            for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
                if point_idx not in (p1, p2):
                    continue
                if -1 in (v1, v2):
                    # Infinite ridge
                    t = vor.points[p2] - vor.points[p1]
                    t = t / np.linalg.norm(t)
                    n = np.array([-t[1], t[0]])  # normal vector

                    midpoint = vor.points[[p1, p2]].mean(axis=0)
                    direction = np.sign(np.dot(midpoint - center, n)) * n
                    finite_vertex = vor.vertices[v1 if v2 == -1 else v2]
                    far_point = finite_vertex + direction * radius
                    region_lines.append(finite_vertex)
                    region_lines.append(far_point)
                else:
                    region_lines.append(vor.vertices[v1])
                    region_lines.append(vor.vertices[v2])

            # Construct polygon from collected points
            coords = np.array(region_lines)
            try:
                poly = Polygon(coords).convex_hull
                clipped = poly.intersection(bbox)
                if clipped.is_valid and not clipped.is_empty:
                    regions.append(clipped)
            except:
                continue
        else:
            # Finite polygon
            coords = vor.vertices[vertices]
            poly = Polygon(coords)
            clipped = poly.intersection(bbox)
            if clipped.is_valid and not clipped.is_empty:
                regions.append(clipped)

    return regions


def get_bounded_voronoi_polygons(vor: Voronoi, elevation_map_shape):
    """
    Efficiently constructs bounded Voronoi polygons (Shapely) clipped to elevation map extent.
    Handles infinite regions robustly.
    """
    from collections import defaultdict
    height, width = elevation_map_shape
    bbox = box(0, 0, width, height)
    center = vor.points.mean(axis=0)
    radius = max(width, height) * 2

    # Build ridge map: point_idx -> list of (neighbor_point_idx, vertex_pair)
    ridge_map = defaultdict(list)
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        ridge_map[p1].append((p2, v1, v2))
        ridge_map[p2].append((p1, v1, v2))

    regions = []

    for p_idx, region_idx in enumerate(vor.point_region):
        vertices = vor.regions[region_idx]
        if not vertices:
            regions.append(Polygon())  # empty polygon
            continue

        if -1 not in vertices:
            poly = Polygon(vor.vertices[vertices])
            poly = poly.intersection(bbox)
            regions.append(poly if poly.is_valid else Polygon())
            continue

        # Reconstruct infinite region polygon
        region_vertices = []
        for neighbor_idx, v1, v2 in ridge_map[p_idx]:
            if v1 >= 0 and v2 >= 0:
                # finite edge
                region_vertices.append(vor.vertices[v1])
                region_vertices.append(vor.vertices[v2])
            else:
                # infinite edge
                t = vor.points[neighbor_idx] - vor.points[p_idx]
                t /= np.linalg.norm(t)
                n = np.array([-t[1], t[0]])
                midpoint = (vor.points[p_idx] + vor.points[neighbor_idx]) / 2
                direction = np.sign(np.dot(midpoint - center, n)) * n
                finite_vertex = vor.vertices[v1 if v1 >= 0 else v2]
                far_point = finite_vertex + direction * radius
                region_vertices.append(finite_vertex)
                region_vertices.append(far_point)

        try:
            poly = Polygon(region_vertices).convex_hull
            poly = poly.intersection(bbox)
            regions.append(poly if poly.is_valid else Polygon())
        except:
            regions.append(Polygon())  # fallback

    return regions







from shapely.geometry import Polygon, box
import numpy as np
from collections import defaultdict

def get_bounded_voronoi_polygons(vor, elevation_map_shape):
    """
    Constructs bounded Voronoi polygons clipped to elevation map extent.
    Also returns a list of neighbor indices per polygon.
    
    Returns:
        - regions: List[Polygon] — clipped polygons
        - neighbors: List[List[int]] — indices of neighboring regions
    """
    height, width = elevation_map_shape
    bbox = box(0, 0, width, height)
    center = vor.points.mean(axis=0)
    radius = max(width, height) * 2

    # Build ridge map and neighbor map
    ridge_map = defaultdict(list)
    neighbor_map = defaultdict(set)

    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        ridge_map[p1].append((p2, v1, v2))
        ridge_map[p2].append((p1, v1, v2))
        neighbor_map[p1].add(p2)
        neighbor_map[p2].add(p1)

    regions = []
    neighbors = []
    centroids = []
    

    for p_idx, region_idx in enumerate(vor.point_region):
        vertices = vor.regions[region_idx]
        point_center = vor.points[p_idx]
        if not vertices:
            regions.append(Polygon())
            neighbors.append([])
            centroids.append(point_center)
            continue

        if -1 not in vertices:
            poly = Polygon(vor.vertices[vertices])
            poly = poly.intersection(bbox)
            regions.append(poly if poly.is_valid else Polygon())
            neighbors.append(sorted(neighbor_map[p_idx]))
            centroids.append(point_center)

            continue

        # Infinite region reconstruction
        region_vertices = []
        for neighbor_idx, v1, v2 in ridge_map[p_idx]:
            if v1 >= 0 and v2 >= 0:
                region_vertices.append(vor.vertices[v1])
                region_vertices.append(vor.vertices[v2])
            else:
                t = vor.points[neighbor_idx] - vor.points[p_idx]
                t /= np.linalg.norm(t)
                n = np.array([-t[1], t[0]])
                midpoint = (vor.points[p_idx] + vor.points[neighbor_idx]) / 2
                direction = np.sign(np.dot(midpoint - center, n)) * n
                finite_vertex = vor.vertices[v1 if v1 >= 0 else v2]
                far_point = finite_vertex + direction * radius
                region_vertices.append(finite_vertex)
                region_vertices.append(far_point)

        try:
            poly = Polygon(region_vertices).convex_hull
            poly = poly.intersection(bbox)
            regions.append(poly if poly.is_valid else Polygon())
            centroids.append([poly.centroid.x, poly.centroid.y])
        except:
            regions.append(Polygon())

        neighbors.append(sorted(neighbor_map[p_idx]))

    return regions, neighbors, centroids



from skimage.draw import polygon2mask


def polygons_to_masks(polygon, elevation_map_shape):
    """
    Given a list of shapely.Polygon objects and elevation map shape,
    return a list of boolean masks (same shape) indicating covered pixels.
    
    Args:
        polygons: list of shapely.geometry.Polygon
        elevation_map_shape: (H, W) tuple
    
    Returns:
        List of (H, W) boolean numpy arrays
    """
    masks = []
    poly = polygon
    if poly.is_empty or poly.geom_type != 'Polygon':
        mask = np.zeros(elevation_map_shape, dtype=bool)
    else:
        coords = np.array(poly.exterior.coords)
        mask = polygon2mask(elevation_map_shape, coords[:, [1, 0]])  # (row, col)
    return mask


class RegionBuilderPatches(RegionBuilder):
    def __init__(self, landcover_map, elevation_map, transform, map_name, region_count=1000):
        super().__init__(landcover_map, elevation_map, transform, map_name, region_count)

        self.centroids = []

    def get_polygon_data(self, polygon):

        mask = polygons_to_masks(polygon, self.elevation_map.shape)
        if mask is None:
            return None, None, mask
        x_region = self.xv[mask]
        y_region = self.yv[mask]
        z_region = self.elevation_map[mask]
        landcover_region = self.landcover_map[mask]

        if np.all(mask == False):
            return None, None, mask
        if np.any(np.isnan(z_region)) or np.any(np.isnan(landcover_region)):
            return None, None, mask
    
        points = np.column_stack((x_region, y_region, z_region))
        return points, landcover_region, mask



    def get_region_attributes(self, region_idx, points=None, landcover_region=None):
        attributes = {}

        if points is None:
            polygon = self.polygons[region_idx]
            points, landcover_region, mask = self.get_polygon_data(polygon)

        z = points[:, 2]
        attributes["mean_elevation"] = np.mean(z)
        attributes["landcover"] = np.bincount(landcover_region).argmax() if landcover_region is not None else None
        max_dist = np.max(np.linalg.norm(points[:, :2] - points[:, :2].mean(axis=0), axis=1))
        max_z_dist = np.max(np.abs(z - np.mean(z)))
        attributes["max_distance"] = max_dist
        attributes["max_z_distance"] = max_z_dist
        attributes["region_size"] = len(points)

        from sklearn.linear_model import LinearRegression

        def fit_plane(points):
            X = points[:, :2]  # x, y
            X[:,0] = X[:,0]*self.map_scale
            X[:,1] = X[:,1]*self.map_scale
            y = points[:, 2]   # z (elevation)
            model = LinearRegression().fit(X, y)
            a, b = model.coef_
            c = model.intercept_
            return a, b, c  # z = ax + by + c
        def compute_grade_and_direction(points):
            a, b, _ = fit_plane(points)
            
            # Tangent of slope angle (rise/run)
            tan_theta = np.sqrt(a**2 + b**2)
            
            # Percent grade
            percent_grade = tan_theta * 100
            
            # Grade direction (downslope)
            direction_rad = np.arctan2(-a, -b)
            direction_deg = np.degrees(direction_rad) % 360
            
            return percent_grade, direction_deg

        attributes["grade"], attributes["direction"] = compute_grade_and_direction(points)


        return attributes



    def decompose(self,  decomposition="voronoi" , elevation_bins=10, min_area=1, flatness_ratio=0.7):

        st_time = time.time()
        
        if decomposition == "voronoi":

            # print("==================Decomposing landcover map using Voronoi decomposition... min_area: ", min_area)
            # self.voronoi = get_landcover_boundary_region_count(self.landcover_map, region_count=self.region_count)
            # self.voronoi = get_landcover_boundary_and_elevation_bin_region_count_v2(self.landcover_map, self.elevation_map, region_count=self.region_count, elevation_bins=elevation_bins)
            self.voronoi = get_landcover_boundary_and_elevation_bin_region_count_v7(self.landcover_map, self.elevation_map, region_count=self.region_count, elevation_bins=elevation_bins, min_area=min_area, flatness_ratio=flatness_ratio)


        elif decomposition == "grid":
            self.voronoi = get_landcover_grid_samples(self.landcover_map, region_count=self.region_count)
        elif decomposition == "hex":
            self.voronoi = get_landcover_hex_samples(self.landcover_map, region_count=self.region_count)
        else:
            raise ValueError("Invalid decomposition method. Choose 'voronoi', 'grid', or 'hex'.")
        

        self.polygons, self.neighbors, self.centroids = get_bounded_voronoi_polygons(self.voronoi, self.elevation_map.shape)

        # print("Voronoi decomposition took: ", time.time() - st_time)

        decomposition_time = time.time() - st_time
        self.all_output["decomposition_time"] = decomposition_time



    def process_polygons(self, elevation_abstraction_method, error_thresh=10.0, max_planes=100):

        from joblib import Parallel, delayed
        from scipy.spatial import ConvexHull, QhullError

        def process_polygon(i, polygon, get_data, elevation_abstractor, get_attributes):
            points, landcover_region, _ = get_data(polygon)
            if points is None:
                return i, None

            method_used, plane_regions = elevation_abstractor.build_abstraction(points, landcover_array=landcover_region)

            try:
                hull = ConvexHull(points[:, :2])
                polygon_pts = points[hull.vertices, :2]
            except QhullError:
                polygon_pts = points[:, :2]

            centroid = np.mean(points[:, :2], axis=0)
            if method_used == "surface":
                region = SurfaceRegion(plane_regions, centroid, polygon_pts)
            else:
                region = PlaneRegion(plane_regions, centroid, polygon_pts)

            region.attributes = get_attributes(i, points=points, landcover_region=landcover_region)
            return i, region



        elevation_abstractor = ElevationAbstractor(abstraction_method=elevation_abstraction_method, error_thresh=error_thresh, max_planes=max_planes)

        # Main loop using joblib
        results = Parallel(n_jobs=-1)(
            delayed(process_polygon)(i, self.polygons[i], self.get_polygon_data, elevation_abstractor, self.get_region_attributes)
            for i in tqdm(range(len(self.polygons)))
        )

        # Assign results
        for i, region in results:
            if region is not None:
                self.regions[i] = region






    def build_regions(self, decomposition="voronoi", elevation_abstraction_method="plane", elevation_bins=10, min_area=1, max_planes=100, error_thresh=10.0, flatness_ratio=0.7):

        self.decompose(decomposition=decomposition, elevation_bins=elevation_bins, min_area=min_area, flatness_ratio=flatness_ratio)

        print("Number of polygons Extracted: ", len(self.polygons))

        st_time= time.time()

        compute_parellel = False
        if compute_parellel:

            self.process_polygons(elevation_abstraction_method, error_thresh=error_thresh, max_planes=max_planes)
        else:
            elevation_abstractor = ElevationAbstractor(abstraction_method=elevation_abstraction_method, error_thresh=error_thresh, max_planes=max_planes)

            # print("Number of polygons: ", len(self.polygons))
            for i in tqdm(range(len(self.polygons))):
                points, landcover_region, _  = self.get_polygon_data(self.polygons[i])

                centroid = self.centroids[i]

                if points is None:
                    continue

                method_used, plane_regions = elevation_abstractor.build_abstraction(points, landcover_array=landcover_region)
                try:
                    convex_hull = ConvexHull(points[:, :2])
                    polygon_pts = np.array(convex_hull.vertices)
                    polygon_pts = points[polygon_pts, :2]

                except QhullError as e:

                    polygon_pts = np.array(points[:, :2])

                # centroid = np.mean(points[:, :2], axis=0)

                if method_used=="surface":
                    region_abstraction = SurfaceRegion(plane_regions, centroid, polygon_pts)
                else:
                    region_abstraction = PlaneRegion(plane_regions, centroid, polygon_pts)

                region_abstraction.attributes = self.get_region_attributes(i, points=points, landcover_region=landcover_region)

                self.regions[i]=region_abstraction

        self.all_output["elevation_abstraction_time"] = time.time() - st_time

    


    def reconstruction_evaluation(self, debug = False):

        st_time= time.time()
        elevation_data = self.elevation_map
        landcover_data = self.landcover_map
        
        elev_rmse = []
        elev_std = []
        elev_max = []

        plane_counts = []
        region_indices = []

        elevation_fitted = np.full(elevation_data.shape, np.nan)

        landcover_data = landcover_data.astype(np.int32)
        landcover_fitted = np.full(landcover_data.shape, -1).astype(np.int32)

        total_memory_original = []
        total_memory_planes = []
        em = EvaluationMetrics()

        for i in tqdm(range(len(self.polygons))):

            points, _, mask = self.get_polygon_data(self.polygons[i])
            if points is None:
                continue

            abst_region = self.regions[i]

            x_region, y_region, z_region = points[:, 0], points[:, 1], points[:, 2]

            points = np.column_stack((x_region, y_region, z_region))

            # print("points", points)
            reconstructed_z, landcover_reconstructed = abst_region.reconstruct_region(x_region, y_region)

            
            metrics = abst_region.evaluate_region(points)
            
            memory_points_MB = metrics["memory_points_MB"]
            memory_planes_MB = metrics["memory_plane_regions_MB"]
            # coverage = metrics["coverage"]
            total_memory_original.append(memory_points_MB)
            total_memory_planes.append(memory_planes_MB)


            plane_count = abst_region.get_plane_count()

            elevation_fitted[mask] = reconstructed_z

            landcover_fitted[mask] = landcover_reconstructed

            rmse, var, max_res = em.elevation_info_loss(z_region, reconstructed_z)
        
            elev_rmse.append(rmse)
            elev_std.append(np.sqrt(var))
            elev_max.append(max_res)
            plane_counts.append(plane_count)
            region_indices.append(i)


        # print("Plane fitting took: ", time.time() - st_time)

        loss_df = pd.DataFrame(
            { "Region Index": region_indices,
            "Elevation RMSE": elev_rmse,
            "Elevation (Std Dev)": elev_std,
            "Elevation Max Residual": elev_max,
            "Plane Count": plane_counts,

        })

        values1 = loss_df["Plane Count"].dropna()

        # elevation statistics
        values2 = loss_df["Elevation RMSE"].dropna()
        values3 = loss_df["Elevation (Std Dev)"].dropna()
        values4 = loss_df["Elevation Max Residual"].dropna()

        # memory statistics
        values5 = np.array(total_memory_original)
        original_memory = values5.sum()

        values6 = np.array(total_memory_planes)
        plane_memory = values6.sum()


        
        # classification error 
        result = em.evaluate_landcover_classification(landcover_data.flatten(), landcover_fitted.flatten())


        if debug:
            print(f"Plane Count: min={values1.min()}, max={values1.max()}, mean={values1.mean():.2f}")

            print(f"Elevation RMSE: min={values2.min():.4f}, max={values2.max():.4f}, mean={values2.mean():.4f}")
            print(f"Elevation Std Dev: min={values3.min():.4f}, max={values3.max():.4f}, mean={values3.mean():.4f}")
            print(f"Elevation Max Residual: min={values4.min():.4f}, max={values4.max():.4f}, mean={values4.mean():.4f}")
            print(f"Original Memory: min={values5.min():.4f} MB, max={values5.max():.4f} MB, mean={values5.mean():.4f} MB, total={values5.sum():.4f} MB")
            print(f"Plane Memory: min={values6.min():.4f} MB, max={values6.max():.4f} MB, mean={values6.mean():.4f} MB, total={values6.sum():.4f} MB")
            print(f"Landcover Classification Accuracy: {result['accuracy']:.4f}")

        all_output = {
            "loss_df": loss_df,
            "elevation_fitted": elevation_fitted,
            "landcover_fitted": landcover_fitted,
            "original_memory": original_memory,
            "plane_memory": plane_memory,
            "landcover_accuracy": result["accuracy"],
        }
        for key, value in all_output.items():
            self.all_output[key] = value

        return all_output



class RegionBuilderQuadtree(RegionBuilderPatches):
    def __init__(self, landcover_map, elevation_map, transform, map_name, region_count=1000):
        super().__init__(landcover_map, elevation_map, transform, map_name, region_count)
        self.quadtree = None


    def get_patches(self):
        """
        Get the patches for the regions.
        """
        return self.patches

    def get_region_attributes(self, region_idx, points=None, landcover_region=None):
        attributes = {}

        patches = self.get_patches()
        if points is None:
            patch = patches[region_idx]
            x, y, w, h, label, avg_elevation = patch
            points = np.column_stack((self.xv[y:y+h, x:x+w].ravel(), self.yv[y:y+h, x:x+w].ravel(), self.elevation_map[y:y+h, x:x+w].ravel()))
            landcover_region = self.landcover_map[y:y+h, x:x+w].ravel()
            
        z = points[:, 2]
        attributes["mean_elevation"] = np.mean(z)
        attributes["landcover"] = np.bincount(landcover_region).argmax() if landcover_region is not None else None
        max_dist = np.max(np.linalg.norm(points[:, :2] - points[:, :2].mean(axis=0), axis=1))
        max_z_dist = np.max(np.abs(z - np.mean(z)))
        attributes["max_distance"] = max_dist
        attributes["max_z_distance"] = max_z_dist
        attributes["region_size"] = len(points)


        from sklearn.linear_model import LinearRegression

        def fit_plane(points):
            X = points[:, :2]  # x, y
            X[:,0] = X[:,0]*self.map_scale
            X[:,1] = X[:,1]*self.map_scale
            y = points[:, 2]   # z (elevation)
            model = LinearRegression().fit(X, y)
            a, b = model.coef_
            c = model.intercept_
            return a, b, c  # z = ax + by + c
        def compute_grade_and_direction(points):
            a, b, _ = fit_plane(points)
            
            # Tangent of slope angle (rise/run)
            tan_theta = np.sqrt(a**2 + b**2)
            
            # Percent grade
            percent_grade = tan_theta * 100
            
            # Grade direction (downslope)
            direction_rad = np.arctan2(-a, -b)
            direction_deg = np.degrees(direction_rad) % 360
            
            return percent_grade, direction_deg

        attributes["grade"], attributes["direction"] = compute_grade_and_direction(points)


        return attributes


    def decompose(self,  min_area=1):

        st_time = time.time()
        self.quadtree = self.quadtree_decompose_full_coverage(self.landcover_map, min_size=min_area)

        patches = self.get_patch_descriptors_with_elevation(self.landcover_map, self.elevation_map, self.quadtree)
        self.set_patches(patches)




        from rtree import index

        def get_patch_neighbors_from_boxes(patches, connectivity=4):
            """
            Fast neighbor detection using bounding boxes (R-tree).
            `patches` is a list of (x, y, w, h, label, elevation).
            Returns: List[List[int]] of neighbor indices.
            """
            # Build R-tree index
            rtree_idx = index.Index()
            for i, (x, y, w, h, *_ ) in enumerate(patches):
                rtree_idx.insert(i, (x, y, x + w, y + h))

            neighbors = [[] for _ in patches]

            for i, (x1, y1, w1, h1, *_ ) in enumerate(patches):
                cx1, cy1 = x1 + w1 / 2, y1 + h1 / 2
                avg_w, avg_h = w1, h1

                # Expand search box slightly to include adjacent boxes
                search_box = (x1 - 1, y1 - 1, x1 + w1 + 1, y1 + h1 + 1)

                for j in rtree_idx.intersection(search_box):
                    if i == j:
                        continue

                    x2, y2, w2, h2, *_ = patches[j]
                    cx2, cy2 = x2 + w2 / 2, y2 + h2 / 2
                    dx, dy = abs(cx1 - cx2), abs(cy1 - cy2)
                    avg_wj, avg_hj = (w1 + w2) / 2, (h1 + h2) / 2

                    if connectivity == 4:
                        adjacent = (dx <= avg_wj and dy < 1e-6) or (dy <= avg_hj and dx < 1e-6)
                    else:
                        adjacent = dx <= avg_wj and dy <= avg_hj

                    if adjacent:
                        neighbors[i].append(j)
                        neighbors[j].append(i)

            return neighbors





        self.neighbors = get_patch_neighbors_from_boxes(patches, connectivity=8)

        print("Number of neighbors: ", len(self.neighbors))

        # # neighbors 
        # # Add edges
        # connectivity = 4

        # self.neighbors = defaultdict(list)

        # for i in range(len(patches)):
        #     for j in range(i+1, len(patches)):
        #         x1, y1, w1, h1, _, _ = patches[i]
        #         x2, y2, w2, h2, _, _ = patches[j]

        #         cx1 = x1 + w1 / 2
        #         cy1 = y1 + h1 / 2

        #         cx2 = x2 + w2 / 2
        #         cy2 = y2 + h2 / 2

        #         dx = abs(cx1 - cx2)
        #         dy = abs(cy1 - cy2)
        #         avg_w = (w1 + w2) / 2
        #         avg_h = (h1 + h2) / 2

        #         if connectivity == 4:
        #             adjacent = (dx <= avg_w and dy <= 1e-6) or (dy <= avg_h and dx <= 1e-6)
        #         else:  # 8-connectivity
        #             adjacent = (dx <= avg_w) and (dy <= avg_h)

        #         if adjacent:
        #             self.neighbors[i].append(j)
        #             self.neighbors[j].append(i)

        quadtree_time = time.time() - st_time
        self.all_output["decomposition_time"] = quadtree_time



    def build_regions(self, decomposition="voronoi", elevation_abstraction_method="plane", elevation_bins=10, min_area=1, max_planes=100, error_thresh=10.0, flatness_ratio=0.7):
        # Placeholder for quadtree region building logic
        # st_time = time.time()
        # self.quadtree = self.quadtree_decompose_full_coverage(self.landcover_map, min_size=min_area)


        
        # patches = self.get_patch_descriptors_with_elevation(self.landcover_map, self.elevation_map, self.quadtree)
        # self.set_patches(patches)

        # quadtree_time = time.time() - st_time
        # self.all_output["decomposition_time"] = quadtree_time

        self.decompose(min_area=min_area)

        st_time= time.time()


        elevation_abstractor = ElevationAbstractor(abstraction_method=elevation_abstraction_method, error_thresh=error_thresh, max_planes=max_planes)

        patches = self.get_patches()

        complete_points = []
        


        def points_inside_rectangle(x, y, w, h):
            xs = np.arange(x, x + w + 1)
            ys = np.arange(y, y + h + 1)
            grid = np.array([[xi, yi] for xi in xs for yi in ys])
            return grid

        for i, (x, y, w, h, label, avg_elevation) in tqdm(enumerate(patches), total=len(patches)):
            # Get the polygon points for the quadtree region

            polygon_pts = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]])

            x, y, w, h = map(int, (x, y, w, h))
            # fit the plane to the quadtree region
            points = np.column_stack((self.xv[y:y+h, x:x+w].ravel(), self.yv[y:y+h, x:x+w].ravel(), self.elevation_map[y:y+h, x:x+w].ravel()))
            landcover_region = self.landcover_map[y:y+h, x:x+w].ravel()

            method_used, plane_regions = elevation_abstractor.build_abstraction(points, landcover_array=landcover_region)

            polygon_pts = points[:, :2]

            all_pts_inside = points_inside_rectangle(x, y, w-1, h-1)
            complete_points.extend(all_pts_inside)
            # print("all_pts_inside", all_pts_inside)
            # print("points", points)
            # print("plane_regions", plane_regions)

            centeroid = (x + (w / 2), y + (h / 2))
            # if len(points)>1:
            #     centeroid = (x + (w / 2), y + (h / 2))
            # else:
            #     centeroid = (x, y)

            if method_used=="surface":
                region_abstraction = SurfaceRegion(plane_regions, centeroid, polygon_pts)
            else:
                region_abstraction = PlaneRegion(plane_regions, centeroid, polygon_pts)

            region_abstraction.attributes = self.get_region_attributes(i, points=points, landcover_region=landcover_region)

            self.regions[i]=region_abstraction

            # print(region_abstraction)

        # check if all the points are covered

        complete_points = np.array(complete_points)
        complete_points = np.unique(complete_points, axis=0)
        # points in elevation map
        points_actual = np.column_stack((self.xv.ravel(), self.yv.ravel()))
        

        def points_not_in(list_a, list_b):
            a = np.asarray(list_a)
            b_set = set(map(tuple, list_b))
            return [pt for pt in a if tuple(pt) not in b_set]
        all_points = points_not_in(complete_points, points_actual)
        # print("All points in quadtree: ", len(all_points))
        # print(all_points)
        

        self.all_output["elevation_abstraction_time"] = time.time() - st_time




    def is_homogeneous(self, region):
        return np.all(region == region[0, 0])

    def quadtree_decompose_full_coverage(self, landcover_map, min_size=1):
        h, w = landcover_map.shape
        tree = []

        def split(x, y, w_, h_):
            if x >= w or y >= h or w_ <= 0 or h_ <= 0:
                return
            region = landcover_map[y:y+h_, x:x+w_]
            if region.size == 0:
                return
            if self.is_homogeneous(region) or (w_ <= min_size and h_ <= min_size):
                tree.append((x, y, w_, h_, region[0, 0]))
            else:
                w_half = w_ // 2
                h_half = h_ // 2
                # split into 4 subregions
                split(x, y, w_half, h_half)
                split(x + w_half, y, w_ - w_half, h_half)
                split(x, y + h_half, w_half, h_ - h_half)
                split(x + w_half, y + h_half, w_ - w_half, h_ - h_half)

        split(0, 0, w, h)
        return tree

    def get_patch_descriptors_with_elevation(self, landcover_map, elevation_map, quadtree_regions):
        patches = []
        polygons = []
        for x, y, w, h, label in quadtree_regions:
            lc_region = landcover_map[y:y+h, x:x+w]
            el_region = elevation_map[y:y+h, x:x+w]
            
            center_x = x + w / 2
            center_y = y + h / 2
            
            # Dominant landcover label
            if np.all(lc_region == lc_region[0, 0]):
                dominant_label = label
            else:
                unique, counts = np.unique(lc_region, return_counts=True)
                dominant_label = unique[np.argmax(counts)]
            
            # Average elevation
            avg_elevation = np.mean(el_region)
            
            patches.append((x, y, w, h, dominant_label, avg_elevation))
            polygon = Polygon([(x, y), (x + w, y), (x + w, y + h), (x, y + h)])
            polygons.append(polygon)
        self.polygons = polygons
        return patches




    def compute_coverage(self, tree, landcover):
        # Check coverage
        covered_pixels_full = np.zeros_like(landcover, dtype=bool)
        for x, y, w_, h_, label in tree:
            covered_pixels_full[y:y+h_, x:x+w_] = True

        coverage_full = 100.0 * covered_pixels_full.sum() / (landcover.shape[0] * landcover.shape[1])
        return coverage_full



    def get_quadtree_data(self, region_idx):
        """
        Get data for a specific quadtree region index.
        Returns the 3D points, landcover values, and region mask.
        """
        patches = self.get_patches()
        try:
            x, y, w, h, label, avg_elevation = patches[region_idx]
            x = int(round(x))
            y = int(round(y))
            w = int(round(w))
            h = int(round(h))

        except IndexError:
            return None, None, None

        if w == 0 or h == 0:
            return None, None, None

        mask = np.zeros(self.elevation_map.shape, dtype=bool)
        mask[y:y+h, x:x+w] = True

        x_region = self.xv[mask]
        y_region = self.yv[mask]
        z_region = self.elevation_map[mask]
        landcover_region = self.landcover_map[mask]

        if np.any(np.isnan(z_region)) or np.any(np.isnan(landcover_region)):
            return None, None, mask

        points = np.column_stack((x_region, y_region, z_region))
        return points, landcover_region, mask


    def reconstruction_evaluation(self, debug = False):

        st_time= time.time()
        elevation_data = self.elevation_map
        landcover_data = self.landcover_map
        
        elev_rmse = []
        elev_std = []
        elev_max = []

        plane_counts = []
        region_indices = []

        elevation_fitted = np.full(elevation_data.shape, np.nan)

        landcover_data = landcover_data.astype(np.int32)
        landcover_fitted = np.full(landcover_data.shape, -1).astype(np.int32)

        total_memory_original = []
        total_memory_planes = []
        em = EvaluationMetrics()

        patches = self.get_patches()
        for i in tqdm(range(len(patches))):

            points, landcover_label, mask = self.get_quadtree_data(i)

            abst_region = self.regions[i]

            x_region, y_region, z_region = points[:, 0], points[:, 1], points[:, 2]

            points = np.column_stack((x_region, y_region, z_region))

            # print("points", points)
            reconstructed_z, landcover_reconstructed = abst_region.reconstruct_region(x_region, y_region)

            
            metrics = abst_region.evaluate_region(points)
            
            memory_points_MB = metrics["memory_points_MB"]
            memory_planes_MB = metrics["memory_plane_regions_MB"]
            # coverage = metrics["coverage"]
            total_memory_original.append(memory_points_MB)
            total_memory_planes.append(memory_planes_MB)


            plane_count = abst_region.get_plane_count()


            elevation_fitted[mask] = reconstructed_z
            landcover_fitted[mask] = landcover_reconstructed


            # elevation_fitted[y:y+h, x:x+w] = reconstructed_z.reshape((h, w))
            # landcover_fitted[y:y+h, x:x+w] = landcover_reconstructed.reshape((h, w))

            rmse, var, max_res = em.elevation_info_loss(z_region, reconstructed_z)
        
            elev_rmse.append(rmse)
            elev_std.append(np.sqrt(var))
            elev_max.append(max_res)
            plane_counts.append(plane_count)
            region_indices.append(i)


        # print("Plane fitting took: ", time.time() - st_time)

        loss_df = pd.DataFrame(
            { "Region Index": region_indices,
            "Elevation RMSE": elev_rmse,
            "Elevation (Std Dev)": elev_std,
            "Elevation Max Residual": elev_max,
            "Plane Count": plane_counts,

        })

        values1 = loss_df["Plane Count"].dropna()

        # elevation statistics
        values2 = loss_df["Elevation RMSE"].dropna()
        values3 = loss_df["Elevation (Std Dev)"].dropna()
        values4 = loss_df["Elevation Max Residual"].dropna()

        # memory statistics
        values5 = np.array(total_memory_original)
        original_memory = values5.sum()

        values6 = np.array(total_memory_planes)
        plane_memory = values6.sum()

        
        # classification error 
        result = em.evaluate_landcover_classification(landcover_data.flatten(), landcover_fitted.flatten())

        if debug:
            print(f"Plane Count: min={values1.min()}, max={values1.max()}, mean={values1.mean():.2f}")

            print(f"Elevation RMSE: min={values2.min():.4f}, max={values2.max():.4f}, mean={values2.mean():.4f}")
            print(f"Elevation Std Dev: min={values3.min():.4f}, max={values3.max():.4f}, mean={values3.mean():.4f}")
            print(f"Elevation Max Residual: min={values4.min():.4f}, max={values4.max():.4f}, mean={values4.mean():.4f}")
            print(f"Original Memory: min={values5.min():.4f} MB, max={values5.max():.4f} MB, mean={values5.mean():.4f} MB, total={values5.sum():.4f} MB")
            print(f"Plane Memory: min={values6.min():.4f} MB, max={values6.max():.4f} MB, mean={values6.mean():.4f} MB, total={values6.sum():.4f} MB")
            print(f"Landcover Classification Accuracy: {result['accuracy']:.4f}")

        all_output = {
            "loss_df": loss_df,
            "elevation_fitted": elevation_fitted,
            "landcover_fitted": landcover_fitted,
            "original_memory": original_memory,
            "plane_memory": plane_memory,
            "landcover_accuracy": result["accuracy"],
        }
        for key, value in all_output.items():
            self.all_output[key] = value

        return all_output





import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d

def plot_voronoi_comparison(voronoi_list, titles, landcover_map, outfile="results/voronoi_comparison.png"):
    """
    Plot multiple Voronoi diagrams over a landcover map.
    
    Args:
        voronoi_list: list of Voronoi objects
        titles: list of strings for subplot titles
        landcover_map: 2D array (H, W)
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    for ax, vor, title in zip(axes, voronoi_list, titles):
        ax.imshow(landcover_map, cmap='tab20b', alpha=0.3)
        voronoi_plot_2d(vor, ax=ax, alpha=0.3, show_vertices=False, show_points=False)
        ax.set_title(title)
        ax.set_xlim(0, landcover_map.shape[1])
        ax.set_ylim(landcover_map.shape[0], 0)
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(outfile, dpi=300)








# nalcms_color_pallet = [
#       '033e00',  // 1  Temperate or sub-polar needleleaf forest
#       '939b71',  // 2  Sub-polar taiga needleleaf forest
#       '196d12',  // 3  Tropical or sub-tropical broadleaf evergreen forest
#       '1fab01',  // 4  Tropical or sub-tropical broadleaf deciduous forest
#       '5b725c',  // 5  Temperate or sub-polar broadleaf deciduous forest
#       '6b7d2c',  // 6  Mixed forest
#       'b29d29',  // 7  Tropical or sub-tropical shrubland
#       'b48833',  // 8  Temperate or sub-polar shrubland
#       'e9da5d',  // 9  Tropical or sub-tropical grassland
#       'e0cd88',  // 10  Temperate or sub-polar grassland
#       'a07451',  // 11  Sub-polar or polar shrubland-lichen-moss
#       'bad292',  // 12  Sub-polar or polar grassland-lichen-moss
#       '3f8970',  // 13  Sub-polar or polar barren-lichen-moss
#       '6ca289',  // 14  Wetland
#       'e6ad6a',  // 15  Cropland
#       'a9abae',  // 16  Barren land
#       'db2126',  // 17  Urban and built-up
#       '4c73a1',  // 18  Water
#       'fff7fe',  // 19  Snow and ice
#     ]



import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Color and label dictionaries
nalcms_palette = {
    1:  '#033e00', 2:  '#939b71', 3:  '#196d12', 4:  '#1fab01', 5:  '#5b725c',
    6:  '#6b7d2c', 7:  '#b29d29', 8:  '#b48833', 9:  '#e9da5d', 10: '#e0cd88',
    11: '#a07451', 12: '#bad292', 13: '#3f8970', 14: '#6ca289', 15: '#e6ad6a',
    16: '#a9abae', 17: '#db2126', 18: '#4c73a1', 19: '#fff7fe',
}
nalcms_classes = {
    1:  'Temperate or sub-polar needleleaf forest',
    2:  'Sub-polar taiga needleleaf forest',
    3:  'Tropical or sub-tropical broadleaf evergreen forest',
    4:  'Tropical or sub-tropical broadleaf deciduous forest',
    5:  'Temperate or sub-polar broadleaf deciduous forest',
    6:  'Mixed forest',
    7:  'Tropical or sub-tropical shrubland',
    8:  'Temperate or sub-polar shrubland',
    9:  'Tropical or sub-tropical grassland',
    10: 'Temperate or sub-polar grassland',
    11: 'Sub-polar or polar shrubland-lichen-moss',
    12: 'Sub-polar or polar grassland-lichen-moss',
    13: 'Sub-polar or polar barren-lichen-moss',
    14: 'Wetland',
    15: 'Cropland',
    16: 'Barren land',
    17: 'Urban and built-up',
    18: 'Water',
    19: 'Snow and ice',
}

# Build colormap and normalization
sorted_keys = sorted(nalcms_palette.keys())
cmap_list = [nalcms_palette[k] for k in sorted_keys]
nalcms_cmap = mcolors.ListedColormap(cmap_list)


def plot_landcover_reconstruction(list_reconstruced_landcover, titles, output_file="results/landcover_reconstruction.png"):
    """
    Plot multiple landcover reconstructions side by side.
    
    Args:
        list_reconstruced_landcover: list of 2D arrays (H, W)
        titles: list of strings for subplot titles
    """
    fig, axes = plt.subplots(1, len(list_reconstruced_landcover), figsize=(18, 6))
    
    for ax, lc_map, title in zip(axes, list_reconstruced_landcover, titles):
        ax.imshow(lc_map, cmap=nalcms_cmap)
        ax.set_title(title)
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)



import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

def plot_elevation_reconstruction(elevation_data, list_elevation_fitted, titles, output_file="results/elevation_reconstruction.png"):
    all_errors = [np.abs(elevation_data - fit) for fit in list_elevation_fitted]

    # Compute global vmin, vmax
    vmax = 0
    vmin = 1e6
    for error in all_errors:
        error_fixed = np.nan_to_num(error, nan=0)
        vmax = max(vmax, np.max(error_fixed))
        vmin = min(vmin, np.min(error_fixed))

    fig, axes = plt.subplots(1, len(list_elevation_fitted), figsize=(18, 6), layout='constrained')
    for ax, elevation_error, title in zip(axes, all_errors, titles):
        # elevation_error = np.abs(elevation_data - elevatio_fitted)

        img = ax.imshow(elevation_error, cmap=plt.cm.inferno, vmin=vmin, vmax=vmax)
        ax.set_title(title)
        ax.axis('off')   

    fig.colorbar(img, ax=axes, shrink=0.55, pad=0.005, location='right', aspect=30, label="Elevation Error")
    plt.savefig(output_file, dpi=300)
    plt.close()



import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

def plot_combined_reconstruction(    
    elevation_data,
    list_elevation_fitted,
    list_reconstructed_landcover,
    elevation_titles,
    landcover_titles,
    output_file="results/combined_reconstruction.png"
):
    """
    Plot elevation errors (with colorbar) and landcover reconstructions in 2-row layout.
    """
    n = len(list_elevation_fitted)
    assert len(list_reconstructed_landcover) == n
    assert len(elevation_titles) == n
    assert len(landcover_titles) == n

    # Compute elevation errors and global vmin/vmax
    all_errors = [np.abs(elevation_data - fit) for fit in list_elevation_fitted]
    vmax = max(np.nanmax(np.nan_to_num(err)) for err in all_errors)
    vmin = min(np.nanmin(np.nan_to_num(err)) for err in all_errors)

    # fig = plt.figure(figsize=(4 * n + 1.5, 9), layout='constrained')

    fig, axes = plt.subplots(2, n , figsize=(4 * n , 6), layout='constrained')
    # gs = gridspec.GridSpec(2, n + 1, width_ratios=[1]*n + [0.05], height_ratios=[1, 1], wspace=0.1, hspace=0.15)

    # Top row: elevation error maps
    img = None
    for i in range(n):
        ax = axes[0, i]
        img = ax.imshow(all_errors[i], cmap='inferno', vmin=vmin, vmax=vmax)
        ax.set_title(elevation_titles[i])
        ax.axis('off')

    # Bottom row: landcover maps
    for i in range(n):
        ax = axes[1, i]
        ax.imshow(list_reconstructed_landcover[i], cmap=nalcms_cmap)
        ax.set_title(landcover_titles[i])
        ax.axis('off')

    # Shared colorbar for elevation error only
    cbar = fig.colorbar(img, ax=axes[0, :], shrink=0.9, pad=0.005, location='right', aspect=30)
    cbar.set_label("Elevation Error")

    plt.savefig(output_file, dpi=300)
    plt.close()



import rasterio

# ---------------- Utility Functions ---------------- #
def pixel_to_coords(xs, ys, transform):
    coords = [transform * (x, y) for x, y in zip(xs, ys)]
    return np.array(coords)


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



def load_data_region_count_data(example, region_count):


    # ---------------- Load Data ---------------- #
    if example == 1:
        # Load example 1 data
        elevation_data, elev_transform, landcover_data, lc_transform, example_name = example_1()
    elif example == 2:
        elevation_data, elev_transform, landcover_data, lc_transform, example_name = example_2()
    elif example == 3:
        elevation_data, elev_transform, landcover_data, lc_transform, example_name = example_3()
    elif example == 4:
        elevation_data, elev_transform, landcover_data, lc_transform, example_name = example_4()
    else:
        raise ValueError("Invalid example number. Choose 1 or 2.")

    return elevation_data, elev_transform, landcover_data, lc_transform, example_name


def load_data_region_count(example, region_count):

    elevation_data, elev_transform, landcover_data, lc_transform, example_name = load_data_region_count_data(example, region_count)

    # ---------------- Execute Pipeline ---------------- #

    # print("Voronoi generation took: ", time.time() - st_time_total)
    # rb = RegionBuilderPatches(landcover_data, elevation_data, elev_transform, example_name, region_count=region_count)
    rb = RegionBuilder(landcover_data, elevation_data, elev_transform, example_name, region_count=region_count)
    return rb






def plot_reconstruction_with_decomposition(original_landcover, reconstructed_landcover,
                                           original_elevation, reconstructed_elevation,
                                           landcover_error, elevation_rmse,
                                           quadtree_regions):
    """
    Plot decomposition, reconstruction, and error maps for landcover and elevation.
    """
    landcover_diff = (original_landcover != reconstructed_landcover)
    elevation_diff = np.abs(original_elevation - reconstructed_elevation)

    fig, axs = plt.subplots(2, 3, figsize=(24, 12))

    # (0,0) Original Landcover
    axs[0, 0].imshow(original_landcover, cmap='tab20', interpolation='none')
    axs[0, 0].set_title('Original Landcover')
    axs[0, 0].axis('off')

    for x, y, w, h, label, elev in quadtree_regions:
        rect = plt.Rectangle((x, y), w, h, edgecolor='blue', facecolor='none', linewidth=0.5)
        axs[0, 0].add_patch(rect)

    # (0,1) Reconstructed Landcover
    axs[0, 1].imshow(reconstructed_landcover, cmap='tab20', interpolation='none')
    axs[0, 1].set_title('Reconstructed Landcover')
    axs[0, 1].axis('off')

    # (0,2) Landcover Error
    axs[0, 2].imshow(landcover_diff, cmap='Reds', interpolation='none')
    axs[0, 2].set_title(f'Landcover Error Map\n(Error = {landcover_error:.2f}%)')
    axs[0, 2].axis('off')


    # (1,0) Original Elevation
    axs[1, 0].imshow(original_elevation, cmap='terrain')

    for x, y, w, h, label, elev in quadtree_regions:
        rect = plt.Rectangle((x, y), w, h, edgecolor='blue', facecolor='none', linewidth=0.5)
        axs[1, 0].add_patch(rect)


    axs[1, 0].set_title('Original Elevation')
    axs[1, 0].axis('off')

    # (1,1) Reconstructed Elevation
    axs[1, 1].imshow(reconstructed_elevation, cmap='terrain')
    axs[1, 1].set_title('Reconstructed Elevation')
    axs[1, 1].axis('off')

    # (1,2) Elevation Error
    elev_img = axs[1, 2].imshow(elevation_diff, cmap='plasma')
    axs[1, 2].set_title(f'Elevation Error Map\n(RMSE = {elevation_rmse:.2f} m)')
    axs[1, 2].axis('off')
    fig.colorbar(elev_img, ax=axs[1, 2], fraction=0.046, pad=0.04)

    # (1,3) Empty or could show another analysis
    # axs[1, 3].axis('off')

    plt.tight_layout()
    plt.savefig('reconstruction_comparison.png', dpi=300)


import argparse
import pickle
import time
import os
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Evaluate landcover reconstruction and classification.")
    parser.add_argument("--example", type=int, default=1, help="Example number (1 or 2)")
    parser.add_argument("--region_count", type=int, default=1000, help="Step size for Voronoi generation")
    parser.add_argument("--d", type=str, default="voronoi", help="Decomposition method options: voronoi, grid, hex")
    parser.add_argument("--compute", action="store_true", help="Compute results instead of loading")
    parser.add_argument("--method", type=str, default="plane", help="Method for elevation abstraction: surface or plane")


    args = parser.parse_args()
    example = args.example
    decomposition = args.d
    region_count = args.region_count
    method = args.method

    compute_results = args.compute

    region_count_to_min_size = {
        60000: 1,
        30000: 1,
        20000: 2,
        10000: 4,
        5000: 8,
        1000: 16,
    }
    if example == 1:
        region_count_list = [30000, 20000, 10000, 5000, 1000] #[60000, 
    elif example == 2:
        region_count_list = [60000, 30000, 20000, 10000, 5000, 1000]
    else:
        raise ValueError("Invalid example number. Choose 1 or 2.")
    

    region_count_list = [1,2,4,8,16]

    print("args.compute_results", compute_results)
    st_time = time.time()

    decomposition_list = ["quadtree", "voronoi", "grid", "hex"]

    base_dir = f"./data/results/evaluated_{method}"
    if compute_results:

        if not os.path.exists(base_dir):
            os.makedirs(base_dir, exist_ok=True)
        print(f"Base directory: {base_dir}")
        print("Computing results...")

        for min_region_size in region_count_list:
            region_count = 31441 #min_region_size
            for decomposition in decomposition_list:

                # Save the output to a file
                output_file = f"{base_dir}/output_ex{example}_r{min_region_size}_{decomposition}.pkl"

                if os.path.exists(output_file):
                    print(f"Output file {output_file} already exists. Skipping computation.")
                    continue

                rb = load_data_region_count(example, region_count)


                if decomposition == "quadtree":
                
                    rb = RegionBuilderQuadtree(rb.landcover_map, rb.elevation_map, rb.transform, rb.map_name, region_count=region_count)
                    rb.build_regions(elevation_abstraction_method=method, min_area=min_region_size)

                    region_count = len(rb.regions)
                    print(f"Quadtree: {region_count} regions")

                else:
                    rb.build_regions(decomposition=decomposition, elevation_abstraction_method=method, elevation_bins=5)

                all_output = rb.reconstruction_evaluation(debug = True)


                with open(output_file, "wb") as f:
                    pickle.dump(rb, f)
                print(f"Output saved to {output_file}")


    else:
        print("Loading results...")
        for region_count_i in region_count_list:


            
            voronoi_list = []
            titles = []
            decomposition_map = {
                "quadtree": "Quadtree",
                "voronoi": "Boundary",
                "grid": "Grid",
                "hex": "Hexagonal"
            }

            reconstructed_landcover_list = []
            titles_landcover = []
            reconstructed_elevation_list = []
            titles_elevation = []
            for decomposition in decomposition_list:

                # Save the output to a file
                output_file = f"{base_dir}/output_ex{example}_r{region_count_i}_{decomposition}.pkl"

                if not os.path.exists(output_file):
                    print(f"Output file {output_file} does not exist. Skipping.")
                    continue


                with open(output_file, "rb") as f:
                    rb = pickle.load(f)

                all_output = rb.all_output
                # print(f"Loaded output from {output_file}")
                loss_df = all_output["loss_df"]
                elevation_fitted = all_output["elevation_fitted"]
                landcover_fitted = all_output["landcover_fitted"]
                

                original_memory = all_output["original_memory"]
                plane_memory = all_output["plane_memory"]


                landcover_data = rb.landcover_map
                elevation_data = rb.elevation_map
                vor = rb.voronoi
                # classification error 
                # print("all_output", all_output.keys())

                landcover_accuracy = all_output["landcover_accuracy"]

                em = EvaluationMetrics()
                mean_iou_landcover = em.mean_iou(landcover_data, landcover_fitted)

                voronoi_list.append(vor)

                memory_string = f"Memory: {original_memory:.4f} MB -> {plane_memory:.4f} MB"

                region_count = loss_df["Region Index"].size
                curr_title = f"Region {region_count} - Voronoi: {decomposition_map[decomposition]}, {memory_string}"

                # print(f"Voronoi: {decomposition_map[decomposition]} - Region {region_count} - Class Accuracy {landcover_accuracy:.4f}")
                print(f"Voronoi: {decomposition_map[decomposition]} - Region {region_count} - Mean IoU {mean_iou_landcover:.4f}")

                titles.append(curr_title)

                reconstructed_landcover_list.append(landcover_fitted)

                # titles_landcover.append(f"{decomposition_map[decomposition]} - Region {region_count}: Class Accuracy {landcover_accuracy:.4f}")
                titles_landcover.append(f"{decomposition_map[decomposition]} - Region {region_count}: Mean IoU {mean_iou_landcover:.4f}")



                reconstructed_elevation_list.append(elevation_fitted)

                titles_elevation.append(f"{decomposition_map[decomposition]} - Region {region_count}: RMSE {loss_df['Elevation RMSE'].mean():.4f}")

            # plot_voronoi_comparison(voronoi_list, titles, landcover_data, outfile=f"./results_new/voronoi_comparison_ex{example}_r{region_count_i}.png")

            # plot_landcover_reconstruction(reconstructed_landcover_list, titles_landcover, output_file=f"./results_new/landcover_reconstruction_ex{example}_r{region_count_i}.png")

            # plot_elevation_reconstruction(elevation_data, reconstructed_elevation_list, titles_elevation, output_file=f"./results_new/elevation_reconstruction_ex{example}_r{region_count_i}.png")
            
            
            plot_combined_reconstruction(elevation_data, reconstructed_elevation_list, reconstructed_landcover_list, titles_elevation,
                                  titles_landcover, output_file=f"./results_new/reconstruction_ex{example}_r{region_count_i}.png")


        print(f"Total time : {time.time() - st_time:.2f} seconds")
