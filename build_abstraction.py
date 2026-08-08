#!/usr/bin/env python3
"""Build the standalone CLEAR terrain-abstraction graph used by the paper."""

from __future__ import annotations

import argparse
import pickle
import sys
import time
from pathlib import Path

import numpy as np

from paper_config import (
    DECOMPOSITION_BOUNDARY_EMPHASIS,
    ELEVATION_BINS,
    MAX_PLANES,
    PAPER_MAPS,
    PLANE_RMSE_TOLERANCE_M,
)


HERE = Path(__file__).resolve().parent
CORE = HERE / "clear_core"
sys.path.insert(0, str(CORE))

from cost_models import VehicleObjective  # noqa: E402
from dataloader import load_data_region_count_older  # noqa: E402
from pathplanning import GraphPathPlanningPipeline  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map", default="wharton", choices=PAPER_MAPS, help="Paper map profile.")
    parser.add_argument("--map-file", type=Path, help="Override the profile's maps/<map>.pkl input.")
    parser.add_argument("--region-count", type=int, help="Override the paper CLEAR region budget.")
    parser.add_argument("--output", type=Path, help="Default: cache/<map>_clear.pkl.")
    parser.add_argument("--purpose", default="planning", choices=("planning", "decomposition"),
                        help="Sets alpha_bdy=0 for paper planning or 1 for paper decomposition metrics.")
    parser.add_argument("--clear-flatness-ratio", type=float, help="Explicit alpha_bdy override.")
    parser.add_argument("--min-area", type=int, help="Override the map/method paper setting.")
    parser.add_argument("--rmse-tolerance", type=float, default=PLANE_RMSE_TOLERANCE_M)
    return parser.parse_args()


def load_map(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with path.open("rb") as handle:
        data = pickle.load(handle)
    # Preserve the paper inputs' exact dtypes. Casting Wharton from int16/uint8
    # to float64/platform-int changes ordering among tied samples and therefore
    # changes the Voronoi seeds, retained regions, graph, and planned paths.
    return np.asarray(data["elevation"]), np.asarray(data["landcover"])


def main() -> None:
    args = parse_args()
    profile = PAPER_MAPS[args.map]
    map_file = args.map_file or profile.default_map_file
    if not map_file.exists():
        raise FileNotFoundError(
            f"Missing {profile.label} input: {map_file}\n"
            "Copy the paper map pickle there or pass --map-file explicitly."
        )
    elevation, landcover = load_map(map_file)
    decomposition = "voronoi"
    region_count = args.region_count or profile.region_budget
    min_area = args.min_area or profile.clear_min_area
    alpha_bdy = args.clear_flatness_ratio
    if alpha_bdy is None:
        alpha_bdy = profile.planning_boundary_emphasis if args.purpose == "planning" else DECOMPOSITION_BOUNDARY_EMPHASIS
    output = args.output or HERE / "cache" / f"{args.map}_clear.pkl"

    start = time.perf_counter()
    # The archived paper CLEAR graphs used the legacy RegionBuilder conversion.
    rb = load_data_region_count_older(
        example=profile.example,
        region_count=region_count,
        decomposition=decomposition,
        landcover_data=landcover,
        elevation_data=elevation,
    )
    rb.build_regions(
        decomposition=decomposition,
        elevation_abstraction_method="plane",
        elevation_bins=ELEVATION_BINS,
        min_area=min_area,
        max_planes=MAX_PLANES,
        error_thresh=args.rmse_tolerance,
        flatness_ratio=alpha_bdy,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    planner = GraphPathPlanningPipeline(
        example=profile.example,
        region_count=region_count,
        decomposition=decomposition,
        method="plane",
        cost_model=VehicleObjective,
        base_path=str(output.parent),
        region_count_to_save=region_count,
    )
    planner.rb = rb
    planner._build_model_and_graph(min_area=min_area, flatness_ratio=alpha_bdy)
    with output.open("wb") as handle:
        pickle.dump(planner.convex_rb, handle)
    print(
        f"Built CLEAR abstraction from seed budget {region_count}; "
        f"retained {len(rb.regions)} planning-graph regions in {time.perf_counter() - start:.3f} s"
    )
    print(f"paper profile={args.map}, alpha_bdy={alpha_bdy}, epsilon={args.rmse_tolerance} m, min_area={min_area}")
    print(output.resolve())


if __name__ == "__main__":
    main()
