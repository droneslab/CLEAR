"""Fixed configurations used for the CLEAR paper planning experiments."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class PaperMap:
    key: str
    label: str
    example: int
    region_budget: int
    clear_min_area: int
    planning_boundary_emphasis: float
    archived_retained_regions: int
    coordinates: tuple[tuple[int, int], ...]

    @property
    def default_map_file(self) -> Path:
        return ROOT / "maps" / f"{self.key}.pkl"

    def queries(self) -> tuple[tuple[tuple[int, int], tuple[int, int]], ...]:
        points = self.coordinates + self.coordinates[-2::-1]
        return tuple(zip(points[:-1], points[1:]))


PAPER_MAPS = {
    "wharton": PaperMap(
        "wharton", "Wharton", 1, 31_440, 2, 0.0, 30_885,
        ((99, 262), (98, 9), (327, 93), (26, 192), (140, 22),
         (99, 260), (304, 106), (92, 11), (63, 202), (318, 100), (59, 200)),
    ),
    "humphreys": PaperMap(
        "humphreys", "Humphreys", 3, 156_004, 4, 0.7, 155_000,
        ((1408, 1319), (898, 99), (188, 113), (1440, 398), (1553, 1202),
         (1354, 239), (1237, 1459), (834, 375), (53, 131), (1630, 1541), (1594, 398)),
    ),
    "rainier": PaperMap(
        "rainier", "Mount Rainier", 4, 218_074, 4, 0.7, 216_835,
        ((2404, 170), (693, 1400), (2255, 2015), (2029, 432), (3090, 1860),
         (2178, 319), (495, 234), (778, 1839), (2305, 92), (608, 2397), (608, 403)),
    ),
}

# Paper-wide fixed settings.
DECOMPOSITION_BOUNDARY_EMPHASIS = 1.0
PLANE_RMSE_TOLERANCE_M = 10.0
ELEVATION_BINS = 5
MAX_PLANES = 10
MAP_SCALE_M_PER_PIXEL = 30.0
GRADE_LIMIT_PERCENT = 35.0
GRADE_BARRIER = 1_000_000.0
