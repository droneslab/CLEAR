#!/usr/bin/env python3
"""Run the paper's raw CLEAR graph paths and report their statistics."""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np

from paper_config import MAP_SCALE_M_PER_PIXEL, PAPER_MAPS


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "clear_core"))

import pathplanning  # noqa: E402,F401 (registers canonical classes for pickle)


def point(value: str) -> tuple[int, int]:
    try:
        x, y = value.split(",")
        return int(x), int(y)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected coordinates as x,y") from exc


def query_selection(value: str) -> str | int:
    if value == "all":
        return value
    try:
        index = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected 'all' or an index from 0 to 19") from exc
    if not 0 <= index < 20:
        raise argparse.ArgumentTypeError("query index must be from 0 to 19")
    return index


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map", default="wharton", choices=PAPER_MAPS, help="Paper map profile.")
    parser.add_argument("--query", type=query_selection, default="all", help="Paper query index (0-19); default: all 20.")
    parser.add_argument("--abstraction", type=Path, help="Default: cache/<map>_clear.pkl.")
    parser.add_argument("--start", type=point, help="Override the paper query start.")
    parser.add_argument("--goal", type=point, help="Override the paper query goal.")
    parser.add_argument("--output", type=Path, help="Default: results/<map>_clear_query_N.npy.")
    parser.add_argument("--scale-xy", type=float, default=MAP_SCALE_M_PER_PIXEL)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    profile = PAPER_MAPS[args.map]
    abstraction = args.abstraction or HERE / "cache" / f"{args.map}_clear.pkl"
    with abstraction.open("rb") as handle:
        graph = pickle.load(handle)
    model = graph.model
    query_indices = range(20) if args.query == "all" else (args.query,)
    records = []
    for index in query_indices:
        paper_start, paper_goal = profile.queries()[index]
        start = args.start or paper_start
        goal = args.goal or paper_goal
        if args.output and len(query_indices) == 1:
            output = args.output
        else:
            output = HERE / "results" / f"{args.map}_clear_query_{index}.npy"

        started = time.perf_counter()
        path, region_path = graph.compute_nx_path(start, goal)
        planning_time = time.perf_counter() - started
        path_array = np.asarray(path, dtype=float)
        length = float(np.linalg.norm(np.diff(path_array[:, :2], axis=0), axis=1).sum() * args.scale_xy)
        output.parent.mkdir(parents=True, exist_ok=True)
        np.save(output, path_array)
        records.append({
            "solved": True, "map": args.map, "method": "CLEAR", "query": index,
            "start": start, "goal": goal, "path_nodes": len(path_array),
            "region_nodes": len(region_path),
            "cost": float(model.compute_path_cost(path, use_step_cost=True)),
            "cost_scaled_x1e4": float(model.compute_path_cost(path, use_step_cost=True)) / 1e4,
            "path_length_m": length, "planning_time_s": planning_time,
            "path_file": str(output.resolve()),
        })
    summary_path = HERE / "results" / f"{args.map}_clear_summary.json"
    summary_path.write_text(json.dumps(records, indent=2), encoding="utf-8")
    if len(records) == 1:
        report = records[0]
    else:
        mean_cost = float(np.mean([r["cost_scaled_x1e4"] for r in records]))
        mean_length = float(np.mean([r["path_length_m"] for r in records]))
        cost_sample_std = float(np.std([r["cost_scaled_x1e4"] for r in records], ddof=1))
        length_sample_std = float(np.std([r["path_length_m"] for r in records], ddof=1))
        report = {
            "map": args.map, "method": "CLEAR", "queries": len(records),
            "mean_cost_scaled_x1e4": mean_cost,
            "mean_path_length_m": mean_length,
            "cost_sample_std_scaled_x1e4": cost_sample_std,
            "path_length_sample_std_m": length_sample_std,
            "mean_planning_time_s": float(np.mean([r["planning_time_s"] for r in records])),
            "summary_file": str(summary_path.resolve()),
        }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
