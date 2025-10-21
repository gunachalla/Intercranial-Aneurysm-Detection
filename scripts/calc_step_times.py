#!/usr/bin/env python3

import argparse
import re
import statistics
from collections.abc import Callable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute per-step inference time statistics from a log file."
    )
    parser.add_argument(
        "log_path",
        help="Path to the inference log file (e.g., /workspace/log_inference.txt).",
    )
    return parser.parse_args()


def extract_step_times(log_path: str) -> list[dict[int, float]]:
    pattern = re.compile(r"DEBUG: \[DEBUG\]\[STEP(?P<step>\d)[^\]]*\] (?P<sec>\d+(?:\.\d+)?)s")
    cases: list[dict[int, float]] = []
    current: dict[int, float] = {}
    last_step = 0

    with open(log_path, encoding="utf-8") as fp:
        for line in fp:
            match = pattern.search(line)
            if not match:
                continue

            step = int(match.group("step"))
            sec = float(match.group("sec"))

            if current and step <= last_step:
                cases.append(current)
                current = {}

            current[step] = sec
            last_step = step

    if current:
        cases.append(current)

    return cases


def compute_stats(values: list[float]) -> tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    if len(values) == 1:
        return values[0], 0.0
    mean_val = statistics.mean(values)
    std_val = statistics.pstdev(values)
    return mean_val, std_val


def main() -> None:
    args = parse_args()
    cases = extract_step_times(args.log_path)
    if len(cases) > 1:
        cases = cases[1:]

    aggregations: dict[str, Callable[[dict[int, float]], float | None]]
    aggregations = {
        "STEP1+2": lambda case: case.get(1, 0.0) + case.get(2, 0.0) if 1 in case or 2 in case else None,
        "STEP3": lambda case: case.get(3),
        "STEP4+5": lambda case: case.get(4, 0.0) + case.get(5, 0.0) if 4 in case or 5 in case else None,
    }

    for label, fn in aggregations.items():
        values = [value for case in cases if (value := fn(case)) is not None]
        mean_val, std_val = compute_stats(values)
        print(
            f"{label}: count={len(values)}, mean={mean_val:.3f}s, std={std_val:.3f}s"
        )


if __name__ == "__main__":
    main()
