"""Follow a likelihood-free nested sampling run from its run directory.

    uv run python tools/watch_run.py --run-directory <folder of the run>
    uv run python tools/watch_run.py --run-directory <folder> --problem <folder>/problem.yaml

Reads the checkpoint files the run writes after every iteration
(``config.json``, ``reports.jsonl``, ``live.json``) and prints one line per
finished iteration: the prior volume removed so far, the likelihood
threshold, the best and the median live point, the acceptance rate, the
likelihood evaluations of the iteration, and the evidence estimate. Lines
appear as iterations finish; stop it with Ctrl-C, the run is not affected.

With ``--problem`` the best live point is also written, in parameter
values, to ``best_live.json`` in the run directory each time it improves.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

CONFIG_FILE = "config.json"
REPORTS_FILE = "reports.jsonl"
LIVE_FILE = "live.json"
BEST_LIVE_FILE = "best_live.json"
HEADER = (f"{'time':>8} {'iter':>5} {'-ln X':>7} {'threshold':>12} {'median live':>12} {'best live':>12} "
          f"{'accept':>8} {'evaluations':>11} {'ln Z':>12}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-directory", type=Path, required=True, help="the run directory to follow")
    parser.add_argument("--problem", type=Path, default=None,
                        help="the PEtab YAML file, to write the best live point in parameter values")
    parser.add_argument("--interval", type=float, default=5.0, help="seconds between two looks at the files")
    return parser.parse_args()


def read_reports(run_directory: Path) -> list[dict]:
    """The reports of the iterations recorded so far; a line still being written is left for later."""
    path = run_directory / REPORTS_FILE
    if not path.is_file():
        return []
    reports = []
    for line in path.read_text().splitlines():
        try:
            reports.append(json.loads(line))
        except json.JSONDecodeError:
            break
    return reports


def read_live(run_directory: Path) -> dict | None:
    """The live particles of the last completely recorded iteration."""
    path = run_directory / LIVE_FILE
    return json.loads(path.read_text()) if path.is_file() else None


def depth_per_iteration(config: dict) -> float:
    """Expected prior volume removed by one iteration, in nats: the sum of 1/(N - i) over the r removed points."""
    n_live, n_replaced = config["population"]["n_live"], config["population"]["n_replaced"]
    return sum(1.0 / (n_live - index) for index in range(n_replaced))


def write_best_live(problem, live: dict, run_directory: Path) -> None:
    """Writes the best live point in parameter values."""
    import numpy as np

    best = max(live["particles"], key=lambda particle: particle["log_l"])
    values = problem.to_parameters(np.asarray([best["u"]]))[0]
    record = {"iteration": live["iteration"], "log_likelihood": best["log_l"],
              "parameters": dict(zip(problem.parameter_ids, map(float, values)))}
    (run_directory / BEST_LIVE_FILE).write_text(json.dumps(record, indent=2))


def main() -> None:
    args = parse_args()
    while not (args.run_directory / CONFIG_FILE).is_file():
        print(f"waiting for {args.run_directory / CONFIG_FILE} ...", flush=True)
        time.sleep(args.interval)
    config = json.loads((args.run_directory / CONFIG_FILE).read_text())
    step = depth_per_iteration(config)
    problem = None
    if args.problem is not None:
        import lfns

        problem = lfns.PetabProblem(str(args.problem))
    print(f"{config['population']['n_live']} live points, {config['population']['n_replaced']} replaced per "
          f"iteration ({step:.3f} nats of prior volume each)")
    print(HEADER, flush=True)
    n_printed, best_written = 0, -math.inf
    while True:
        reports, live = read_reports(args.run_directory), read_live(args.run_directory)
        if live is not None and live["iteration"] >= len(reports) > n_printed:
            log_likelihoods = sorted(particle["log_l"] for particle in live["particles"])
            median, best = log_likelihoods[len(log_likelihoods) // 2], log_likelihoods[-1]
            for report in reports[n_printed:]:
                is_latest = report["iteration"] == len(reports)
                print(f"{time.strftime('%H:%M:%S'):>8} {report['iteration']:>5d} {report['iteration'] * step:>7.2f} "
                      f"{report['log_threshold']:>12.1f} "
                      f"{median if is_latest else float('nan'):>12.1f} {best if is_latest else float('nan'):>12.1f} "
                      f"{report['n_accepted'] / report['n_proposals']:>8.4f} {report['n_proposals']:>11d} "
                      f"{report['evidence']['log_evidence']:>12.1f}", flush=True)
            n_printed = len(reports)
            if problem is not None and best > best_written:
                write_best_live(problem, live, args.run_directory)
                best_written = best
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
