"""Simulate a PEtab problem at its nominal parameters with RoadRunner.

A reference for checking this repository's ODE simulation against an
independent simulator. Run it with an environment that has `roadrunner`,
`pandas`, and `pyyaml` (the Cultzyme playground's environment has them):

    python tools/simulate_with_roadrunner.py --problem <folder>/problem.yaml \
        --condition exp1 --end 480 --points 41 --out reference.npz

The output holds `times`, `species_ids`, and `species` (one row per time),
with every species read as formulas read it (a concentration for
concentration species).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import roadrunner
import yaml

RELATIVE_TOLERANCE = 1e-10
ABSOLUTE_TOLERANCE = 1e-12
MAX_STEPS = 2_000_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--problem", type=Path, required=True, help="the PEtab YAML file")
    parser.add_argument("--condition", required=True, help="the condition to simulate")
    parser.add_argument("--end", type=float, required=True, help="last output time")
    parser.add_argument("--points", type=int, required=True, help="number of output times from 0 to --end")
    parser.add_argument("--out", type=Path, required=True, help="the .npz file to write")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    folder = args.problem.parent
    spec = yaml.safe_load(args.problem.read_text())
    files = spec["problems"][0]
    runner = roadrunner.RoadRunner(str(folder / files["sbml_files"][0]))
    model_parameters = set(runner.model.getGlobalParameterIds())
    species_ids = list(runner.model.getFloatingSpeciesIds())

    parameters = pd.read_csv(folder / spec["parameter_file"], sep="\t")
    for name, value in zip(parameters["parameterId"], parameters["nominalValue"]):
        if name in model_parameters:
            runner.setValue(f"init({name})", float(value))
    conditions = pd.read_csv(folder / files["condition_files"][0], sep="\t").set_index("conditionId")
    for name, value in conditions.loc[args.condition].items():
        if name == "conditionName" or pd.isna(value):
            continue
        runner.setValue(f"init([{name}])" if name in species_ids else f"init({name})", float(value))
    # Events that trigger at time zero read the values set above only if the
    # model is re-initialized after setting them.
    runner.reset()

    integrator = runner.integrator
    integrator.relative_tolerance = RELATIVE_TOLERANCE
    integrator.absolute_tolerance = ABSOLUTE_TOLERANCE
    integrator.maximum_num_steps = MAX_STEPS
    runner.timeCourseSelections = ["time", *(f"[{name}]" for name in species_ids)]
    result = np.asarray(runner.simulate(0.0, args.end, args.points))
    np.savez(args.out, times=result[:, 0], species_ids=np.array(species_ids), species=result[:, 1:])
    print(f"wrote {args.out}: {len(species_ids)} species at {args.points} times")


if __name__ == "__main__":
    main()
