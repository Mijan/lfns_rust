"""Export a culture-fit campaign of the Cultzyme playground as a PEtab problem.

Run this with the PLAYGROUND's Python environment, not this repository's:

    <playground>/.venv/bin/python tools/export_campaign_petab.py \
        --playground <playground> \
        --campaign <playground>/jm/merck_ambr_transfer/campaigns/<campaign> \
        --out <folder outside any git repository>

It rebuilds the joint multi-run problem exactly as the campaign's fit did
(``ambr.build_joint_problem`` over the training runs of ``holdout_split.json``,
scored on the channels the fit scored) and writes it as PEtab files:
``problem.yaml``, the SBML model with the feed events compiled in, and the
parameter, observable, condition, and measurement tables.

Two options adapt the problem to nested sampling:

- ``--theta <theta.json>`` replaces the nominal values of the parameter table
  by a fit's values (the campaign's own fit is ``iter1/1_fit/theta.json``), so
  parameters that are not estimated are held at the fitted values.
- ``--free NAME ...`` estimates only the named parameters and holds all
  others at their nominal values. Nested sampling over all parameters of a
  large model is expensive; a subset is the way to start.

The written folder contains the measurements of the export. If the export is
confidential, so is the folder: write it next to the data, never into a git
repository, and do not hand it to an AI agent.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SETTINGS_FILE = "fit_settings.json"
SPLIT_FILE = "holdout_split.json"
DEFAULT_DATA = Path("data") / "restricted" / "Cultzyme_ambrdata_v2.csv"
PROBLEM_FILE = "problem.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--playground", type=Path, required=True, help="root of ctz-model-training-playground")
    parser.add_argument("--campaign", type=Path, required=True, help="campaign folder with fit_settings.json")
    parser.add_argument("--out", type=Path, required=True, help="folder to write the PEtab problem into")
    parser.add_argument("--data", type=Path, default=None, help="the export; default: the restricted export")
    parser.add_argument("--runs", nargs="+", default=None, metavar="RUN",
                        help="export these runs instead of the campaign's training runs")
    parser.add_argument("--theta", type=Path, default=None, metavar="THETA_JSON",
                        help="replace the nominal parameter values by this fit's values")
    parser.add_argument("--free", nargs="+", default=None, metavar="NAME",
                        help="estimate only these parameters; hold the others at their nominal values")
    return parser.parse_args()


def training_runs(campaign: Path, override: list[str] | None) -> list[str]:
    """The runs to export: the named ones, or the campaign's training runs."""
    if override:
        return list(override)
    split = json.loads((campaign / SPLIT_FILE).read_text())
    return list(split["training"])


def scored_channels(model, unscored: list[str]) -> list[str]:
    """The model's observables the fit scored."""
    return sorted(set(model.spec.observable_of_column.values()) - set(unscored))


def with_fitted_nominals(parameter_table, theta_file: Path):
    """The parameter table with nominal values replaced by a fit's values.

    Raises:
        SystemExit: If the file names a parameter the table does not have.
    """
    values = {name: float(value) for name, value in json.loads(theta_file.read_text())["parameters"].items()}
    unknown = sorted(set(values) - set(parameter_table.index))
    if unknown:
        raise SystemExit(f"{theta_file} names {unknown}, which this model's parameter table does not have. "
                         f"Use a theta.json of this model version.")
    table = parameter_table.copy()
    for name, value in values.items():
        table.loc[name, "nominalValue"] = value
    return table


def with_only_free(parameter_table, free: list[str]):
    """The parameter table with only the named parameters estimated.

    Raises:
        SystemExit: If a name is not in the table.
    """
    unknown = sorted(set(free) - set(parameter_table.index))
    if unknown:
        raise SystemExit(f"--free names {unknown}, which are not parameters of this model. "
                         f"Known: {list(parameter_table.index)}")
    table = parameter_table.copy()
    table["estimate"] = [1 if name in free else 0 for name in table.index]
    return table


def main() -> None:
    args = parse_args()
    sys.path.insert(0, str(args.playground / "jm"))
    import ambr
    from cultzyme.models.petab.scope import PetabProblemScope

    settings = json.loads((args.campaign / SETTINGS_FILE).read_text())
    runs = training_runs(args.campaign, args.runs)
    dataset = ambr.ExperimentDataset.from_file(args.data or args.playground / DEFAULT_DATA)
    model = ambr.load_model(settings["model"], root=args.campaign / "models")

    problem = ambr.build_joint_problem(model, dataset, runs, hold={})
    PetabProblemScope(problem).observe_only(scored_channels(model, settings["unscored_channels"]))

    if args.theta is not None:
        problem.parameter_df = with_fitted_nominals(problem.parameter_df, args.theta)
    if args.free is not None:
        problem.parameter_df = with_only_free(problem.parameter_df, args.free)

    args.out.mkdir(parents=True, exist_ok=True)
    problem.to_files_generic(str(args.out))
    condition_of_run = {f"exp{index + 1}": run for index, run in enumerate(runs)}
    (args.out / "runs.json").write_text(json.dumps(condition_of_run, indent=2))
    n_estimated = int((problem.parameter_df["estimate"] == 1).sum())
    print(f"wrote {args.out / PROBLEM_FILE}: {len(runs)} runs, {len(problem.measurement_df)} measurements, "
          f"{n_estimated} estimated parameters")


if __name__ == "__main__":
    main()
