# Fed tank example

A stirred tank whose feed rate follows a schedule that differs per run. It
exercises the parts of SBML and PEtab that models with scheduled inputs
need:

- a **rate rule** (`S' = ...`) instead of reactions;
- **assignment rules** (`dilution`, `consumption`), with `max` and `piecewise`;
- **events** that set the input `feed` at fixed times to the interval
  parameters `feed_0`, `feed_1`, `feed_2`;
- a **condition table** that gives each run its own initial substrate and
  its own values of the interval parameters.

Within an interval the equation is linear, `S' = D (S_feed - S) - k_use S`
with `D = feed / tank_volume`, so the solution is known in closed form. The tests
of `lfns-model` check the simulation and the likelihood against it.

`measurements.tsv` holds synthetic data: the closed-form solution at
`k_use = 0.3`, `S_feed = 10` plus normal noise of standard deviation 0.2.
`k_use` and `S_feed` are estimated; the noise parameter `sigma_S` is fixed
and exists only in the PEtab tables.

The SBML is generated from `fed_tank.ant` with the `antimony` package, as
described in `../birth_death/README.md`. Being rule- and event-based, the
model can only be simulated deterministically.
