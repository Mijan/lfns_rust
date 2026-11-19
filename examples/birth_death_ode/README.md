# Birth-death example, deterministic

The same model and data as `../birth_death`, treated as an ordinary
differential equation: `d mRNA / dt = k - gamma_mRNA * mRNA`.

A deterministic model cannot produce the fluctuations of a single cell, so
the standard deviation of the read-out is no longer fixed at the
measurement noise of 2 but estimated as the parameter `sigma`, which then
absorbs the intrinsic fluctuations as well. Three parameters are estimated,
each with a log-uniform prior: `k` on [0.1, 10], `gamma_mRNA` on [0.01, 1],
and `sigma` on [0.5, 20].

`sigma` appears only in the PEtab tables (as the `noiseFormula` of the
observable), not in the SBML model. The model, measurements, and conditions
are read from `../birth_death`.
