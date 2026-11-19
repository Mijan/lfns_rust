# Birth-death example

The birth-death model of the LF-NS paper (appendix S7.1) as a PEtab problem:
one mRNA species produced at rate `k` and degraded at rate `gamma_mRNA` per
molecule, observed with additive normal noise of standard deviation 2.

- `birth_death.ant`: the model in Antimony; the source of truth.
- `birth_death.xml`: the SBML generated from it (see below).
- `problem.yaml` and the four `.tsv` tables: the PEtab problem. `k` is
  estimated with a log-uniform prior on [0.631, 2.5119]; `gamma_mRNA` is
  fixed at 0.1.
- `measurements.tsv`: the single trajectory of 21 measurements used in the
  paper (`bd_data.txt` of the original C++ example).

Regenerate the SBML after editing the Antimony file with

``` bash
uv run --with antimony python -c "
import antimony
assert antimony.loadAntimonyFile('birth_death.ant') >= 0, antimony.getLastError()
open('birth_death.xml', 'w').write(antimony.getSBMLString(antimony.getMainModuleName()))"
```

`gamma` is a reserved word in Antimony, hence the name `gamma_mRNA`.
