# Building instances

The released data make these steps unnecessary for reproducing the tables.

## Rebuilding the benchmark instance files

The archive does not contain the instance files built from public benchmark files, nor the user messages of the 405 recorded prompts of the data variant on the benchmark instances, which embed a complete benchmark instance; the prompt builder (`llmdw.prompts.propose.build_messages` with the `with_data` template) recreates these messages from the rebuilt instance files, identical to the recorded ones for all 405. `scripts/fetch_benchmarks.sh` downloads the public benchmark files from their original locations (the OR-Library, CommaLAB and suerie.de, with the test list of the GCG repository, and the OR-Library files gap1 to gap12 and binpack1 of the generalized assignment and bin-packing instances) into `data/benchmark_raw/` and verifies every data file against the SHA-256 recorded when the instances were built. `python -m llmdw.benchmark.standard build --raw data/benchmark_raw --out work/benchmark_std` recreates the 219 instance files of the standard formulations, and `python -m llmdw.benchmark.standard check --out work/benchmark_std --recorded data/work/benchmark_std` verifies each against the SHA-256 recorded in its `meta.json` (219 of 219 identical on a fresh clone); `python -m llmdw.benchmark.extra_classes build --raw data/benchmark_raw --out work/no_joint_group` and `python -m llmdw.benchmark.extra_classes check --out work/no_joint_group --recorded data/work/no_joint_group` do the same for the 80 instance files of generalized assignment and bin packing (80 of 80 identical). [DATA.md](../DATA.md) names the sources and their terms.

## Instance collections

The templates (lot sizing, network design, facility location, generalized assignment) and the hidden-group programs are fixed instances with exact gains for every set of families:

```bash
python -m llmdw.instances.export --out data/work/templates
```

The rich models are random draws of three classes (lot sizing, network design, facility location). A draw is accepted when it is feasible, its LP gap is at least two percent, and no single family closes more than half of the best gain among the sets of at most three families; the first eight accepted draws of each class are the instances, with exact gains for every set of at most three families:

```bash
python -m llmdw.instances.rich_models_build --out data/work/rich_models --parallel 8
```

The rich models with 12 to 16 families add standard modeling features on the same variables to the three classes: lot sizing and facility location with 12 and 16 families, network design with 12 and 14. The menu with *k* families is the families of the smaller model followed by the first new families in a fixed order (`llmdw.instances.rich_models_large.NEW_FAMILIES`). The data of draw *i* of a class and *k* come from `random.Random(7000 + 100 k + 10 c + i)`, with *c* the index of the class; a draw is accepted under the rule of the smaller rich models, an infeasible network-design draw is repaired by redrawing the new families of a minimal conflicting family set up to ten times, and the first ten accepted draws of each class and *k* are the 60 instances, with exact gains for every set of at most three families (at most two for network design with 14 families). The part `rich_models_large` of `scripts/rerun.sh` builds them (`python -m llmdw.instances.rich_models_large build` per class and *k*, then `finalize`) and computes their five-dual lookahead:

```bash
scripts/rerun.sh rich_models_large
```

The MIPLIB-NL models are exported by running each MIPLIB-NL script with the optimize call replaced by an MPS export that keeps row and column names; no model is solved, so the size-limited gurobipy license suffices. Families are the row names with digit runs removed, and the paper keeps the 108 models with two to six families:

```bash
git clone https://github.com/optsuite/MIPLIB-NL.git
git -C MIPLIB-NL checkout 9f14ad1782eaa547f7dad982d1edb186321ffd88
python -m llmdw.instances.miplibnl_export --dataset MIPLIB-NL/dataset --out data/work/mps --jobs 4 --timeout 1800
python -m llmdw.gcg.batch oracle    --mps-dir data/work/mps --out data/work/oracle --max-families 6 --time 120 --parallel 8
python -m llmdw.gcg.batch lookahead --mps-dir data/work/mps --oracle data/work/oracle --out data/work/lookahead --block-time 60 --parallel 8
python -m llmdw.scoring.structure --oracle-root data/work/oracle --lookahead-root data/work/lookahead --out data/work/structure
```

The benchmark instances use standard formulations of three classes on public data, with families named by their role:

- lot sizing: the 20 smallest usable instances of Trigeiro, Thomas and McClain (1989) by items times periods, with the families balance, setup linking and capacity with setup times;
- network design: the R instances of Crainic, Frangioni and Gendron (2001) with the families flow balance, arc linking and arc capacity, every instance with a feasible LP and a positive LP gap, up to the 20 smallest by number of variables;
- facility location: the OR-Library capacitated instances cap41 to cap134 (Beasley, 1990) with split demand and the families assignment, capacity and linking, the 19 instances with a positive LP gap.

The proved optima come from SCIP with one thread, seed 0 and a one-hour limit; the ground truth evaluates every set of at most two families with the GCG root-node settings of the census (no presolve, cuts or heuristics, SCIP propagation off, 120 seconds, one thread, seed 0); branch-and-price runs default GCG with one thread, seed 0 and 1800 seconds, once more with the next seed after a failure, and identical reformulations are run once and shared between methods:

```bash
scripts/fetch_benchmarks.sh
scripts/rerun.sh benchmark
scripts/rerun.sh time_optimal
```

`scripts/rerun.sh time_optimal` runs branch-and-price on every nonempty family set of the 59 instances, with GCG as shipped and SCIP alone, seeds 0, 1 and 2, one thread and 1800 seconds; a run that ends abnormally is repeated once with the same seed.

The instances without a joint group use standard formulations on OR-Library files, with families named by their role:

- generalized assignment: every instance of gap1 to gap12 (60) with the families Assign and Capacity; the files are maximization problems, written with negated objective coefficients and minimized;
- bin packing: the Kantorovich formulation of binpack1 with the families Assign (each item in exactly one bin), Capacity (the sizes in a bin at most the capacity if the bin is used) and SymmetryBreaking (y<sub>j</sub> ≥ y<sub>j+1</sub>), with as many bins as the first-fit-decreasing packing uses; the 16 of the 20 instances whose LP relaxation is feasible and whose LP gap against the SCIP reference of one hour is positive.

`scripts/rerun.sh no_joint_group` builds them, computes the SCIP reference, the gain of every set of at most *k* - 1 families and the lookahead, and runs branch-and-price on the sets of the methods and on GCG as shipped (seeds 0, 1 and 2, 1800 seconds), with the feasibility check of a solution better than the reference optimum:

```bash
scripts/fetch_benchmarks.sh
scripts/rerun.sh no_joint_group
```

`scripts/rerun.sh` also reruns the census, the five-dual lookahead, the large models, the two reference optima and the time of one evaluation on the templates and the rich models (`scripts/rerun.sh census|five_duals|large_models|reference_optima|eval_time`); these runs are optional and take days.

## Solver settings of the bound evaluations

Bound evaluations solve the root node only, with presolving, separation, primal heuristics and SCIP propagation off, one thread and seed 0. [EXPERIMENTS.md](EXPERIMENTS.md) gives the settings of every experiment.
