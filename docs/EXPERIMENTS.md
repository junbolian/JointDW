# Experiments: settings

This file states the settings of every experiment behind the tables and figures. The records and their layout are
described in `DATA.md`; the commands are in `scripts/rerun.sh` (solver runs) and `scripts/reproduce.sh` (tables).

## Machine and solvers

| Item | Setting |
|---|---|
| machine | one workstation, 24-core Intel processor, 48 GB memory |
| solvers | SCIP 10 with GCG through PyGCGOpt 1.0.0b0 on PySCIPOpt 6.0.0; HiGHS for the LP duals of the lookahead; SoPlex for the LP bound |
| threads | one per solver process |
| memory | an address-space cap per solver process: 5 GB, 10 GB for the high-memory models (`config/high_memory_models.txt`); a shared budget of 40 GB on the sum of the caps of the running processes (a process waits until its cap fits, and starts only with at least 10 GB available) |
| solver processes | at most 22 active solver processes on the machine during the time-optimal decompositions, the rich models with 12 and 16 families and the classes without a joint group |

## Instance collections and their selection

| Collection | Instances | Selection |
|---|---|---|
| templates | 15 | the hand-built templates of `llmdw.instances.templates` |
| MIPLIB-NL | 108 models with two to six families, 102 of them scored | families are the row names with digit runs removed; the scored models are the 102 of the scorable set |
| rich models with 7 to 9 families | 24 (8 per class) | random draws of lot sizing, network design and facility location; a draw is accepted when it is feasible, its LP gap is at least two percent, and no single family closes more than half of the best gain among the sets of at most three families; the first eight accepted draws of each class |
| rich models with 12 and 16 families | 60 (10 per class and k) | the same classes with further families on the same variables (lot sizing and facility location with k = 12 and 16, network design with k = 12 and 14), the same acceptance rule; the first ten accepted draws of each class and k; the k = 12 menu of lot sizing is its 9 families plus the first 3 new ones, of network design and facility location their 7 families plus the first 5 new ones; an infeasible network-design draw is repaired by redrawing the new families of a minimal conflicting family set (deletion filter) up to ten times |
| models with more than six families | 30 MIPLIB-NL models with 7 to 26 families | every exported model in that range |
| benchmark instances | 59 | lot sizing: the 20 smallest usable Trigeiro, Thomas and McClain instances by items times periods; network design: the Canad R instances with a feasible LP and a positive LP gap, up to the 20 smallest by number of variables; facility location: the OR-Library capacitated instances cap41 to cap134 with a positive LP gap (19) |
| classes without a joint group | 76 | generalized assignment: every instance of OR-Library gap1 to gap12 (60; the files are maximization problems, written with negated objective coefficients and minimized); bin packing: the instances of OR-Library binpack1 (Kantorovich formulation with as many bins as first-fit-decreasing uses) whose LP relaxation is feasible and whose LP gap against the SCIP reference of one hour is positive (16 of 20) |

## Gain evaluations

| Item | Setting |
|---|---|
| mode | GCG with the given decomposition, root node only, no presolve, no cutting planes, no primal heuristics, no rounding |
| limit | 120 s per evaluation; wall-clock limit 240 s for a given decomposition, 360 s for GCG's detected decompositions |
| propagation | off in the original and the master problem (`propagating/maxrounds` and `propagating/maxroundsroot` 0) |
| seed | 0 |
| failed evaluation | retried twice with the same settings, then once with seed 1; four abnormal ends give the status failed and no bound |
| sets | every set of at most k - 1 families without linking variables (MIPLIB-NL census); at most two families on the benchmark instances, at most k - 1 on the classes without a joint group; at most three on the rich models (exact gains in rational arithmetic) and on the models with more than six families |
| LP bound | SCIP LP (SoPlex), every variable continuous, presolve off; HiGHS interior point with crossover for graph40-40-1rand and graph40-80-1rand |
| reference optimum | SCIP defaults, one thread, seed 0, 3600 s (benchmark collections); see Reference optimum below |

## Lookahead

| Item | Setting |
|---|---|
| lookahead | one pricing round per candidate set at an optimal LP dual (HiGHS simplex; interior point for the two graph40 models), block MIP limit 60 s, instance limit 3600 s, block propagation off |
| five-dual lookahead | the HiGHS dual and four further optimal duals of the dual face (dual seeds 1 to 4, face tolerance 1e-9, box factor 10); the value of a set is the best of the five |
| ties | lookahead values within 1e-9 are tied; ties by fewer families, then name |

## Branch-and-price

| Item | Setting |
|---|---|
| solver | GCG with SCIP defaults (propagation default), the decomposition given as a file, one thread, 1800 s; GCG as shipped (presolve, detection, default score) for GCG's own decomposition; SCIP alone on the original formulation with the same limit for the time-optimal decompositions |
| seeds | seed 0 for the standard formulations; seeds 0, 1 and 2 for the time-optimal decompositions and the classes without a joint group |
| failure rule | standard formulations: a run that ends abnormally is repeated with the next seed; time-optimal decompositions and classes without a joint group: it is repeated once with the same seed; a run that fails twice counts as unsolved at the limit with its root bound |
| runs | standard formulations: the set of each method; time-optimal decompositions: every nonempty family set of the 59 instances, GCG as shipped and SCIP alone; classes without a joint group: the distinct sets of the methods (the bound-optimal set, both lookaheads, GCG's detected set, the anchor of every answer) and GCG as shipped |
| concurrency | 8 runs at a time under the shared memory budget |
| final gap | a final gap below 1e-9 counts as 0 |

## Reference optimum and validity rules

| Item | Setting |
|---|---|
| reference optimum z* | the best of the value proved or found by SCIP, for bin packing the best-known count of the OR-Library file, and a branch-and-price solution accepted by the feasibility check below; used for the gains, the gain shares and both validity rules |
| tolerance | 1e-4 times the larger of the LP gap and max(1, abs(z*)); without a reference optimum 1e-4 times max(1, abs(primal value)) |
| bound validity | a recorded dual bound counts as no bound when it is not finite or at least 1e19 in absolute value, when it is beyond z*, or when it is beyond the best primal solution value recorded by SCIP for the model, each by more than the tolerance |
| validity of branch-and-price runs | a run that reports optimality is solved only when its value is within the tolerance of z*; a run that reports optimality at a value worse than z*, or ends with a dual bound beyond z*, by more than the tolerance, counts as failed (unsolved at 1800 s, 1801 s in the time statistic, root bound kept only when valid, final gap from it or else from the LP bound); a run with a solution better than z* by more than the tolerance is checked against every row of the model, the variable bounds and integrality (absolute tolerance 1e-6), and a feasible solution becomes z* |

## Ground truth of the benchmark collections

| Item | Definition |
|---|---|
| bound-optimal set | a set of maximal gain among the evaluated sets within the budget (gain values within 1e-9 tied; ties by fewer families, then name); on an instance where every set has gain 0 nothing is scored against it |
| time statistic | the geometric mean over seeds 0, 1 and 2 of t + 1 s, t the total time, an unsolved run at 1801 s |
| time-optimal set | the first set in the order: more seeds solved first; then the smaller time statistic; among sets unsolved on every seed, the smaller mean final gap; on the classes without a joint group among the planned sets within the budget |
| time-optimal set within tolerance | every set that solves on as many seeds as the time-optimal set and whose time statistic is at most 1.1 times its time statistic; when no set solves on any seed, every set lies within tolerance (one rule for every time-optimal set below) |
| time-optimal set within the budget | on the benchmark instances, the first set of the same ranking among the sets of at most two families (k - 1): the set of all three families leaves an empty master, so its pricing problem is the program itself, and the census and the gain tables exclude it as well; a method whose set has all three families is a miss and keeps its own time; the bound-optimal set ("best pair") is reported as a reference method |
| budget | each evaluation convexifies at most k - 1 families (the sets of the gain table); a method's first set beyond it has gain 0 in every pooled gain and is counted in its own column |

## Search costs and statistics

| Item | Setting |
|---|---|
| family orders | 1,000 random orders of the families per instance, seed 0 (keyed by the instance name) |
| size limits | s = 2, 3 and k - 1 (s = min(s, k - 1) where k - 1 is smaller) |
| staged order of JointDW | anchored at the first predicted joint support, or else at the families convexified by the top-ranked reformulation; an anchor with more than s families keeps its first s families in the family order (the staged order O1) |
| alternation | the JointDW order takes the entries of O1 and of the covering order O2 in turn, and an entry that was already evaluated (or lies outside the gain table) is skipped; the interleaved orders take the entries of the JointDW order and of the lookahead order in turn in the same way; a cost is the number of evaluations until a set of maximal gain is evaluated |
| misses | e = min over the minimal best sets W of the number of families of W outside the anchor, counted on the anchor that the search uses (after the cut to s families); stage e of O1 covers every combination of all but e families of the anchor P with e other families, so O1 reaches a best set W within stages 0 to e when W has at most as many families as P |
| coverings | greedy: at each step the first set, in the lexicographic order of positions in the family order, among the sets that cover the most uncovered subsets |
| bootstrap | two stages: prompt groups with replacement, then the instances within each drawn group with replacement; instances with byte-identical prompts form one prompt group and are drawn together; 10,000 resamples, seed 0; 95 percent percentile intervals; an asterisk when the interval of a difference excludes 0 |
| time ratios | geometric mean over instances, standard error of the mean log ratio times the geometric mean |
| census shares | Wilson 95 percent intervals |
| first sets (Table 31) | units: the instances with a true joint support (benchmark instances: with a best pair); outcomes of a sample: hit@1 of the language model (0 or 1) and hit@1 of the lookahead order (the share of the family orders under which its first set is a true joint support; benchmark instances: its set is a bound-optimal set); the four joint outcomes are the products of the two, averaged over the samples of an instance and then over the instances; bootstrap as above, with the prompt groups of the text prompt and every instance its own group with every coefficient |

## Language models

| Configuration | Endpoint identifier | Request options | Samples | Streaming | JSON mode | Temperature | Timeout | Reasoning required |
|---|---|---|---|---|---|---|---|---|
| Claude Sonnet 5.5 | `claude-sonnet-5-5` | none | 3 | no | yes | default | 1800 s | yes |
| GPT-6.1 Sol | `gpt-6.1-sol` | `{"reasoning_effort": "medium"}` | 3 | yes | yes | default | 1800 s | yes |
| Grok 4.20 (Reasoning) | `grok-4-20-reasoning` | none | 3 | yes | yes | default | 1800 s | yes |
| Grok 4.20 (Non-Reasoning) | `grok-4-20-non-reasoning` | none | 3 | yes | yes | default | 1800 s | no |
| DeepSeek-V4-Pro (Non-Thinking) | `deepseek-v4-pro` | `{"thinking": {"type": "disabled"}}` | 3 | yes | yes | default | 600 s | no |

| Item | Setting |
|---|---|
| prompts | one system message and one user message (`prompts/decomp_v2.md`); the data variant adds the instance in CPLEX LP format; the ranked variant adds ranked groups and relevant families |
| presentations | templates and MIPLIB-NL: original, names only, permuted, anonymized; rich models: original and the data variant; control presentations: the data variant with the family names permuted and with the names replaced by labels (per-instance random labels, the same for both models) |
| configurations per experiment | all five on the templates, MIPLIB-NL and the rich models with 7 to 9 families; GPT-6.1 Sol and Claude Sonnet 5.5 on the control presentations, the benchmark instances (original presentation and data variant), the rich models with 12 and 16 families (original presentation and data variant) and the classes without a joint group (original presentation) |
| prompt-token guard | an answer whose reported prompt tokens exceed a factor per prompt character plus 300 tokens is rejected (`prompt_tokens_per_char` in `config/public_settings.json`; for the data variant on the benchmark instances 0.50 for GPT-6.1 Sol and 0.85 for Claude Sonnet 5.5); repeated samples of one request must report the same prompt tokens within 5 tokens |
| cache | every request has a cache entry keyed by the request; a byte-identical request is answered from the cache |
| requests in flight | GPT-6.1 Sol 4 and Claude Sonnet 5.5 8 on the rich models with 12 and 16 families; one per configuration on the classes without a joint group |

## Time-optimal decompositions on the benchmark instances

Every nonempty family set of the 59 benchmark instances, GCG as shipped and SCIP alone, with the branch-and-price
settings above and seeds 0, 1 and 2. From each run: total time, solved, final gap, root bound and root gap closed, root
time, master LP iterations and pricing calls (the root column-generation iterations are not in GCG's statistics),
columns, nodes, pricing problems and aggregated blocks, pricing time and its share, and the pricing solvers. Every
method's first set is measured against the time-optimal set: hit@1 strict and within tolerance, and the ratio of time
statistics.

## Rich models with 12 and 16 families

The draws, the acceptance rule and the menus are given above. The exact gain of every set of at most three families is
computed for each accepted instance (network design with k = 14: at most two). JointDW with GPT-6.1 Sol and Claude
Sonnet 5.5 under the original presentation and the data variant, three samples per instance; hit@1 and recall of the
first predicted joint support; the evaluations until a best set is evaluated for s = 2 and s = 3 under the covering
order, the lookahead order at the HiGHS dual, the five-dual lookahead order, JointDW, JointDW interleaved with the
five-dual order and the staged search anchored at GCG's own decomposition.

## Classes without a joint group

Generalized assignment (families Assign and Capacity) and bin packing (Assign, Capacity, SymmetryBreaking). The gain of
every set of at most k - 1 families at the root with the settings above; on bin packing {Assign, Capacity} and on
generalized assignment the pair of both families lie outside the budget. Branch-and-price as above. Measured: hit@1
against the bound-optimal and the time-optimal set, the samples that predict a joint support where the gain table has
none within the budget, the time ratios and the paired differences.

## Cost accounting

| Item | Setting |
|---|---|
| requests | every request of the sample of record from the call records: prompt, cached prompt, cache-write, completion and reasoning tokens, latency |
| prices | the list price on the day of the request (start time = log time minus latency, local date) from the provider's pricing page: captured on 2026-11-08 and 2026-11-09, earlier days from Internet Archive snapshots of the same pages (the last snapshot on the day, else the nearest before, else after); cached-input prices only where cached tokens are recorded; cache writes at the 5-minute write price; DeepSeek-V4-Pro (Non-Thinking) at its peak or off-peak rate by the UTC start time; no price changed between 2026-10-30 and 2026-11-09 |
| one prediction | the request that produced the answer of a cell; an answer read from the cache takes the latency and cost of its sent request |
| time of one evaluation | templates and rich models: the census time of a set (the wall time of the solver process, else GCG's solving time, capped at 120 s) on every set the search orders visit; MIPLIB-NL: the census evaluation times; benchmark instances: the mean time of one branch-and-price run of a family set over seeds 0 to 2 (unsolved at 1800 s) |
| measures | evaluations saved by JointDW and by JointDW interleaved with the five-dual order against the covering order and the five-dual lookahead; the solver time they represent; the latency and cost of one prediction; the net time saved; the break-even evaluation time (latency over evaluations saved) |
| benchmark instances | net time saved on each instance's own mean evaluation time, and the share of instances where it is positive; the orders built on the five-dual values are charged the stored time of the five-dual lookahead on the instance |

## Evaluation time against the number of convexified families

The census time of one evaluation by the number of convexified families: the MIPLIB-NL census, the timed evaluations
of the templates and the rich models with 7 to 9 families, and the root and pricing times of the branch-and-price runs
of the time-optimal decompositions. Per collection and size: counts, median and quartiles; per model the least-squares
slope of the log median time on the log size, over the models with at least three sizes.

## Models with more than six families

| Item | Setting |
|---|---|
| models | 30 MIPLIB-NL models with 7 to 26 families |
| sets | at most three families, no linking variables |
| budget | 10 evaluations per order |
| orders | covering, greedy forward selection, lookahead, random (seeds 0, 1 and 2) |
| evaluation | as the census: 120 s, propagation off, seed 0, the failure rule above |

## Learned predictor and combined orders

| Item | Setting |
|---|---|
| learner | XGBoost regression of the gain share: max_depth 13, learning_rate 0.1, n_estimators 100, random_state 0, 2 threads |
| splits | within class; leave one class out; MIPLIB-NL leave one out; training size (1, 2, 4 and 7 training instances, 20 draws) |
| size limits | s = 2, 3 (combined orders: 2, 3 and k - 1) |
| features | the structural features of `llmdw/scoring/learned.py` (FEATURES), and in the second variant the lookahead values |
