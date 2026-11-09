"""llmdw: which constraint families must be convexified together, and do language models know?

Modules
-------
miplibnl_export     Export MIPLIB-NL reference models (gurobipy scripts) to MPS files with names.
families            Group constraints and variables of an MPS model into named families.
dec_io              Materialize a family-level decomposition into a GCG .dec file.
gcg_run             Run GCG / SCIP on a model with a given decomposition or baseline settings.
oracle              Evaluate every family-level decomposition (GCG root bound) and GCG's own choices.
lookahead           One-pricing-call lookahead bound of every family set at the LP duals.
batch               Resumable batch runs of the oracle and the lookahead.
structure           Census of zero-marginal groups (joint supports) from oracle and lookahead values.
search_sim          Offline simulation of search policies (random, greedy, covering, staged, JointDW).
relabel             Search costs averaged over random orders of the families.
theory_sim          Covering bounds and simulated search on hidden-group instances.
figure2             Values of Figure 2 and Table 3.
llm_client          OpenAI-compatible chat client with request cache, call log and model guard.
propose             Build the prompt, call the model, validate the answer.
counterfactual      Presentations of a model (original, names_only, permuted, anonymized).
support_run         Prediction requests per instance, presentation, prompt and sample.
smoke               Smoke test of one model configuration (model id, reasoning setting, endpoint cache).
analyze_support     Scoring of predictions: support detection, prediction error, search cost.
cluster             Prompt groups and the two-stage bootstrap over them.
usage               Tokens used per configuration, identical samples and retried attempts.
large_models        Budgeted search on MIPLIB-NL models with more than six families.
rich_models_build   Generation of the rich models and their exact gains.
rich_models         Scoring and search on the rich models.
results             All result tables and RESULTS.md.
support             Exact ground truth on small instances (rational LP certificates, templates).
check_outputs       Find result files cut off by an interrupted write, and remove them before a resume.
time_truth          Time-optimal decompositions on the benchmark instances: branch-and-price of every family set.
validity            Reference optimum and validity of branch-and-price runs.
extra_classes       Generalized assignment and bin packing from the OR-Library files.
no_joint_group      The classes without a joint group: the sets of the methods, branch-and-price and scoring.
rich_models_large   Rich models with 12 and 16 families (instances: generation; scoring: search costs and tables).
eval_time           Time of one evaluation of every set that the search orders visit.
eval_growth         Evaluation time against the number of convexified families.
cost_accounting     Cost of the language-model predictions against the solver time they save.
rich_figure_data    Figure data of the rich models across the number of families.
"""

__version__ = "1.2.0"
