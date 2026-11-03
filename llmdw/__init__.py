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
cluster             Prompt groups and the cluster bootstrap over them.
usage               Tokens used per configuration, identical samples and retried attempts.
large_models        Budgeted search on MIPLIB-NL models with more than six families.
rich_models_build   Generation of the rich models and their exact gains.
rich_models         Scoring and search on the rich models.
results             All result tables and RESULTS.md.
support             Exact ground truth on small instances (rational LP certificates, templates).
check_outputs       Find result files cut off by an interrupted write, and remove them before a resume.
"""

__version__ = "1.0.0"
