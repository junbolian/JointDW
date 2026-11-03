"""The constants that fix every random draw and tolerance of the result tables (Section 6 of the paper).

FAMILY_ORDERS        random orders of the families per instance; every search cost is the mean over them
FAMILY_ORDER_SEED    entropy of numpy's SeedSequence; each instance has its own stream, keyed by its name
BOOTSTRAP_RESAMPLES  resamples of the cluster bootstrap, which resamples prompt groups
BOOTSTRAP_SEED       seed of numpy's default_rng for every bootstrap
TIE_TOL              lookahead values within this distance are tied
"""
FAMILY_ORDERS = 1000
FAMILY_ORDER_SEED = 0
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 0
TIE_TOL = 1e-9
