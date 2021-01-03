Like BALD10 but with patience=10 instead of patience=3 to ensure that the model has indeed converged.
(FYI, turns out convergence wasn't the issue; BALD performs worse than random acq. because
the uncertainty estimates were inaccurate. Using ensemble seems to help.)
