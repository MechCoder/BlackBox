# Release history

## Version 0.1

First light!

### New features

* Bayesian optimization via `gp_minimize`.
* Tree-based sequential model-based optimization via `forest_minimize` and `gbrt_minimize`, with support for multi-threading.
* Support of LCB, EI and PI as acquisition functions.
* Plotting functions for inspecting convergence, evaluations and the objective function.
* API for specifying and sampling from a parameter space.

## Version 0.2

### New features

* Speed improvements when using `gp_minimize` with `acq_optimizer='lbfgs'` and
`acq_optimizer='auto'` when all the search-space dimensions are Real.
* Persistence of minimization results using `skopt.dump` and `skopt.load`.
* Support for using arbitrary estimators that implement a
`return_std` argument in their `predict` method by means of `base_minimize` from `skopt.optimizer.`
* Support for tuning noise in `gp_minimize` using the `noise` argument.
* `TimerCallback` in `skopt.callbacks` to log the time between iterations of
the minimization loop.

# Contributors

See `AUTHORS.md`.
