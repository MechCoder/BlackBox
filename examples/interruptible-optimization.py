"""
================================================
Interruptible optimization runs with checkpoints
================================================

Christian Schell, Mai 2018
Reformatted by Holger Nahrstaedt 2020

.. currentmodule:: skopt

Problem statement
=================

Optimization runs can take a very long time and even run for multiple days.
If for some reason the process has to be interrupted results are irreversibly
lost, and the routine has to start over from the beginning.

With the help of the :class:`callbacks.CheckpointSaver` callback the optimizer's current state
can be saved after each iteration, allowing to restart from that point at any
time.

This is useful, for example,

* if you don't know how long the process will take and cannot hog computational resources forever
* if there might be system failures due to shaky infrastructure (or colleagues...)
* if you want to adjust some parameters and continue with the already obtained results

"""
print(__doc__)
import sys
import numpy as np
np.random.seed(777)
import os

#############################################################################
# Simple example
# ==============
#
# We will use pretty much the same optimization problem as in the
# :ref:`sphx_glr_auto_examples_bayesian-optimization.py`
# notebook. Additionally we will instantiate the :class:`callbacks.CheckpointSaver`
# and pass it to the minimizer:

from skopt import gp_minimize
from skopt import callbacks
from skopt.callbacks import CheckpointSaver

noise_level = 0.1


def obj_fun(x, noise_level=noise_level):
    return np.sin(5 * x[0]) * (1 - np.tanh(x[0] ** 2)) + np.random.randn() \
        * noise_level

checkpoint_saver = CheckpointSaver("./checkpoint.pkl", compress=9) # keyword arguments will be passed to `skopt.dump`

gp_minimize(obj_fun,            # the function to minimize
            [(-20.0, 20.0)],    # the bounds on each dimension of x
            x0=[-20.],          # the starting point
            acq_func="LCB",     # the acquisition function (optional)
            n_calls=10,         # number of evaluations of f including at x0
            n_random_starts=3,  # the number of random initial points
            callback=[checkpoint_saver],
            # a list of callbacks including the checkpoint saver
            random_state=777)

#############################################################################
# Now let's assume this did not finish at once but took some long time: you
# started this on Friday night, went out for the weekend and now, Monday
# morning, you're eager to see the results. However, instead of the
# notebook server you only see a blank page and your colleague Garry
# tells you that he had had an update scheduled for Sunday noon – who
# doesn't like updates?
#
# :class:`gp_minimize` did not finish, and there is no `res` variable with the
# actual results!
#
# Restoring the last checkpoint
# =============================
#
# Luckily we employed the :class:`callbacks.CheckpointSaver` and can now restore the latest
# result with :class:`skopt.load`
# (see :ref:`sphx_glr_auto_examples_store-and-load-results.py` for more
# information on that)

from skopt import load

res = load('./checkpoint.pkl')

res.fun

#############################################################################
# Continue the search
# ===================
#
# The previous results can then be used to continue the optimization process:

x0 = res.x_iters
y0 = res.func_vals

# To ensure that the base estimator is loaded properly and that the next 
# parameters are new ones:
base_estimator = res.specs['args']['base_estimator']
random_state = res.random_state

gp_minimize(obj_fun,                        # the function to minimize
            [(-20.0, 20.0)],                # the bounds on each dimension of x
            base_estimator=base_estimator,  # warm-started base-estimator from checkpoint
            x0=x0,                          # already examined values for x
            y0=y0,                          # observed values for x0
            acq_func="LCB",                 # the acquisition function (optional)
            n_calls=10,                     # number of evaluations of f including at x0
            n_random_starts=3,              # the number of random initialization points
            random_state=random_state       # warm-started random state from checkpoint
            callback=[checkpoint_saver],
            random_state=777)

#############################################################################
# Possible problems
# =================
#
# * **changes in search space:** You can use this technique to interrupt
#   the search, tune the search space and continue the optimization. Note
#   that the optimizers will complain if `x0` contains parameter values not
#   covered by the dimension definitions, so in many cases shrinking the
#   search space will not work without deleting the offending runs from
#   `x0` and `y0`.
# * see :ref:`sphx_glr_auto_examples_store-and-load-results.py`
#
# for more information on how the results get saved and possible caveats
