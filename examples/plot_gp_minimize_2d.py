"""
=============================================================
Lower confidence bound as a function of number of iterations.
=============================================================

``gp_minimize`` approximates a costly function with a GaussianProcess prior
and instead of minimizing this costly function, it minimizes an acquisition
function which is relatively cheaper to minimize. Before every iteration of
``gp_minimize``, a fixed number of points are used to approximate the
GaussianProcess prior. The acquisition function combines information
about

1. New points that are close to the previous points used to approximate the GP
prior and have a very low expected value. (Posterior mean)
2. New points that are far away from these previous points and hence have a
very high uncertainty. (Posterior std)

The plot shows the "Lower Confidence Bound" after 2, 5 and 10 iterations of
``gp_minimize`` with the branin function. It may be worth noting that the areas
around the points previously sampled have very low std values.
"""
print(__doc__)
import numpy as np
import matplotlib.pyplot as plt

from skopt import gp_minimize
from skopt.benchmarks import branin
from skopt.gp_opt import acquisition

fig = plt.gcf()
plt.set_cmap("viridis")
bounds = np.asarray([[-5, 10], [0, 15]])

x1_values = np.linspace(-5, 10, 100)
x2_values = np.linspace(0, 15, 100)
x_ax, y_ax = np.meshgrid(x1_values, x2_values)
vals = np.c_[x_ax.ravel(), y_ax.ravel()]
subplot_no = 221

res = gp_minimize(
    branin, bounds, search='lbfgs', maxiter=10, random_state=0,
    acq="LCB", n_start=1, n_restarts_optimizer=2)
gp_model = res.models[-1]
opt_points = res['x_iters']

posterior_mean, posterior_std = gp_model.predict(vals, return_std=True)
acquis_values = acquisition(vals, gp_model, method="LCB")
acquis_values = acquis_values.reshape(100, 100)
posterior_mean = posterior_mean.reshape(100, 100)
posterior_std = posterior_std.reshape(100, 100)

plt.subplot(subplot_no)
plt.pcolormesh(x_ax, y_ax, posterior_mean)
plt.colorbar()
plt.xlabel('X1')
plt.xlim([-5, 10])
plt.ylabel('X2')
plt.ylim([0, 15])
plt.title("Posterior mean for acq=LCB")
subplot_no += 1

plt.subplot(subplot_no)
plt.pcolormesh(x_ax, y_ax, posterior_std)
plt.colorbar()
plt.xlabel('X1')
plt.xlim([-5, 10])
plt.ylabel('X2')
plt.ylim([0, 15])
plt.title("Posterior std for acq=LCB")
subplot_no += 1

plt.subplot(subplot_no)
plt.pcolormesh(x_ax, y_ax, acquis_values)
plt.plot(opt_points[:, 0], opt_points[:, 1], 'ro', markersize=2)
plt.colorbar()
plt.xlabel('X1')
plt.xlim([-5, 10])
plt.ylabel('X2')
plt.ylim([0, 15])
plt.title("LCB after 20 iterations.")
subplot_no += 1

plt.subplot(subplot_no)
func_values = np.reshape([branin(val) for val in vals], (100, 100))
plt.pcolormesh(x_ax, y_ax, func_values)
plt.colorbar()
plt.xlabel('X1')
plt.xlim([-5, 10])
plt.ylabel('X2')
plt.ylim([0, 15])
plt.title("Function values")

plt.suptitle("2-D acquisition values on the branin function")
plt.show()
