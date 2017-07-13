"""Plotting functions."""

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from matplotlib.ticker import LogLocator
from matplotlib.ticker import MaxNLocator

from scipy.optimize import OptimizeResult


def plot_convergence(*args, **kwargs):
    """Plot one or several convergence traces.

    Parameters
    ----------
    args[i] : ``OptimizeResult``, list of ``OptimizeResult``, or tuple
        The result(s) for which to plot the convergence trace.

        - if ``OptimizeResult``, then draw the corresponding single trace;
        - if list of ``OptimizeResult``, then draw the corresponding
          convergence traces in transparency, along with the average
          convergence trace;
        - if tuple, then ``args[i][0]`` should be a string label and
          ``args[i][1]`` an ``OptimizeResult`` or a list of ``OptimizeResult``.

    ax : `Axes`, optional (default=None)
        The matplotlib axes on which to draw the plot, or `None` to create
        a new one.

    true_minimum : float, optional (default=None)
        The true minimum value of the function, if known.

    yscale : None or string, optional (default=None)
        The scale for the y-axis.

    Returns
    -------
    ax : ``Axes``
        The matplotlib axes.
    """
    # <3 legacy python
    ax = kwargs.get("ax", None)
    true_minimum = kwargs.get("true_minimum", None)
    yscale = kwargs.get("yscale", None)

    if ax is None:
        ax = plt.gca()

    ax.set_title("Convergence plot")
    ax.set_xlabel("Number of calls $n$")
    ax.set_ylabel(r"$\min f(x)$ after $n$ calls")
    ax.grid()

    if yscale is not None:
        ax.set_yscale(yscale)

    colors = cm.viridis(np.linspace(0.25, 1.0, len(args)))

    for results, color in zip(args, colors):
        if isinstance(results, tuple):
            name, results = results
        else:
            name = None

        if isinstance(results, OptimizeResult):
            n_calls = len(results.x_iters)
            mins = [np.min(results.func_vals[:i])
                    for i in range(1, n_calls + 1)]
            ax.plot(range(1, n_calls + 1), mins, c=color,
                    marker=".", markersize=12, lw=2, label=name)

        elif isinstance(results, list):
            n_calls = len(results[0].x_iters)
            iterations = range(1, n_calls + 1)
            mins = [[np.min(r.func_vals[:i]) for i in iterations]
                    for r in results]

            for m in mins:
                ax.plot(iterations, m, c=color, alpha=0.2)

            ax.plot(iterations, np.mean(mins, axis=0), c=color,
                    marker=".", markersize=12, lw=2, label=name)

    if true_minimum:
        ax.axhline(true_minimum, linestyle="--",
                   color="r", lw=1,
                   label="True minimum")

    if true_minimum or name:
        ax.legend(loc="best")

    return ax


def _format_scatter_plot_axes(ax, space, ylabel):
    # Work out min, max of y axis for the diagonal so we can adjust
    # them all to the same value
    diagonal_ylim = (np.min([ax[i, i].get_ylim()[0]
                             for i in range(space.n_dims)]),
                     np.max([ax[i, i].get_ylim()[1]
                             for i in range(space.n_dims)]))

    # Deal with formatting of the axes
    for i in range(space.n_dims):  # rows
        for j in range(space.n_dims):  # columns
            ax_ = ax[i, j]

            if j > i:
                ax_.axis("off")

            # off-diagonal axis
            if i != j:
                # plots on the diagonal are special, like Texas. They have
                # their own range so do not mess with them.
                ax_.set_ylim(*space.dimensions[i].bounds)
                ax_.set_xlim(*space.dimensions[j].bounds)
                if j > 0:
                    ax_.set_yticklabels([])
                else:
                    ax_.set_ylabel("$X_{%i}$" % i)

                # for all rows except ...
                if i < space.n_dims - 1:
                    ax_.set_xticklabels([])
                # ... the bottom row
                else:
                    [l.set_rotation(45) for l in ax_.get_xticklabels()]
                    ax_.set_xlabel("$X_{%i}$" % j)

            else:
                ax_.set_ylim(*diagonal_ylim)
                ax_.yaxis.tick_right()
                ax_.yaxis.set_label_position('right')
                ax_.yaxis.set_ticks_position('both')
                ax_.set_ylabel(ylabel)

                ax_.xaxis.tick_top()
                ax_.xaxis.set_label_position('top')
                ax_.set_xlabel("$X_{%i}$" % j)

            ax_.xaxis.set_major_locator(MaxNLocator(6, prune='both'))
            ax_.yaxis.set_major_locator(MaxNLocator(6, prune='both'))

    return ax


def partial_dependence(space, model, i, j=None, sample_points=None,
                       n_samples=250, n_points=40):
    """Calculate the partial dependence for dimensions ``i`` and ``j`` with
    respect to the objective value, as approximated by ``model``.

    The partial dependence plot shows how the value of the dimensions
    ``i`` and ``j`` influence the ``model`` predictions after "averaging out"
    the influence of all other dimensions.

    Parameters
    ----------
    space : ``Space``
        The parameter space over which the minimization was performed.

    model : skopt regressor
        Surrogate model for the objective function.

    i : int
        The first dimension for which to calculate the partial dependence.

    j : int, optional (default=None)
        The second dimension for which to calculate the partial dependence.
        To calculate the 1D partial dependence on ``i`` alone set ``j=None``.

    sample_points : np.array, shape=[n_points, n_dims], optional (default=None)
        Randomly sampled and transformed points to use when averaging
        the model function at each of the ``n_points``.

    n_samples : int, optional (default=100)
        Number of random samples to use for averaging the model function
        at each of the ``n_points``. Only used when ``sample_points=None``.

    n_points : int, optional (default=40)
        Number of points at which to evaluate the partial dependence
        along each dimension ``i`` and ``j``.

    Returns
    -------
    For 1D partial dependence:

    xi : np.array
        The points at which the partial dependence was evaluated.

    yi : np.array
        The value of the model at each point ``xi``.

    For 2D partial dependence:

    xi : np.array, shape=[n_points]
        The points at which the partial dependence was evaluated.
    yi : np.array, shape=[n_points]
        The points at which the partial dependence was evaluated.
    zi : np.array, shape=[n_points, n_points]
        The value of the model at each point ``(xi, yi)``.
    """
    if sample_points is None:
        sample_points = space.transform(space.rvs(n_samples=n_samples))

    if j is None:
        bounds = space.dimensions[i].bounds
        # XXX use linspace(*bounds, n_points) after python2 support ends
        xi = np.linspace(bounds[0], bounds[1], n_points)
        xi_transformed = space.dimensions[i].transform(xi)

        yi = []
        for x_ in xi_transformed:
            rvs_ = np.array(sample_points)
            rvs_[:, i] = x_
            yi.append(np.mean(model.predict(rvs_)))

        return xi, yi

    else:
        # XXX use linspace(*bounds, n_points) after python2 support ends
        bounds = space.dimensions[j].bounds
        xi = np.linspace(bounds[0], bounds[1], n_points)
        xi_transformed = space.dimensions[j].transform(xi)

        bounds = space.dimensions[i].bounds
        yi = np.linspace(bounds[0], bounds[1], n_points)
        yi_transformed = space.dimensions[i].transform(yi)

        zi = []
        for x_ in xi_transformed:
            row = []
            for y_ in yi_transformed:
                rvs_ = np.array(sample_points)
                rvs_[:, (j, i)] = (x_, y_)
                row.append(np.mean(model.predict(rvs_)))
            zi.append(row)

        return xi, yi, np.array(zi).T


def plot_objective(result, levels=10, n_points=40, n_samples=250,
                   zscale='linear'):
    """Pairwise partial dependence plot of the objective function.

    The diagonal shows the partial dependence for dimension ``i`` with
    respect to the objective function. The off-diagonal shows the
    partial dependence for dimensions ``i`` and ``j`` with
    respect to the objective function. The objective function is
    approximated by ``result.model``.

    Pairwise scatter plots of the points at which the objective
    function was directly evaluated are shown on the off-diagonal.
    A red point indicates the found minimum.

    Note: search spaces that contain ``Categorical`` dimensions are
          currently not supported by this function.

    Parameters
    ----------
    result : `OptimizeResult``
        The result for which to create the scatter plot matrix.

    levels : int, optional (default=10)
        Number of levels to draw on the contour plot, passed directly
        to ``plt.contour()``.

    n_points : int, optional (default=40)
        Number of points at which to evaluate the partial dependence
        along each dimension.

    n_samples : int, optional (default=250)
        Number of random samples to use for averaging the model function
        at each of the ``n_points``.

    zscale : str, optional (default='linear')
        Scale to use for the z axis of the contour plots. Either 'linear'
        or 'log'.

    Returns
    -------
    ax : ``Axes``
        The matplotlib axes.
    """
    space = result.space
    samples = np.asarray(result.x_iters)
    rvs_transformed = space.transform(space.rvs(n_samples=n_samples))

    if zscale == 'log':
        locator = LogLocator()
    elif zscale == 'linear':
        locator = None
    else:
        raise ValueError("Valid values for zscale are 'linear' and 'log',"
                         " not '%s'." % zscale)

    fig, ax = plt.subplots(space.n_dims, space.n_dims,
                           figsize=(2 * space.n_dims, 2 * space.n_dims))

    fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95,
                        hspace=0.1, wspace=0.1)

    for i in range(space.n_dims):
        for j in range(space.n_dims):
            if i == j:
                xi, yi = partial_dependence(space, result.models[-1], i,
                                            j=None,
                                            sample_points=rvs_transformed,
                                            n_points=n_points)

                ax[i, i].plot(xi, yi)
                ax[i, i].axvline(result.x[i], linestyle="--", color="r", lw=1)

            # lower triangle
            elif i > j:
                xi, yi, zi = partial_dependence(space, result.models[-1],
                                                i, j,
                                                rvs_transformed, n_points)
                ax[i, j].contourf(xi, yi, zi, levels,
                                  locator=locator, cmap='viridis_r')
                ax[i, j].scatter(samples[:, j], samples[:, i],
                                 c='k', s=10, lw=0.)
                ax[i, j].scatter(result.x[j], result.x[i],
                                 c=['r'], s=20, lw=0.)

    return _format_scatter_plot_axes(ax, space, "Partial dependence")


def plot_evaluations(result, bins=20):
    """Visualize the order in which points where sampled.

    The scatter plot matrix shows at which points in the search
    space and in which order samples were evaluated. Pairwise
    scatter plots are shown on the off-diagonal for each
    dimension of the search space. The order in which samples
    were evaluated is encoded in each point's color.
    The diagonal shows a histogram of sampled values for each
    dimension. A red point indicates the found minimum.

    Note: search spaces that contain ``Categorical`` dimensions are
          currently not supported by this function.

    Parameters
    ----------
    result : ``OptimizeResult``
        The result for which to create the scatter plot matrix.

    bins : int, optional (default=20)
        Number of bins to use for histograms on the diagonal.

    Returns
    -------
    ax : ``Axes``
        The matplotlib axes.
    """
    space = result.space
    samples = np.asarray(result.x_iters)
    order = range(samples.shape[0])
    fig, ax = plt.subplots(space.n_dims, space.n_dims,
                           figsize=(2 * space.n_dims, 2 * space.n_dims))

    fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95,
                        hspace=0.1, wspace=0.1)

    for i in range(space.n_dims):
        for j in range(space.n_dims):
            if i == j:
                ax[i, i].hist(samples[:, j], bins=bins,
                              range=space.dimensions[j].bounds)

            # lower triangle
            elif i > j:
                ax[i, j].scatter(samples[:, j], samples[:, i], c=order,
                                 s=40, lw=0., cmap='viridis')
                ax[i, j].scatter(result.x[j], result.x[i],
                                 c=['r'], s=20, lw=0.)

    return _format_scatter_plot_axes(ax, space, "Number of samples")
