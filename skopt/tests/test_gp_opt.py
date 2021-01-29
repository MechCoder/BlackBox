import numpy as np
from numpy.testing import assert_array_equal
from numpy.testing import assert_array_almost_equal
import pytest
import numpy as np
from skopt import gp_minimize
from skopt.benchmarks import bench1
from skopt.benchmarks import bench2
from skopt.benchmarks import bench3
from skopt.benchmarks import bench4
from skopt.benchmarks import branin
from skopt.space.space import Real, Categorical, Space
from skopt.utils import cook_estimator


def check_minimize(func, y_opt, bounds, acq_optimizer, acq_func,
                   margin, n_calls, n_initial_points=10, init_gen="random"):
    r = gp_minimize(func, bounds, acq_optimizer=acq_optimizer,
                    acq_func=acq_func, n_initial_points=n_initial_points,
                    n_calls=n_calls, random_state=1,
                    initial_point_generator=init_gen,
                    noise=1e-10)
    assert r.fun < y_opt + margin


SEARCH = ["sampling", "lbfgs"]
ACQUISITION = ["LCB", "EI"]
INITGEN = ["random", "lhs", "halton", "hammersly", "sobol"]


@pytest.mark.slow_test
@pytest.mark.parametrize("search", SEARCH)
@pytest.mark.parametrize("acq", ACQUISITION)
def test_gp_minimize_bench1(search, acq):
    check_minimize(bench1, 0.,
                   [(-2.0, 2.0)], search, acq, 0.05, 20)


@pytest.mark.slow_test
@pytest.mark.parametrize("search", ["sampling"])
@pytest.mark.parametrize("acq", ["LCB"])
@pytest.mark.parametrize("initgen", INITGEN)
def test_gp_minimize_bench1_initgen(search, acq, initgen):
    check_minimize(bench1, 0.,
                   [(-2.0, 2.0)], search, acq, 0.05, 20, init_gen=initgen)


@pytest.mark.slow_test
@pytest.mark.parametrize("search", SEARCH)
@pytest.mark.parametrize("acq", ACQUISITION)
def test_gp_minimize_bench2(search, acq):
    check_minimize(bench2, -5,
                   [(-6.0, 6.0)], search, acq, 0.05, 20)


@pytest.mark.slow_test
@pytest.mark.parametrize("search", SEARCH)
@pytest.mark.parametrize("acq", ACQUISITION)
def test_gp_minimize_bench3(search, acq):
    check_minimize(bench3, -0.9,
                   [(-2.0, 2.0)], search, acq, 0.05, 20)


@pytest.mark.fast_test
@pytest.mark.parametrize("search", ["sampling"])
@pytest.mark.parametrize("acq", ACQUISITION)
def test_gp_minimize_bench4(search, acq):
    # this particular random_state picks "2" twice so we can make an extra
    # call to the objective without repeating options
    check_minimize(bench4, 0,
                   [("-2", "-1", "0", "1", "2")], search, acq, 1.05, 20)


@pytest.mark.fast_test
def test_n_jobs():
    r_single = gp_minimize(bench3, [(-2.0, 2.0)], acq_optimizer="lbfgs",
                           acq_func="EI", n_calls=4, n_initial_points=2,
                           random_state=1, noise=1e-10)
    r_double = gp_minimize(bench3, [(-2.0, 2.0)], acq_optimizer="lbfgs",
                           acq_func="EI", n_calls=4, n_initial_points=2,
                           random_state=1, noise=1e-10, n_jobs=2)
    assert_array_equal(r_single.x_iters, r_double.x_iters)


@pytest.mark.fast_test
def test_gpr_default():
    """Smoke test that gp_minimize does not fail for default values."""
    gp_minimize(branin, ((-5.0, 10.0), (0.0, 15.0)), n_initial_points=2,
                n_calls=2)


@pytest.mark.fast_test
def test_random_state():
    """Test that checks if random_state leads to reproducible results"""
    def objective(X):
        return (X[0] - 2) ** 2 + (X[1] - 4) ** 2
    space = [(-4., 4.), (-10., 10)]
    random_state = 42
    # Set seed
    np.random.seed(random_state)
    res1 = gp_minimize(objective, space, acq_optimizer="sampling",
                       acq_func="EI", n_initial_points=2,
                       n_calls=15, random_state=random_state,
                       n_restarts_optimizer=5,
                       xi=0.01, kappa=1.96, noise='gaussian')
    assert_array_almost_equal(res1.x, [2.0455744369142526, 4.083563545515538])
    res2 = gp_minimize(objective, space, acq_optimizer="sampling",
                       acq_func="gp_hedge", n_initial_points=2,
                       n_calls=15, random_state=random_state,
                       n_restarts_optimizer=5,
                       xi=0.01, kappa=1.96, noise='gaussian')
    assert_array_almost_equal(res2.x, [1.8623414734159072, 3.133240079201766])
    res3 = gp_minimize(objective, space, acq_optimizer="lbfgs",
                       acq_func="EI", n_initial_points=2,
                       n_calls=15, random_state=random_state,
                       n_restarts_optimizer=5,
                       xi=0.01, kappa=1.96, noise='gaussian')
    assert_array_almost_equal(res3.x, [2.004762986035108, 4.004300784961838])
    res4 = gp_minimize(objective, space, acq_optimizer="lbfgs",
                       acq_func="gp_hedge", n_initial_points=2,
                       n_calls=15, random_state=random_state,
                       n_restarts_optimizer=5,
                       xi=0.01, kappa=1.96, noise='gaussian')
    assert_array_almost_equal(res4.x, [2.208269686475764, 2.7125254537751253])


@pytest.mark.fast_test
def test_use_given_estimator():
    """ Test that gp_minimize does not use default estimator if one is passed
    in explicitly. """
    domain = [(1.0, 2.0), (3.0, 4.0)]
    noise_correct = 1e+5
    noise_fake = 1e-10
    estimator = cook_estimator("GP", domain, noise=noise_correct)
    res = gp_minimize(branin, domain, n_calls=4, n_initial_points=2,
                      base_estimator=estimator, noise=noise_fake)

    assert res['models'][-1].noise == noise_correct


@pytest.mark.fast_test
def test_use_given_estimator_with_max_model_size():
    """ Test that gp_minimize does not use default estimator if one is passed
    in explicitly. """
    domain = [(1.0, 2.0), (3.0, 4.0)]
    noise_correct = 1e+5
    noise_fake = 1e-10
    estimator = cook_estimator("GP", domain, noise=noise_correct)
    res = gp_minimize(branin, domain, n_calls=4, n_initial_points=2,
                      base_estimator=estimator, noise=noise_fake,
                      model_queue_size=1)
    assert len(res['models']) == 1
    assert res['models'][-1].noise == noise_correct


@pytest.mark.fast_test
def test_categorical_integer():
    def f(params):
        return np.random.uniform()

    dims = [[1]]
    res = gp_minimize(f, dims, n_calls=2, n_initial_points=2,
                      random_state=1)
    assert res.x_iters[0][0] == dims[0][0]


@pytest.mark.parametrize("initgen", INITGEN)
def test_mixed_categoricals(initgen):

    space = Space([
        Categorical(name="x", categories=["1", "2", "3"]),
        Categorical(name="y", categories=[4, 5, 6]),
        Real(name="z", low=1.0, high=5.0)
    ])

    def objective(param_list):
        x = param_list[0]
        y = param_list[1]
        z = param_list[2]
        loss = int(x) + y * z
        return loss

    res = gp_minimize(objective, space, n_calls=20, random_state=1,
                      initial_point_generator=initgen)
    assert res["x"] in [['1', 4, 1.0], ['2', 4, 1.0]]


@pytest.mark.parametrize("initgen", INITGEN)
def test_mixed_categoricals2(initgen):
    space = Space([
        Categorical(name="x", categories=["1", "2", "3"]),
        Categorical(name="y", categories=[4, 5, 6])
    ])

    def objective(param_list):
        x = param_list[0]
        y = param_list[1]
        loss = int(x) + y
        return loss

    res = gp_minimize(objective, space, n_calls=12, random_state=1,
                      initial_point_generator=initgen)
    assert res["x"] == ['1', 4]
