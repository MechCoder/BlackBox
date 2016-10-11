import numpy as np
from scipy import optimize
from sklearn.utils.testing import assert_array_almost_equal

from skopt.learning.gp_kernels import ExpSineSquared
from skopt.learning.gp_kernels import Matern
from skopt.learning.gp_kernels import RationalQuadratic
from skopt.learning.gp_kernels import RBF

KERNELS = [RBF(length_scale=[1.0, 2.0, 3.0]),
           Matern(length_scale=[1.0, 2.0, 3.0], nu=0.5),
           Matern(length_scale=[1.0, 2.0, 3.0], nu=1.5),
           Matern(length_scale=[1.0, 2.0, 3.0], nu=2.5),
           RationalQuadratic(alpha=2.0, length_scale=2.0),
           ExpSineSquared(length_scale=2.0, periodicity=3.0)]

rng = np.random.RandomState(0)
X = rng.randn(3)
Y = rng.randn(10, 3)

def func(X, Y, kernel):
    x = np.expand_dims(X, axis=0)
    y = np.expand_dims(Y, axis=0)
    return kernel(x, y)[0][0]

def numerical_gradient(X, func, Y, kernel):
    grad = []
    for y in Y:
        num_grad = optimize.approx_fprime(X, func, 1e-4, y, kernel)
        grad.append(num_grad)
    return np.asarray(grad)

def test_gradient_correctness():
    for kernel in KERNELS:
        X_grad = kernel.gradient_X(X, Y)
        num_grad = numerical_gradient(X, func, Y, kernel)
        assert_array_almost_equal(X_grad, num_grad, decimal=3)
