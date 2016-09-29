import numbers
import numpy as np

from scipy.stats.distributions import randint
from scipy.stats.distributions import rv_discrete
from scipy.stats.distributions import uniform

from sklearn.utils import check_random_state
from sklearn.utils.fixes import sp_version

from skopt.transformers import CategoricalEncoder
from skopt.transformers import HyperCube
from skopt.transformers import Identity
from skopt.transformers import Log10
from skopt.transformers import Pipeline

# helper class to be able to print [1, ..., 4] instead of [1, '...', 4]
class _Ellipsis:
    def __repr__(self):
        return '...'


def check_dimensions(dimensions, transform=None):
    """
    Parameters
    ----------
    * `dimensions` [list]:
        List of search space dimensions.
        Each search dimension can be defined either as

        - a `(upper_bound, lower_bound)` tuple (for `Real` or `Integer`
          dimensions),
        - a `(upper_bound, lower_bound, "prior")` tuple (for `Real`
          dimensions),
        - as a list of categories (for `Categorical` dimensions), or
        - an instance of a `Dimension` object (`Real`, `Integer` or
          `Categorical`).

    * `transform` ["hypercube", optional] :
        If transform is set to "hypercube", it is assumed that
        the optimizer space is [0, 1] and hence calling "transform"
        on that particular dimension returns a value scaled between
        0 and 1.

    Returns
    -------
    * `dimensions` [list, shape=(n_dims,)]
        A list of dimension instances.
    """
    _dimensions = []

    for dim in dimensions:
        if isinstance(dim, Dimension):
            _dimensions.append(dim)

        elif (len(dim) == 3 and
              isinstance(dim[0], numbers.Real) and
              isinstance(dim[2], str)):
            _dimensions.append(Real(*dim, transform=transform))

        elif len(dim) > 2 or isinstance(dim[0], str):
            _dimensions.append(Categorical(dim))

        elif isinstance(dim[0], numbers.Integral):
            _dimensions.append(Integer(*dim))

        elif isinstance(dim[0], numbers.Real):
            _dimensions.append(Real(*dim, transform=transform))
        else:
            raise ValueError("Invalid grid component (got %s)." % dim)
    return _dimensions


class Dimension(object):
    """Base class for search space dimensions."""

    def rvs(self, n_samples=1, random_state=None):
        """Draw random samples.

        Parameters
        ----------
        * `n_samples` [int or None]:
            The number of samples to be drawn.

        * `random_state` [int, RandomState instance, or None (default)]:
            Set random state to something other than None for reproducible
            results.
        """
        rng = check_random_state(random_state)
        samples = self._rvs.rvs(size=n_samples, random_state=rng)
        return self.inverse_transform(samples)

    def transform(self, X):
        """Transform samples form the original space to a warped space."""
        return self.transformer.transform(X)

    def inverse_transform(self, Xt):
        """Inverse transform samples from the warped space back into the
           original space.
        """
        return self.transformer.inverse_transform(Xt)

    @property
    def size(self):
        return 1

    @property
    def transformed_size(self):
        return 1

    @property
    def bounds(self):
        raise NotImplementedError

    @property
    def transformed_bounds(self):
        raise NotImplementedError


class Real(Dimension):
    def __init__(self, low, high, prior="uniform", transform=None):
        """Search space dimension that can take on any real value.

        Parameters
        ----------
        * `low` [float]:
            Lower bound (inclusive).

        * `high` [float]:
            Upper bound (exclusive).

        * `prior` ["uniform" or "log-uniform", default="uniform"]:
            Distribution to use when sampling random points for this dimension.
            - If `"uniform"`, points are sampled uniformly between the lower
              and upper bounds.
            - If `"log-uniform"`, points are sampled uniformly between
              `log10(lower)` and `log10(upper)`.`

        * `transform` [None or "hypercube", optional]:
            If `transform=hypercube`, the `transform` method returns points
            sampled uniform / log-uniformly from the given bounds and then
            scaled to [0, 1]
        """
        self.low = low
        self.high = high
        self.prior = prior
        self.transform_ = transform

        if self.transform_ and self.transform_ != "hypercube":
            raise ValueError(
                "transform should be hypercube, got %s" % self.transform_)

        if prior == "uniform":
            if self.transform_ == "hypercube":
                self._rvs = uniform(0.0, 1.0)
                pipe = [Identity(), HyperCube(low, high)]
                self.transformer = Pipeline(pipe)
            else:
                self._rvs = uniform(self.low, self.high - self.low)
                self.transformer = Identity()

        elif prior == "log-uniform":
            if self.transform_ == "hypercube":
                self._rvs = uniform(0.0, 1.0)
                pipe = [Log10(), HyperCube(np.log10(low), np.log10(high))]
                self.transformer = Pipeline(pipe)
            else:
                self._rvs = uniform(
                    np.log10(self.low),
                    np.log10(self.high) - np.log10(self.low))
                self.transformer = Log10()

        else:
            raise ValueError(
                "Prior should be either 'uniform' or 'log-uniform', "
                "got '%s'." % self._rvs)

    def __eq__(self, other):
        return (type(self) is type(other)
                and np.allclose([self.low], [other.low])
                and np.allclose([self.high], [other.high])
                and self.prior == other.prior
                and self.transform_ == other.transform_)

    def __repr__(self):
        return "Real(low={}, high={}, prior={}, transform={})".format(
            self.low, self.high, self.prior, self.transform_)

    def inverse_transform(self, Xt):
        """Inverse transform samples from the warped space back into the
           orignal space.
        """
        return super(Real, self).inverse_transform(Xt).astype(np.float)

    @property
    def bounds(self):
        return (self.low, self.high)

    @property
    def transformed_bounds(self):
        if self.transform_ == "hypercube":
            return (0.0, 1.0)
        elif self.prior == "uniform":
            return (self.low, self.high)
        else:  # self.prior == "log-uniform"
            return (np.log10(self.low), np.log10(self.high))


class Integer(Dimension):
    def __init__(self, low, high):
        """Search space dimension that can take on integer values.

        Parameters
        ----------
        * `low` [float]:
            Lower bound (inclusive).

        * `high` [float]:
            Upper bound (inclusive).
        """
        self.low = low
        self.high = high
        self._rvs = randint(self.low, self.high + 1)
        self.transformer = Identity()

    def __eq__(self, other):
        return (type(self) is type(other)
                and np.allclose([self.low], [other.low])
                and np.allclose([self.high], [other.high]))

    def __repr__(self):
        return "Integer(low={}, high={})".format(self.low, self.high)

    def inverse_transform(self, Xt):
        """Inverse transform samples from the warped space back into the
           orignal space.
        """
        # The concatenation of all transformed dimensions makes Xt to be
        # of type float, hence the required cast back to int.
        return super(Integer, self).inverse_transform(Xt).astype(np.int)

    @property
    def bounds(self):
        return (self.low, self.high)

    @property
    def transformed_bounds(self):
        return (self.low, self.high)


class Categorical(Dimension):
    def __init__(self, categories, prior=None):
        """Search space dimension that can take on categorical values.

        Parameters
        ----------
        * `categories` [list, shape=(n_categories,)]:
            Sequence of possible categories.

        * `prior` [list, shape=(categories,), default=None]:
            Prior probabilities for each category. By default all categories
            are equally likely.
        """
        self.categories = categories
        self.transformer = CategoricalEncoder()
        self.transformer.fit(self.categories)
        self.prior = prior

        if prior is None:
            self.prior_ = np.tile(1. / len(self.categories), len(self.categories))
        else:
            self.prior_ = prior

        # XXX check that sum(prior) == 1
        self._rvs = rv_discrete(values=(range(len(self.categories)), self.prior_))

    def __eq__(self, other):
        return (type(self) is type(other)
                and self.categories == other.categories
                and np.allclose(self.prior_, other.prior_))

    def __repr__(self):
        if len(self.categories) > 7:
            cats = self.categories[:3] + [_Ellipsis()] + self.categories[-3:]
        else:
            cats = self.categories

        if self.prior is not None and len(self.prior) > 7:
            prior = self.prior[:3] + [_Ellipsis()] + self.prior[-3:]
        else:
            prior = self.prior

        return "Categorical(categories={}, prior={})".format(
            cats, prior)

    def rvs(self, n_samples=None, random_state=None):
        choices = self._rvs.rvs(size=n_samples, random_state=random_state)

        if isinstance(choices, numbers.Integral):
            return self.categories[choices]
        else:
            return [self.categories[c] for c in choices]

    @property
    def transformed_size(self):
        size = len(self.categories)
        # when len(categories) == 2, CategoricalEncoder outputs a single value
        return size if size != 2 else 1

    @property
    def bounds(self):
        return self.categories

    @property
    def transformed_bounds(self):
        if self.transformed_size == 1:
            return (0.0, 1.0)
        else:
            return [(0.0, 1.0) for i in range(self.transformed_size)]

class Space:
    """Search space."""

    def __init__(self, dimensions):
        """Initialize a search space from given specifications.

        Parameters
        ----------
        * `dimensions` [list, shape=(n_dims,)]:
            List of search space dimensions.
            Each search dimension can be defined either as

            - a `(upper_bound, lower_bound)` tuple (for `Real` or `Integer`
              dimensions),
            - a `(upper_bound, lower_bound, "prior")` tuple (for `Real`
              dimensions),
            - as a list of categories (for `Categorical` dimensions), or
            - an instance of a `Dimension` object (`Real`, `Integer` or
              `Categorical`).
        """
        self.dimensions = check_dimensions(dimensions)

    def __eq__(self, other):
        return all([a == b for a, b in zip(self.dimensions, other.dimensions)])

    def __repr__(self):
        if len(self.dimensions) > 31:
            dims = self.dimensions[:15] + [_Ellipsis()] + self.dimensions[-15:]
        else:
            dims = self.dimensions
        return "Space([{}])".format(
            ',\n       '.join(map(str, dims)))

    @property
    def is_real(self):
        """
        Returns true if all dimensions are Real
        """
        return all([isinstance(dim, Real) for dim in self.dimensions])

    def rvs(self, n_samples=1, random_state=None):
        """Draw random samples.

        The samples are in the original space. They need to be transformed
        before being passed to a model or minimizer by `space.transform()`.

        Parameters
        ----------
        * `n_samples` [int, default=1]:
            Number of samples to be drawn from the space.

        * `random_state` [int, RandomState instance, or None (default)]:
            Set random state to something other than None for reproducible
            results.

        Returns
        -------
        * `points`: [list of lists, shape=(n_points, n_dims)]
           Points sampled from the space.
        """
        rng = check_random_state(random_state)

        # Draw
        columns = []

        for dim in self.dimensions:
            if sp_version < (0, 16):
                columns.append(dim.rvs(n_samples=n_samples))
            else:
                columns.append(dim.rvs(n_samples=n_samples, random_state=rng))

        # Transpose
        rows = []

        for i in range(n_samples):
            r = []
            for j in range(self.n_dims):
                r.append(columns[j][i])

            rows.append(r)

        return rows

    def transform(self, X):
        """Transform samples from the original space into a warped space.

        Note: this transformation is expected to be used to project samples
              into a suitable space for numerical optimization.

        Parameters
        ----------
        * `X` [list of lists, shape=(n_samples, n_dims)]:
            The samples to transform.

        Returns
        -------
        * `Xt` [array of floats, shape=(n_samples, transformed_n_dims)]
            The transformed samples.
        """
        # Pack by dimension
        columns = []
        for dim in self.dimensions:
            columns.append([])

        for i in range(len(X)):
            for j in range(self.n_dims):
                columns[j].append(X[i][j])

        # Transform
        for j in range(self.n_dims):
            columns[j] = self.dimensions[j].transform(columns[j])

        # Repack as an array
        Xt = np.hstack([np.asarray(c).reshape((len(X), -1)) for c in columns])
        Xt = Xt.astype(np.float)

        return Xt

    def inverse_transform(self, Xt):
        """Inverse transform samples from the warped space back to the
           original space.

        Parameters
        ----------
        * `Xt` [array of floats, shape=(n_samples, transformed_n_dims)]:
            The samples to inverse transform.

        Returns
        -------
        * `X` [list of lists, shape=(n_samples, n_dims)]
            The original samples.
        """
        # Inverse transform
        columns = []
        start = 0

        for j in range(self.n_dims):
            dim = self.dimensions[j]
            offset = dim.transformed_size

            if offset == 1:
                columns.append(dim.inverse_transform(Xt[:, start]))
            else:
                columns.append(
                    dim.inverse_transform(Xt[:, start:start+offset]))

            start += offset

        # Transpose
        rows = []

        for i in range(len(Xt)):
            r = []
            for j in range(self.n_dims):
                r.append(columns[j][i])

            rows.append(r)

        return rows

    @property
    def n_dims(self):
        """The dimensionality of the original space."""
        return len(self.dimensions)

    @property
    def transformed_n_dims(self):
        """The dimensionality of the warped space."""
        return sum([dim.transformed_size for dim in self.dimensions])

    @property
    def bounds(self):
        """The dimension bounds, in the original space."""
        b = []

        for dim in self.dimensions:
            if dim.size == 1:
                b.append(dim.bounds)
            else:
                b.extend(dim.bounds)

        return b

    @property
    def transformed_bounds(self):
        """The dimension bounds, in the warped space."""
        b = []

        for dim in self.dimensions:
            if dim.transformed_size == 1:
                b.append(dim.transformed_bounds)
            else:
                b.extend(dim.transformed_bounds)

        return b
