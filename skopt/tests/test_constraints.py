
import pytest
from pytest import raises
import numpy as np
from skopt.space.constraints import Constraints, Single, Exclusive, Inclusive, Sum, Conditional, \
    check_constraints, check_bounds, check_value
from skopt import Optimizer
from skopt.space import Real, Integer, Categorical, Space

from sklearn.utils.testing import assert_false, assert_not_equal
from numpy.testing import assert_array_equal
from numpy.testing import assert_equal

# Combination of base estimators and acquisition optimizers
ACQ_OPTIMIZERS = ['sampling', 'lbfgs']


@pytest.mark.fast_test
def test_constraints_equality():
    # Same constraints should be equal
    space_a = Space([(0.0, 5.0), (1.0, 5.0)])
    space_b = Space([(0.0, 5.0), (1.0, 5.0)])
    cons_list_a = [Single(0, 4.0, 'real'), Single(1, 4.0, 'real')]
    cons_list_b = [Single(0, 4.0, 'real'), Single(1, 4.0, 'real')]
    cons_a = Constraints(cons_list_a, space_a)
    cons_b = Constraints(cons_list_b, space_b)
    assert_equal(cons_a, cons_b)

    # Different lengths of constraints_list should not be equal
    space_a = Space([(0.0, 5.0)])
    space_b = Space([(0.0, 5.0), (1.0, 5.0)])
    cons_list_a = [Single(0, 4.0, 'real')]
    cons_list_b = [Single(0, 4.0, 'real'), Single(1, 4.0, 'real')]
    cons_a = Constraints(cons_list_a, space_a)
    cons_b = Constraints(cons_list_b, space_b)
    assert_not_equal(cons_a, cons_b)

    # Different dimension types in constraints_list should not be equal
    space_a = Space([(0.0, 5.0)])
    space_b = Space([(0, 5)])
    cons_list_a = [Single(0, 4.0, 'real')]
    cons_list_b = [Single(0, 4, 'integer')]
    cons_a = Constraints(cons_list_a, space_a)
    cons_b = Constraints(cons_list_b, space_b)
    assert_not_equal(cons_a, cons_b)

    # Different values in constraints should not be equal
    space_a = Space([(0.0, 5.0)])
    space_b = Space([(0.0, 5.0)])
    cons_list_a = [Single(0, 4.0, 'real')]
    cons_list_b = [Single(0, 4.1, 'real')]
    cons_a = Constraints(cons_list_a, space_a)
    cons_b = Constraints(cons_list_b, space_b)
    assert_not_equal(cons_a, cons_b)


@pytest.mark.fast_test
def test_single_inclusive_and_exclusive():
    # Test that valid constraints can be initialized
    Single(0, 1.0, 'real')
    Single(0, -1, 'integer')
    Single(0, 'a', 'categorical')
    Inclusive(0, (1.0, 2.0), 'real')
    Inclusive(0, [1.0, 2.0], 'real')
    Inclusive(0, (-1, 2), 'integer')
    Inclusive(0, ('a', 'b', 'c', 1, 0.1), 'categorical')
    Exclusive(0, (1.0, 2.0), 'real')
    Exclusive(0, [1.0, 2.0], 'real')
    Exclusive(0, (-1, 1), 'integer')
    Exclusive(0, ('a', 'b', 'c', 1, 0.1), 'categorical')

    # A tuple or list should be passed
    with raises(TypeError):
        Inclusive(0, 'a', 'real')
    with raises(TypeError):
        Exclusive(0, dict(), 'real')

    # Length of bounds should be 2
    with raises(ValueError):
        Inclusive(0, [0], 'integer')
    with raises(ValueError):
        Inclusive(0, (0, 1, 2), 'integer')

    # Dimenion type and bounds should be of the same type i.e 'real' -> float, 'integer' -> int
    with raises(TypeError):
        Single(0, 1, 'real')
    with raises(TypeError):
        Single(0, 1.0, 'integer')
    with raises(TypeError):
        Inclusive(0, (1, 2.0), 'real')
    with raises(TypeError):
        Inclusive(0, (1.0, 2), 'integer')

    # Dimension should be int
    with raises(TypeError):
        Single('a', 1.0, 'real')
    with raises(TypeError):
        Inclusive(0.1, (1.0, 2.0), 'real')
    with raises(TypeError):
        Exclusive('b', (1.0, 2.0), 'real')

    # Dimension should not be negative
    with raises(ValueError):
        Single(-1, 1.0, 'real')
    with raises(ValueError):
        Inclusive(-1, (1.0, 2.0), 'real')
    with raises(ValueError):
        Exclusive(-1, (1.0, 2.0), 'real')

    # Dimension_type should be valid
    with raises(ValueError):
        Single('a', 1.0, 'not a proper value')


@pytest.mark.fast_test
def test_Sum():
    # Test that Sym type constraintcan be initialized
    Sum((1, 2, 3), 5, less_than=False)
    Sum([3, 2, 1], -10.0, less_than=True)

    with raises(TypeError):
        Sum('a', 5)
    # `less_than` should be a bool
    with raises(TypeError):
        Sum((1, 2, 3), 5, less_than=1)
    # Value should be int or float
    with raises(TypeError):
        Sum((1, 2, 3), True)
    # Dimensions should be of lenght > 1
    with raises(ValueError):
        Sum([0], True)
    # Dimensions can not be a negative int
    with raises(ValueError):
        Sum([-10, 1, 2], True)

    space = Space([[0.0, 10.0], [0, 10], ['abcdef']])
    # A dimension value of 4 is out of bounds for a space with only 3 dimensions
    cons_list = [Sum((4, 3), 5)]
    with raises(IndexError):
        Constraints(cons_list, space)
    # Cannot apply sum constraint to categorical dimensions
    cons_list = [Sum((1, 2), 5)]
    with raises(ValueError):
        Constraints(cons_list, space)

    # Check that validate_sample validates samples correctly
    cons = Constraints([Sum((0, 1), 6)], space)
    assert_false(cons.validate_sample([0.0, 7, 'a']))
    assert_false(cons.validate_sample([7.0, 0, 'a']))
    assert_false(cons.validate_sample([3.00001, 3, 'a']))
    assert cons.validate_sample([2.99999, 3, 'a'])

    # Check that validate_sample validates samples correctly for less_than = False
    cons = Constraints([Sum((0, 1), 6, less_than=False)], space)
    assert cons.validate_sample([0.0, 7, 'a'])
    assert cons.validate_sample([7.0, 0, 'a'])
    assert cons.validate_sample([3.00001, 3, 'a'])
    assert_false(cons.validate_sample([2.99999, 3, 'a']))

    # Check that only valid samples are drawn
    samples = cons.rvs(n_samples=1000)
    for sample in samples:
        assert cons.validate_sample(sample)


@pytest.mark.fast_test
def test_Conditional():
    # Test conditional constraints
    space = Space([list('ab'), list('ab'), list('ab'), [1, 4], [1, 4], [1, 4]])
    condition = Single(0, 'a', 'categorical')
    if_true = Inclusive(3, [1, 2], 'integer')
    if_false = Exclusive(3, [1, 2], 'integer')

    # Test init of constriants
    Conditional(condition, if_true, if_false)
    Conditional(condition, if_true=if_true, if_false=if_false)
    cons_list = [Conditional(condition, if_true=if_true, if_false=if_false)]
    cons = Constraints(cons_list, space)

    # Check that only valid samples are being drawn
    samples = cons.rvs(n_samples=100)
    for sample in samples:
        if sample[0] == 'a':
            assert sample[3] < 3
        else:
            assert sample[3] > 2

    # Test that validate_sample works
    assert_false(cons.validate_sample(['a', 'a', 'a', 3, 3, 3]))
    assert cons.validate_sample(['a', 'a', 'a', 2, 3, 3])
    assert_false(cons.validate_sample(['b', 'a', 'a', 2, 3, 3]))
    assert cons.validate_sample(['b', 'a', 'a', 3, 3, 3])

    # Test list of constraints
    cons = Constraints([
        Conditional(
            Single(0, 'a', 'categorical'),
            if_true=[Single(1, 'a', 'categorical'), Single(2, 'a', 'categorical')],
            if_false=[Single(1, 'b', 'categorical'), Single(2, 'b', 'categorical')]
        )
    ], space)
    assert_false(cons.validate_sample(['a', 'a', 'b', 3, 3, 3]))
    assert cons.validate_sample(['a', 'a', 'a', 3, 3, 3])
    assert_false(cons.validate_sample(['b', 'a', 'b', 3, 3, 3]))
    assert cons.validate_sample(['b', 'b', 'b', 3, 3, 3])

    # Test nested contraints
    cons = Constraints([
        Conditional(
            Inclusive(3, [1, 2], 'integer'),
            Conditional(
                Single(0, 'a', 'categorical'),
                if_true=[Single(1, 'a', 'categorical'), Single(2, 'a', 'categorical')],
                if_false=[Single(1, 'b', 'categorical'), Single(2, 'b', 'categorical')]
            ),
            Conditional(
                Single(0, 'a', 'categorical'),
                if_true=[Single(1, 'b', 'categorical'), Single(2, 'b', 'categorical')],
                if_false=[Single(1, 'a', 'categorical'), Single(2, 'a', 'categorical')]
            ),
        )
    ], space)

    assert_false(cons.validate_sample(['a', 'a', 'b', 2, 3, 3]))
    assert cons.validate_sample(['a', 'a', 'a', 2, 3, 3])
    assert_false(cons.validate_sample(['b', 'a', 'b', 2, 3, 3]))
    assert cons.validate_sample(['b', 'b', 'b', 2, 3, 3])

    assert cons.validate_sample(['a', 'b', 'b', 3, 3, 3])
    assert_false(cons.validate_sample(['a', 'a', 'a', 3, 3, 3]))
    assert cons.validate_sample(['b', 'a', 'a', 3, 3, 3])
    assert_false(cons.validate_sample(['b', 'b', 'b', 3, 3, 3]))

    samples = cons.rvs(n_samples=100)
    for sample in samples:
        assert cons.validate_sample(sample)


@pytest.mark.fast_test
def test_check_constraints():
    space = Space([(0.0, 5.0), (1.0, 5.0)])
    cons_list = Single(0, 1, 'integer')

    # Constraints_list must be a list
    with raises(TypeError):
        check_constraints(space, cons_list)

    # Constraints dimension must be less than number of dimensions in space
    cons_list = [Single(2, 1, 'integer')]
    with raises(IndexError):
        check_constraints(space, cons_list)

    # Constraints dimension_types must be the same as corresponding dimension in space
    space = Space([(0.0, 5.0), (0, 5), ['a', 'b', 'c']])
    cons_list = [Single(2, 1, 'integer')]
    with raises(TypeError):
        check_constraints(space, cons_list)
    cons_list = [Single(1, 1.0, 'real')]
    with raises(TypeError):
        check_constraints(space, cons_list)
    cons_list = [Single(0, 'a', 'categorical')]
    with raises(TypeError):
        check_constraints(space, cons_list)

    # Only one Single constraint per dimension
    cons_list = [Single(0, 1.0, 'real'), Single(0, 2.0, 'real')]
    with raises(IndexError):
        check_constraints(space, cons_list)


@pytest.mark.fast_test
def test_check_bounds():
    # Check that no error is raised when using valid bounds
    space = Space([(0.0, 5.0), (0, 5), ['a', 'b', 'c', 1, 1.0]])
    check_bounds(space.dimensions[0], (1.0, 2.0))
    check_bounds(space.dimensions[1], (1, 3))
    check_bounds(space.dimensions[2], ('a', 'b'))
    check_bounds(space.dimensions[2], ('a', 1.0))
    # Check that error is raised when using invalid bounds
    with raises(ValueError):
        check_bounds(space.dimensions[0], (-1.0, 2.0))
    with raises(ValueError):
        check_bounds(space.dimensions[0], (2.0, 10.0))
    with raises(ValueError):
        check_bounds(space.dimensions[1], (-1, 2))
    with raises(ValueError):
        check_bounds(space.dimensions[1], (2, 10))
    with raises(ValueError):
        check_bounds(space.dimensions[2], ('k', -1.0))
    with raises(ValueError):
        check_bounds(space.dimensions[2], ('a', 'b', 'c', 1.2))


@pytest.mark.fast_test
def test_check_value():
    # Test that no error is raised when using a valid value
    space = Space([(0.0, 5.0), (0, 5), ['a', 'b', 'c', 1, 1.0]])
    check_value(space.dimensions[0], 1.0)
    check_value(space.dimensions[1], 1)
    check_value(space.dimensions[2], 'b')
    check_value(space.dimensions[2], 1)
    check_value(space.dimensions[2], 1.0)

    # Test that error is raised when using an invalid value
    with raises(ValueError):
        check_value(space.dimensions[0], 10.0)
    with raises(ValueError):
        check_value(space.dimensions[0], -1.0)
    with raises(ValueError):
        check_value(space.dimensions[1], 10)
    with raises(ValueError):
        check_value(space.dimensions[1], -1)
    with raises(ValueError):
        check_value(space.dimensions[2], 'wow')
    with raises(ValueError):
        check_value(space.dimensions[2], 1.2)


@pytest.mark.fast_test
def test_single_validate_constraint():
    # Test categorical
    cons = Single(0, 1.0, 'categorical')
    assert cons._validate_constraint(1.0)
    assert_false(cons._validate_constraint(1.1))

    cons = Single(0, 'a', 'categorical')
    assert cons._validate_constraint('a')
    assert_false(cons._validate_constraint('b'))

    # Test real
    cons = Single(0, 1.0, 'real')
    assert cons._validate_constraint(1.0)
    assert_false(cons._validate_constraint(1.1))

    # Test integer
    cons = Single(0, 1, 'integer')
    assert cons._validate_constraint(1)
    assert_false(cons._validate_constraint(2))


@pytest.mark.fast_test
def test_Constraints_init():
    space = Space([
        Real(1, 10),
        Real(1, 10),
        Real(1, 10),
        Integer(0, 10),
        Integer(0, 10),
        Integer(0, 10),
        Categorical(list('abcdefg')),
        Categorical(list('abcdefg')),
        Categorical(list('abcdefg'))
    ])

    cons_list = [Single(0, 5.0, 'real'),
                 Inclusive(1, (3.0, 5.0), 'real'),
                 Exclusive(2, (3.0, 5.0), 'real'),
                 Single(3, 5, 'integer'),
                 Inclusive(4, (3, 5), 'integer'),
                 Exclusive(5, (3, 5), 'integer'),
                 Single(6, 'b', 'categorical'),
                 Inclusive(7, ('c', 'd', 'e'), 'categorical'),
                 Exclusive(8, ('c', 'd', 'e'), 'categorical'),
                 # Note that two ocnstraints are being added to dimension 4 and 5
                 Inclusive(4, (7, 9), 'integer'),
                 Exclusive(5, (7, 9), 'integer'),
                 ]

    cons = Constraints(cons_list, space)

    # Test that space and constriants_list are being saved in object
    assert_equal(cons.space, space)
    assert_equal(cons.constraints_list, cons_list)

    # Test that a correct list of single constraints have been made
    assert_equal(len(cons.single), space.n_dims)
    assert_equal(cons.single[1], None)
    assert_equal(cons.single[-1], None)
    assert_not_equal(cons.single[0], None)
    assert_not_equal(cons.single[6], None)

    # Test that a correct list of inclusive constraints have been made
    assert_equal(len(cons.inclusive), space.n_dims)
    assert_equal(cons.inclusive[0], [])
    assert_equal(cons.inclusive[2], [])
    assert_not_equal(not cons.inclusive[1], [])
    assert_not_equal(not cons.inclusive[7], [])
    assert_equal(len(cons.inclusive[4]), 2)

    # Test that a correct list of exclusive constraints have been made
    assert_equal(len(cons.exclusive), space.n_dims)
    assert_equal(cons.exclusive[3], [])
    assert_equal(cons.exclusive[7], [])
    assert_not_equal(cons.exclusive[2], [])
    assert_not_equal(cons.exclusive[5], [])
    assert_equal(len(cons.exclusive[5]), 2)


@pytest.mark.fast_test
def test_Constraints_validate_sample():
    space = Space([
        Real(1, 10),
        Real(1, 10),
        Real(1, 10),
        Integer(0, 10),
        Integer(0, 10),
        Integer(0, 10),
        Categorical(list('abcdefg')),
        Categorical(list('abcdefg')),
        Categorical(list('abcdefg'))
    ])

    # Test validation of single constraints
    cons_list = [Single(0, 5.0, 'real')]
    cons = Constraints(cons_list, space)
    sample = [0] * space.n_dims
    sample[0] = 5.0
    assert cons.validate_sample(sample)
    sample[0] = 5.00001
    assert_false(cons.validate_sample(sample))
    sample[0] = 4.99999
    assert_false(cons.validate_sample(sample))

    cons_list = [Single(3, 5, 'integer')]
    cons = Constraints(cons_list, space)
    sample = [0] * space.n_dims
    sample[3] = 5
    assert cons.validate_sample(sample)
    sample[3] = 6
    assert_false(cons.validate_sample(sample))
    sample[3] = -5
    assert_false(cons.validate_sample(sample))
    sample[3] = 5.000001
    assert_false(cons.validate_sample(sample))

    cons_list = [Single(6, 'a', 'categorical')]
    cons = Constraints(cons_list, space)
    sample = [0] * space.n_dims
    sample[6] = 'a'
    assert cons.validate_sample(sample)
    sample[6] = 'b'
    assert_false(cons.validate_sample(sample))
    sample[6] = -5
    assert_false(cons.validate_sample(sample))
    sample[6] = 5.000001
    assert_false(cons.validate_sample(sample))

    # Test validation of inclusive constraints
    cons_list = [Inclusive(0, (5.0, 7.0), 'real')]
    cons = Constraints(cons_list, space)
    sample = [0] * space.n_dims
    sample[0] = 5.0
    assert cons.validate_sample(sample)
    sample[0] = 7.0
    assert cons.validate_sample(sample)
    sample[0] = 7.00001
    assert_false(cons.validate_sample(sample))
    sample[0] = 4.99999
    assert_false(cons.validate_sample(sample))
    sample[0] = -10
    assert_false(cons.validate_sample(sample))

    cons_list = [Inclusive(3, (5, 7), 'integer')]
    cons = Constraints(cons_list, space)
    sample = [0] * space.n_dims
    sample[3] = 5
    assert cons.validate_sample(sample)
    sample[3] = 6
    assert cons.validate_sample(sample)
    sample[3] = 7
    assert cons.validate_sample(sample)
    sample[3] = 8
    assert_false(cons.validate_sample(sample))
    sample[3] = 4
    assert_false(cons.validate_sample(sample))
    sample[3] = -4
    assert_false(cons.validate_sample(sample))

    cons_list = [Inclusive(6, ('c', 'd', 'e'), 'categorical')]
    cons = Constraints(cons_list, space)
    sample = [0] * space.n_dims
    sample[6] = 'c'
    assert cons.validate_sample(sample)
    sample[6] = 'e'
    assert cons.validate_sample(sample)
    sample[6] = 'f'
    assert_false(cons.validate_sample(sample))
    sample[6] = -5
    assert_false(cons.validate_sample(sample))
    sample[6] = 3.3
    assert_false(cons.validate_sample(sample))
    sample[6] = 'a'
    assert_false(cons.validate_sample(sample))

    # Test validation of exclusive constraints
    cons_list = [Exclusive(0, (5.0, 7.0), 'real')]
    cons = Constraints(cons_list, space)
    sample = [0] * space.n_dims
    sample[0] = 5.0
    assert_false(cons.validate_sample(sample))
    sample[0] = 7.0
    assert_false(cons.validate_sample(sample))
    sample[0] = 7.00001
    assert cons.validate_sample(sample)
    sample[0] = 4.99999
    assert cons.validate_sample(sample)
    sample[0] = -10
    assert cons.validate_sample(sample)

    cons_list = [Exclusive(3, (5, 7), 'integer')]
    cons = Constraints(cons_list, space)
    sample = [0] * space.n_dims
    sample[3] = 5
    assert_false(cons.validate_sample(sample))
    sample[3] = 6
    assert_false(cons.validate_sample(sample))
    sample[3] = 7
    assert_false(cons.validate_sample(sample))
    sample[3] = 8
    assert cons.validate_sample(sample)
    sample[3] = 4
    assert cons.validate_sample(sample)
    sample[3] = -4
    assert cons.validate_sample(sample)

    cons_list = [Exclusive(3, (5, 5), 'integer')]
    cons = Constraints(cons_list, space)
    sample = [0] * space.n_dims
    sample[3] = 5
    assert_false(cons.validate_sample(sample))

    cons_list = [Exclusive(6, ('c', 'd', 'e'), 'categorical')]
    cons = Constraints(cons_list, space)
    sample = [0] * space.n_dims
    sample[6] = 'c'
    assert_false(cons.validate_sample(sample))
    sample[6] = 'e'
    assert_false(cons.validate_sample(sample))
    sample[6] = 'f'
    assert cons.validate_sample(sample)
    sample[6] = -5
    assert cons.validate_sample(sample)
    sample[6] = 3.3
    assert cons.validate_sample(sample)
    sample[6] = 'a'
    assert cons.validate_sample(sample)

    # Test more than one constraint per dimension
    cons_list = [Inclusive(0, (1.0, 2.0), 'real'), Inclusive(0, (3.0, 4.0), 'real'), Inclusive(0, (5.0, 6.0), 'real')]
    cons = Constraints(cons_list, space)
    sample = [0] * space.n_dims
    sample[0] = 1.3
    assert cons.validate_sample(sample)
    sample[0] = 6.0
    assert cons.validate_sample(sample)
    sample[0] = 5.0
    assert cons.validate_sample(sample)
    sample[0] = 3.0
    assert cons.validate_sample(sample)
    sample[0] = 4.0
    assert cons.validate_sample(sample)
    sample[0] = 5.5
    assert cons.validate_sample(sample)
    sample[0] = 2.1
    assert_false(cons.validate_sample(sample))
    sample[0] = 4.9
    assert_false(cons.validate_sample(sample))
    sample[0] = 7.0
    assert_false(cons.validate_sample(sample))

    cons_list = [Exclusive(0, (1.0, 2.0), 'real'), Exclusive(0, (3.0, 4.0), 'real'), Exclusive(0, (5.0, 6.0), 'real')]
    cons = Constraints(cons_list, space)
    sample = [0] * space.n_dims
    sample[0] = 1.3
    assert_false(cons.validate_sample(sample))
    sample[0] = 6.0
    assert_false(cons.validate_sample(sample))
    sample[0] = 5.0
    assert_false(cons.validate_sample(sample))
    sample[0] = 3.0
    assert_false(cons.validate_sample(sample))
    sample[0] = 4.0
    assert_false(cons.validate_sample(sample))
    sample[0] = 5.5
    assert_false(cons.validate_sample(sample))
    sample[0] = 2.1
    assert cons.validate_sample(sample)
    sample[0] = 4.9
    assert cons.validate_sample(sample)
    sample[0] = 7.0
    assert cons.validate_sample(sample)


@pytest.mark.slow_test
def test_constraints_rvs():
    space = Space([
        Real(1, 10),
        Real(1, 10),
        Real(1, 10),
        Integer(0, 10),
        Integer(0, 10),
        Integer(0, 10),
        Categorical(list('abcdefg')),
        Categorical(list('abcdefg')),
        Categorical(list('abcdefg'))
    ])

    cons_list = [Single(0, 5.0, 'real'),
                 Inclusive(1, (3.0, 5.0), 'real'),
                 Exclusive(2, (3.0, 5.0), 'real'),
                 Single(3, 5, 'integer'),
                 Inclusive(4, (3, 5), 'integer'),
                 Exclusive(5, (3, 5), 'integer'),
                 Single(6, 'b', 'categorical'),
                 Inclusive(7, ('c', 'd', 'e'), 'categorical'),
                 Exclusive(8, ('c', 'd', 'e'), 'categorical'),
                 # Note that two constraints are being added to dimension 4 and 5
                 Inclusive(4, (7, 9), 'integer'),
                 Exclusive(5, (7, 9), 'integer'),
                 ]

    # Test lenght of samples
    constraints = Constraints(cons_list, space)
    samples = constraints.rvs(n_samples=100)
    assert_equal(len(samples), 100)
    assert_equal(len(samples[0]), space.n_dims)
    assert_equal(len(samples[-1]), space.n_dims)

    # Test random state
    samples_a = constraints.rvs(n_samples=100, random_state=1)
    samples_b = constraints.rvs(n_samples=100, random_state=1)
    samples_c = constraints.rvs(n_samples=100, random_state=2)
    assert_equal(samples_a, samples_b)
    assert_not_equal(samples_a, samples_c)

    # Test invalid constraint combinations
    space = Space([Real(0, 1)])
    cons_list = [Exclusive(0, (0.3, 0.7), 'real'), Inclusive(0, (0.5, 0.6), 'real')]
    constraints = Constraints(cons_list, space)
    with raises(RuntimeError):
        samples = constraints.rvs(n_samples=10)


@pytest.mark.slow_test
@pytest.mark.parametrize('acq_optimizer', ACQ_OPTIMIZERS)
def test_optimizer_with_constraints(acq_optimizer):
    base_estimator = 'GP'
    space = Space([
        Real(1, 10),
        Real(1, 10),
        Real(1, 10),
        Integer(0, 10),
        Integer(0, 10),
        Integer(0, 10),
        Categorical(list('abcdefg')),
        Categorical(list('abcdefg')),
        Categorical(list('abcdefg'))
    ])

    cons_list = [Single(0, 5.0, 'real'), Single(3, 5, 'integer')]
    cons_list_2 = [Single(0, 4.0, 'real'), Single(3, 4, 'integer')]
    cons = Constraints(cons_list, space)
    cons_2 = Constraints(cons_list_2, space)

    # Test behavior when not adding constraitns
    opt = Optimizer(space, base_estimator, acq_optimizer=acq_optimizer, n_initial_points=5)

    # Test that constraint is None when no constraint has been set so far
    assert_equal(opt._constraints, None)

    # Test constraints are still None
    for _ in range(6):
        next_x = opt.ask()
        f_val = np.random.random() * 100
        opt.tell(next_x, f_val)
    assert_equal(opt._constraints, None)
    opt.remove_constraints()
    assert_equal(opt._constraints, None)

    # Test behavior when adding constraints in an optimization setting
    opt = Optimizer(space, base_estimator, acq_optimizer=acq_optimizer, n_initial_points=3)
    opt.set_constraints(cons)
    assert_equal(opt._constraints, cons)
    next_x = opt.ask()
    assert_equal(next_x[0], 5.0)
    assert_equal(next_x[3], 5)
    f_val = np.random.random() * 100
    opt.tell(next_x, f_val)
    assert_equal(opt._constraints, cons)
    opt.set_constraints(cons_2)
    next_x = opt.ask()
    assert_equal(opt._constraints, cons_2)
    assert_equal(next_x[0], 4.0)
    assert_equal(next_x[3], 4)
    f_val = np.random.random() * 100
    opt.tell(next_x, f_val)
    assert_equal(opt._constraints, cons_2)
    opt.remove_constraints()
    assert_equal(opt._constraints, None)
    next_x = opt.ask()
    assert_not_equal(next_x[0], 4.0)
    assert_not_equal(next_x[0], 5.0)
    f_val = np.random.random() * 100
    opt.tell(next_x, f_val)
    assert_equal(opt._constraints, None)

    # Test that next_x is changed when adding constraints
    opt = Optimizer(space, base_estimator, acq_optimizer=acq_optimizer, n_initial_points=3)
    assert_false(hasattr(opt, '_next_x'))
    for _ in range(4):  # We exhaust initial points
        next_x = opt.ask()
        f_val = np.random.random() * 100
        opt.tell(next_x, f_val)
    assert hasattr(opt, '_next_x')  # Now next_x should be in optimizee
    assert_not_equal(next_x[0], 4.0)
    assert_not_equal(next_x[0], 5.0)
    next_x = opt._next_x
    opt.set_constraints(cons)
    assert_not_equal(opt._next_x, next_x)  # Check that next_x has been changed
    assert_equal(opt._next_x[0], 5.0)
    assert_equal(opt._next_x[3], 5)
    next_x = opt._next_x
    opt.set_constraints(cons_2)
    assert_not_equal(opt._next_x, next_x)
    assert_equal(opt._next_x[0], 4.0)
    assert_equal(opt._next_x[3], 4)

    # Test that adding a Constraint or constraint_list gives the same
    opt = Optimizer(space, base_estimator, acq_optimizer=acq_optimizer, n_initial_points=3)
    opt.set_constraints(cons_list)
    opt2 = Optimizer(space, base_estimator, acq_optimizer=acq_optimizer, n_initial_points=3)
    opt2.set_constraints(cons)
    assert_equal(opt._constraints, opt2._constraints)

    # Test that constraints are satisfied
    opt = Optimizer(space, base_estimator, acq_optimizer=acq_optimizer, n_initial_points=2)
    opt.set_constraints(cons)
    next_x = opt.ask()
    assert_equal(next_x[0], 5.0)

    opt = Optimizer(space, base_estimator, acq_optimizer=acq_optimizer, n_initial_points=2)
    next_x = opt.ask()
    assert_not_equal(next_x[0], 5.0)
    f_val = np.random.random() * 100
    opt.tell(next_x, f_val)
    opt.set_constraints(cons)
    next_x = opt.ask()
    assert_equal(next_x[0], 5.0)
    assert_equal(next_x[3], 5)
    opt.set_constraints(cons)
    next_x = opt.ask()
    f_val = np.random.random() * 100
    opt.tell(next_x, f_val)
    opt.set_constraints(cons_2)
    next_x = opt.ask()
    assert_equal(next_x[0], 4.0)
    assert_equal(next_x[3], 4)
    f_val = np.random.random() * 100
    opt.tell(next_x, f_val)
    assert_equal(next_x[0], 4.0)
    assert_equal(next_x[3], 4)


@pytest.mark.fast_test
def test_get_constraints():
    # Test that the get_constraints() function returns the constraints that were set byt set_constraints()
    space = Space([Real(1, 10)])
    cons_list = [Single(0, 5.0, 'real')]
    cons = Constraints(cons_list, space)
    opt = Optimizer(space, "ET")
    opt.set_constraints(cons)
    assert_equal(cons, opt.get_constraints())

