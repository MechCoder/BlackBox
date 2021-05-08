"""Test scikit-optimize based implementation of hyperparameter
search with interface similar to those of GridSearchCV
"""

import pytest

import sklearn as skl
from sklearn.datasets import load_iris, make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.base import clone
from sklearn.base import BaseEstimator
from packaging.version import parse as parse_version
from scipy.stats import rankdata
import numpy as np
from numpy.testing import assert_array_equal
from skopt.space import Real, Categorical, Integer
from skopt import BayesSearchCV


def _fit_svc(n_jobs=1, n_points=1, cv=None):
    """
    Utility function to fit a larger classification task with SVC
    """

    X, y = make_classification(n_samples=1000, n_features=20, n_redundant=0,
                               n_informative=18, random_state=1,
                               n_clusters_per_class=1)

    opt = BayesSearchCV(
        SVC(),
        {
            'C': Real(1e-3, 1e+3, prior='log-uniform'),
            'gamma': Real(1e-3, 1e+1, prior='log-uniform'),
            'degree': Integer(1, 3),
        },
        n_jobs=n_jobs, n_iter=11, n_points=n_points, cv=cv,
        random_state=42,
    )

    opt.fit(X, y)
    assert opt.score(X, y) > 0.9

    opt2 = BayesSearchCV(
        SVC(),
        {
            'C': Real(1e-3, 1e+3, prior='log-uniform'),
            'gamma': Real(1e-3, 1e+1, prior='log-uniform'),
            'degree': Integer(1, 3),
        },
        n_jobs=n_jobs, n_iter=11, n_points=n_points, cv=cv,
        random_state=42,
    )

    opt2.fit(X, y)

    assert opt.score(X, y) == opt2.score(X, y)


def test_raise_errors():

    # check if empty search space is raising errors
    with pytest.raises(ValueError):
        BayesSearchCV(SVC(), {})

    # check if invalid dimensions are raising errors
    with pytest.raises(ValueError):
        BayesSearchCV(SVC(), {'C': '1 ... 100.0'})

    with pytest.raises(TypeError):
        BayesSearchCV(SVC(), ['C', (1.0, 1)])


@pytest.mark.parametrize("surrogate", ['gp', None])
@pytest.mark.parametrize("n_jobs", [1, -1])  # test sequential and parallel
@pytest.mark.parametrize("n_points", [1, 3])  # test query of multiple points
def test_searchcv_runs(surrogate, n_jobs, n_points, cv=None):
    """
    Test whether the cross validation search wrapper around sklearn
    models runs properly with available surrogates and with single
    or multiple workers and different number of parameter settings
    to ask from the optimizer in parallel.

    Parameters
    ----------

    * `surrogate` [str or None]:
        A class of the scikit-optimize surrogate used. None means
        to use default surrogate.

    * `n_jobs` [int]:
        Number of parallel processes to use for computations.

    """

    X, y = load_iris(True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=0.75, random_state=0
    )

    # create an instance of a surrogate if it is not a string
    if surrogate is not None:
        optimizer_kwargs = {'base_estimator': surrogate}
    else:
        optimizer_kwargs = None

    opt = BayesSearchCV(
        SVC(),
        {
            'C': Real(1e-6, 1e+6, prior='log-uniform'),
            'gamma': Real(1e-6, 1e+1, prior='log-uniform'),
            'degree': Integer(1, 8),
            'kernel': Categorical(['linear', 'poly', 'rbf']),
        },
        n_jobs=n_jobs, n_iter=11, n_points=n_points, cv=cv,
        optimizer_kwargs=optimizer_kwargs
    )

    opt.fit(X_train, y_train)

    # this normally does not hold only if something is wrong
    # with the optimizaiton procedure as such
    assert opt.score(X_test, y_test) > 0.9


@pytest.mark.slow_test
def test_parallel_cv():

    """
    Test whether parallel jobs work
    """

    _fit_svc(n_jobs=1, cv=5)
    _fit_svc(n_jobs=2, cv=5)


def test_searchcv_runs_multiple_subspaces():
    """
    Test whether the BayesSearchCV runs without exceptions when
    multiple subspaces are given.
    """

    X, y = load_iris(True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=0.75, random_state=0
    )

    # used to try different model classes
    pipe = Pipeline([
        ('model', SVC())
    ])

    # single categorical value of 'model' parameter sets the model class
    lin_search = {
        'model': Categorical([LinearSVC()]),
        'model__C': Real(1e-6, 1e+6, prior='log-uniform'),
    }

    dtc_search = {
        'model': Categorical([DecisionTreeClassifier()]),
        'model__max_depth': Integer(1, 32),
        'model__min_samples_split': Real(1e-3, 1.0, prior='log-uniform'),
    }

    svc_search = {
        'model': Categorical([SVC()]),
        'model__C': Real(1e-6, 1e+6, prior='log-uniform'),
        'model__gamma': Real(1e-6, 1e+1, prior='log-uniform'),
        'model__degree': Integer(1, 8),
        'model__kernel': Categorical(['linear', 'poly', 'rbf']),
    }

    opt = BayesSearchCV(
        pipe,
        [(lin_search, 1), (dtc_search, 1), svc_search],
        n_iter=2
    )

    opt.fit(X_train, y_train)

    # test if all subspaces are explored
    total_evaluations = len(opt.cv_results_['mean_test_score'])
    assert total_evaluations == 1+1+2, "Not all spaces were explored!"
    assert len(opt.optimizer_results_) == 3
    assert isinstance(opt.optimizer_results_[0].x[0], LinearSVC)
    assert isinstance(opt.optimizer_results_[1].x[0], DecisionTreeClassifier)
    assert isinstance(opt.optimizer_results_[2].x[0], SVC)


def test_searchcv_sklearn_compatibility():
    """
    Test whether the BayesSearchCV is compatible with base sklearn methods
    such as clone, set_params, get_params.
    """

    X, y = load_iris(True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=0.75, random_state=0
    )

    # used to try different model classes
    pipe = Pipeline([
        ('model', SVC())
    ])

    # single categorical value of 'model' parameter sets the model class
    lin_search = {
        'model': Categorical([LinearSVC()]),
        'model__C': Real(1e-6, 1e+6, prior='log-uniform'),
    }

    dtc_search = {
        'model': Categorical([DecisionTreeClassifier()]),
        'model__max_depth': Integer(1, 32),
        'model__min_samples_split': Real(1e-3, 1.0, prior='log-uniform'),
    }

    svc_search = {
        'model': Categorical([SVC()]),
        'model__C': Real(1e-6, 1e+6, prior='log-uniform'),
        'model__gamma': Real(1e-6, 1e+1, prior='log-uniform'),
        'model__degree': Integer(1, 8),
        'model__kernel': Categorical(['linear', 'poly', 'rbf']),
    }

    opt = BayesSearchCV(
        pipe,
        [(lin_search, 1), svc_search],
        n_iter=2
    )

    opt_clone = clone(opt)

    params, params_clone = opt.get_params(), opt_clone.get_params()
    assert params.keys() == params_clone.keys()

    for param, param_clone in zip(params.items(), params_clone.items()):
        assert param[0] == param_clone[0]
        assert isinstance(param[1], type(param_clone[1]))

    opt.set_params(search_spaces=[(dtc_search, 1)])

    opt.fit(X_train, y_train)
    opt_clone.fit(X_train, y_train)

    total_evaluations = len(opt.cv_results_['mean_test_score'])
    total_evaluations_clone = len(opt_clone.cv_results_['mean_test_score'])

    # test if expected number of subspaces is explored
    assert total_evaluations == 1
    assert total_evaluations_clone == 1+2


def test_searchcv_reproducibility():
    """
    Test whether results of BayesSearchCV can be reproduced with a fixed
    random state.
    """

    X, y = load_iris(True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=0.75, random_state=0
    )

    random_state = 42

    opt = BayesSearchCV(
        SVC(random_state=random_state),
        {
            'C': Real(1e-6, 1e+6, prior='log-uniform'),
            'gamma': Real(1e-6, 1e+1, prior='log-uniform'),
            'degree': Integer(1, 8),
            'kernel': Categorical(['linear', 'poly', 'rbf']),
        },
        n_iter=11, random_state=random_state
    )

    opt.fit(X_train, y_train)
    best_est = opt.best_estimator_
    optim_res = opt.optimizer_results_[0].x

    opt2 = clone(opt).fit(X_train, y_train)
    best_est2 = opt2.best_estimator_
    optim_res2 = opt2.optimizer_results_[0].x

    assert getattr(best_est, 'C') == getattr(best_est2, 'C')
    assert getattr(best_est, 'gamma') == getattr(best_est2, 'gamma')
    assert getattr(best_est, 'degree') == getattr(best_est2, 'degree')
    assert getattr(best_est, 'kernel') == getattr(best_est2, 'kernel')
    # dict is sorted by alphabet
    assert optim_res[0] == getattr(best_est, 'C')
    assert optim_res[2] == getattr(best_est, 'gamma')
    assert optim_res[1] == getattr(best_est, 'degree')
    assert optim_res[3] == getattr(best_est, 'kernel')
    assert optim_res2[0] == getattr(best_est, 'C')
    assert optim_res2[2] == getattr(best_est, 'gamma')
    assert optim_res2[1] == getattr(best_est, 'degree')
    assert optim_res2[3] == getattr(best_est, 'kernel')


@pytest.mark.fast_test
def test_searchcv_rank():
    """
    Test whether results of BayesSearchCV can be reproduced with a fixed
    random state.
    """

    X, y = load_iris(True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=0.75, random_state=0
    )

    random_state = 42

    opt = BayesSearchCV(
        SVC(random_state=random_state),
        {
            'C': Real(1e-6, 1e+6, prior='log-uniform'),
            'gamma': Real(1e-6, 1e+1, prior='log-uniform'),
            'degree': Integer(1, 8),
            'kernel': Categorical(['linear', 'poly', 'rbf']),
        },
        n_iter=11, random_state=random_state, return_train_score=True
    )

    opt.fit(X_train, y_train)
    results = opt.cv_results_

    test_rank = np.asarray(rankdata(-np.array(results["mean_test_score"]),
                                    method='min'), dtype=np.int32)
    train_rank = np.asarray(rankdata(-np.array(results["mean_train_score"]),
                                     method='min'), dtype=np.int32)

    assert_array_equal(np.array(results['rank_test_score']), test_rank)
    assert_array_equal(np.array(results['rank_train_score']), train_rank)


def test_searchcv_refit():
    """
    Test whether results of BayesSearchCV can be reproduced with a fixed
    random state.
    """

    X, y = load_iris(True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=0.75, random_state=0
    )

    random_state = 42

    opt = BayesSearchCV(
        SVC(random_state=random_state),
        {
            'C': Real(1e-6, 1e+6, prior='log-uniform'),
            'gamma': Real(1e-6, 1e+1, prior='log-uniform'),
            'degree': Integer(1, 8),
            'kernel': Categorical(['linear', 'poly', 'rbf']),
        },
        n_iter=11, random_state=random_state
    )

    opt2 = BayesSearchCV(
        SVC(random_state=random_state),
        {
            'C': Real(1e-6, 1e+6, prior='log-uniform'),
            'gamma': Real(1e-6, 1e+1, prior='log-uniform'),
            'degree': Integer(1, 8),
            'kernel': Categorical(['linear', 'poly', 'rbf']),
        },
        n_iter=11, random_state=random_state, refit=True
    )

    opt.fit(X_train, y_train)
    opt2.best_estimator_ = opt.best_estimator_

    opt2.fit(X_train, y_train)
    # this normally does not hold only if something is wrong
    # with the optimizaiton procedure as such
    assert opt2.score(X_test, y_test) > 0.9


def test_searchcv_callback():
    # Test whether callback is used in BayesSearchCV and
    # whether is can be used to interrupt the search loop

    X, y = load_iris(True)
    opt = BayesSearchCV(
        DecisionTreeClassifier(),
        {
            'max_depth': [3],  # additional test for single dimension
            'min_samples_split': Real(0.1, 0.9),
        },
        n_iter=5
    )
    total_iterations = [0]

    def callback(opt_result):
        # this simply counts iterations
        total_iterations[0] += 1

        # break the optimization loop at some point
        if total_iterations[0] > 2:
            return True  # True == stop optimization

        return False

    opt.fit(X, y, callback=callback)

    assert total_iterations[0] == 3

    # test whether final model was fit
    opt.score(X, y)


def test_searchcv_total_iterations():
    # Test the total iterations counting property of BayesSearchCV

    opt = BayesSearchCV(
        DecisionTreeClassifier(),
        [
            ({'max_depth': (1, 32)}, 10),  # 10 iterations here
            {'min_samples_split': Real(0.1, 0.9)}  # 5 (default) iters here
        ],
        n_iter=5
    )

    assert opt.total_iterations == 10 + 5


def test_search_cv_internal_parameter_types():
    # Test whether the parameters passed to the
    # estimator of the BayesSearchCV are of standard python
    # types - float, int, str

    # This is estimator is used to check whether the types provided
    # are native python types.
    class TypeCheckEstimator(BaseEstimator):
        def __init__(self, float_param=0.0, int_param=0, str_param=""):
            self.float_param = float_param
            self.int_param = int_param
            self.str_param = str_param

        def fit(self, X, y):
            assert isinstance(self.float_param, float)
            assert isinstance(self.int_param, int)
            assert isinstance(self.str_param, str)
            return self

        def score(self, X, y):
            return np.random.uniform()

    # Below is example code that used to not work.
    X, y = make_classification(10, 4)

    model = BayesSearchCV(
        estimator=TypeCheckEstimator(),
        search_spaces={
            'float_param': [0.0, 1.0],
            'int_param': [0, 10],
            'str_param': ["one", "two", "three"],
        },
        n_iter=11
    )

    model.fit(X, y)


def test_searchcv_multimetric_scoring():
    # test that multi-metric scoring works as intened
    # for BayesSearchCV

    random_state = 42

    X, y = make_classification(n_classes=2, random_state=random_state)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=0.75, random_state=0
    )
    # test iterable scoring
    opt = BayesSearchCV(
        SVC(random_state=random_state),
        {
            'C': Real(1e-6, 1e+6, prior='log-uniform'),
            'gamma': Real(1e-6, 1e+1, prior='log-uniform'),
            'degree': Integer(1, 8),
            'kernel': Categorical(['linear', 'poly', 'rbf']),
        },
        scoring=["accuracy", "f1"],
        refit="f1",
        n_iter=11,
        random_state=random_state
    )
    opt.fit(X_train, y_train)
    y_pred = opt.predict(X_test)
    assert opt._target_score == "f1"
    assert f1_score(y_test, y_pred) > 0.9
    assert (
        len(opt.cv_results_["mean_test_accuracy"])
        == len(opt.cv_results_["mean_test_f1"])
    )

    # test dict scoring
    opt = BayesSearchCV(
        SVC(random_state=random_state),
        {
            'C': Real(1e-6, 1e+6, prior='log-uniform'),
            'gamma': Real(1e-6, 1e+1, prior='log-uniform'),
            'degree': Integer(1, 8),
            'kernel': Categorical(['linear', 'poly', 'rbf']),
        },
        scoring={
            "f1": "f1",
            "accuracy": "accuracy",
        },
        refit="f1",
        n_iter=11,
        random_state=random_state
    )
    opt.fit(X_train, y_train)
    y_pred = opt.predict(X_test)
    assert opt._target_score == "f1"
    assert f1_score(y_test, y_pred) > 0.9
    assert (
        len(opt.cv_results_["mean_test_accuracy"])
        == len(opt.cv_results_["mean_test_f1"])
    )

    # test callable scoring
    # Scoring functions returning dicts are only supported in sklearn >=0.24
    if parse_version(skl.__version__) >= parse_version("0.24dev0"):
        # sample code taken from scikit-learn
        def confusion_matrix_score(clf, X, y):
            y_pred = clf.predict(X)
            cm = confusion_matrix(y, y_pred)
            return {'tn': cm[0, 0], 'fp': cm[0, 1],
                    'fn': cm[1, 0], 'tp': cm[1, 1]}
        opt = BayesSearchCV(
            SVC(random_state=random_state),
            {
                'C': Real(1e-6, 1e+6, prior='log-uniform'),
                'gamma': Real(1e-6, 1e+1, prior='log-uniform'),
                'degree': Integer(1, 8),
                'kernel': Categorical(['linear', 'poly', 'rbf']),
            },
            scoring=confusion_matrix_score,
            refit="tp",
            n_iter=11,
            random_state=random_state
        )
        opt.fit(X_train, y_train)
        assert opt._target_score == "tp"
        assert confusion_matrix_score(opt, X_test, y_test)["tp"] > 0.9
        assert (
            len(opt.cv_results_["mean_test_tp"])
            == len(opt.cv_results_["mean_test_fp"])
        )
