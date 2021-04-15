#!/bin/bash
#
# Copied from https://github.com/scikit-learn/scikit-learn/blob/master/build_tools/travis/test_script.sh
# The scikit-learn developers.
# License: BSD-style
#
# This script is meant to be called by the "script" step defined in
# .travis.yml. See https://docs.travis-ci.com/ for more details.
# The behavior of the script is controlled by environment variabled defined
# in the .travis.yml in the top level folder of the project.

# License: 3-clause BSD

set -e

python --version
python -c "import numpy; print('numpy %s' % numpy.__version__)"
python -c "import scipy; print('scipy %s' % scipy.__version__)"
python -c "import sklearn; print('sklearn %s' % sklearn.__version__)"
python -c "\
try:
    import skopt
    print('skopt %s' % skopt.__version__)
except ImportError:
    pass
"
python -c "import multiprocessing as mp; print('%d CPUs' % mp.cpu_count())"

run_tests() {
    TEST_CMD="pytest --showlocals --durations=20 --pyargs"

    # Get into a temp directory to run test from the installed scikit-learn and
    # check if we do not leave artifacts
    mkdir -p $TEST_DIR
    # We need the setup.cfg for the pytest settings
    cp setup.cfg $TEST_DIR
    cd $TEST_DIR

    # Skip tests that require large downloads over the network to save bandwidth
    # usage as travis workers are stateless and therefore traditional local
    # disk caching does not work.
    export SKOPT_SKIP_NETWORK_TESTS=1

    if [[ "$COVERAGE" == "true" ]]; then
        TEST_CMD="$TEST_CMD --cov skopt"
    fi

    if [[ -n "$CHECK_WARNINGS" ]]; then
        TEST_CMD="$TEST_CMD -Werror::DeprecationWarning -Werror::FutureWarning"
    fi

    set -x  # print executed commands to the terminal

    $TEST_CMD skopt
}

run_package_check() {

    TEST_CMD="twine check dist/*"
    set -x
    $TEST_CMD
}

if [[ "$SDIST" == "true" ]]; then
    run_package_check
else
    run_tests
fi