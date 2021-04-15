#!/bin/bash

set -e
set -x
if [[ "$SDIST" != "true" ]]; then
    make test-doc
fi
