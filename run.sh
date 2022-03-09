#!/bin/bash
set -e
set -x

virtualenv -p python3
source ./bin/activate

pip install -e jax_dft
python3 -m examples.solve_non_interacting_system