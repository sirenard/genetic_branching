# genetic_branching

## Installation

Configure environment
```shell
conda install pip scip==9.1.0 pybind11 fmt
export CMAKE_PREFIX_PATH="${CONDA_PREFIX}" 
export CPLUS_INCLUDE_PATH="${CONDA_PREFIX}/include/"
export LIBRARY_PATH=${CONDA_PREFIX}/lib
export LD_LIBRARY_PATH=${CONDA_PREFIX}/lib
```

Install pip dependencies
```shell
pip install boundml
pip install deap objproxies mpipool
```

Build c++
```shell
mkdir build
cmake ..
make install
```

## Features of interest

### Static features

- Objective coefficient (1)
- Number of constraint in which coef is non-zero (1)
- Degree of a constraint: For positive/negative matrix coefficient the variable participates (count, mean) (4)
- Statistics on constraints degrees (considering only constraint in which the var is active) (mean, stdev, min, max) (4)

### Tree features

- Gap
- Leaf Frequency
- Open Node
- Sum Of Subtree Gaps
- Tree Weight

### Dynamic features

- Pseudocosts: upward and downward values weighted by x fractionality of x_j
- reducedcost
- Dynamic statistics on constraints degree: A dynamic variant of the static version above. Here, the constraint degrees
  are on the current node’s LP.
- Minimum and maximum ratios across positive and negative right-hand-sides (RHS)
- Min/max for one-to-all coefficient ratios: The statistics are over the ratios of a variable’s coefﬁcient, to the sum
  over all other variables’ coefficients, for a given constraint. Four versions of these ratios are considered:
  positive (negative) coefficient to sum of positive (negative) coefficients