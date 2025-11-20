This document describes how to perform simulations of plasma channel formation using Castro.

# Installation

To setup the folders:
```
git clone git@github.com:RemiLehe/castro_sim.git
cd castro_sim
git clone --recursive https://github.com/RemiLehe/Castro.git --branch 2T_25.10
```

## Setup a conda environment

```
conda create -n castro_sim
conda activate castro_sim
conda install -c conda-forge compilers "hdf5=*=mpi_openmpi*" openmpi make zlib
```

In order to analyze the results, create a Python environment with `numpy`, `scipy`, `Jupyter` and `yt`.

## Switch between two-temperature and single-temperature model

The choice of a single-temperature model or two-temperature model depends on the ex file that you are using to run Castro. 
To compiled both model use :
```
cd sim_folder/build
make -j 4 -s EOS_DIR=gamma_law DIM=1
make -j 4 -s EOS_DIR=gamma_law_2T DIM=1
```
It will create two files with a sufix coresponding to the model : gamma_law for single-temperature, gamma_law_2T for two-temperature. The DIM flag change the dimension (here 1D).

## Tutorial 2D rz

First compile the code with 2d single-temperature model :
```
cd sim_folder/build
make -j 4 -s EOS_DIR=gamma_law DIM=2
```
(for GPU, use `make USE_CUDA=TRUE -j 4` ; on MacOS, use `make COMP=clang -j 4`)

```
cd ../analysis
jupyter notebook Tutorial_2D_rz_v0.ipynb
```

## For 1D Cylindrical sims

```
cd sim_folder/build
make -j 4 -s EOS_DIR=gamma_law DIM=1
```
(for GPU, use `make USE_CUDA=TRUE -j 4 -s EOS_DIR=gamma_law DIM=1` ; on MacOS, use `make COMP=clang -j 4 -s EOS_DIR=gamma_law DIM=1`)

```
cd ../run
../build/Castro1d.gnu.MPI.gamma_law.ex inputs.1d.cyl
```

```
cd ../analysis
jupyter notebook Analysis.ipynb
```

# Test the code

To run the test suite:

- In the folder `sim_folder/build`, compile the code in 1D

- In the folder `tests`, run:
```
py.test
```

## Add a new test

- In `test_1d.py`, add a new `test_<test name>` function similar to the existing ones.

- Make a new file at `tests/checksum/benchmarks_json/<test name>.json` containing `{}`.

- Run all tests using `py.test`. The checksum of the new test should fail and
  print the new json file to the console.

- Copy the json from the console output into the `<test name>.json` file.

- Verify that `py.test` now passes.


# More info on the simulations

Castro documentation: https://amrex-astro.github.io/Castro/docs/
Microphysics documentation: https://amrex-astro.github.io/Microphysics/docs/

Note that all units in the input script and output are CGS.
