# Castro-based Hofi Expansion with QUasineutral Plasma (CHEQUP)

This document describes how to perform simulations of plasma channel formation using Castro.

## Installation

#### Download source code

```
git clone git@github.com:chequp-org/chequp.git
cd chequp
git clone --recursive https://github.com/chequp-org/Castro.git --branch 2T_25.10
```

#### Setup the environment

With Conda:
```sh
conda create -n chequp
conda activate chequp
conda install -c conda-forge compilers "hdf5=*=mpi_openmpi*" openmpi make zlib
```

With Homebrew, on MacOS:
```sh
brew update
brew install make
brew install fftw
brew install hdf5 # for .h5 file support
# Or to run in parallel
# brew install hdf5-mpi
# brew install open-mpi
```

Install dependencies for analysis:
```py
conda install -y -c conda-forge scipy numpy numba tqdm pandas openpmd-api yt h5py Jupyter
```

#### Compile 

The choice of a single-temperature model or two-temperature model depends on the ex file that you are using to run Castro.
To compiled both the single-temperature and the two-temperature models:
```sh
cd sim_folder/build
make -j 4 -s EOS_DIR=gamma_law DIM=1 # single-temperature
make -j 4 -s EOS_DIR=gamma_law_2T DIM=1 # two-temperature
make -j 4 -s EOS_DIR=gamma_law DIM=2 # 2D
```
It will create two files with a sufix coresponding to the model: gamma_law for single-temperature, gamma_law_2T for two-temperature. The DIM flag change the dimension (here 1D).

For MacOS, you may need to define the path to HDF5 by hand
```sh
# Serial
export HDF5_DIR=/opt/homebrew/Cellar/hdf5/2.1.1/
make COMP=clang -j 4 -s EOS_DIR=gamma_law DIM=1 USE_MPI=FALSE
make COMP=clang -j 4 -s EOS_DIR=gamma_law_2T DIM=1 USE_MPI=FALSE
make COMP=clang -j 4 -s EOS_DIR=gamma_law DIM=2 USE_MPI=FALSE # 2D
# Parallel
export HDF5_DIR=/opt/homebrew/Cellar/hdf5-mpi/2.1.1/
make COMP=clang -j 4 -s EOS_DIR=gamma_law DIM=1 USE_MPI=TRUE
make COMP=clang -j 4 -s EOS_DIR=gamma_law_2T DIM=1 USE_MPI=TRUE
make COMP=clang -j 4 -s EOS_DIR=gamma_law DIM=2 USE_MPI=TRUE # 2D
```

For GPU (assuming the environment is ready)

```sh
make USE_CUDA=TRUE -j 4 -s EOS_DIR=gamma_law DIM=1
```

## Run

First, create the initial conditions for the 1D simulation: 

```sh
cd sim_folder/run
python3 generate_initial_conditions.py
```

This will create the file with the initial conditions ```example_1d_initial_conditions.h5```. Then run the simulation with the 1D inputs.
```
../build/Castro1d.gnu.MPI.gamma_law.ex inputs.1d.cyl
# You can find an analysis Jupyter Notebook at ../analysis/Analysis.ipynb
```

## Run the test suite

Setup environment and compile Castro as described above. For the following, we assume conda was used.
```sh
conda activate chequp
conda install -y -c conda-forge pytest
cd tests
py.test
```

## Add a new test

- In `test_1d.py`, add a new `test_<test name>` function similar to the existing ones.

- Make a new file at `tests/checksum/benchmarks_json/<test name>.json` containing `{}`.

- Run all tests using `py.test`. The checksum of the new test should fail and
  print the new json file to the console.

- Copy the json from the console output into the `<test name>.json` file.

- Verify that `py.test` now passes.


## More info on the simulations

Castro documentation: https://amrex-astro.github.io/Castro/docs/
Microphysics documentation: https://amrex-astro.github.io/Microphysics/docs/

Note that all units in the input script and output are CGS.
