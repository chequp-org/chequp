This document describes how to perform simulations of plasma channel formation using Castro.

# Installation

To setup the folders:
```
git clone git@github.com:RemiLehe/castro_sim.git
cd castro_sim
git clone --recursive https://github.com/AMReX-Astro/Castro.git
cd Castro
git checkout development
cd ..
```

On Linux, I used the same compilation environment as for WarpX i.e.
```
spack env activate warpx-openmp-dev
```
or (for GPU)
```
spack env activate warpx-cuda-dev
```

On MacOS, I followed the instruction here:
https://github.com/AMReX-Astro/Castro/issues/2195
```
brew install gcc make
brew install --build-from-source open-mpi --cc=gcc-11
```
and used `gmake` instead of `make` in the instructions below.

In order to analyze the results, create a Python environment with `numpy`, `scipy`, `Jupyter` and `yt`.

## For 2D Cartesian sims

```
cd sim_folder/build
make -j 4
```
(for GPU, use `make USE_CUDA=TRUE -j 4`)

```
cd ../run
../build/Castro2d.gnu.MPI.ex inputs.2d.cyl_in_cartcoords
```

```
cd ../analysis
jupyter notebook Analysis.ipynb
```

## For 1D Cylindrical sims

```
cd sim_folder/build
make DIM=1 -j 4
```
(for GPU, use `make DIM=1 USE_CUDA=TRUE -j 4`)

```
cd ../run
../build/Castro1d.gnu.MPI.ex inputs.1d.cyl
```

```
cd ../analysis
jupyter notebook Analysis.ipynb
```

# More info on the simulations

Castro documentation: https://amrex-astro.github.io/Castro/docs/
Microphysics documentation: https://amrex-astro.github.io/Microphysics/docs/

Note that all units in the input script and output are CGS.