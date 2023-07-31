This document describes how to perform simulations of plasma channel formation using Castro.

# Installation

On Linux, I used the same compilation environment as for WarpX i.e.
```
spack env activate warpx-openmp-dev
```
or (for GPU)
```
spack env activate warpx-cuda-dev
```

To setup the folders:
```
git clone git@github.com:RemiLehe/castro_sim.git
cd castro_sim
git clone --recursive https://github.com/RemiLehe/Castro.git
cd Castro
git checkout avoid_out_of_bound
cd ..
```

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