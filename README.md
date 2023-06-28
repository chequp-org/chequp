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
git clone --recursive https://github.com/AMReX-Astro/Castro.git
```

## For 2D Cartesian sims

```
cd 2d_cartesian/build
make
```
(for GPU, use `make USE_CUDA=TRUE)

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
cd 1d_cylindrical/build
make
```
(for GPU, use `make USE_CUDA=TRUE)

```
cd ../run
../build/Castro1d.gnu.MPI.ex inputs.1d.cyl
```

```
cd ../analysis
jupyter notebook Analysis.ipynb
```