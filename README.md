This document describes how to perform simulations of plasma channel formation using Castro.

# Installation

Follow the instructions [here](https://amrex-astro.github.io/Castro/docs/getting_started.html) to download and compile the code.

On Linux, I used the same compilation environment as for WarpX i.e.
```
spack env activate warpx-openmp-dev
make
```
or (for GPU)
```
spack env activate warpx-cuda-dev
make USE_CUDA=TRUE
```