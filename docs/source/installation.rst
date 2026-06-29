Installation
------------

Download source code
~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

    git clone git@github.com:chequp-org/chequp.git
    cd chequp
    git clone --recursive https://github.com/chequp-org/Castro.git --branch 2T_25.10

Setup the environment
~~~~~~~~~~~~~~~~~~~~~

With Conda:

.. code-block:: sh

    conda create -n chequp
    conda activate chequp
    conda install -c conda-forge compilers "hdf5=*=mpi_openmpi*" openmpi make zlib

With Homebrew, on MacOS:

.. code-block:: sh

    brew update
    brew install make
    brew install fftw
    brew install hdf5 # for .h5 file support
    # Or to run in parallel
    # brew install hdf5-mpi
    # brew install open-mpi

Install dependencies for analysis:

.. code-block:: python

    conda install -y -c conda-forge scipy numpy numba tqdm pandas openpmd-api yt h5py Jupyter

Compile
~~~~~~~

The choice of a single-temperature model or two-temperature model depends on the ex file that you are using to run Castro.
To compiled both the single-temperature and the two-temperature models:

.. code-block:: sh

    cd sim_folder/build
    make -j 4 -s EOS_DIR=gamma_law DIM=1 # single-temperature
    make -j 4 -s EOS_DIR=gamma_law_2T DIM=1 # two-temperature
    make -j 4 -s EOS_DIR=gamma_law DIM=2 # 2D

It will create two files with a sufix coresponding to the model: gamma_law for single-temperature, gamma_law_2T for two-temperature. The DIM flag change the dimension (here 1D).

For MacOS, you may need to define the path to HDF5 by hand

.. code-block:: sh

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

For GPU (assuming the environment is ready)

.. code-block:: sh

    make USE_CUDA=TRUE -j 4 -s EOS_DIR=gamma_law DIM=1
