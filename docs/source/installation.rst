Installation
------------

Download source code
~~~~~~~~~~~~~~~~~~~~

CHEQUP relies on a fork of Castro (branch ``2T_25.10``) that provides the two-temperature
equation of state. It must be cloned inside the CHEQUP directory, as the build system expects
it at ``../../Castro`` relative to ``sim_folder/build`` (see ``CASTRO_HOME`` in the ``GNUmakefile``).

.. code-block:: sh

    git clone https://github.com/chequp-org/chequp.git
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

On Debian/Ubuntu (this is what the continuous integration uses):

.. code-block:: sh

    sudo apt-get install -y build-essential mpich libmpich-dev make libhdf5-mpich-dev
    export CPATH="/usr/include/hdf5/mpich:${CPATH}"
    export LIBRARY_PATH="/usr/lib/x86_64-linux-gnu/hdf5/mpich:${LIBRARY_PATH}"

Install the Python dependencies used to generate initial conditions, analyse the results and run the tests:

.. code-block:: sh

    conda install -y -c conda-forge scipy "numpy<2.0.0" numba tqdm pandas matplotlib openpmd-api openpmd-viewer yt h5py jupyter pytest
    pip install pytools  # only needed by hipace_to_chequp.py

Compile
~~~~~~~

The executable is built from ``sim_folder/build``. The main build options are:

* ``EOS_DIR``: the equation of state, which selects the temperature model.

  * ``gamma_law``: single-temperature model.
  * ``gamma_law_2T``: two-temperature model (separate electron and heavy-particle temperatures).

* ``DIM``: the dimension of the simulation (``1``, ``2`` or ``3``).
* ``USE_MPI``: ``TRUE`` by default (set in the ``GNUmakefile``).
* ``USE_CUDA``: set to ``TRUE`` to build for NVIDIA GPUs.

.. code-block:: sh

    cd sim_folder/build
    make -j 4 -s EOS_DIR=gamma_law DIM=1 # single-temperature, 1D
    make -j 4 -s EOS_DIR=gamma_law_2T DIM=1 # two-temperature, 1D
    make -j 4 -s EOS_DIR=gamma_law DIM=2 # single-temperature, 2D

The name of the executable encodes the build options, with the equation of state as a suffix, e.g.
``Castro1d.gnu.MPI.gamma_law.ex``, ``Castro1d.gnu.MPI.gamma_law_2T.ex`` or ``Castro2d.gnu.MPI.gamma_law.ex``
(``MPI`` is absent from the name when compiling with ``USE_MPI=FALSE``, and ``CUDA`` is added for GPU builds).

.. note::

    The test suite looks for exactly one executable matching ``Castro<DIM>d*.<EOS_DIR>.ex`` in
    ``sim_folder/build``. Remove stale executables (e.g. a serial and an MPI build of the same model)
    before running the tests.

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

For GPU (assuming the CUDA toolkit is installed). Set ``CUDA_ARCH`` to the compute capability of your GPU
(e.g. ``80`` for A100, which is what the CI uses):

.. code-block:: sh

    make USE_CUDA=TRUE CUDA_ARCH=80 -j 4 -s EOS_DIR=gamma_law_2T DIM=1

What is compiled
~~~~~~~~~~~~~~~~

The CHEQUP-specific sources live in ``sim_folder/build`` and are compiled on top of Castro:

.. list-table::
   :widths: 35 65
   :header-rows: 1

   * - File
     - Role
   * - ``species.net``
     - List of species (ionization stages) evolved by the code, with their atomic mass and charge, plus the
       auxiliary variable ``f_heavies`` (fraction of internal energy carried by heavy particles).
   * - ``_prob_params``
     - Problem runtime parameters (``problem.*`` in the inputs file).
   * - ``problem_initialize.H`` / ``problem_initialize_state_data.H``
     - Read the openPMD/HDF5 initial-condition file and interpolate it onto the AMR grid
       (see :doc:`initial_conditions`).
   * - ``problem_source.H``
     - External source term (``castro.add_ext_src = 1``): collisional ionization, three-body recombination,
       and the associated energy exchanges, integrated with the matrix formalism (see :doc:`matrix_form`).
   * - ``collisional_ionization.H``
     - Atomic data (ionization energies, degeneracies, oscillator strengths, Gaunt-factor fits) and rate
       coefficients for direct ionization, excitation-ionization and three-body recombination
       (rate tables are built at start-up by ``coll_ion::initialize()``).
   * - ``collision_frequencies.H``
     - Electron-ion and electron-neutral collision frequencies (fitted momentum-transfer cross sections).
   * - ``actual_conductivity.H``
     - Electron thermal conductivity used when ``castro.diffuse_temp = 1``.

.. warning::

    The atomic data in ``collisional_ionization.H`` is hard-coded for the species of ``species.net``
    (H, N up to N\ :sup:`5+`, He, Ar up to Ar\ :sup:`8+`), in this order. Changing ``species.net`` requires
    updating these tables accordingly.
