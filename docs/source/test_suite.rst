Testing the code
------------------

The tests are in the ``tests/`` folder. Each test generates its initial conditions with ``save_to_openpmd``,
runs a compiled CHEQUP executable from ``sim_folder/build`` with the inputs files of ``sim_folder/run``
(overriding some parameters on the command line), and compares the result to a theory or reference solution.

Available tests
~~~~~~~~~~~~~~~

Each test needs exactly one executable matching its dimension and equation of state in ``sim_folder/build``:

.. list-table::
   :widths: 40 15 15 30
   :header-rows: 1

   * - Test
     - ``DIM``
     - ``EOS_DIR``
     - What is checked
   * - ``test_1d.py::test_1d_sedov_taylor``
     - 1
     - ``gamma_law``
     - Blast-wave radius and density profile against the Sedov-Taylor solution, energy conservation, checksum.
   * - ``test_1d.py::test_1d_desy_benchmark``
     - 1
     - ``gamma_law_2T``
     - HOFI channel expansion against a COMSOL/HYQUP benchmark (close to Mewes et al., PRR 5, 033112, 2023),
       checksum.
   * - ``test_2d.py::test_2d_sedov_taylor``
     - 2
     - ``gamma_law``
     - Blast-wave radius, isotropy and density profile against the Sedov-Taylor solution (2D Cartesian,
       ``inputs.2d.cyl_in_cartcoords``), energy conservation, checksum.
   * - ``test_atomic_process.py::test_0D_Ar_H_mix``
     - 1
     - ``gamma_law_2T``
     - Ionization/recombination of a uniform Ar-H mixture against an independent ODE solution of the rate
       equations.

Running the tests
~~~~~~~~~~~~~~~~~

Setup the environment and compile CHEQUP as described in :doc:`installation`. For the following, we assume conda was used.

.. code-block:: sh

    conda activate chequp
    conda install -y -c conda-forge pytest
    cd tests
    export OMP_NUM_THREADS=1
    pytest -v test_1d.py::test_1d_sedov_taylor   # run a single test
    pytest -v                                     # run all tests (requires all executables above)

The tests must be run from the ``tests`` folder, as they use relative paths to ``sim_folder``.

Checksums
~~~~~~~~~

On top of the physics checks, some tests compare the final plotfile to a reference stored in
``tests/checksum/benchmarks_json/<test name>.json`` (``evaluate_checksum`` in ``tests/checksum/checksumAPI.py``).
If a change of the code intentionally modifies the results, the reference must be updated, either by copying
the json printed by the failing test, or with:

.. code-block:: sh

    cd tests
    python -m checksum.checksumAPI --reset-benchmark --test-name <test name> --file_name <path/to/plotfile>

Continuous integration
~~~~~~~~~~~~~~~~~~~~~~

The GitHub Actions workflows in ``.github/workflows`` run on every pull request to ``main``:

* ``CI_physics_test.yml``: compiles the required executable and runs each of the tests above in a separate job.
* ``cpu.yml``: compiles the 1D two-temperature model with MPI.
* ``cuda.yml``: compiles the 1D two-temperature model for NVIDIA GPUs (``USE_CUDA=TRUE``).

Add a new test
~~~~~~~~~~~~~~~

* In the ``tests/`` folder, create a new file, for example ``test_<test name>.py``, and add a new ``test_<test name>`` function similar to the existing ones (see ``test_1d.py`` for an example).

* Make a new file at ``tests/checksum/benchmarks_json/<test name>.json`` containing ``{}``.

* Run the test using ``pytest``. The checksum of the new test should fail and print the new json file to the console.

* Copy the json from the console output into the ``<test name>.json`` file.

* Verify that ``pytest`` now passes.

* Add the test to the matrix of ``.github/workflows/CI_physics_test.yml``.
