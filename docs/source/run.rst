Run
---

First, create the initial conditions for the 1D simulation: 

Create initial conditions
~~~~~~~~~~~~~

.. code-block:: sh

    cd sim_folder/run
    python3 generate_initial_conditions.py

This will create the file with the initial conditions ``example_1d_initial_conditions.h5``. Then run the simulation with the 1D inputs.

Running in 1D
~~~~~~~~~~~~~

If you compiled the 1D executable (using ``DIM=1``), you can run the simulation using the 1D inputs file.
For a standard or serial execution:

.. code-block:: sh

    ../build/Castro1d.gnu.gamma_law.ex inputs.1d.cyl

Running in 2D
~~~~~~~~~~~~~

To run a 2D simulation, you must first ensure you compiled the 2D executable (using ``DIM=2``).
For a standard or serial execution:

.. code-block:: sh

    ../build/Castro2d.gnu.gamma_law.ex inputs.2d.cyl