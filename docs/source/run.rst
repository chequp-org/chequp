Run
---

Units in the inputs file and in the outputs are CGS.

.. warning::

    ``sim_folder/run/generate_initial_conditions.py`` predates the current initial-condition format: it writes
    a ``T`` mesh instead of ``Te``/``Th`` and no momentum meshes, so its output cannot be read by the current
    code. Use ``save_to_openpmd`` instead (see :doc:`initial_conditions`).

Inputs files
~~~~~~~~~~~~

* ``inputs.1d.cyl``: 1D cylindrical (radial) geometry.
* ``inputs.2d.cyl``: 2D cylindrical (r-z) geometry, expects an initial-condition file with ``r, z`` axes.
* ``inputs.2d.cyl_in_cartcoords``: 2D Cartesian (x-y) geometry, used e.g. for the 2D Sedov-Taylor test.

Runtime parameters
~~~~~~~~~~~~~~~~~~

Any parameter of the inputs file can be overridden on the command line by appending ``key=value`` after the
inputs file (this is how the tests run, e.g. ``castro.add_ext_src=0 castro.diffuse_temp=0``).
The most relevant parameters are:

.. list-table::
   :widths: 35 65
   :header-rows: 1

   * - Parameter
     - Description
   * - ``problem.initial_conditions_file``
     - Path to the openPMD initial-condition file (default ``initial_conditions.h5``).
   * - ``problem.p_ambient``
     - Ambient pressure (default ``1e-5``).
   * - ``problem.temp_ambient``
     - Initial guess for the temperature state variable (default ``300``); the actual temperatures are set
       from the ``Te`` and ``Th`` fields of the initial-condition file.
   * - ``castro.add_ext_src``
     - ``1`` to enable the CHEQUP source terms (collisional ionization, recombination and the
       associated energy exchanges). ``castro.do_react`` should stay ``0``.
   * - ``castro.diffuse_temp``
     - ``1`` to enable electron thermal conduction (conductivity from ``actual_conductivity.H``).
   * - ``castro.diffuse_use_amrex_mlmg``
     - Keep at ``0`` to get the correct geometric terms in cylindrical coordinates
       (see `Castro#3099 <https://github.com/AMReX-Astro/Castro/issues/3099>`_).
   * - ``geometry.coord_sys``
     - ``0`` Cartesian, ``1`` cylindrical (r-z).
   * - ``geometry.prob_lo`` / ``geometry.prob_hi``
     - Domain extent in cm.
   * - ``castro.lo_bc`` / ``castro.hi_bc``
     - Boundary conditions (``3`` = symmetry on axis, ``2`` = outflow).
   * - ``amr.max_level``, ``amr.refinement_indicators``
     - Adaptive mesh refinement settings (refinement on density and pressure gradients by default).
   * - ``amr.plot_file``, ``amr.plot_int``, ``amr.plot_per``
     - Prefix and frequency (in steps or in simulated time) of the plotfiles.
   * - ``eos.eos_gamma``
     - Adiabatic index (``5/3``).
   * - ``eos.eos_assume_neutral``
     - Keep at ``0`` so that the free electrons are accounted for in the equation of state.

See the `Castro documentation <https://amrex-astro.github.io/Castro/docs/>`_ for the full list of parameters.

Outputs
~~~~~~~

The simulation writes AMReX plotfiles (``plt_1d_00000``, ``plt_1d_00010``, ...) in the run directory.
They contain, among others, the total density, the partial density of each species (``rho_H0``, ``rho_H1``,
...), the momentum, the energy and the temperature. See :doc:`analyze_basic` to read them with Python, and
:doc:`chequp_to_fbpic` / :doc:`openpmd_conversion` to export them to PIC codes.
