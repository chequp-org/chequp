Initial conditions
------------------

CHEQUP does not hard-code an initial state: it reads the plasma profiles from an
`openPMD <https://www.openpmd.org>`_ HDF5 file, given by the runtime parameter
``problem.initial_conditions_file``. At start-up, the file is loaded once
(``read_initial_conditions`` in ``sim_folder/build/problem_initialize_state_data.H``) and the data is then
linearly (1D), bilinearly (2D) or trilinearly (3D) interpolated onto every AMR cell.
Points outside of the file's grid take the value of the closest grid point.

File format
~~~~~~~~~~~

The file must contain a single iteration (``data/0``) with the following meshes, all defined on the same grid:

.. list-table::
   :widths: 25 20 55
   :header-rows: 1

   * - Mesh
     - Unit
     - Description
   * - ``Te``
     - K
     - Electron temperature. Its ``gridSpacing`` and ``gridGlobalOffset`` attributes (in m) define the grid
       for all the other meshes.
   * - ``Th``
     - K
     - Heavy-particle (ion and neutral) temperature.
   * - ``<species>_density``
     - m\ :sup:`-3`
     - Number density of each species listed in ``sim_folder/build/species.net``
       (``H0_density``, ``H1_density``, ``N0_density``, ..., ``Ar8_density``). All species must be present,
       even with zero density.
   * - ``xmom``, ``ymom``, ``zmom``
     - see note
     - Momentum density, copied directly into Castro's momentum state variables.

The axes of the meshes must follow the geometry of the simulation: ``r`` (1D cylindrical), ``r, z``
(2D cylindrical) or ``x, y`` (2D Cartesian), with the array index order matching the axis order.

From these fields, CHEQUP sets the mass density
:math:`\rho = \sum_s n_s A_s m_u`, the species partial densities, the internal energy
:math:`\rho e = (n_h k_B T_h + n_e k_B T_e)/(\gamma - 1)` with :math:`n_e = \sum_s Z_s n_s` (quasi-neutrality),
and the heavy-particle energy fraction :math:`f = e_h/e`.

.. note::

    The ``xmom``, ``ymom``, ``zmom`` values are used without unit conversion, i.e. they are interpreted
    in Castro's CGS units (g cm\ :sup:`-2` s\ :sup:`-1`). Use zero (the default of ``save_to_openpmd``)
    for a plasma initially at rest.

.. note::

    Regions with a vanishing density or electron density can lead to numerical issues. It is common
    practice to add a small floor (e.g. a residual ionization fraction of :math:`10^{-3}`) to the profiles.

Writing the file with ``save_to_openpmd``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The helper ``save_to_openpmd`` in ``initial_condition/ionization_routines.py`` writes a file in the
expected format. Temperatures are given in eV and converted to K; densities are given in m\ :sup:`-3`.
If ``Th_eV`` is not given, the heavy-particle temperature is set to 348 K (0.03 eV).

.. code-block:: python

    import re, sys
    import numpy as np
    sys.path.append("../../initial_condition")
    from ionization_routines import save_to_openpmd

    # Read the species for which CHEQUP was compiled
    with open("../build/species.net") as f:
        species_keys = re.findall(r'\n\s.*\s([A-Z][a-z]*\d)', f.read())

    r = np.linspace(0, 600e-6, 601)                    # m
    T_eV = 27 * np.exp(-(r**2 / (2 * 35e-6**2))**3) + 0.17
    ioniz = (1 - 1e-3) * np.exp(-(r**2 / (2 * 38e-6**2))**12) + 1e-3

    densities = np.zeros((len(r), len(species_keys)))  # shape: (*grid, n_species)
    densities[:, species_keys.index("H0")] = (1 - ioniz) * 1e24
    densities[:, species_keys.index("H1")] = ioniz * 1e24

    save_to_openpmd({"r": [r.min(), r.max()]}, densities, T_eV,
                    "1d_initial_conditions.h5", species_keys)

For a 2D r-z grid, pass ``{'r': [rmin, rmax], 'z': [zmin, zmax]}`` and arrays of shape ``(Nr, Nz)``
(``(Nr, Nz, n_species)`` for the densities).

Species
~~~~~~~

The species evolved by CHEQUP are defined in ``sim_folder/build/species.net``:

.. list-table::
   :widths: 20 30 50
   :header-rows: 1

   * - Element
     - Short names
     - Ionization stages
   * - Hydrogen
     - ``H0`` - ``H1``
     - neutral to fully ionized
   * - Nitrogen
     - ``N0`` - ``N5``
     - neutral to N\ :sup:`5+`
   * - Helium
     - ``He0`` - ``He2``
     - neutral to fully ionized
   * - Argon
     - ``Ar0`` - ``Ar8``
     - neutral to Ar\ :sup:`8+`

Other ways to generate initial conditions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* From a laser intensity profile, using the ADK field-ionization model:
  ``process_intensity_array_multispecies`` in ``initial_condition/ionization_routines.py`` (requires FBPIC
  for the atomic data). See also ``initial_condition/Plasma_heating.ipynb``.
* From a HiPACE++ simulation (laser-ionized plasma): see :doc:`hipace_to_chequp`.

API reference
~~~~~~~~~~~~~

.. automodule:: ionization_routines
   :members: save_to_openpmd, load_intensity_profile, process_intensity_array_multispecies
