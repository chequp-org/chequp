===================
Analyse & visualize
===================

Overview
--------
``CastroSimulation`` (in ``sim_folder/analysis/analysis_tool.py``) is a Python post-processing class built on top of **yt** and **NumPy**. It streamlines the extraction and analysis of hydrodynamic and plasma properties from `Castro <https://amrex-astro.github.io/Castro/>`_ plotfiles.

It handles 1D (cylindrical), 2D (Cartesian and cylindrical) and 3D (Cartesian) geometries and computes derived quantities, total energies, and species particle counts. All quantities are in CGS units.


Core Capabilities
-----------------
* **Automatic Time-Series Loading:** Loads all plotfiles matching a prefix (e.g. ``plt_1d_*``) and extracts the array of output times.
* **Field Extraction:** Returns any plotfile field on a uniform grid at a given AMR level, optionally as a 1D slice of 2D data.
* **Derived Plasma Quantities:** Computes the electron temperature ``T_e`` and heavy-particle temperature ``T_h`` from the internal energy and the heavy-particle energy fraction.
* **Energy Accounting:** Integrates internal, kinetic, ionization, and total energy over the domain.
* **Particle Inventory:** Computes the total number of particles per species by integrating the species mass densities.


Prerequisites & Setup
---------------------

.. code-block:: bash

    conda install -c conda-forge numpy yt scipy tqdm matplotlib

``analysis_tool.py`` is not an installed package: add ``sim_folder/analysis`` to your Python path. It reads the species masses and ionization energies from ``database_species.json`` in the same folder.

Code Usage Guide
----------------

1. Initialization and Simulation Summary
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    import sys
    sys.path.append("path/to/chequp/sim_folder/analysis")
    from analysis_tool import CastroSimulation

    # Load all plotfiles starting with 'plt_1d_' in the run directory
    sim = CastroSimulation(run_dir="./", file_start="plt_1d_")

    # Print dimensionality, geometry, available fields, AMR levels, and time range
    sim.sim_info()

Useful attributes: ``sim.output_times`` (array of output times, in s), ``sim.fields_list``, ``sim.species_list``, ``sim.dim``, ``sim.geo`` and ``sim.max_level``.


2. Field Extraction (``get_field``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
``get_field`` uses the output closest to the requested time and returns a dictionary with the exact time ``'t'``, the values ``'q'`` and the cell-center coordinates (``'r'``/``'z'`` in cylindrical geometry, ``'x'``/``'y'``/``'z'`` in Cartesian geometry, in cm). If the requested level is out of bounds, the finest level is used.

.. code-block:: python

    # Species mass density of H+ at t = 1 ns, on AMR level 1
    field_data = sim.get_field(t=1.0e-9, quantity="rho_H1", level=1)
    q_vals = field_data['q']    # Field values
    r_coords = field_data['r']  # Radial coordinates (cm)

    # Derived electron temperature (K)
    Te = sim.get_field(t=1.0e-9, quantity="T_e", level=1)

    # Extract a 1D radial slice from a 2D r-z dataset at z = 0.5 cm
    slice_data = sim.get_field(t=1.0e-9, quantity="density", level=2, positions={'z': 0.5})

.. note::

    The derived ``T_e`` assumes a hydrogen plasma (the electron fraction is taken from ``rho_H1``).

3. Energy Integration (``get_energy``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
``get_energy(t, level, energy_type)`` returns ``(energy, time)`` for a single time, or two lists for an array of times. The available energy types are:

* ``'total'``: total energy (``rho_E``), plus the ionization energy when ``castro.add_ext_src = 1``.
* ``'thermal'``: internal energy (``rho_e``).
* ``'kinetic'``: ``rho_E - rho_e``.
* ``'ion'``: ionization energy only (requires ``castro.add_ext_src = 1``).

Energies are in erg (erg/cm in 1D cylindrical and 2D Cartesian geometry).

.. code-block:: python

    import matplotlib.pyplot as plt

    times = sim.output_times
    e_total, t_eval = sim.get_energy(times, level=0, energy_type='total')
    e_thermal, _ = sim.get_energy(times, level=0, energy_type='thermal')
    e_kinetic, _ = sim.get_energy(times, level=0, energy_type='kinetic')
    e_ion, _ = sim.get_energy(times, level=0, energy_type='ion')

    plt.plot(t_eval, e_total, label="Total", color="black")
    plt.plot(t_eval, e_thermal, label="Thermal", linestyle="--")
    plt.plot(t_eval, e_kinetic, label="Kinetic", linestyle=":")
    plt.plot(t_eval, e_ion, label="Ionization", linestyle="-.")
    plt.xlabel("Time [s]")
    plt.ylabel("Energy [erg]")
    plt.legend()
    plt.show()


4. Particle Tracking (``get_particle_number``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Integrates the mass density of a species and divides by its mass to get the total number of particles:

.. code-block:: python

    n_H0, t_eval = sim.get_particle_number(times, species="H0", level=1)
    n_H1, _      = sim.get_particle_number(times, species="H1", level=1)

    plt.plot(t_eval, n_H0, label="Neutral H (H0)")
    plt.plot(t_eval, n_H1, label="Ionized H (H1)")
    plt.xlabel("Time [s]")
    plt.ylabel("Total Particle Count")
    plt.legend()
    plt.show()

5. Plotting a 2D field
^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    data = sim.get_field(t=2.5e-9, quantity="density", level=2)

    fig, ax = plt.subplots(figsize=(8, 6))
    mesh = ax.pcolormesh(data['z'], data['r'], data['q'], cmap="viridis", shading="auto")
    fig.colorbar(mesh, ax=ax, label=r"Density $\left[\mathrm{g\cdot cm^{-3}}\right]$")
    ax.set_xlabel("Axial Position z [cm]")
    ax.set_ylabel("Radius r [cm]")
    ax.set_title(f"Density (t = {data['t']:.2e} s)")
    plt.show()

To convert a species mass density ``rho_<species>`` to a number density, divide by the species mass, e.g. ``sim.data_species['H1']['mass']`` (in g).
