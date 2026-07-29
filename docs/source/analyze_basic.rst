===================
Analyse & visualize
===================

Overview
--------
``CastroSimulation`` is a Python post-processing class built on top of **yt** and **NumPy**. It streamlines the extraction, analysis, and visualization of hydrodynamic and plasma properties from `Castro <https://amrex-astro.github.io/Castro/>`_ simulation outputs. 

It handles 1D, 2D, and 3D geometries (both Cartesian and cylindrical) and automatically computes derived physical quantities, total energies, and species particle counts.


Core Capabilities
-----------------
* **Automatic Time-Series Loading:** Loads sequence files (``plt00000``, ``plt_1d_*``) with automatic time-array extraction and metadata parsing.
* **Derived Plasma Quantities:** Computes derived field properties such as electron temperature (:math:`T_e`) and heavy-particle temperature (:math:`T_h`) dynamically.
* **Energy Accounting:** Integrates internal, kinetic, ionization, and total energy densities across complex geometries.
* **Particle Inventory:** Computes total particle counts per species by integrating spatial density fields.
* **AMR Grid Visualization:** Overlays Adaptive Mesh Refinement (AMR) grid structures and cell boundaries on matplotlib plots.


Prerequisites & Setup
---------------------

Required Libraries
^^^^^^^^^^^^^^^^^^
Ensure you have the required Python libraries installed:

.. code-block:: bash

    pip install numpy yt scipy tqdm matplotlib

Code Usage Guide
----------------

1. Initialization and Simulation Summary
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
To load output files and inspect general metadata:

.. code-block:: python

    from castro_analysis import CastroSimulation

    # Initialize time series from a run directory
    sim = CastroSimulation(run_dir="./sim_output/", file_start="plt")

    # Print dimensionalities, available fields, AMR levels, and simulation time range
    sim.sim_info()


2. Field Extraction (``get_field``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Extract scalar fields at specific refinement levels or slice through higher dimensions.

.. code-block:: python

    # Extract electron temperature at t = 1 ns at AMR level 1
    field_data = sim.get_field(t=1.0e-9, quantity="Temp", level=1)

    # Inspect output contents
    q_vals = field_data['q']    # Field array values
    r_coords = field_data['r']  # Spatial coordinates (if cylindrical 'x' in cartesian)

    # Extract a 1D slice from a 2D dataset at z = 0.5 cm
    slice_data = sim.get_field(t=1.0e-9, quantity="density", level=2, positions={'z': 0.5})


3. Energy Integration (``get_energy``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Calculate total energy evolution across different energy forms over time:

.. code-block:: python

    import numpy as np
    import matplotlib.pyplot as plt

    # Select time targets covering the entire simulation
    times = sim.output_times

    # Integrate total, thermal, and kinetic energy over the domain
    e_total, t_eval = sim.get_energy(t=times, level=0, energy_type='total')
    e_thermal, _  = sim.get_energy(t=times, level=0, energy_type='thermal')
    e_kinetic, _  = sim.get_energy(t=times, level=0, energy_type='kinetic')
    # If ionization is enabled (castro.add_ext_src = 1), add the ionization energy
    e_ion, _ = sim.get_energy(t=times, level=0, energy_type='ion')

    # Plot energy conservation over time
    plt.figure(figsize=(8, 5))
    plt.plot(t_eval, e_total, label="Total Energy", color="black")
    plt.plot(t_eval, e_thermal, label="Thermal Energy", linestyle="--")
    plt.plot(t_eval, e_kinetic, label="Kinetic Energy", linestyle=":")
    plt.xlabel("Time [s]")
    plt.ylabel("Energy [erg]")
    plt.title("Energy Evolution")
    plt.legend()
    plt.grid(True)
    plt.show()


4. Particle Tracking (``get_particle_number``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Integrate mass densities to calculate total particle counts for target species:

.. code-block:: python

    # Track species counts over time
    n_H0, t_eval = sim.get_particle_number(t=times, species="H0", level=1)
    n_H1, _      = sim.get_particle_number(t=times, species="H1", level=1)

    # Plot ionization progress
    plt.figure(figsize=(8, 5))
    plt.plot(t_eval, n_H0, label="Neutral H (H0)")
    plt.plot(t_eval, n_H1, label="Ionized H (H1)")
    plt.xlabel("Time [s]")
    plt.ylabel("Total Particle Count")
    plt.title("Ionization Dynamics")
    plt.legend()
    plt.grid(True)
    plt.show()

5. Plotting Fields with AMR Grids
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Combine field extractions with ``plot_AMR_grid`` to visualize physical fields overlaid with adaptive grid levels:

.. code-block:: python

    import matplotlib.pyplot as plt

    # 1. Target evaluation time and extraction
    target_time = 2.5e-9
    data = sim.get_field(t=target_time, quantity="density", level=2)

    # 2. Render 2D Pseudocolor Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    mesh = ax.pcolormesh(data['r'], data['z'], data['q'].T, cmap="viridis", shading="auto")
    fig.colorbar(mesh, ax=ax, label=r"Density $\left[\mathrm{g\cdot cm^{-3}}\right]$")

    # 3. Overlay AMR Patch Hierarchy (Levels 1 and higher)
    sim.plot_AMR_grid(
        t=target_time, 
        ax=ax, 
        scale_z=1, # Only for 2D plots
        scale_r=1,
        plot_patches=True, 
        plot_cells=False, 
        color_patch='red', 
        linewidth_patch=1.0, 
        min_plot_level=1
    )

    ax.set_xlabel("Radius r [cm]")
    ax.set_ylabel("Axial Position z [cm]")
    ax.set_title(f"Density Field & AMR Patches (t = {data['t']:.2e} s)")
    plt.tight_layout()
    plt.show()

.. note:: 
    
    It may be necessary to adjust the ``scale_z`` and ``scale_r`` parameters to ensure the correct plotting range.