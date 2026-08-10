================
Example Tutorial
================

This tutorial walks through creating initial conditions, running, and analyzing a 2D cylindrical (rz) Sedov-Taylor hydrodynamics expansion simulation. The simulation utilizes the Castro code, and requires the ``Castro2d`` executable compiled with the 2-temperature model (``gamma_law_2T``).

.. rubric:: Environment Setup

.. code-block:: python

    import os
    import sys
    import re
    import numpy as np
    import matplotlib.pyplot as plt

    # Ensure custom modules are in the path
    sys.path.append("../../initial_condition")
    from ionization_routines import save_to_openpmd
    from analysis_tool import CastroSimulation


.. rubric:: Generating Initial Conditions

.. code-block:: python

    # Define a 2D cylindrical grid: 100 um radius by 1 cm length
    r = np.linspace(0.0, 100e-6, 100)
    z = np.linspace(0.0, 1e-2, 100)
    R, Z = np.meshgrid(r, z, indexing='ij')

    # Define a Gaussian temperature profile (peak at 10 eV)
    width = 10e-6 # 10 um width
    T_max = 10.0  # eV
    T_eV = T_max * np.exp(- R**2 / width**2) + 0.06

    # Extract species keys from Castro build
    with open('../build/species.net', 'r') as f:
        species_keys = re.findall(r'\n\s.*\s([A-Z][a-z]*\d)', f.read())

    n_total = 1e24 # m^-3 (equivalent to 1e18 cm^-3)
    ionization_fraction = np.exp(-R**2/width**2)

    # Allocate densities and assign fractions
    densities = np.zeros((R.shape[0], Z.shape[1], len(species_keys)))
    densities[:, :, species_keys.index('H1')] = ionization_fraction * n_total + 1e20
    densities[:, :, species_keys.index('H0')] = (1 - ionization_fraction) * n_total + 1e20

    # Export initial conditions to HDF5
    save_to_openpmd(
        {'r': [r.min(), r.max()], 'z': [z.min(), z.max()]},
        densities, 
        T_eV, 
        '2d_inputs.h5', # The name of the input file
        species_keys
    )


.. rubric:: Executing the Simulation

The input file ``inputs.2d.cyl``::

    # ------------------  INPUTS TO MAIN PROGRAM  -------------------
    max_step = 10000
    stop_time = 5e-9

    # PROBLEM SIZE & GEOMETRY
    geometry.is_periodic =  0 0 # Not periodic in either x or z
    geometry.coord_sys   =  1       # r-z coordinates
    geometry.prob_lo     =  0    0
    geometry.prob_hi     =  0.01 1.0 # 100 um radius by 1 cm length
    amr.n_cell           = 8   8 #number of cells in each direction at the coarsest level

    # >>>>>>>>>>>>>  BC FLAGS <<<<<<<<<<<<<<<<
    # 0 = Interior           3 = Symmetry
    # 1 = Inflow             4 = SlipWall
    # 2 = Outflow            5 = NoSlipWall
    # >>>>>>>>>>>>>  BC FLAGS <<<<<<<<<<<<<<<<
    castro.lo_bc       =  3   3
    castro.hi_bc       =  2   2 #Outflow in all directions
    castro.allow_non_unit_aspect_zones = 1

    # WHICH PHYSICS
    castro.do_hydro = 1 #time-advance the fluid dynamical equations
    castro.do_react = 0 // Here, reactions (ionization) are included as ext_src
    castro.add_ext_src = 1
    castro.diffuse_temp = 0

    # Get correct geometric terms in cylindrical
    # (See https://github.com/AMReX-Astro/Castro/issues/3099)
    castro.diffuse_use_amrex_mlmg = 0

    # TIME STEP CONTROL
    castro.cfl            = 0.5     # cfl number for hyperbolic system
    castro.init_shrink    = 0.01    # scale back initial timestep
    castro.change_max     = 100.     # maximum increase in dt over successive steps

    # DIAGNOSTICS & VERBOSITY
    castro.sum_interval   = 1       # timesteps between computing mass
    castro.v              = 1       # verbosity in Castro.cpp
    amr.v                 = 1       # verbosity in Amr.cpp

    # REFINEMENT / REGRIDDING
    amr.max_level       = 2       # maximum level number allowed
    amr.ref_ratio       = 2 2 2 2 # refinement ratio
    amr.regrid_int      = 5       # how often to regrid
    amr.blocking_factor = 8       # block factor in grid generation
    amr.max_grid_size   = 256

    amr.refinement_indicators = dengrad pressgrad

    amr.refine.dengrad.max_level = 2
    amr.refine.dengrad.relative_gradient = 0.4
    amr.refine.dengrad.field_name = density

    amr.refine.pressgrad.max_level = 2
    amr.refine.pressgrad.relative_gradient = 0.4
    amr.refine.pressgrad.field_name = pressure

    # PLOTFILES
    amr.plot_file       = plt_2d_
    amr.plot_int        = 10

    # PROBLEM PARAMETERS
    problem.initial_conditions_file = "2d_inputs.h5"
    problem.p_ambient = 1.e-6

    # EOS
    eos.eos_assume_neutral = 0
    eos.eos_gamma = 1.66666667


Run the 2D simulation

.. code-block:: bash

   ../build/Castro2d.gnu.MPI.gamma_law_2T.ex inputs.2d.cyl

.. rubric:: Data Initialization

.. code-block:: python

    # Load the simulation data
    cs = CastroSimulation(sim_folder, 'plt_2d_')
    cs.sim_info()          # Prints grid, domain, and available fields
    max_level = cs.max_level # Identifies the maximum Adaptive Mesh Refinement (AMR) level
    atomic_mass = 1.66e-30


.. rubric:: Plotting the 2D Density Field

.. code-block:: python

    # --- Plot 2D Density Field ---
    t_2d = 3.0e-9
    m = cs.get_field(t_2d, quantity='rho_H1', level=max_level)

    plt.figure(figsize=(6,3))
    plt.imshow(m['q']/atomic_mass, 
               extent=[m['z'][0]*1e1, m['z'][-1]*1e1, m['r'][0]*1e4, m['r'][-1]*1e4], 
               origin='lower', aspect='auto', cmap='Blues')
    plt.colorbar()
    plt.title(f'2D Density Profile at {t_2d} s')
    plt.xlabel('z [mm]')
    plt.ylabel('r [um]')
    plt.show()


.. rubric:: Plotting the 1D Density Slice

.. code-block:: python

    # --- Plot 1D Slice ---
    t_1d = 2.0e-9
    m_slice = cs.get_field(t_1d, quantity='density', level=max_level, positions={'z': 0.03})

    plt.figure(figsize=(6,4))
    plt.plot(m_slice['r'], m_slice['q']/atomic_mass)
    plt.title(f'Density Profile at z=0.03 cm (t={t_1d} s)')
    plt.xlabel('r (um)')
    plt.ylabel('Density (m^-3)')
    plt.show()


.. rubric:: Tracking Energy Over Time

.. code-block:: python

    # --- Track Energy Over Time ---
    t_array = cs.output_times

    thermal = cs.get_energy(t_array, level=max_level, energy_type='thermal')[0]
    kinetic = cs.get_energy(t_array, level=max_level, energy_type='kinetic')[0]
    ion = cs.get_energy(t_array, level=max_level, energy_type='ion')[0]
    total = cs.get_energy(t_array, level=max_level)[0]

    plt.figure()
    plt.plot(t_array, thermal, label='Thermal Energy')
    plt.plot(t_array, kinetic, label='Kinetic Energy')
    plt.plot(t_array, ion, label='Ion Energy')
    plt.plot(t_array, total, label='Total Energy', linestyle='--', color='black')
    plt.xlabel('Time (s)')
    plt.ylabel('Total Energy (erg)')
    plt.legend()
    plt.show()


.. rubric:: Tracking Particle Numbers

.. code-block:: python

    # --- Track Particles Over Time ---
    part_H1_array = cs.get_particle_number(t_array, species='H1', level=2)[0]

    plt.figure()
    plt.plot(t_array, part_H1_array, label='H1 Particle Number')
    plt.xlabel('Time (s)')
    plt.ylabel('Particle Number')
    plt.yscale('log')
    plt.legend()
    plt.show()


.. rubric:: AMR Grid Overlay Plot

.. code-block:: python

    # --- Plot AMR Grid Overlay ---
    t_amr = 3.5e-9
    fig, ax = plt.subplots(figsize=(8, 6))

    n_H1 = cs.get_field(t_amr, quantity='rho_H1', level=4)
    ax.imshow(n_H1['q']/atomic_mass, 
              extent=[n_H1['z'][0]*1e1, n_H1['z'][-1]*1e1, n_H1['r'][0]*1e4, n_H1['r'][-1]*1e4], 
              origin='lower', aspect='auto', cmap='Blues')

    # Overlay the mesh patches and individual cells
    cs.plot_AMR_grid(t_amr, ax, scale_z=10, scale_r=10000, 
                     plot_patches=True, plot_cells=True, 
                     swap_axes=True, linewidth_cell=0.14, linewidth_patch=1.0)
                     
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 100)
    ax.set_title(f'AMR Grid Overlay at {t_amr} s')
    plt.show()