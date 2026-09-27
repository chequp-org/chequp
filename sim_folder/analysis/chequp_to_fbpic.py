"""
chequp_to_fbpic.py
==================

This module handles the extraction and processing of plasma density profiles from 
CHEQUP hydrodynamics simulations and exports them into an openPMD 
HDF5 format compatible with FBPIC (in 2D cylindrical :math:`r-z` geometry).

Typical usage::

    from chequp_to_fbpic import write_FBPIC_profile

    fbpic_input = write_FBPIC_profile(input_path='./path_to_CHEQUP_sim', 
                                    output_path='./path_to_fbpic_input', 
                                    t_hydro=3.5e-9)
"""

import os
import sys
import numpy as np
import openpmd_api as io
from analysis_tool import CastroSimulation


def write_FBPIC_profile(input_path='.', output_path='.', t_hydro=0.0e-9):
    """Extract plasma species density profiles from a Castro hydrodynamics simulation.

    Processes species density distributions and exports them to an openPMD-compliant
    HDF5 file formatted for FBPIC in 2D cylindrical (:math:`r-z`) geometry.

    Args:
        input_path (str, optional): Path to the directory containing Castro
            simulation plotfiles (e.g., ``plt_2d_*``). Defaults to ``'.'``.
        output_path (str, optional): Directory where the output HDF5 file
            (``plasma_density.h5``) will be saved. Defaults to ``'.'``.
        t_hydro (float, optional): Hydrodynamic simulation time (in seconds)
            at which to extract density profiles. Defaults to ``0.0e-9``.

    Note:
        The generated file adheres to openPMD standards using the ``thetaMode``
        geometry (cylindrical :math:`r-z`) and records scalar density datasets
        for Hydrogen, Helium, Argon, Nitrogen, and calculated free electrons.
    """

    def store_species_in_file_rz(z_1d, r_1d, series, density_data, species_name):
        """Store a single 2D species density grid into an openPMD series.

        Data is saved using cylindrical :math:`r-z` coordinates.

        Args:
            z_1d (numpy.ndarray): 1D array representing spatial grid points
                along the longitudinal Z-axis (in meters).
            r_1d (numpy.ndarray): 1D array representing spatial grid points
                along the radial R-axis (in meters).
            series (openpmd_api.Series): Active openPMD file series instance.
            density_data (numpy.ndarray): 2D array of species density values
                with shape ``(Nz, Nr)``.
            species_name (str): Name of the mesh record to write in the openPMD
                file (e.g., ``'density_e'``, ``'density_H_n'``).

        Raises:
            AssertionError: If ``density_data.shape`` does not match ``(len(z_1d), len(r_1d))``.
        """
        assert density_data.shape == (len(z_1d), len(r_1d)), \
            f"Data shape {density_data.shape} does not match grids ({len(z_1d)}, {len(r_1d)})"

        it = series.iterations[0]
        density = it.meshes[species_name]

        # Z is index 0, R is index 1
        density.grid_spacing = np.array([
            (z_1d[-1] - z_1d[0]) / (len(z_1d) - 1),
            (r_1d[-1] - r_1d[0]) / (len(r_1d) - 1)
        ])
        
        density.grid_global_offset = [z_1d[0], r_1d[0]]
        density.axis_labels = ["z", "r"]
        density.geometry = io.Geometry.thetaMode
        density.unit_dimension = {
            io.Unit_Dimension.L: -3,
        }
        
        density_d = density[io.Mesh_Record_Component.SCALAR]
        density_d.position = [0, 0, 0]

        dataset = io.Dataset(density_data.dtype, density_data.shape)
        density_d.reset_dataset(dataset)
        density_d.store_chunk(density_data)

    def write_plasma_profile(output_dir):
        """Create target HDF5 file and write calculated species density profiles."""
        print('Loading CHEQUP...')
        cs = CastroSimulation(input_path, 'plt_2d_*')
        atomic_mass = 1.66e-30
    
        # Hydrogen
        density_H_n = cs.get_field(t_hydro, quantity='rho_H0', level=0)['q'] / atomic_mass
        density_H_ion = cs.get_field(t_hydro, quantity='rho_H1', level=0)['q'] / atomic_mass
        # Helium
        density_He_n = cs.get_field(t_hydro, quantity='rho_He0', level=0)['q'] / (4.00 * atomic_mass)
        density_He_ion = cs.get_field(t_hydro, quantity='rho_He1', level=0)['q'] / (4.00 * atomic_mass)
        density_He_ion += cs.get_field(t_hydro, quantity='rho_He2', level=0)['q'] / (4.00 * atomic_mass)
        # Argon
        density_Ar_n = cs.get_field(t_hydro, quantity='rho_Ar0', level=0)['q'] / (39.9 * atomic_mass)
        density_Ar_ion = cs.get_field(t_hydro, quantity='rho_Ar1', level=0)['q'] / (39.9 * atomic_mass)
        for Z in range(2, 9):
            density_Ar_ion += cs.get_field(t_hydro, quantity=f'rho_Ar{Z}', level=0)['q'] / (39.9 * atomic_mass)
        # Nitrogen
        density_N_n = cs.get_field(t_hydro, quantity='rho_N0', level=0)['q'] / (14.0 * atomic_mass)
        density_N_ion = cs.get_field(t_hydro, quantity='rho_N1', level=0)['q'] / (14.0 * atomic_mass)
        for Z in range(2, 6):
            density_N_ion += cs.get_field(t_hydro, quantity=f'rho_N{Z}', level=0)['q'] / (14.0 * atomic_mass)
    
        density_e = density_H_ion + density_Ar_ion * 8.0 + 5.0 * density_N_ion + 2 * density_He_ion
        # Geometry
        r_max, z_max = get_domain_extents(f"{input_path}/plt_2d_00000/Header")
        r = np.linspace(0, r_max * 1e-2, density_e.shape[0])
        z = np.linspace(0, z_max * 1e-2, density_e.shape[0])
    
        file_path = os.path.join(output_dir, "plasma_density.h5")
        if os.path.exists(file_path):
            os.remove(file_path)
        
        # Create the openPMD series
        series = io.Series(file_path, io.Access.create)
        series.set_software("PIP4_Profile_Generator", "1.0") 

        store_species_in_file_rz(z, r, series, density_H_n, "density_H_n")
        store_species_in_file_rz(z, r, series, density_H_ion, "density_H_ion")
        store_species_in_file_rz(z, r, series, density_He_n, "density_He_n")
        store_species_in_file_rz(z, r, series, density_He_ion, "density_He_ion")
        store_species_in_file_rz(z, r, series, density_Ar_n, "density_Ar_n")
        store_species_in_file_rz(z, r, series, density_Ar_ion, "density_Ar_ion")
        store_species_in_file_rz(z, r, series, density_N_n, "density_N_n")
        store_species_in_file_rz(z, r, series, density_N_ion, "density_N_ion")
        store_species_in_file_rz(z, r, series, density_e, "density_e")

        # Flush and safely close
        series.flush()
        del series

    def get_domain_extents(header_path):
        """Parse the Castro plotfile Header file to extract spatial domain limits.

        Args:
            header_path (str): Full file path to the plotfile ``Header`` file.

        Returns:
            tuple[float, float]: Upper spatial extents ``(r_max, z_max)``.
        """
        with open(header_path, "r") as f:
            lines = [line.strip() for line in f if line.strip()]
        num_vars = int(lines[1])
        prob_hi_idx = 2 + num_vars + 4
        r_max, z_max = map(float, lines[prob_hi_idx].split())
        return r_max, z_max

    write_plasma_profile(output_dir=output_path)
    print(f'Wrote {output_path}/plasma_density.h5')
