"""
hipace_to_chequp.py
-------------------
Utility to extract ionization fields and temperature from a HiPACE++ simulation (via openPMD)
and write CHEQUP initial-condition files in either 1-D (r) or 2-D (r-z)
openPMD/HDF5 format.

Typical usage
-------------
from hipace_to_chequp import HipaceToChequpConverter

converter = HipaceToChequpConverter(
    path_to_hipace_sim="/data/runs/my_hipace_run/",
    path_to_chequp_code="/data/chequp/",
    path_to_chequp_input="/data/chequp/my_case/",
    species=['H', 'Ar'],
    dim=2,
    r_zoom_um=(0, 50),
    z_zoom_cm=(10, 15),
    N_new=(200, 500)
)
converter.convert(plot=True)
"""

import json
import os
import re
import sys
import numpy as np
import scipy.constants as scc
import tqdm
from openpmd_viewer import OpenPMDTimeSeries
from scipy.interpolate import RegularGridInterpolator, interp1d
from pytools import norm_p


class HipaceToChequpConverter:
    def __init__(
        self,
        path_to_hipace_sim,
        path_to_chequp_code,
        path_to_chequp_input,
        species=None,
        dim=2,
        r_zoom_um=(0, 0),
        z_zoom_cm=(0, 0),
        N_new=(300, 300)
    ):
        """
        Initialize the converter with path and grid parameters.
        
        Parameters
        ----------
        path_to_hipace_sim : str
            Path to the openPMD output directory of the HiPACE++ simulation.
        path_to_chequp_code : str
            Path to the CHEQUP root directory (needed to read species.net and access routines).
        path_to_chequp_input : str
            Directory where the generated 1d_input.h5 or 2d_input.h5 will be saved.
        species : list of str
            Base elements to extract (e.g., ['H', 'Ar']). H, He, N, and Ar are supported.
        dim : int
            Dimensionality of the output (1 for radial only, 2 for r-z grid).
        r_zoom_um : tuple of floats
            Radial window to extract in micrometers (min, max). Defaults to (0,0) which extracts the full grid.
        z_zoom_cm : tuple of floats
            Longitudinal window to extract in centimeters (min, max). Defaults to (0,0) which extracts the full grid.
        N_new : tuple of ints
            Resolution (Nr, Nz) to interpolate the zoomed grid onto.
        """
        self.path_to_hipace_sim = path_to_hipace_sim
        self.path_to_chequp_code = path_to_chequp_code
        self.path_to_chequp_input = path_to_chequp_input
        self.species = species if species is not None else ['H', 'Ar']
        self.dim = dim
        self.r_zoom_um = r_zoom_um
        self.z_zoom_cm = z_zoom_cm
        self.N_new = N_new

    @staticmethod
    def _get_atom_level(species='H'):
        """Returns the supported ionization levels for a given base species."""
        if species == 'H':
            ion_levels = ['0', '1']
        elif species == 'Ar':
            ion_levels = ['0', '1', '2', '3']
        elif species == 'He':
            ion_levels = ['0', '1', '2']
        elif species == 'N':
            ion_levels = ['0', '1', '2', '3', '4', '5']
        else:
            raise ValueError(f'Species {species} not supported')
        return ion_levels

    def _extract_fields_from_hipace(self, ts):
        """
        Loop over every iteration in z_list and collect:
          - electron temperature T_eV  (Nr x Nz)
          - ion weight density n_rz    per species (Nr x Nz)

        Returns a dict with the same layout as species_field_hipace.json.
        """
        species_field = {}
        z_list = ts.iterations  # In HiPACE++ 2D outputs, iterations represent z-slices
        
        # Grab a sample field to establish grid dimensions
        sample_field, m = ts.get_field(field="grid_ionization_ux^2_elec", iteration=ts.iterations[0])
        species_field['r'] = m.x  # m.x is the radial array
        species_field['z'] = m.zmin + scc.c * ts.t  # Convert time to longitudinal z-coordinate
        Nr, Nz = sample_field[0, :, :].shape[0], len(z_list)

        # electron temperature
        T_eV = np.zeros((Nr, Nz))
        for idx, it in tqdm.tqdm(enumerate(z_list), desc="Extracting T_eV", total=Nz):
            # Fetch relativistic momenta (ux, uy, uz) and statistical weights (w)
            ux2 = ts.get_field(field="grid_ionization_ux^2_elec", iteration=it)[0][0, :, :]
            uy2 = ts.get_field(field="grid_ionization_uy^2_elec", iteration=it)[0][0, :, :]
            uz2 = ts.get_field(field="grid_ionization_uz^2_elec", iteration=it)[0][0, :, :]
            w = ts.get_field(field="grid_ionization_w_elec", iteration=it)[0][0, :, :]
            
            # Protect against division by zero where particle weight is 0
            w_inv = np.where(w != 0, 1.0 / w, 0.0)
            
            # Calculate relativistic kinetic energy / temperature in eV
            T_ij = (
                np.sqrt(1.0 + (ux2 * w_inv) + (uy2 * w_inv) + (uz2 * w_inv)) - 1.0
            ) * scc.m_e * scc.c**2 / scc.e
            
            # Take the 1D radial slice at the center of the domain
            T_eV[:, idx] = T_ij[:, ux2.shape[1] // 2]

        # ion densities
        species_field['n'] = {sp + i: {} for sp in self.species for i in self._get_atom_level(sp)}
        species_field['Te_eV'] = T_eV
        
        for atom in self.species:
            for i_level in self._get_atom_level(atom):
                field_name = f"grid_ionization_w_ion_{atom}_{i_level}"
                if field_name not in ts.avail_fields:
                    print(f"Field {field_name} not found in ts.fields")
                    continue
                
                sample = ts.get_field(field=field_name, iteration=ts.iterations[0])[0][0, :, :]
                n_rz = np.zeros((sample.shape[0], Nz))
                
                for idx, it in tqdm.tqdm(enumerate(z_list), desc=f"Extracting {atom+i_level}", total=Nz):
                    # Extract density and take the center slice radially
                    rho = ts.get_field(field=field_name, iteration=it)[0][0, :, :]
                    n_rz[:, idx] = rho[:, rho.shape[1] // 2]
                
                species_field['n'][atom + i_level] = n_rz

        return species_field

    def _plot_fields_1d(self, r_inputs, densities_inputs, T_inputs, r_max_zoom, species_keys):
        """
        Plot 1D (radial) density profiles, grouping all ionization levels of the 
        same species onto a single shared r-axis subplot with species-specific 
        color shading.
        """
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
        import numpy as np

        # Map species to a specific base colormap for visual grouping
        species_colormaps = {
            'H': plt.cm.Reds,
            'He': plt.cm.Blues,
            'N': plt.cm.Greens,
            'Ar': plt.cm.Purples
        }

        # One row per base species, plus one for Temperature at the bottom
        n_rows = len(self.species) + 1
        n_cols = 1

        fig = plt.figure(figsize=(7, 2.5 * n_rows))
        gs = gridspec.GridSpec(n_rows, n_cols, hspace=0.55)

        r_um = r_inputs * 1e6

        # Plot each species on its own subplot
        for i, base_sp in enumerate(self.species):
            ax = fig.add_subplot(gs[i, 0])
            levels = self._get_atom_level(base_sp)
            n_levels = len(levels)
            
            # Select the appropriate colormap (fallback to viridis if not mapped)
            cmap = species_colormaps.get(base_sp, plt.cm.viridis)
            
            # Generate a color gradient for the ionization levels
            # Starting at 0.4 ensures the lightest color is still visible
            colors = cmap(np.linspace(0.4, 0.95, n_levels))
            
            for j, lvl in enumerate(levels):
                sp_key = f"{base_sp}{lvl}"
                if sp_key in species_keys:
                    # Look up the data using the species.net index
                    data = densities_inputs[:, species_keys.index(sp_key)] 
                    ax.plot(r_um, data, linewidth=1.5, color=colors[j], label=sp_key)
            
            ax.set_xlabel('r (µm)')
            ax.set_ylabel('n (arb. units)')
            ax.set_title(f'Density  –  {base_sp}')
            ax.set_xlim(0, r_max_zoom * 1e6)
            ax.grid(True, linewidth=0.4, alpha=0.5)
            ax.legend(loc='upper right', fontsize='small')

        # Plot Electron Temperature at the bottom
        ax_T = fig.add_subplot(gs[len(self.species), 0])
        ax_T.plot(r_um, T_inputs, linewidth=1.5, color='crimson')
        ax_T.set_xlabel('r (µm)')
        ax_T.set_ylabel('Tₑ (eV)')
        ax_T.set_title('Electron temperature')
        ax_T.set_xlim(0, r_max_zoom * 1e6)
        ax_T.grid(True, linewidth=0.4, alpha=0.5)

        plt.tight_layout()
        plt.show()

    def _plot_fields_2d(self, r_inputs, z_new, densities_inputs, T_inputs, r_max_zoom, z_min_zoom, z_max_zoom, species_keys):
        """
        Plot 2D (r-z) density maps for each ionization level of the species
        plus the electron temperature using imshow.
        """
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
        from mpl_toolkits.axes_grid1 import make_axes_locatable

        species_to_plot = []
        for base_sp in self.species:
            for lvl in self._get_atom_level(base_sp):
                species_to_plot.append(f"{base_sp}{lvl}")

        n_species = len(species_to_plot)
        n_cols = 1
        n_rows = n_species + 1

        r_um = r_inputs * 1e6
        z_cm = z_new * 1e2
        plot_extent = [z_cm.min(), z_cm.max(), r_um.min(), r_um.max()]

        fig = plt.figure(figsize=(6, 4.5 * n_rows))
        gs = gridspec.GridSpec(n_rows, n_cols, hspace=0.45)

        # density panels
        for i, sp_key in enumerate(species_to_plot):
            ax = fig.add_subplot(gs[i, 0])
            # Look up the data using the species.net index
            data = densities_inputs[:, :, species_keys.index(sp_key)]
            im = ax.imshow(data, extent=plot_extent, aspect='auto', origin='lower', cmap='viridis')
            ax.set_xlabel('z (cm)')
            ax.set_ylabel('r (µm)')
            ax.set_title(f'Density  –  {sp_key}')
            ax.set_xlim(z_min_zoom * 1e2, z_max_zoom * 1e2)
            ax.set_ylim(0, r_max_zoom * 1e6)
            div = make_axes_locatable(ax)
            cax = div.append_axes('right', size='5%', pad=0.05)
            fig.colorbar(im, cax=cax, label='n (arb. units)')

        # temperature panel
        ax_T = fig.add_subplot(gs[n_species, 0])
        im_T = ax_T.imshow(T_inputs, extent=plot_extent, aspect='auto', origin='lower', cmap='inferno')
        ax_T.set_xlabel('z (cm)')
        ax_T.set_ylabel('r (µm)')
        ax_T.set_title('Electron temperature')
        ax_T.set_xlim(z_min_zoom * 1e2, z_max_zoom * 1e2)
        ax_T.set_ylim(0, r_max_zoom * 1e6)
        div_T = make_axes_locatable(ax_T)
        cax_T = div_T.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(im_T, cax=cax_T, label='Tₑ (eV)')
        
        plt.tight_layout(rect=[0, 0, 1, 0.98])
        plt.show()

    def convert(self, plot=False):
        """
        Reads HiPACE++ simulation data, interpolates onto a zoomed grid, 
        slices for r >= 0, and saves to an OpenPMD format for CHEQUP.
        """
        sys.path.append(f"{self.path_to_chequp_code}/initial_condition")
        from ionization_routines import save_to_openpmd
        
        # 1. Load species keys and atomic weights from CHEQUP code
        species_net_path = f'{self.path_to_chequp_code}/sim_folder/build/species.net'
        with open(species_net_path, 'r') as f:
            content = f.read()

        # Capture groups for short name (e.g., 'H0') and aion (e.g., '1.0078')
        pattern = r'^\s*\w+\s+([A-Z][a-z]*\d)\s+([0-9.]+)'
        matches = re.findall(pattern, content, re.MULTILINE)
        # Store the short names defined in CHEQUP in a list
        species_keys = [match[0] for match in matches]
        # Store the atomic weights in a dictionary
        aion = {match[0]: float(match[1]) for match in matches}

        # 2. Extract field data from HiPACE++ simulation
        print('Loading HiPACE++ data...')
        ts = OpenPMDTimeSeries(self.path_to_hipace_sim)
        species_field = self._extract_fields_from_hipace(ts)

        # Define geometry from the extracted fields
        if self.dim == 1:
            r = np.array(species_field['r'])
        if self.dim == 2:
            r, z = np.array(species_field['r']), np.array(species_field['z'])

        # 3. Handle Zoom / Coordinate Conversion
        # If the user specified a custom zoom window, convert it from um/cm to SI units (meters).
        # We also ensure the requested zoom window doesn't exceed the actual simulation bounds.
        if self.r_zoom_um != (0, 0) and self.z_zoom_cm != (0, 0):
            if np.abs(r.max() * 1e6) > self.r_zoom_um[1] and self.z_zoom_cm[1] < z.max() * 1e2:
                r_min_zoom, r_max_zoom = self.r_zoom_um[0] * 1e-6, self.r_zoom_um[1] * 1e-6
                z_min_zoom, z_max_zoom = self.z_zoom_cm[0] * 1e-2, self.z_zoom_cm[1] * 1e-2
                Nr_new, Nz_new = self.N_new
            else:
                raise ValueError("r_zoom_um and z_zoom_cm are not compatible with the field.")
        else:
            # Fallback to full grid if no zoom was requested
            r_min_zoom, r_max_zoom = r.min(), r.max()
            if self.dim == 2:
                z_min_zoom, z_max_zoom = z.min(), z.max()
                Nr_new, Nz_new = len(r), len(z)
            else:
                Nr_new = len(r)

        # 4. 2D Interpolation Logic
        if self.dim == 2:
            # Create the new r-z grid based on user requested zoom and resolution
            r_new = np.linspace(r_min_zoom, r_max_zoom, Nr_new)
            z_new = np.linspace(z_min_zoom, z_max_zoom, Nz_new)
            R_new, Z_new = np.meshgrid(r_new, z_new, indexing='ij')
            points = np.stack([R_new.ravel(), Z_new.ravel()], axis=-1)
            
            # Slicing index: The HiPACE++ radial grid spans from -r to +r, but 
            # CHEQUP operates on r >= 0. We slice the grid exactly in half.
            half_idx = Nr_new // 2
            r_inputs = r_new[half_idx:]
            densities_inputs = np.zeros((len(r_inputs), Nz_new, len(species_keys)))
            
            # Interpolate Temperature onto the new grid and slice r >= 0
            Te_eV = species_field['Te_eV']
            interp_T = RegularGridInterpolator((r, z), Te_eV, method='linear', bounds_error=False, fill_value=0)
            T_zoom = interp_T(points).reshape(Nr_new, Nz_new)
            T_inputs = T_zoom[half_idx:, :]
            
            # Interpolate Densities Dynamically
            for sp_key in species_field['n'].keys():
                if len(species_field['n'][sp_key]) != 0:
                    n_orig = np.array(species_field['n'][sp_key])  
                    interp_n = RegularGridInterpolator((r, z), n_orig, method='linear', bounds_error=False, fill_value=0)
                    n_zoom = interp_n(points).reshape(Nr_new, Nz_new)
                    n_inputs_sp = n_zoom[half_idx:, :]
                    
                    # CHEQUP needs physical number density divided by atomic mass.
                    # We also apply a floor of 1% of the maximum density to prevent 
                    # zero-density numerical instabilities in CHEQUP.
                    densities_inputs[:, :, species_keys.index(sp_key)] = (n_inputs_sp + 0.01 * np.max(n_inputs_sp)) / aion[sp_key]
            
            # Save to file
            os.makedirs(os.path.dirname(self.path_to_chequp_input), exist_ok=True)
            save_to_openpmd(
                {'r': [0, r_max_zoom], 'z': [z_min_zoom, z_max_zoom]},
                densities_inputs,
                T_inputs + 1e-2 * np.max(T_inputs), # Add small baseline temperature floor 
                f'{self.path_to_chequp_input}/2d_input.h5',
                species_keys
            )
            
            if plot:
                print('Plotting...')
                self._plot_fields_2d(
                    r_inputs, z_new, densities_inputs, T_inputs,
                    r_max_zoom, z_min_zoom, z_max_zoom, species_keys
                )

        # 5. 1D Interpolation Logic
        elif self.dim == 1:
            r_new = np.linspace(r_min_zoom, r_max_zoom, Nr_new)
            
            # Slice for r >= 0 (same logic as 2D)
            half_idx = Nr_new // 2
            r_inputs = r_new[half_idx:]
            densities_inputs = np.zeros((len(r_inputs), len(species_keys)))
            
            # Interpolate Temperature for the center slice
            Te_eV_1d = species_field['Te_eV'][:, 0]
            interp_T = interp1d(r, Te_eV_1d, kind='linear', bounds_error=False, fill_value=0)
            T_zoom = interp_T(r_new)
            T_inputs = T_zoom[half_idx:]
            
            for sp_key in species_field['n'].keys():
                if len(species_field['n'][sp_key]) != 0:
                    n_orig = species_field['n'][sp_key][:, 0]
                    interp_n = interp1d(r, n_orig, kind='linear', bounds_error=False, fill_value=0)
                    n_zoom = interp_n(r_new)
                    n_inputs_sp = n_zoom[half_idx:]
                    
                    # Convert to CHEQUP compatible density with 1% stability floor
                    densities_inputs[:, species_keys.index(sp_key)] = (n_inputs_sp + 0.01 * np.max(n_inputs_sp)) / aion[sp_key]
                
            os.makedirs(os.path.dirname(self.path_to_chequp_input), exist_ok=True)
            save_to_openpmd(
                {'r': [0, r_max_zoom]}, 
                densities_inputs,
                T_inputs + 1e-2 * np.max(T_inputs), 
                f'{self.path_to_chequp_input}/1d_input.h5',
                species_keys
            )
            
            if plot:
                print('Plotting...')
                self._plot_fields_1d(
                    r_inputs, densities_inputs, T_inputs, r_max_zoom, species_keys
                )