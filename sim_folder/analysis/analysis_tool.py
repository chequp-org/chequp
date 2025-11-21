import os
import re
import numpy as np
import yt
import tqdm
import json
yt.set_log_level("error")
from scipy.constants import m_p, k, elementary_charge

# Convert physical constants to CGS units
m_H = m_p * 1e3  # Hydrogen mass in grams
kB = k * 1e7     # Boltzmann constant in erg/K
qe = elementary_charge * 1e7  # Elementary charge in erg

MODULE_DIR = Path(__file__).resolve().parent
json_file = MODULE_DIR / "database_species.json"

class CastroSimulation(object):
    """
    Class to postprocess and analyze results from Castro simulations.
    
    This class provides methods to:
    - Load simulation data from Castro output files
    - Extract physical quantities at different refinement levels
    - Calculate energies (kinetic, thermal, ionization, total)
    - Compute particle numbers for different species
    - Handle 1D/2D/3D simulations with Cartesian or cylindrical geometry
    """

    def __init__(self, run_dir, file_start):
        """
        Initialize the Castro simulation analysis object.

        Parameters
        ----------
        run_dir : str
            Path to the directory containing Castro output files
        file_start : str
            Prefix of the output filenames (e.g., 'plt_1d_', 'plt00000')
            
        Attributes
        ----------
        ts : yt.DatasetSeries
            Time series of all simulation outputs
        output_times : np.ndarray
            Array of simulation times for each output
        parameters_list : dict
            Simulation parameters from the first output file
        fields_list : list
            List of available physical quantities in the simulation
        dim : int
            Dimensionality of the simulation (1D, 2D, or 3D)
        geo : str
            Geometry type ('Cartesian' or 'cylindrical')
        """
        # Load all simulation outputs as a time series
        self.ts = yt.load(os.path.join(run_dir, file_start + '*'), hint="castro")
        
        # Extract simulation times from each output file
        self.output_times = np.array([float(ds.current_time) for ds in tqdm.tqdm(self.ts)])
        
        # Get simulation metadata from the first output
        self.parameters_list = self.ts[0].parameters
        self.fields_list = [f for _, f in self.ts[0].field_list] + ['T_e', 'T_h'] # T_e and T_h are derived quantities
        self.species_list = [f[-2:] for f in self.fields_list if f[-1].isdigit()]
        self.dim = self.ts[0].dimensionality
        self.geo = self.ts[0].geometry
        self.max_level = self.ts[0].max_level  # Number of AMR levels
        with open(json_file) as f: # Load species data: energies in eV, masses in g
            self.data_species = json.load(f)

    def sim_info(self):
        """
        Display comprehensive information about the loaded simulation.
        
        Prints simulation dimensionality, geometry, available fields,
        number of outputs, and time range.
        """
        print("\n Simulation info:")
        print(f" - Dimension: {self.dim}D")
        print(f" - Geometry: {self.geo}")
        print(f" - Available fields: {self.fields_list}")
        print(f" - Number of AMR levels: {self.max_level}")
        print(f" - Number of outputs: {len(self.ts)}")
        print(f" - Time range: {self.output_times[0]} s to {self.output_times[-1]} s")

    def get_field(self, t, quantity, level=0, positions={}):
        """
        Extract a physical quantity from the simulation at a specific time and refinement level.

        Parameters
        ----------
        t : float
            Time at which to extract the quantity (finds closest output)
        quantity : str
            Name of the physical quantity to extract
            Special computed quantities: 'T_e' (electron temperature), 'T_h' (heavy particle temperature)
        level : int, optional
            AMR refinement level to extract data from (default: 0, coarsest level)
        positions : dict, optional
            Dictionary to extract 1D slices from 2D data (e.g., {'r': 0.5} or {'z': 1.0})

        Returns
        -------
        dict
            Dictionary containing:
            - 't': exact simulation time
            - 'q': the requested quantity values
            - coordinate arrays ('r', 'z', 'x', 'y') depending on geometry and dimension
            
        Raises
        ------
        ValueError
            If the requested quantity is not available in the simulation
        """
        if level < 0 or level > self.max_level:
            level = self.max_level
            print(f"Requested level out of bounds. Using max level: {self.max_level}")
        # Find the output closest to the requested time
        i_output = np.argmin(abs(t - self.output_times))
        ds = self.ts[i_output]
        
        # Initialize result dictionary
        m = {'t': ds.current_time.to_value()}

        # Validate that the requested quantity exists
        if quantity not in self.fields_list and quantity not in ['T_e', 'T_h']: 
            raise ValueError(f"Quantity {quantity} not found in the simulation outputs")

        # Handle 1D cylindrical geometry
        if self.dim == 1 and self.geo == 'cylindrical':
            # Create covering grid at specified refinement level
            ad = ds.covering_grid(level=level,
                                left_edge=ds.domain_left_edge,
                                dims=[ds.domain_dimensions[0] * 2**level, 1, 1])
            
            # Calculate derived temperatures if requested (not directly stored)
            if quantity in ['T_e', 'T_h']:
                f = ad['rho_f_heavies'].to_ndarray().squeeze() / ad['density'].to_ndarray().squeeze() # Heavy particle fraction
                e_ = ad['rho_e'].to_ndarray().squeeze()             # Internal energy density
                X_H = ad['rho_H1'].to_ndarray().squeeze() / ad['density'].to_ndarray().squeeze() # Hydrogen mass fraction
                if quantity == 'T_e':
                    # Electron temperature: T_e = (2/3) * (internal energy per electron) / k_B
                    m['q'] = 2 * m_H * e_ * (1 - f) / (3 * X_H * kB)
                else:
                    # Heavy particle temperature
                    m['q'] = 2 * m_H * e_ * f / (3 * kB)
            else:
                # Extract field directly from simulation
                m['q'] = ad[quantity].to_ndarray().squeeze()
            
            # Calculate radial coordinates of cell centers
            m_edges = np.linspace(ds.domain_left_edge[0], ds.domain_right_edge[0],
                                ds.domain_dimensions[0] * 2**level + 1)
            m['r'] = np.array(0.5 * (m_edges[1:] + m_edges[:-1]), dytype=float)

        # Handle 2D simulations
        elif self.dim == 2:
            # Create 2D covering grid
            ad = ds.covering_grid(level=level,
                                left_edge=ds.domain_left_edge,
                                dims=[ds.domain_dimensions[0] * 2**level, 
                                      ds.domain_dimensions[1] * 2**level, 1])
            
            # Calculate derived temperatures (same as 1D case)
            if quantity in ['T_e', 'T_h']:
                f = ad['rho_f_heavies'].to_ndarray().squeeze() / ad['density'].to_ndarray().squeeze()
                e_ = ad['rho_e'].to_ndarray().squeeze()
                X_H = ad['rho_H1'].to_ndarray().squeeze() / ad['density'].to_ndarray().squeeze()
                if quantity == 'T_e':
                    m['q'] = 2 * m_H * e_ * (1 - f) / (3 * X_H * kB)
                else:
                    m['q'] = 2 * m_H * e_ * f / (3 * kB)
            else:
                m['q'] = ad[quantity].to_ndarray().squeeze()
            
            # Calculate coordinate arrays for both directions
            m_edges_x = np.linspace(ds.domain_left_edge[0], ds.domain_right_edge[0],
                                  ds.domain_dimensions[0] * 2**level + 1)
            m_edges_y = np.linspace(ds.domain_left_edge[1], ds.domain_right_edge[1],
                                  ds.domain_dimensions[1] * 2**level + 1)
            
            # Set coordinate names based on geometry
            if self.geo == 'cylindrical':
                m['r'] = np.array(0.5 * (m_edges_x[1:] + m_edges_x[:-1]), dtype=float)  # Radial coordinate
                m['z'] = np.array(0.5 * (m_edges_y[1:] + m_edges_y[:-1]), dtype=float)  # Axial coordinate
            elif self.geo == 'cartesian':
                m['x'] = np.array(0.5 * (m_edges_x[1:] + m_edges_x[:-1]), dtype=float)  # x-coordinate
                m['y'] = np.array(0.5 * (m_edges_y[1:] + m_edges_y[:-1]), dtype=float)  # y-coordinate
        
        # Handle 3D Cartesian simulations
        elif self.dim == 3 and self.geo == 'cartesian':
            # Create 3D covering grid
            ad = ds.covering_grid(level=level,
                                left_edge=ds.domain_left_edge,
                                dims=[ds.domain_dimensions[0] * 2**level, 
                                      ds.domain_dimensions[1] * 2**level,
                                      ds.domain_dimensions[2] * 2**level])
            
            # Calculate derived temperatures (same as previous cases)
            if quantity in ['T_e', 'T_h']:
                f = ad['rho_f_heavies'].to_ndarray().squeeze() / ad['density'].to_ndarray().squeeze()
                e_ = ad['rho_e'].to_ndarray().squeeze()
                X_H = ad['rho_H1'].to_ndarray().squeeze() / ad['density'].to_ndarray().squeeze()
                
                if quantity == 'T_e':
                    m['q'] = 2 * m_H * e_ * (1 - f) / (3 * X_H * kB)
                else:
                    m['q'] = 2 * m_H * e_ * f / (3 * kB)
            else:
                m['q'] = ad[quantity].to_ndarray().squeeze()
            
            # Calculate all three coordinate arrays
            m_edges_x = np.linspace(ds.domain_left_edge[0], ds.domain_right_edge[0],
                                  ds.domain_dimensions[0] * 2**level + 1)
            m_edges_y = np.linspace(ds.domain_left_edge[1], ds.domain_right_edge[1],
                                  ds.domain_dimensions[1] * 2**level + 1)
            m_edges_z = np.linspace(ds.domain_left_edge[2], ds.domain_right_edge[2],
                                  ds.domain_dimensions[2] * 2**level + 1)
            
            m['x'] = np.array(0.5 * (m_edges_x[1:] + m_edges_x[:-1]), dtype=float)
            m['y'] = np.array(0.5 * (m_edges_y[1:] + m_edges_y[:-1]), dtype=float)
            m['z'] = np.array(0.5 * (m_edges_z[1:] + m_edges_z[:-1]), dtype=float)

        # Extract 1D slices from 2D data if positions are specified
        if bool(positions.keys() & m.keys()):
            if self.dim == 2:
                # Find the coordinate key that matches the positions dictionary
                key = [k for k in positions.keys() if k in m.keys()][0]
                # Find the index closest to the requested position
                idx = (np.abs(np.array(m[key]) - positions[key])).argmin()
                
                # Extract slice depending on which coordinate is fixed
                if key == 'r' or key == 'x':
                    m['q'] = m['q'][idx, :]  # Fix first coordinate, vary second
                    m[key[1]] = m[key[1]]    # Keep the varying coordinate
                elif key == 'z' or key == 'y':
                    m['q'] = m['q'][:, idx]  # Fix second coordinate, vary first
                    m[key[0]] = m[key[0]]    # Keep the varying coordinate
        
        return m

    def get_energy(self, t, level, energy_type='total'):
        """
        Calculate the total energy content of the simulation at specified time(s).

        Parameters
        ----------
        t : float or array-like
            Time(s) at which to calculate energy
        level : int
            AMR refinement level for energy calculation
        energy_type : str, optional
            Type of energy to calculate:
            - 'total': kinetic + thermal + ionization energy
            - 'thermal': internal energy only
            - 'kinetic': kinetic energy only
            - 'ion': ionization energy only

        Returns
        -------
        energy : float or list
            Total energy in erg (or erg/cm for 1D cylindrical)
        time : float or list
            Exact time(s) at which energy was calculated
            
        Raises
        ------
        ValueError
            If invalid energy_type is specified or ionization energy is requested
            but not available in the simulation
        """
        if level < 0 or level > self.max_level:
            level = self.max_level
            print(f"Requested level out of bounds. Using max level: {self.max_level}")
        energies, lt = [], []
        
        # Ensure t is iterable
        t = [t] if np.isscalar(t) else t
        
        for _t in tqdm.tqdm(t):
            lt.append(_t)
            
            # Extract appropriate energy density based on requested type
            if energy_type == 'total':
                # Start with total energy density (kinetic + thermal)
                m_rho_E = self.get_field(_t, 'rho_E', level)
                # Add ionization energy if available
                if bool(self.parameters_list["castro.add_ext_src"]):
                    for spe in [f for f in self.fields_list if f[-1].isdigit()]:
                        m_rho_X = self.get_field(_t, f'{spe}', level)
                        # Add ionization energy: n_species * ionization_potential
                        m_rho_E['q'] += self.data_species[spe[-2:]]['ion_energy'] * qe * m_rho_X['q'] / m_p * 1e-3
                        
            elif energy_type == 'ion':
                # Calculate only ionization energy
                m_rho_E = self.get_field(_t, 'rho_E', level)
                m_rho_E['q'] *= 0.0  # Zero out the array
                if bool(self.parameters_list["castro.add_ext_src"]):
                    for spe in [f for f in self.fields_list if f[-1].isdigit()]:
                        m_rho_X = self.get_field(_t, f'{spe}', level)
                        m_rho_E['q'] += self.data_species[spe[-2:]]['ion_energy'] * qe * m_rho_X['q'] / m_p * 1e-3
                else:
                    raise ValueError("Ionization energy requested but 'castro.add_ext_src' is not activated")
                    
            elif energy_type == 'thermal':
                # Internal (thermal) energy density only
                m_rho_E = self.get_field(_t, 'rho_e', level)
                
            elif energy_type == 'kinetic':
                # Kinetic energy = total energy - internal energy
                m_rho_E = self.get_field(_t, 'rho_E', level)
                m_rho_E_2 = self.get_field(_t, 'rho_e', level)
                m_rho_E['q'] -= m_rho_E_2['q']
            else:
                raise ValueError(f"Invalid energy type: {energy_type}")

            # Integrate energy density over the simulation domain
            if self.dim == 1 and self.geo == 'cylindrical':
                # Cylindrical volume integration: ∫ 2πr * energy_density * dr
                dr = m_rho_E['r'][1] - m_rho_E['r'][0]
                energy = np.sum(np.pi * ((m_rho_E['r'] + 0.5*dr)**2 - 
                                       (m_rho_E['r'] - 0.5*dr)**2) * m_rho_E['q'])
                energies.append(energy)
                
            elif self.dim == 2 and self.geo == 'cartesian':
                # 2D Cartesian integration: ∫∫ energy_density * dx * dy
                dx = m_rho_E['x'][1] - m_rho_E['x'][0]
                dy = m_rho_E['y'][1] - m_rho_E['y'][0]
                energy = np.sum((dx * dy) * m_rho_E['q'])
                energies.append(energy)
                
            elif self.dim == 2 and self.geo == 'cylindrical':
                # 2D cylindrical integration: ∫∫ 2πr * energy_density * dr * dz
                dr = m_rho_E['r'][1] - m_rho_E['r'][0]
                dz = m_rho_E['z'][1] - m_rho_E['z'][0]
                energy = np.sum(2 * np.pi * m_rho_E['r'][:, None] * 
                              (dr * dz) * m_rho_E['q'])
                energies.append(energy)
        
        # Return single values if only one time was requested
        if len(energies) == 1:
            return float(energies[0]), float(lt[0])
        return energies, lt

    def get_particle_number(self, t, species, level):
        """
        Calculate the total number of particles of a specific species in the simulation.

        Parameters
        ----------
        t : float or array-like
            Time(s) at which to calculate particle number
        species : str
            Species identifier (e.g., 'H0', 'H1', 'N0', 'N1', etc.)
        level : int
            AMR refinement level for calculation

        Returns
        -------
        particle_number : float or list
            Total number of particles of the specified species
        time : float or list
            Exact time(s) at which particle number was calculated
            
        Raises
        ------
        ValueError
            If the specified species is not found in the simulation
        """
        if level < 0 or level > self.max_level:
            level = self.max_level
            print(f"Requested level out of bounds. Using max level: {self.max_level}")
        part_array, lt = [], []
        
        # Ensure t is iterable
        t = [t] if np.isscalar(t) else t
        
        for t in tqdm.tqdm(t):
            lt.append(t)
            # Check if species exists in simulation
            if species not in self.species_list:
                print(f"Available species: {self.species_list}")
                raise ValueError(f"Species {species} not found in the simulation outputs")
            
            # Extract mass density of the specified species
            m = self.get_field(t, f'rho_{species}', level)

            # Integrate mass density over the simulation domain
            if self.dim == 1 and self.geo == 'cylindrical':
                dr = m['r'][1] - m['r'][0]
                mass_density = np.sum(np.pi * ((m['r'] + 0.5*dr)**2 - 
                                             (m['r'] - 0.5*dr)**2) * m['q'])
                                             
            elif self.dim == 2 and self.geo == 'cartesian':
                dx = m['x'][1] - m['x'][0]
                dy = m['y'][1] - m['y'][0]
                mass_density = np.sum((dx * dy) * m['q'])
                
            elif self.dim == 2 and self.geo == 'cylindrical':
                dr = m['r'][1] - m['r'][0]
                dz = m['z'][1] - m['z'][0]
                mass_density = np.sum(2 * np.pi * m['r'][:, None] * (dr * dz) * m['q'])

            elif self.dim == 3 and self.geo == 'cartesian':
                dx = m['x'][1] - m['x'][0]
                dy = m['y'][1] - m['y'][0]
                dz = m['z'][1] - m['z'][0]
                mass_density = np.sum((dx * dy * dz) * m['q'])

            # Convert total mass to particle number using species mass
            particle_number = mass_density / self.data_species[species]['mass']
            part_array.append(float(particle_number))
        
        # Return single values if only one time was requested
        if len(part_array) > 1:
            return part_array, lt
        return float(part_array[0]), lt[0]
