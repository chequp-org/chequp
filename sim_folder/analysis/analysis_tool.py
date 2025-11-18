import os
import re
import numpy as np
import yt
import tqdm
yt.set_log_level("error")
from scipy.constants import m_p, k, elementary_charge
# Convert constants to cgs
m_H = m_p * 1e3 # g
kB = k * 1e7 # erg/K
qe = elementary_charge * 1e7 # erg

class CastroSimulation(object):
    """
    Class to postprocess the results of a Castro simulation
    """

    def __init__(self, run_dir, file_start):
        """
        Create object to analyze/plot simulation results

        Parameters:
        -----------
        run_dir: string
            path to the folder that contains the output files

        file_start: string
            beginning of the filenames, e.g. 'plt_1d_'
        """
        # Then extract time for each output
        self.ts = yt.load(os.path.join(run_dir, file_start + '*'), hint="castro")
        self.output_times = np.array([ float(ds.current_time) for ds in tqdm.tqdm(self.ts) ])
        self.parameters_list = self.ts[0].parameters
        self.fields_list = [f for _,f in self.ts[0].field_list]
        self.dim = self.ts[0].dimensionality
        self.geo = self.ts[0].geometry

    def sim_info(self):
        """
        Print some information about the simulation
        """
        print("\n Simulation info:")
        print(f" - Dimension: {self.dim}D")
        print(f" - Geometry: {self.geo}")
        print(f" - Available fields: {self.fields_list}")
        print(f" - Species: {self.species}")
        print(f" - Number of outputs: {len(self.ts)}")
        print(f" - Time range: {self.output_times[0]} s to {self.output_times[-1]} s \n")

    def get_field(self, t, quantity, level = 0, positions = {}):
        """
        Extract the quantity `quantity` at the required refinement level

        Parameters:
        -----------
        ds: yt.Dataset
            dataset to extract the quantity from
        quantity: string
            quantity to extract
        level: int
            refinement level at which to extract the quantity
        """
        i_output = np.argmin( abs(t-self.output_times) )
        ds = self.ts[i_output]
        m = {}
        m['t'] = ds.current_time.to_value()

        if quantity not in self.fields_list and quantity not in ['T_e', 'T_h']:
            raise ValueError(f"Quantity {quantity} not found in the simulation outputs")

        if self.dim==1 and self.geo == 'cylindrical':
            ad = ds.covering_grid( level=level,
                            left_edge=ds.domain_left_edge,
                            dims=[ds.domain_dimensions[0]*2**level, 1, 1] )
            
            if quantity in ['T_e', 'T_h']:
                f = ad['rho_f_heavies'].to_ndarray().squeeze()
                e_ = ad['rho_e'].to_ndarray().squeeze()
                X_H = ad['rho_H1'].to_ndarray().squeeze()
                if quantity == 'T_e':
                    m['q'] = 2*m_H*e_*(1-f)/(3*X_H*kB)
                else:
                    m['q'] = 2*m_H*e_*f/(3*kB)
            else:
                m['q'] = ad[quantity].to_ndarray().squeeze()
            # Find r values of the cell centers
            m_edges = np.linspace(
                ds.domain_left_edge[0],
                ds.domain_right_edge[0],
                ds.domain_dimensions[0]*2**level + 1)
            m['r'] = 0.5*(m_edges[1:] + m_edges[:-1])

        elif self.dim==2:
            
            ad = ds.covering_grid( level=level,
                            left_edge=ds.domain_left_edge,
                            dims=[ds.domain_dimensions[0]*2**level, ds.domain_dimensions[1]*2**level, 1] )
            
            if quantity in ['T_e', 'T_h']:
                f = ad['rho_f_heavies'].to_ndarray().squeeze()
                e_ = ad['rho_e'].to_ndarray().squeeze()
                X_H = ad['rho_H1'].to_ndarray().squeeze()
                if quantity == 'T_e':
                    m['q'] = 2*m_H*e_*(1-f)/(3*X_H*kB)
                else:
                    m['q'] = 2*m_H*e_*f/(3*kB)
            else:
                m['q'] = ad[quantity].to_ndarray().squeeze()
            # Find r values of the cell centers
            m_edges_x, m_edges_y = np.linspace(
                ds.domain_left_edge[0],
                ds.domain_right_edge[0],
                ds.domain_dimensions[0]*2**level + 1), np.linspace(
                ds.domain_left_edge[1],
                ds.domain_right_edge[1],
                ds.domain_dimensions[1]*2**level + 1)
            
            if self.geo == 'cylindrical':
                m['r'] = 0.5*(m_edges_x[1:] + m_edges_x[:-1])
                m['z'] = 0.5*(m_edges_y[1:] + m_edges_y[:-1])
            elif self.geo == 'cartesian':
                m['x'] = 0.5*(m_edges_x[1:] + m_edges_x[:-1])
                m['y'] = 0.5*(m_edges_y[1:] + m_edges_y[:-1])
        
        elif self.dim==3 and self.geo == 'cartesian':
            
            ad = ds.covering_grid(level=level,
                            left_edge=ds.domain_left_edge,
                            dims=[ds.domain_dimensions[0]*2**level, 
                                  ds.domain_dimensions[1]*2**level,
                                  ds.domain_dimensions[2]*2**level])
            
            if quantity in ['T_e', 'T_h']:
                f = ad['rho_f_heavies'].to_ndarray().squeeze()
                e_ = ad['rho_e'].to_ndarray().squeeze()
                X_H = ad['rho_H1'].to_ndarray().squeeze()
                if quantity == 'T_e':
                    m['q'] = 2*m_H*e_*(1-f)/(3*X_H*kB)
                else:
                    m['q'] = 2*m_H*e_*f/(3*kB)
            else:
                m['q'] = ad[quantity].to_ndarray().squeeze()
            # Find r values of the cell centers
            m_edges_x, m_edges_y, m_edges_z = np.linspace(
                ds.domain_left_edge[0],
                ds.domain_right_edge[0],
                ds.domain_dimensions[0]*2**level + 1), np.linspace(
                ds.domain_left_edge[1],
                ds.domain_right_edge[1],
                ds.domain_dimensions[1]*2**level + 1), np.linspace(
                ds.domain_left_edge[2],
                ds.domain_right_edge[2],
                ds.domain_dimensions[2]*2**level + 1)
            m['x'] = 0.5*(m_edges_x[1:] + m_edges_x[:-1])
            m['y'] = 0.5*(m_edges_y[1:] + m_edges_y[:-1])
            m['z'] = 0.5*(m_edges_z[1:] + m_edges_z[:-1])

        if bool(positions.keys() & m.keys()):

            if self.dim == 2:
                key = [k for k in positions.keys() if k in m.keys()][0]
                idx = (np.abs(np.array(m[key]) - positions[key])).argmin()
                if key == 'r' or key == 'x':
                    m['q'] = m['q'][idx,:]
                    m[key[1]] = m[key[1]]
                elif key == 'z' or key == 'y':
                    m['q'] = m['q'][:,idx]
                    m[key[0]] = m[key[0]]
        
        return m

    def get_energy(self, t, level, energy_type='total'):
        """
        Get the energy (either kinetic, thermal, or sum of both)
        at time `t`, at the required refinement level

        Parameters:
        -----------
        energy_type: string
            type of energy to get; possible values: 'kinetic', 'thermal', 'total';
            'total is the sum of the kinetic and thermal energies
        t: float
            time at which to get the energy
        level: int
            refinement level at which to get the energy

        Returns:
        --------
        energy: float, in erg/cm
            The energy per unit length (in 1D cylindrical)
        t: float, in s
            The exact time at which the energy was extracted
        """
        energies, lt = [], []
        data_ion = {'H0': 0.0, 'H1': 13.598, 'N0': 0.0, 'N1':14.5341, 'N2':29.6013, 'N3':47.4453, 'N4':77.4735, 'N5':97.8901}
        t = [t] if np.isscalar(t) else t
        for _t in tqdm.tqdm(t):
            lt.append(_t)
            # Extract the right energy density, depending on the requested energy type
            if energy_type == 'total':
                m_rho_E = self.get_field(_t, 'rho_E', level)
                if bool(self.parameters_list["castro.add_ext_src"]): # Check if ionization is taking into account  
                    for spe in [f for f in self.fields_list if f[-1].isdigit()]:
                        m_rho_X = self.get_field(_t, f'{spe}', level)
                        m_rho_E['q'] += data_ion[spe[-2:]] * qe * m_rho_X['q'] / m_p * 1e-3
            elif energy_type == 'ion':
                m_rho_E = self.get_field(_t, 'rho_E', level)
                m_rho_E['q'] *= 0.0
                if bool(self.parameters_list["castro.add_ext_src"]): # Check if ionization is taking into account  
                    for spe in [f for f in self.fields_list if f[-1].isdigit()]:
                        m_rho_X = self.get_field(_t, f'{spe}', level)
                        m_rho_E['q'] += data_ion[spe[-2:]] * qe * m_rho_X['q'] / m_p * 1e-3
                else:
                    raise ValueError("Ionization energy requested but 'castro.add_ext_src' is not activated in the simulation")
            elif energy_type == 'thermal':
                m_rho_E = self.get_field(_t, 'rho_e', level)
            elif energy_type == 'kinetic':
                m_rho_E = self.get_field(_t, 'rho_E', level)
                m_rho_E_2 = self.get_field(_t, 'rho_e', level)
                m_rho_E['q'] -= m_rho_E_2['q']
            else:
                raise ValueError("Invalid energy type: {energy_type}")

            # Integrate the energy density over the simulation
            if self.dim == 1 and self.geo == 'cylindrical':
                dr = m_rho_E['r'][1] - m_rho_E['r'][0]
                energy = np.sum( np.pi * ((m_rho_E['r']+0.5*dr)**2 - (m_rho_E['r']-0.5*dr)**2) * m_rho_E['q'])
                energies.append(energy)
            elif self.dim == 2 and self.geo == 'cartesian':
                dx = m_rho_E['x'][1] - m_rho_E['x'][0]
                dy = m_rho_E['y'][1] - m_rho_E['y'][0]
                energy = np.sum( (dx * dy) * m_rho_E['q'])
                energies.append(energy)
            elif self.dim == 2 and self.geo == 'cylindrical':
                dr = m_rho_E['r'][1] - m_rho_E['r'][0]
                dz = m_rho_E['z'][1] - m_rho_E['z'][0]
                energy = np.sum( 2 * np.pi * m_rho_E['r'][:,None] * (dr * dz) * m_rho_E['q'] )
                energies.append(energy)
        
        if len(energies) == 1:
            return float(energies[0]), float(lt[0])
        return energies, lt

    def get_particle_number( self, t, species, level ):
        """
        Get the number of particles of the species `species_name`
        at time `t`, at the required refinement level

        Parameters:
        -----------
        species_name: string
            name of the species to get the number of particles from
        species_mass: float
            mass of the species in g
        t: float
            time at which to get the number of particles
        level: int
            refinement level at which to get the number of particles

        Returns:
        --------
        particle_number: float
            The number of particles of the species `species_name` per unit length (in 1D cylindrical)
        t: float, in s
            The exact time at which the number of particles was extracted
        """

        part_array, lt = [], []
        t = [t] if np.isscalar(t) else t
        for t in tqdm.tqdm(t):
            lt.append(t)
            # Define the species mass
            species_mass = {'H0': 1.00784 * m_H,
                            'H1': 1.00784 * m_H,
                            'N0': 14.0067 * m_H,
                            'N1': 14.0067 * m_H,
                            'N2': 14.0067 * m_H,
                            'N3': 14.0067 * m_H,
                            'N4': 14.0067 * m_H,
                            'N5': 14.0067 * m_H}
            
            if species not in self.species:
                raise ValueError(f"Species {species} not found in the simulation outputs")
            # Extract the species mass density
            m = self.get_field(t, f'rho_{species}', level)

            # Integrate the energy density over the simulation
            if self.dim == 1 and self.geo == 'cylindrical':
                dr = m['r'][1] - m['r'][0]
                mass_density = np.sum( np.pi * ((m['r']+0.5*dr)**2 - (m['r']-0.5*dr)**2) * m['q'] )
            elif self.dim == 2 and self.geo == 'cartesian':
                dx = m['x'][1] - m['x'][0]
                dy = m['y'][1] - m['y'][0]
                mass_density = np.sum( (dx * dy) * m['q'] )
            elif self.dim == 2 and self.geo == 'cylindrical':
                dr = m['r'][1] - m['r'][0]
                dz = m['z'][1] - m['z'][0]
                mass_density = np.sum( 2 * np.pi * m['r'][:,None] * (dr * dz) * m['q'] )

            elif self.dim == 3 and self.geo == 'cartesian':
                dx = m['x'][1] - m['x'][0]
                dy = m['y'][1] - m['y'][0]
                dz = m['z'][1] - m['z'][0]
                mass_density = np.sum( (dx * dy * dz) * m['q'] )

            # Calculate the number of particles
            particle_number = mass_density / species_mass[species]
            part_array.append(float(particle_number))
        
        if len(part_array) > 1:
            return part_array, lt
        return particle_number, t
