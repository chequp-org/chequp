import os
import re
import numpy as np
import yt
import tqdm
yt.set_log_level("error")
from scipy.constants import m_p, k
# Convert constants to cgs
m_H = m_p * 1e3 # g
kB = k * 1e7 # erg/K
e = 1.602e-12 # erg


def get_param_value(filename, param_name):
    """
    Searches for a parameter (like 'problem.a_direct_ioniz') in a Castro-style
    input file and returns its value as a string (or None if not found).
    """
    pattern = rf"^\s*{re.escape(param_name)}\s*=\s*([^\s#//]+)"
    with open(filename) as f:
        for line in f:
            # remove inline comments starting with '#' or '//'
            line = re.split(r'#|//', line)[0]
            match = re.search(pattern, line)
            if match:
                return match.group(1)
    return None


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
        self.output_times = np.array([ float(ds.current_time) for ds in self.ts ])


    def extract_data( self, t, quantity, level ):
        """
        Extract the quantity `quantity` at time `t`, at the required refinement level

        Parameters:
        -----------
        t: float
            time at which to extract the quantity
        quantity: possible values: 'density', 'rho_H0', 'rho_H1', 'pressure', 'Temp', 'T_e', 'T_h'
            quantity to extract
        level: int
            refinement level at which to extract the quantity
        """
        i_output = np.argmin( abs(t-self.output_times) )
        ds = self.ts[i_output]
        if quantity in ['T_e', 'T_h']:
            r, f = _extract_radius_and_quantity( ds, 'f_heavies', level )
            r, e = _extract_radius_and_quantity( ds, 'eint_e', level )
            r, X_H = _extract_radius_and_quantity( ds, 'X(H1)', level )
            if quantity == 'T_e':
                q = 2*m_H*e*(1-f)/(3*X_H*kB)
            else:
                q = 2*m_H*e*f/(3*kB)
        else:
            r, q = _extract_radius_and_quantity( ds, quantity, level )
        return r, q, ds.current_time.to_value()


    def get_energy(self, t, level, energy_type='total', species = ['H0', 'H1']):
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
        # Extract the right energy density, depending on the requested energy type
        if energy_type == 'total':
            r, energy_density, t = self.extract_data(t, 'rho_E', level)
            if bool(get_param_value("../sim_folder/run/inputs.1d.cyl", "castro.add_ext_src")): # Check if ionization is taking into account
                data_ion = {'H0': 0.0, 'H1': 13.598, 'N0': 0.0, 'N1':14.5341, 'N2':29.6013, 'N3':47.4453, 'N4':77.4735, 'N5':97.8901}
                for spe in species:
                    r, ion_density, t = self.extract_data(t, f'rho_{spe}', level)
                    energy_density += data_ion[spe] * e * ion_density / m_p * 1e-3
        elif energy_type == 'thermal':
            r, energy_density, t = self.extract_data(t, 'rho_e', level)
        elif energy_type == 'kinetic':
            r, q1, t = self.extract_data(t, 'rho_E', level)
            r, q2, t = self.extract_data(t, 'rho_e', level)
            energy_density = q1 - q2
        else:
            raise ValueError("Invalid energy type: {energy_type}")

        # Integrate the energy density over the simulation
        dr = r[1] - r[0]
        energy = np.sum( np.pi * ((r+0.5*dr)**2 - (r-0.5*dr)**2) * energy_density )

        return energy, t

    def get_particle_number( self, species_name, species_mass, t, level ):
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
        # Extract the species mass density
        r, species_mass_density, t = self.extract_data(t, f'rho_{species_name}', level)

        # Integrate the energy density over the simulation
        dr = r[1] - r[0]
        mass_density = np.sum( np.pi * ((r+0.5*dr)**2 - (r-0.5*dr)**2) * species_mass_density )

        # Calculate the number of particles
        particle_number = mass_density / species_mass

        return particle_number, t


def _extract_radius_and_quantity( ds, quantity, level ):
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
    if ds.dimensionality==1:
        ad = ds.covering_grid( level=level,
                        left_edge=ds.domain_left_edge,
                        dims=[ds.domain_dimensions[0]*2**level, 1, 1] )
        q = ad[quantity].to_ndarray().squeeze()
        # Find r values of the cell centers
        r_edges = np.linspace(
            ds.domain_left_edge[0],
            ds.domain_right_edge[0],
            ds.domain_dimensions[0]*2**level + 1)
        r = 0.5*(r_edges[1:] + r_edges[:-1])
    elif ds.dimensionality:
        ad = ds.covering_grid( level=level,
                            left_edge=ds.domain_left_edge,
                            dims=[ds.domain_dimensions[0]*2**level, ds.domain_dimensions[1]*2**level, 1] )
        q = ad[quantity].to_ndarray().squeeze()
        q = q[q.shape[0]//2:,q.shape[1]//2]
        r = np.linspace(
            0.5*(ds.domain_left_edge[0] + ds.domain_right_edge[0]),
            ds.domain_right_edge[0],
            ds.domain_dimensions[0]*2**level//2)
        r -= 0.5*(ds.domain_left_edge[0] + ds.domain_right_edge[0])
    return r.to_ndarray(), q