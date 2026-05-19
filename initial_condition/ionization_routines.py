import numpy as np
import numba
from numba import cuda
import math
import tqdm
import os
from scipy.special import gamma
from scipy.constants import m_e, c, e, hbar, physical_constants, epsilon_0, k
import pandas as pd
import openpmd_api as io
import re

def get_adk_parameters():
    """
    Return arrays needed for calculation of ADK ionization rates
    """
    from fbpic.particles.elementary_process.ionization.read_atomic_data import get_ionization_energies

    # Parse the name of the species that are used in Castro
    build_dir = '../sim_folder/build'
    with open(os.path.join(build_dir, 'species.net'), 'r') as f:
        species_keys = re.findall(r'\n\s.*\s([A-Z][a-z]*\d)', f.read())

    # Automatically find information about each transition
    ionization_levels = []
    for species_key in species_keys:
        # Parse the species name and initial charge
        species_name = re.findall(r'[A-Z][a-z]*', species_key)[0]
        initial_charge = int(re.findall(r'\d', species_key)[0])
        # Only consider transitions if the final state is in species_keys
        if species_name + str(initial_charge + 1) in species_keys:
            # Get the ionization energy
            Uion_eV = get_ionization_energies(species_name)[initial_charge] / e
            # Add the transition to the list
            ionization_levels.append( {
                'species_name': species_name,
                'initial_charge': initial_charge,
                'final_charge': initial_charge + 1,
                'Uion_eV': Uion_eV
            } )

    num_species = len(species_keys)
    num_transitions = len(ionization_levels)

    # Calculate ADK parameters for all transitions
    UH = 13.6 * e
    alpha = physical_constants['fine-structure constant'][0]
    r_e = physical_constants['classical electron radius'][0]
    wa = alpha**3 * c / r_e
    Ea = m_e*c**2/e * alpha**4/r_e

    # Calculate l_eff for each species based on their ground state. Generalized for future additions like helium
    species_l_eff = {}
    for transition_dict in ionization_levels:
        species_name = transition_dict['species_name']
        # Only calculate l_eff for ground state (initial_charge = 0)
        if transition_dict['initial_charge'] == 0 and species_name not in species_l_eff:
            ground_state_Uion = transition_dict['Uion_eV'] * e
            ground_state_Z = transition_dict['final_charge']
            ground_state_n_eff = ground_state_Z * np.sqrt(UH/ground_state_Uion)
            species_l_eff[species_name] = ground_state_n_eff - 1

    for transition_dict in ionization_levels:
        transition_dict['Uion'] = transition_dict['Uion_eV'] * e
        transition_dict['Z'] = transition_dict['final_charge']
        transition_dict['n_eff'] = transition_dict['Z'] * np.sqrt( UH/transition_dict['Uion'] )
        # Use the l_eff calculated from the ground state of this species
        transition_dict['l_eff'] = species_l_eff[transition_dict['species_name']]
        transition_dict['C2'] = 2**(2*transition_dict['n_eff']) / (transition_dict['n_eff'] * gamma(transition_dict['n_eff']+transition_dict['l_eff']+1) * gamma(transition_dict['n_eff']-transition_dict['l_eff']))
        transition_dict['adk_power'] = -(2*transition_dict['n_eff'] - 1)
        transition_dict['adk_prefactor'] = wa * transition_dict['C2'] * ( transition_dict['Uion']/(2*UH) ) \
            * ( 2*(transition_dict['Uion']/UH)**(3./2)*Ea )**(2*transition_dict['n_eff'] - 1)
        transition_dict['adk_exp_prefactor'] = -2./3 * ( transition_dict['Uion']/UH )**(3./2) * Ea
    # Create numba-compatible arrays
    adk_prefactors = np.array([d['adk_prefactor'] for d in ionization_levels])
    adk_powers = np.array([d['adk_power'] for d in ionization_levels])
    adk_exp_prefactors = np.array([d['adk_exp_prefactor'] for d in ionization_levels])

    # Create arrays that map each transition to its source and target species index
    source_indices = np.zeros(num_transitions, dtype=np.int64)
    target_indices = np.zeros(num_transitions, dtype=np.int64)
    for i, d in enumerate(ionization_levels):
        source_key = f"{d['species_name']}{d['initial_charge']}"
        target_key = f"{d['species_name']}{d['final_charge']}"
        source_indices[i] = species_keys.index(source_key)
        target_indices[i] = species_keys.index(target_key)

    charges = np.array([float(key[-1]) for key in species_keys], dtype=np.float64)

    return species_keys, adk_prefactors, adk_powers, adk_exp_prefactors, source_indices, target_indices, charges


@numba.njit
def get_fraction_and_temperature_multispecies(a0, tau, lambd, ell,
                                             adk_prefactors, adk_powers, adk_exp_prefactors,
                                             source_indices, target_indices, charges,
                                             initial_populations,
                                             npts_per_wavelength=80):
    
    """
    a0: Peak laser amplitude
    tau: laser FWHM duration
    lambd: laser wavelength
    ell: polarization vector
    n0: electron density if the plasma was fully ionized
    charges : Charge state of each species (e.g., [0, 1, 0, 1, 2, 3] for H0, H1, N0, N1, N2, N3)
    initial_populations : Initial population fractions for each species (should sum to 1.0)
    """
    omega = 2*np.pi*c/lambd
    E0 = m_e*omega*c/e
    inv_tau2 = 1./tau**2

    kin_energy = 0.0
    t = -3*tau
    dt = lambd/c/npts_per_wavelength
    assert len(ell) == 2
    assert abs(ell[0]**2 + ell[1]**2 - 1) < 1.e-10
    while (t < 3*tau):
        a_env = a0 * math.exp(-2 * math.log(2) * inv_tau2*t**2)
        a = a_env * math.sqrt( ell[0]**2*np.cos(omega*t)**2 + ell[1]**2*np.sin(omega*t)**2 )
        E = E0 * a_env * math.sqrt( ell[0]**2*np.sin(omega*t)**2 + ell[1]**2*np.cos(omega*t)**2 )

        total_new_electrons = 0.0
        for i in range(len(adk_prefactors)):
            source_idx = source_indices[i]
            target_idx = target_indices[i]

            w = 0.0
            if E > 0:
                w = adk_prefactors[i] * E**adk_powers[i] * math.exp( adk_exp_prefactors[i]/E )

            dp = 1 - math.exp(-w*dt)
            delta_p = dp * initial_populations[source_idx]

            initial_populations[source_idx] -= delta_p
            initial_populations[target_idx] += delta_p
            total_new_electrons += delta_p

        if total_new_electrons > 0:
            kin_energy += total_new_electrons * m_e*c**2 * (math.sqrt( 1 + a**2 ) - 1)

        t += dt

    T = 0.0
    z_average = 0.0
    for i in range(len(charges)):
        z_average += charges[i] * initial_populations[i]    
    if z_average > 0:
        T = kin_energy / (3/2 * z_average * e)

    return initial_populations, T, t  # Return full populations array

# Automatically detect hardware: use 'cuda' if GPU is available, otherwise 'parallel' (CPU)
if numba.cuda.is_available():
    target_backend = 'cuda'
else:
    target_backend = 'parallel'
    
@numba.guvectorize(
    ['void(float64, float64, float64, float64[:], '
     'float64[:], float64[:], float64[:], '
     'int64[:], int64[:], float64[:], float64[:], '
     'float64[:], float64[:])'],
    '(),(),(),(m),(t),(t),(t),(t),(t),(s),(s)->(s),()',  
    nopython=True,
    target=target_backend
)
def compute_ionization_vectorized(
    a0, tau, lambd, ell,
    adk_prefactors, adk_powers, adk_exp_prefactors,
    source_indices, target_indices, charges, initial_populations,
    populations_out, T_out
):
    for i in range(initial_populations.shape[0]):
        populations_out[i] = initial_populations[i]

    pops, T, _ = get_fraction_and_temperature_multispecies(
        a0, tau, lambd, ell,
        adk_prefactors, adk_powers, adk_exp_prefactors,
        source_indices, target_indices, charges,
        populations_out,
        80
    )

    T_out[0] = T

def save_to_openpmd(grid_extent, all_populations, Te_eV, output_file, species_keys, xmom=0, ymom=0, zmom=0, Th_eV=348.0*(k/e)):
    """
    Save with all species densities (m^-3) to an openPMD file
    """
    # create openpmd file
    series = io.Series(output_file, io.Access.create)
    # only 1 iteration needed
    it = series.iterations[0]
    # Extract information about the grid for openPMD
    grid_spacing = np.array([ (grid_extent[key][1] - grid_extent[key][0]) / (all_populations.shape[i] - 1)
        for i, key in enumerate(grid_extent.keys()) ])
    grid_global_offset = [grid_extent[key][0] for key in grid_extent.keys()]
    axis_labels = list(grid_extent.keys())

    # Save the electron temperature
    Te = it.meshes["Te"]
    Te.grid_spacing = grid_spacing
    Te.grid_global_offset = grid_global_offset
    Te.axis_labels = axis_labels
    Te.unit_dimension = {io.Unit_Dimension.theta: 1}
    dataset = io.Dataset(Te_eV.dtype, Te_eV.shape)
    Te_scalar = Te[io.Mesh_Record_Component.SCALAR]
    Te_scalar.reset_dataset(dataset)
    Te_scalar.position = [0.0] * len(grid_extent)
    Te_scalar.store_chunk(Te_eV * (e / k))  # Convert eV to K
    if np.isscalar(Th_eV): # If Th_eV is a scalar, convert to array
        Th_eV = np.full(Te_eV.shape, Th_eV, dtype=np.float64)
    Th_eV = np.asarray(Th_eV, dtype=np.float64)
    if Th_eV.shape != Te_eV.shape: # Check that the shapes match
        raise ValueError(f"Th_eV shape {Th_eV.shape} does not match Te_eV shape {Te_eV.shape}")
    # Save the heavies temperature
    Th = it.meshes["Th"]
    Th.grid_spacing = grid_spacing
    Th.grid_global_offset = grid_global_offset
    Th.axis_labels = axis_labels
    Th.unit_dimension = {io.Unit_Dimension.theta: 1}
    dataset = io.Dataset(Th_eV.dtype, Th_eV.shape)
    Th_scalar = Th[io.Mesh_Record_Component.SCALAR]
    Th_scalar.reset_dataset(dataset)
    Th_scalar.position = [0.0] * len(grid_extent)
    Th_scalar.store_chunk(Th_eV * (e / k))  # Convert eV to K

    # Save the species densities
    for i, species_key in enumerate(species_keys):
        pop = it.meshes[species_key + "_density"]
        pop.grid_spacing = grid_spacing
        pop.grid_global_offset = grid_global_offset
        pop.axis_labels = axis_labels
        pop.unit_dimension = {io.Unit_Dimension.L: -3}  # m^-3
        dataset = io.Dataset(all_populations[..., i].dtype, all_populations[..., i].shape)
        pop_scalar = pop[io.Mesh_Record_Component.SCALAR]
        pop_scalar.reset_dataset(dataset)
        pop_scalar.position = [0.0] * len(grid_extent)
        pop_scalar.store_chunk(all_populations[..., i].copy())

    # Save the momentum density fields (kg m^-2 s^-1 = rho * v)
    for mom_key, mom_value in zip(["xmom", "ymom", "zmom"], [xmom, ymom, zmom]):
        # Accept either a scalar (constant field) or an array of the same shape as Te_eV
        if np.isscalar(mom_value):
            mom_data = np.full(Te_eV.shape, mom_value, dtype=np.float64)
        else:
            mom_data = np.asarray(mom_value, dtype=np.float64)
            if mom_data.shape != Te_eV.shape:
                raise ValueError(f"{mom_key} shape {mom_data.shape} does not match Te_eV shape {Te_eV.shape}")

        mom_mesh = it.meshes[mom_key]
        mom_mesh.grid_spacing = grid_spacing
        mom_mesh.grid_global_offset = grid_global_offset
        mom_mesh.axis_labels = axis_labels
        mom_mesh.unit_dimension = {
            io.Unit_Dimension.M:  1,
            io.Unit_Dimension.L: -2,
            io.Unit_Dimension.T: -1
        }
        dataset = io.Dataset(mom_data.dtype, mom_data.shape)
        mom_scalar = mom_mesh[io.Mesh_Record_Component.SCALAR]
        mom_scalar.reset_dataset(dataset)
        mom_scalar.position = [0.0] * len(grid_extent)
        mom_scalar.store_chunk(mom_data.copy())

    series.flush()

def load_intensity_profile(filename):
    """
    Load intensity profile from txt or csv file

    Parameters:
    -----------
    filename : str, Path to file containing intensity data

    Returns:
    --------
    intensity_array : 1D or 2D array intensity values (W/m^2)
    """
    if filename.endswith('.csv'):
        intensity_array = np.loadtxt(filename, delimiter=',')
    else:
        intensity_array = np.loadtxt(filename)

    return intensity_array

def process_intensity_array_multispecies(intensity_nd, lambd, tau, ell,
            adk_prefactors, adk_powers, adk_exp_prefactors,
            source_indices, target_indices, charges, species_keys,
            initial_populations, output_file=None, grid_extent=None,
            n_total=1e24):
    """
    Process nD intensity array for multi-species plasma

    Parameters:
    -----------
    intensity_nd : nD array
        Laser intensity profile I in W/m^2
    lambd : float
        Laser wavelength (m)
    tau : float
        Laser pulse FWHM duration (s)
    ell : array-like
        Polarization vector [2-element array]
    initial_populations : dictionary
        Dictionary with elements of species_keys as keys and initial population fractions as values
        If an element from species_keys is not in the dictionary, it is assumed to be 0
    output_file : str, optional
        Filename to save openPMD output with radius, temperature and densities
    grid_extent : dict, optional
        Grid extent for openPMD output
    n_total : float
        Total number density in m^-3 for converting fractions to densities (default: 1e24)

    Returns:
    --------
    all_populations : nD array
        Population fractions for all species (shape: [*intensity_nd.shape, len(species_keys)])
    T_array : nD array
        Electron temperatures in eV, same (shape: intensity_nd)
    """
    # Convert intensity to normalized vector potential a0
    a0_array = e * lambd / (np.pi * m_e * c) * np.sqrt(intensity_nd / (2 * epsilon_0 * c**3))

    # Prepare array of initial populations
    initial_populations = np.array([initial_populations.get(key, 0) for key in species_keys])
    # Check that the sum is 1 to machine precision
    assert np.abs(np.sum(initial_populations) - 1) < 1.e-10

    # Flatten, and prepare arrays for temperature and population
    a0_flat = a0_array.flatten()
    T_flat = np.zeros_like(a0_flat)
    all_populations_flat = np.zeros((len(a0_flat), len(species_keys)))

    compute_ionization_vectorized(
        a0_flat, tau, lambd, ell,
        adk_prefactors, adk_powers, adk_exp_prefactors,
        source_indices, target_indices, charges,
        initial_populations,
        all_populations_flat, T_flat
    )

    # Reshape back to nD arrays
    all_populations = all_populations_flat.reshape(a0_array.shape + (len(species_keys),))
    T_array = T_flat.reshape(a0_array.shape)

    # Save openPMD output with species densities
    if output_file and grid_extent is not None:
        all_densities = all_populations * n_total
        save_to_openpmd(grid_extent, all_densities, T_array, output_file, species_keys)

    return all_populations, T_array
