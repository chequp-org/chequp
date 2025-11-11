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


@cuda.jit(device=True)
def get_fraction_and_temperature_multispecies_device(a0, tau, lambd, ell,
                                             adk_prefactors, adk_powers, adk_exp_prefactors,
                                             source_indices, target_indices, charges,
                                             initial_populations,
                                             populations_out, T_out,
                                             npts_per_wavelength=80):
    """
    GPU device function for ionization calculation
    This version writes directly to output arrays instead of returning values
    """
    omega = 2*math.pi*c/lambd
    E0 = m_e*omega*c/e
    inv_tau2 = 1./tau**2

    # Copy initial populations to output array
    for i in range(len(initial_populations)):
        populations_out[i] = initial_populations[i]
    
    kin_energy = 0.0
    t = -3*tau
    dt = lambd/c/npts_per_wavelength
    
    while (t < 3*tau):
        a_env = a0 * math.exp(-2 * math.log(2) * inv_tau2*t**2)
        a = a_env * math.sqrt( ell[0]**2*math.cos(omega*t)**2 + ell[1]**2*math.sin(omega*t)**2 )
        E = E0 * a_env * math.sqrt( ell[0]**2*math.sin(omega*t)**2 + ell[1]**2*math.cos(omega*t)**2 )

        total_new_electrons = 0.0
        for i in range(len(adk_prefactors)):
            source_idx = source_indices[i]
            target_idx = target_indices[i]

            w = 0.0
            if E > 0:
                w = adk_prefactors[i] * E**adk_powers[i] * math.exp( adk_exp_prefactors[i]/E )

            dp = 1 - math.exp(-w*dt)
            delta_p = dp * populations_out[source_idx]

            populations_out[source_idx] -= delta_p
            populations_out[target_idx] += delta_p
            total_new_electrons += delta_p

        if total_new_electrons > 0:
            kin_energy += total_new_electrons * m_e*c**2 * (math.sqrt( 1 + a**2 ) - 1)

        t += dt

    T = 0.0
    z_average = 0.0
    for i in range(len(charges)):
        z_average += charges[i] * populations_out[i]
    
    if z_average > 0:
        T = kin_energy / (3/2 * z_average * e)
    
    T_out[0] = T


@cuda.jit
def ionization_kernel_gpu(a0_array, tau, lambd, ell,
                         adk_prefactors, adk_powers, adk_exp_prefactors,
                         source_indices, target_indices, charges,
                         initial_populations,
                         all_populations, T_array, npts_per_wavelength):
    """
    CUDA kernel for parallel ionization calculation on GPU
    Each thread processes one spatial point
    
    Local array size is hardcoded to 16 species max.
    """
    idx = cuda.grid(1)
    
    if idx < a0_array.shape[0]:
        # Allocate local arrays for this thread
        num_species = len(initial_populations)
        populations_local = cuda.local.array(16, numba.float64)
        T_local = cuda.local.array(1, numba.float64)
        
        # Call device function
        get_fraction_and_temperature_multispecies_device(
            a0_array[idx], tau, lambd, ell,
            adk_prefactors, adk_powers, adk_exp_prefactors,
            source_indices, target_indices, charges,
            initial_populations,
            populations_local, T_local,
            npts_per_wavelength
        )
        
        # Copy results to global memory
        for i in range(num_species):
            all_populations[idx, i] = populations_local[i]
        T_array[idx] = T_local[0]


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

    populations = initial_populations.copy()
    kin_energy = 0.0
    t = -3*tau
    dt = lambd/c/npts_per_wavelength
    assert len(ell) == 2
    assert abs(ell[0]**2 + ell[1]**2 - 1) < 1.e-10
    while (t < 3*tau):
        a_env = a0 * math.exp(-2 * np.log(2) * inv_tau2*t**2)
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
            delta_p = dp * populations[source_idx]

            populations[source_idx] -= delta_p
            populations[target_idx] += delta_p
            total_new_electrons += delta_p

        if total_new_electrons > 0:
            kin_energy += total_new_electrons * m_e*c**2 * (math.sqrt( 1 + a**2 ) - 1)

        t += dt

    T = 0.0
    z_average = np.sum(charges * populations)
    if z_average > 0:
        T = kin_energy / (3/2 * z_average * e)
    return populations, T, t


@numba.guvectorize(
    [(numba.float64, numba.float64, numba.float64, numba.float64[:],
      numba.float64[:], numba.float64[:], numba.float64[:],
      numba.int64[:], numba.int64[:], numba.float64[:], numba.float64[:],
      numba.float64[:], numba.float64[:])],
    '(),(),(),(p),(t),(t),(t),(t),(t),(s),(s)->(s),()',
    target='parallel',
    nopython=True
)
def _vectorized_ionization_cpu(a0_scalar, tau_scalar, lambd_scalar, ell,
                           adk_prefactors, adk_powers, adk_exp_prefactors,
                           source_indices, target_indices, charges,
                           initial_populations,
                           populations_out, T_out):
    """
    CPU-parallelized wrapper (original version)
    """
    populations, T, _ = get_fraction_and_temperature_multispecies(
        a0_scalar, tau_scalar, lambd_scalar, ell,
        adk_prefactors, adk_powers, adk_exp_prefactors,
        source_indices, target_indices, charges,
        initial_populations
    )
    populations_out[:] = populations
    T_out[0] = T


def process_on_gpu(a0_array, tau, lambd, ell,
                   adk_prefactors, adk_powers, adk_exp_prefactors,
                   source_indices, target_indices, charges,
                   initial_populations,
                   threads_per_block=256,
                   npts_per_wavelength=80):
    """
    Process ionization calculation on GPU
    
    Parameters:
    -----------
    a0_array : 1D array
        Flattened array of normalized vector potentials
    ... (other parameters same as before)
    threads_per_block : int
        Number of threads per CUDA block
    npts_per_wavelength : int
        Number of time steps per laser wavelength for integration
    
    Returns:
    --------
    all_populations : 2D array (n_points, n_species)
    T_array : 1D array (n_points,)
    """
    n_points = len(a0_array)
    n_species = len(initial_populations)
    
    # Check if species count exceeds local array size
    MAX_SPECIES = 16  # Must match cuda.local.array size in kernel
    if n_species > MAX_SPECIES:
        raise ValueError(f"Number of species ({n_species}) exceeds maximum ({MAX_SPECIES}). "
                        f"Edit ionization_kernel_gpu and increase cuda.local.array size.")
    
    
    # Allocate output arrays on CPU
    all_populations = np.zeros((n_points, n_species), dtype=np.float64)
    T_array = np.zeros(n_points, dtype=np.float64)
    
    # Transfer data to GPU
    d_a0 = cuda.to_device(a0_array)
    d_ell = cuda.to_device(ell)
    d_adk_prefactors = cuda.to_device(adk_prefactors)
    d_adk_powers = cuda.to_device(adk_powers)
    d_adk_exp_prefactors = cuda.to_device(adk_exp_prefactors)
    d_source_indices = cuda.to_device(source_indices)
    d_target_indices = cuda.to_device(target_indices)
    d_charges = cuda.to_device(charges)
    d_initial_populations = cuda.to_device(initial_populations)
    d_all_populations = cuda.to_device(all_populations)
    d_T_array = cuda.to_device(T_array)
    
    # Configure kernel launch
    blocks_per_grid = (n_points + threads_per_block - 1) // threads_per_block
    
    # Launch kernel
    ionization_kernel_gpu[blocks_per_grid, threads_per_block](
        d_a0, tau, lambd, d_ell,
        d_adk_prefactors, d_adk_powers, d_adk_exp_prefactors,
        d_source_indices, d_target_indices, d_charges,
        d_initial_populations,
        d_all_populations, d_T_array, npts_per_wavelength
    )
    
    # Copy results back to CPU
    all_populations = d_all_populations.copy_to_host()
    T_array = d_T_array.copy_to_host()
    
    return all_populations, T_array


def save_to_openpmd(grid_extent, all_populations, T_eV, output_file, species_keys):
    """
    Save with all species populations to an openPMD file
    """
    # create openpmd file
    series = io.Series(output_file, io.Access.create)
    # only 1 iteratiion needed
    it = series.iterations[0]

    # Extract information about the grid for openPMD
    grid_spacing = np.array([ (grid_extent[key][1] - grid_extent[key][0]) / (all_populations.shape[i] - 1)
        for i, key in enumerate(grid_extent.keys()) ])
    grid_global_offset = [grid_extent[key][0] for key in grid_extent.keys()]
    axis_labels = list(grid_extent.keys())

    # Save the temperature
    T = it.meshes["T"]
    T.grid_spacing = grid_spacing
    T.grid_global_offset = grid_global_offset
    T.axis_labels = axis_labels
    T.unit_dimension = {io.Unit_Dimension.theta:1}
    dataset = io.Dataset(T_eV.dtype, T_eV.shape)
    T_scalar = T[io.Mesh_Record_Component.SCALAR]
    T_scalar.reset_dataset(dataset)
    T_scalar.position = [0.0] * len(grid_extent)
    T_scalar.store_chunk(T_eV * (e/k))  # Convert eV to K

    # Save the species fractions
    for i, species_key in enumerate(species_keys):
        pop = it.meshes[species_key + "_fraction"]
        pop.grid_spacing = grid_spacing
        pop.grid_global_offset = grid_global_offset
        pop.axis_labels = axis_labels
        dataset = io.Dataset(all_populations[..., i].dtype, all_populations[..., i].shape)
        pop_scalar = pop[io.Mesh_Record_Component.SCALAR]
        pop_scalar.reset_dataset(dataset)
        pop_scalar.position = [0.0] * len(grid_extent)
        pop_scalar.store_chunk(all_populations[..., i].copy())

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
            use_gpu=True, threads_per_block=256, npts_per_wavelength=80):
    """
    Process nD intensity array for multi-species plasma
    
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
        Filename to save openPMD output with radius, temperature and populations
    grid_extent : dict, optional
        Grid extent for openPMD output
    use_gpu : bool
        If True and CUDA is available, use GPU acceleration. Otherwise use CPU.
    threads_per_block : int
        Number of CUDA threads per block (only used if use_gpu=True)
    npts_per_wavelength : int
        Number of time steps per laser wavelength for integration


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

    # Flatten a0 array for processing
    a0_flat = a0_array.flatten()
    original_shape = a0_array.shape
    
    # Check if GPU is available and requested
    gpu_available = cuda.is_available()
    
    if use_gpu and gpu_available:
        print(f"Processing {a0_array.ndim}D multi-species profile on GPU...")
        print(f"GPU: {cuda.gpus[0].name}")
        all_populations, T_array = process_on_gpu(
            a0_flat, tau, lambd, ell,
            adk_prefactors, adk_powers, adk_exp_prefactors,
            source_indices, target_indices, charges,
            initial_populations,
            threads_per_block=threads_per_block,
            npts_per_wavelength=npts_per_wavelength
        )
    elif not use_gpu or not gpu_available:
        if use_gpu:
            print("GPU requested but not available. Falling back to CPU parallelization...")
        all_populations, T_array = _vectorized_ionization_cpu(
            a0_flat, tau, lambd, ell,
            adk_prefactors, adk_powers, adk_exp_prefactors,
            source_indices, target_indices, charges,
            initial_populations
        )
    # Reshape back to nD arrays
    all_populations = all_populations.reshape(original_shape + (len(species_keys),))
    T_array = T_array.reshape(original_shape)

    # Save detailed output with all species
    if output_file and grid_extent is not None:
        save_to_openpmd(grid_extent, all_populations, T_array, output_file, species_keys)

    return all_populations, T_array
