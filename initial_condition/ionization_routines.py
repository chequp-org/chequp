import numpy as np
import numba
import math
import tqdm
from scipy.constants import m_e, c, e, hbar, physical_constants, epsilon_0, k
import pandas as pd
import openpmd_api as io


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

    return populations, T, t  # Return full populations array

    return ioniz_frac, T, t

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
    position = [0]*len(grid_extent)

    # Save the temperature
    T = it.meshes["T"]
    T.grid_spacing = grid_spacing
    T.grid_global_offset = grid_global_offset
    T.axis_labels = axis_labels
    T.position = position
    T.unit_dimension = {io.Unit_Dimension.theta:1}
    dataset = io.Dataset(T_eV.dtype,T_eV.shape)
    T.reset_dataset(dataset)
    T.store_chunk( T_eV * (e/k) ) # Convert eV to K

    # Save the species fractions
    for i, species_key in enumerate(species_keys):
        pop = it.meshes[species_key + "_fraction"]
        pop.grid_spacing = grid_spacing
        pop.grid_global_offset = grid_global_offset
        pop.axis_labels = axis_labels
        pop.position = position
        dataset = io.Dataset(all_populations[:, i].dtype, all_populations[:, i].shape)
        pop.reset_dataset(dataset)
        pop.store_chunk( all_populations[:, i].copy() )

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
            initial_populations, output_file=None, r_coords=None):
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
    initial_populations : array-like
        Initial population fractions [H_0, H_1, N_0, N_1, N_2, N_3, N_4, N_5]
    output_file : str, optional
        Filename to save openPMD output with radius, temperature and populations
    r_coords : array-like, optional
        Radial coordinates (m). Required for 1D data if output_file is specified.
        For 2D data, used to determine radial sampling for CSV output.

    Returns:
    --------
    all_populations : nD array
        Population fractions for all species (shape: [*intensity_nd.shape, len(species_keys)])
    T_array : nD array
        Electron temperatures in eV, same (shape: intensity_nd)
    """
    # Convert intensity to normalized vector potential a0
    a0_array = e * lambd / (np.pi * m_e * c) * np.sqrt(intensity_nd / (2 * epsilon_0 * c**3))

    # Flatten, and prepare arrays for temperature and population
    a0_flat = a0_array.flatten()
    T_flat = np.zeros_like(a0_flat)
    all_populations_flat = np.zeros((len(a0_flat), len(initial_populations)))

    # Process nD profile
    for i in tqdm.tqdm(range(len(a0_flat)), desc=f"Processing {a0_array.ndim}D multi-species profile"):
        all_populations_flat[i, :], T_flat[i], _ = get_fraction_and_temperature_multispecies(
            a0_flat[i],
            tau, lambd, ell,
            adk_prefactors, adk_powers, adk_exp_prefactors,
            source_indices, target_indices, charges,
            initial_populations
        )

    # Reshape back to nD arrays
    all_populations = all_populations_flat.reshape(a0_array.shape + (len(initial_populations),))
    T_array = T_flat.reshape(a0_array.shape)

    # Save detailed CSV output with all species
    if output_file and r_coords is not None:
        save_to_openpmd(r_coords, all_populations, T_array, output_file, species_keys)

    return all_populations, T_array
