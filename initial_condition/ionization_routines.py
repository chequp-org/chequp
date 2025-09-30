import numpy as np
import numba
import math
import tqdm
from scipy.constants import m_e, c, e, hbar, physical_constants, epsilon_0

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

def save_radial_csv(r_coords, all_populations, T_eV, output_file):
    """
    Save radial data with all species populations to CSV file
    """
    # Create dictionary starting with radius and temperature
    data_dict = {
        'Radius (cm)': r_coords * 100,  # Convert m to cm
        'Electron Temperature (K)': T_eV * 11604  # Convert eV to K
    }

    # Add a column for each species population
    for i, species_key in enumerate(species_keys):
        data_dict[species_key] = all_populations[:, i]

    df = pd.DataFrame(data_dict)
    df.to_csv(output_file, index=False, float_format='%.4e')
    print(f"Radial profile data saved to {output_file}")

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

def process_intensity_array_multispecies(intensity_1d, lambd, tau, ell,
            adk_prefactors, adk_powers, adk_exp_prefactors,
            source_indices, target_indices, charges,
            initial_populations, output_file=None, r_coords=None):
    """
    Process 1D intensity array for multi-species plasma

    Parameters:
    -----------
    intensity_1d : 1D array
        Radial laser intensity profile I(r) in W/m^2
    lambd : float
        Laser wavelength (m)
    tau : float
        Laser pulse FWHM duration (s)
    ell : array-like
        Polarization vector [2-element array]
    initial_populations : array-like
        Initial population fractions [H_0, H_1, N_0, N_1, N_2, N_3, N_4, N_5]
    output_file : str, optional
        Filename to save CSV output with radius, temperature and populations
    r_coords : array-like, optional
        Radial coordinates (m). Required for 1D data if output_file is specified.
        For 2D data, used to determine radial sampling for CSV output.

    Returns:
    --------
    all_populations : 2D array
        Population fractions for all species (shape: [len(intensity_1d), len(species_keys)])
    T_array : 1D array
        Electron temperatures in eV
    """
    # Convert intensity to normalized vector potential a0
    a0_array = e * lambd / (np.pi * m_e * c) * np.sqrt(intensity_1d / (2 * epsilon_0 * c**3))

    # Initialize output arrays
    all_populations = np.zeros((len(a0_array), len(initial_populations)))
    T_array = np.zeros_like(a0_array)

    # Process 1D profile
    for i in tqdm.tqdm(range(len(a0_array)), desc="Processing 1D multi-species profile"):
        final_populations, T, _ = get_fraction_and_temperature_multispecies(
            a0_array[i], tau, lambd, ell,
            adk_prefactors, adk_powers, adk_exp_prefactors,
            source_indices, target_indices, charges,
            initial_populations
        )
        all_populations[i, :] = final_populations
        T_array[i] = T

    # Save detailed CSV output with all species
    if output_file and r_coords is not None:
        save_radial_csv(r_coords, all_populations, T_array, output_file)

    return all_populations, T_array
