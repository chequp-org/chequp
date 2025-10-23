"""
This script tests that the 1D code produce the correct Sedov-Taylor blast wave solution.

It assumes that the code has already been compiled in ../sim_folder/build/
"""
import subprocess
import re
import numpy as np
import sys
import yt
import glob
import os
import openpmd_api
import time
import h5py
sys.path.append("../initial_condition")
from ionization_routines import save_to_openpmd
sys.path.append('../sim_folder/analysis/')
from analysis_tool import CastroSimulation
sys.path.append('../theory/sedov_theory/python/')
from sedov_theory import SedovTalorProblem
from checksum.checksumAPI import evaluate_checksum
from scipy.constants import m_p, k
from scipy.optimize import curve_fit

def cleanup_outputs(extra_file = ""):
    # Remove previously generated plotfiles and checkpoints
    os.system(f"rm -rf plt_1d_* chk* amr_diag.out species_diag.out grid_diag.out Backtrace.0" + extra_file)


class physical_test_1d:

    def __init__(self, folder_name, init_param = (5.0 / 3.0, 1205.9, 1.67e-6)):
        self.folder_name = folder_name
        self.file_start = 'plt_1d_'
        self.cs = CastroSimulation(folder_name, self.file_start)
        self.t_arr, self.r_arr, self.rmax_arr, self.q_arr = self.open_rt('density')
        self.sol = SedovTalorProblem(*init_param)
        self.E0 = init_param[1]

    def open_rt(self, data_type):
        level=3
        """Extract rmax for each output time."""
        r_arr, rmax_arr, q_arr,  = [], [], []
        t_arr = np.array(self.cs.output_times)
        for time in t_arr:
            r, q, t = self.cs.extract_data(time, data_type, level=level)
            rmax = r[np.argmax(q)]
            rmax_arr.append(rmax)
            q_arr.append(q)
            r_arr.append(r)
        return np.array(t_arr), np.array(r_arr), np.array(rmax_arr), np.stack(q_arr)
    
    def fit_power_mc(self):
        from scipy.optimize import curve_fit, OptimizeWarning
        import warnings

        # --- Monte Carlo parameters ---
        n_mc = 1000
        rng = np.random.default_rng(42)
        sigma_frac = 0.05  # relative uncertainty (5%)

        # --- Model definition (no t0) ---
        def power_law(t, a):
            return a * np.sqrt(t)

        # --- Data ---
        t_arr = np.asarray(self.t_arr)
        rmax_arr = np.asarray(self.rmax_arr)

        # --- Base fit (initial guess) ---
        a0 = np.max(rmax_arr) / np.sqrt(np.max(t_arr)) if np.max(t_arr) > 0 else 1.0
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=OptimizeWarning)
            popt, _ = curve_fit(power_law, t_arr, rmax_arr, p0=[a0], maxfev=10000)
        a_fit = popt[0]

        # --- Default sigma (5% of signal) ---
        sigma = sigma_frac * rmax_arr

        # --- Monte Carlo loop ---
        a_samples = []
        fits = np.empty((n_mc, len(t_arr)))

        for i in range(n_mc):
            rmax_noisy = rng.normal(rmax_arr, sigma)
            try:
                popt_i, _ = curve_fit(power_law, t_arr, rmax_noisy, p0=[a_fit], maxfev=10000)
                a_i = popt_i[0]
                fits[i] = power_law(t_arr, a_i)
                a_samples.append(a_i)
            except RuntimeError:
                fits[i] = np.nan  # mark failed fits

        # --- Clean failed fits ---
        valid = ~np.isnan(fits).any(axis=1)
        fits = fits[valid]
        a_samples = np.array(a_samples)

        # --- Statistics ---
        rmax_fit_mean = np.mean(fits, axis=0)

        return rmax_fit_mean

    def test_energy(self, tol: float = 1.0):
            """
            Extract the total energy (thermal + kinetic) as a function of time.
            """
            e = 1.60218e-19 # C
            E_kin_thermal = []
            E_tot = []
            time = []
            E_pot = []
            yt_timeseries = yt.load('plt_1d_*')
            for ds in yt_timeseries:
                ad0 = ds.covering_grid(level=0, left_edge=ds.domain_left_edge, dims=ds.domain_dimensions)
                r = np.array(np.linspace(ds.domain_left_edge[0], ds.domain_right_edge[0], ds.domain_dimensions[0]+1))
                r *= 1e-2 # Convert to m

                E = np.array(ad0['rho_E'].to_ndarray().squeeze()) * 1e-7 * 1e6 # Convert to J/m^3
                e_kin_thermal =  (np.pi*(r[1:]**2 - r[:-1]**2)*E).sum()

                rho_Hp = ad0['rho_H1'].to_ndarray().squeeze() * 1e-3 * 1e6 # Convert to kg/m^3
                n_e = rho_Hp / m_p
                Ntot = (np.pi*(r[1:]**2 - r[:-1]**2)*n_e).sum()
                e_pot = 13.6*e*Ntot

                E_kin_thermal.append( e_kin_thermal )
                E_pot.append( e_pot )
                E_tot.append( e_kin_thermal+e_pot )
                time.append(float(ds.current_time))
            time = np.array(time)
            E_kin_thermal = np.array(E_kin_thermal) * 1e3  # Convert to mJ/m
            E_tot = np.array(E_tot) * 1e3  # Convert to mJ/m
            E_pot = np.array(E_pot) * 1e3  # Convert to mJ/m

            rel_dev = np.max(np.abs(E_tot - E_tot[0]) / np.array(E_tot[0]) * 100.)
            test = rel_dev < tol
            value = rel_dev
            return test, value
    
    def test_rho_r(self, tol: int = 15):
        """
        Compare radial density profiles at several output times to the analytical solution.
        Returns True if the mean relative L2 error is below 15%.
        """
        indices = np.arange(0.7, 0.99, 0.05) * len(self.t_arr)
        errors = []
        for idx in np.unique(indices):
            idx = int(idx)
            r = self.r_arr[idx]
            rho_sim = self.q_arr[idx]
            t = self.t_arr[idx]

            rho_analytical = self.sol.evaluate('density', r, t)

            # compare up to the first peak present in both profiles
            peak_idx = min(np.argmax(rho_analytical), np.argmax(rho_sim))
            if r[peak_idx] > 1e-2: # dont compared for low blast radius
                denom = np.linalg.norm(rho_sim[:peak_idx])
                err = np.linalg.norm(rho_analytical[:peak_idx] - rho_sim[:peak_idx]) / denom
                errors.append(err)

        mean_error = float(np.mean(errors)) * 100
        return mean_error < tol, mean_error

    def test_r_t(self, tol: int = 10):
        """
        Compare the fitted shock radius to the analytical Sedov-Taylor radius.

        Returns True if the relative L2 error is below `tol`, False otherwise.
        """
        # Get fitted and analytical radii
        r_sim = np.asarray(self.fit_power_mc())
        r_analytical = np.asarray(self.sol.blast_radius(self.t_arr))

        # Compute relative L2 error, handle zero-norm reference safely
        denom = np.linalg.norm(r_analytical)
        err = np.linalg.norm(r_sim - r_analytical)
        rel_err = err / denom * 100
        return rel_err < tol, rel_err

def run_castro_simulation(runtime_options):
    """
    Run the Castro simulation.

    Raise an error and print stdout/stderr if the command fails.
    """
    # Find the Castro executable
    build_dir = "../sim_folder/build"
    executables = glob.glob( os.path.join(build_dir, "Castro1d*") )
    if len(executables) == 0:
        raise FileNotFoundError(f"No Castro1d executable found in {build_dir}")
    elif len(executables) > 1:
        raise RuntimeError(f"Multiple Castro1d executables found: {executables}")
    executable = executables[0]

    cleanup_outputs()

    # Run the code
    inputs = "../sim_folder/run/inputs.1d.cyl"
    command = f"{executable} {inputs} {runtime_options}"
    try:
        subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            check=True
        )
    except subprocess.CalledProcessError as e:
        print(f"Command failed with exit code {e.returncode}")
        print("STDOUT:", e.stdout)
        print("STDERR:", e.stderr)
        raise

def test_1d_sedov_taylor():
    """
    Test that code produce the exact Sedov-Taylor blast wave solution, in a simplified setup:
    - no ionization reactions (castro.add_ext_src=0)
    - no temperature diffusion (castro.diffuse_temp=0)
    - the initial radius of the hot plasma is small (5 microns)
    """
    print("Generating initial conditions...")
    # Generate openPMD inital conditions for a small-radius plasma
    # 1000eV plasma in the first 5 microns, low-temperature plasma in the rest
    r = np.linspace(0, 10e-6, 1024)
    sigma = 4e-6
    sigma_H = 1.0e-6
    T_eV = np.ones_like(r) * 1000 * np.exp(-r**2/sigma**2)
    # Parse the species names for which Castro has been compiled
    with open('../sim_folder/build/species.net', 'r') as f:
        species_keys = re.findall(r'\n\s.*\s([A-Z][a-z]*\d)', f.read())
    populations = np.zeros((len(r), len(species_keys)))
    # Set fraction to 1 for H+
    populations[:, species_keys.index('H1')] = np.where(r <= 7e-6, 1.0, np.exp(-((r - 7e-6)/sigma_H)**2))
    populations[:, species_keys.index('H0')] = 1 - np.where(r <= 7e-6, 1.0, np.exp(-((r - 7e-6)/sigma_H)**2))
    # Save file
    save_to_openpmd( {'r': [r.min(), r.max()]}, populations,
        T_eV, '1d_sedov_taylor.h5', species_keys)

    # Run the code
    print("Starting simulation...")
    time_s = time.time()
    run_castro_simulation("problem.initial_conditions_file=1d_sedov_taylor.h5")
    time_e = time.time()
    print(f"Simulation completed in {time_e - time_s:.2f} seconds.")
    # Physical tests #
    print("Running physical tests...\n")
    phys_test = physical_test_1d('.', init_param = (5.0 / 3.0, 1205.9, 1.67e-6))
    test_rho, val_rho = phys_test.test_rho_r(tol = 15)
    if test_rho :
        print(f"\t Test density profile : PASSED (rel. err. = {val_rho:.1f} % < 15 % tol.)")
    else :
        print(f"\t Test density profile : FAILED (rel. err. = {val_rho:.1f} % > 15 % tol.)")
    test_r_t, val_r_t = phys_test.test_r_t(tol = 10)
    if test_r_t :
        print(f"\t Test shock radius vs time : PASSED (rel. err. = {val_r_t:.1f} % < 10 % tol.)")
    else :
        print(f"\t Test shock radius vs time : FAILED (rel. err. = {val_r_t:.1f} % > 10 % tol.)")

    test_energy, val_energy = phys_test.test_energy(tol = 1)
    if test_energy :
        print(f"\t Test energy conservation : PASSED (Avg. Deviation = {val_energy:.1e} % < 1% tol.)")
    else :
        print(f"\t Test energy conservation : FAILED (Avg. Deviation = {val_energy:.1e} % > 1% tol.)")

    # Evaluate checksum
    evaluate_checksum("1d_sedov_taylor", "plt_1d_*")

    # Remove generated plotfiles and checkpoints
    cleanup_outputs('1d_sedov_taylor.h5')

def test_1d_desy_benchmark():
    """
    Test the code in the scenario that benchmarked with DESY team
    (close - but not identical - to the one from Mewes et al., PRR 5, 033112, 2023)
    """
    # Generate openPMD inital conditions according to the agreed-upon benchmark
    sigma1 = 38e-6  # in m
    sigma2 = 32e-6  # in m
    Te_max = 14.65 # in eV
    Ta = 0.08 # in eV
    # Create r array from 0 to 6e-4 with 1e-6 increment
    r = np.arange(0, 6e-4 + 1e-6, 1e-6)
    # Calculate ionization fraction, with minimal ionization fraction of 1e-3
    # (the minimal fraction is needed for the electron temperature to be defined everywhere)
    ioniz_fraction = (1. - 1.e-3)*np.exp(-np.power(r*r/(2*sigma1*sigma1), 12)) + 1.e-3
    # Calculate electron temperature, with a minimal temperature of 0.03 eV
    T_eV = (Te_max - Ta) * np.exp(-np.power(r*r/(2*sigma2*sigma2), 3)) + Ta
    # Parse the species names for which Castro has been compiled
    with open('../sim_folder/build/species.net', 'r') as f:
        species_keys = re.findall(r'\n\s.*\s([A-Z][a-z]*\d)', f.read())
    populations = np.zeros((len(r), len(species_keys)))
    # Set H0 and H1 fractions
    populations[:, species_keys.index('H0')] = 1 - ioniz_fraction
    populations[:, species_keys.index('H1')] = ioniz_fraction
    # Save file
    save_to_openpmd( {'r': [r.min(), r.max()]}, populations,
        T_eV, '1d_desy_benchmark.h5', species_keys)

    # Run the code
    run_castro_simulation("problem.initial_conditions_file=1d_desy_benchmark.h5")
    # Evaluate checksum
    evaluate_checksum("1d_desy_benchmark", "plt_1d_*")

    # Remove generated plotfiles and checkpoints
    cleanup_outputs('1d_desy_benchmark.h5')

if __name__ == "__main__":
    print("\n Starting 1D tests... \n")
    test_1d_sedov_taylor()
    print("\n 1D Sedov-Taylor test completed. \n")

    #test_1d_desy_benchmark()