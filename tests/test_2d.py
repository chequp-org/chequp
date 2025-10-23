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
    os.system(f"rm -rf plt_2d_* chk* amr_diag.out species_diag.out grid_diag.out Backtrace.0 " + extra_file)

def clean_cluster_sim():
    # Remove previously generated plotfiles and checkpoints
    os.system(f"rm -rf plt_2d_*.temp.old.* plt_2d_*.temp.* plt_2d_*.old.* amr_diag.out species_diag.out grid_diag.out Backtrace.0 slurm-*.out slurm-*.err " )

class physical_test_2d:

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

    def test_energy(self, tol: int = 10):
        """
        Compute thermal, kinetic, and potential energies over time.
        """
        eps_ion = 13.6 * 1.60218e-12  # erg
        m_H = 1.673e-24            # g
        times = self.cs.output_times
        E_th_K_arr, E_pot_arr = [], []
        t_arr = []

        # conversion factor: erg/cm -> mJ/m
        conv = 1e-2  # erg/cm * 1e-2 = mJ/m
        for time in times:
            # --- Radial grid ---
            r, _, _ = self.cs.extract_data(time, 'rho_H1', 3)
            dr = r[1] - r[0]

            # --- Densities ---
            _, rho_H1, _ = self.cs.extract_data(time, 'rho_H1', 3)  # g/cm³
            _, rho_H0, _ = self.cs.extract_data(time, 'rho_H0', 3)  # g/cm³
            ne = rho_H1 / m_H        # electrons come from ionized H only


            # total energy density from simulation
            _, rho_E, _ = self.cs.extract_data(time, 'rho_E', 3)
            E_total_density = rho_E


            # --- thermal + kinetic energy from rho_E ---
            ethpot_density = E_total_density
            E_th_K = 2 * np.pi * np.sum(ethpot_density * r * dr)

            # --- Potential / ionization energy ---
            q = rho_H1
            E_pot = 2 * np.pi * np.sum(ne * eps_ion * r * dr)
            M_ions = 2*np.pi*dr * np.sum(q*r)
            e = 1.60218e-19 # C
            E_pot = 13.6 * (e*1e7) * M_ions / (m_p * 1e3)

            # Separate E_th and E_kin # 

            E_th = E_th_K
            E_th *= conv
            E_pot *= conv

            # --- Save ---
            t_arr.append(time)
            E_th_K_arr.append(E_th)
            E_pot_arr.append(E_pot)

        E_total_arr = (np.array(E_th_K_arr) +  np.array(E_pot_arr))

        # compute percent relative errors and evaluate test
        test, value = np.mean(np.abs((E_total_arr - np.mean(E_total_arr))) / E_total_arr) * 100 < tol, np.mean(np.abs((E_total_arr - np.mean(E_total_arr))) / E_total_arr) * 100
        return bool(test), float(value)

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
    executables = glob.glob( os.path.join(build_dir, "Castro2d*") )
    if len(executables) == 0:
        raise FileNotFoundError(f"No Castro1d executable found in {build_dir}")
    elif len(executables) > 1:
        raise RuntimeError(f"Multiple Castro1d executables found: {executables}")
    executable = executables[0]

    cleanup_outputs()

    # Run the code
    inputs = "../sim_folder/run/inputs.2d.cyl_in_cartcoords"
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

def run_castro_simulation_cluster(runtime_options=""):

    # --- Define paths ---
    run_dir = "../sim_folder/run"
    script_path = os.path.join(run_dir, "run_castro_job.sh")
    executable = "../sim_folder/build/Castro2d.gnu.MPI.ex"
    input_file = "../sim_folder/run/inputs.2d.cyl_in_cartcoords"

    # --- Make sure run directory exists ---
    if not os.path.isdir(run_dir):
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    # --- Clean outputs if necessary ---
    if "cleanup_outputs" in globals():
        cleanup_outputs()

    # --- Create the SLURM job script content ---
    script_content = f"""#!/bin/bash
#SBATCH --job-name=castro-test        # job name
#SBATCH --output=slurm-%j.out         # standard output (%j = job ID)
#SBATCH --error=slurm-%j.err          # standard error
#SBATCH --time=0-01:00:00             # walltime (D-HH:MM:SS)
#SBATCH --partition=mpa               # queue/partition
#SBATCH --nodes=8                     # number of nodes
#SBATCH --ntasks-per-node=8           # number of MPI ranks per node
#SBATCH --cpus-per-task=1             # threads per rank (OpenMP)

# --- Environment setup ---
export LD_PRELOAD=""                  # avoid preload issues on some nodes
source /etc/profile.d/modules.sh      # enable 'module' command

# --- Load dependencies ---
module purge
module load gcc/12.2.0
module load openmpi/4.1.5
module load hdf5/1.14.0
# module load amrex                    # uncomment if available

# --- Run CASTRO ---
echo "Running Castro simulation on $(hostname) at $(date)"
srun {executable} {input_file} {runtime_options}

# --- Optional: timing info ---
echo "Simulation completed at $(date)"
"""

    # --- Write the script to file ---
    with open(script_path, "w") as f:
        f.write(script_content)

    # Make it executable
    os.chmod(script_path, 0o755)

    # --- Submit the job ---
    try:
        result = subprocess.run(
            ["sbatch", script_path],
            capture_output=True,
            text=True,
            check=True
        )
        # Capture job ID from sbatch output
        job_id = result.stdout.strip().split()[-1]
        print(f"Job submitted successfully (ID: {job_id})")
    except subprocess.CalledProcessError as e:
        print("Job submission failed.")
        print("STDOUT:\n", e.stdout)
        print("STDERR:\n", e.stderr)
        raise

def test_2d_sedov_taylor():
    """
    Test that code produce the exact Sedov-Taylor blast wave solution, in a simplified setup:
    - no ionization reactions (castro.add_ext_src=0)
    - no temperature diffusion (castro.diffuse_temp=0)
    - the initial radius of the hot plasma is small (5 microns)
    """
    print("Generating initial conditions...")
    # Grid
    r = np.linspace(0, 10e-6, 64)
    X, Y = np.meshgrid(r, r, indexing='ij')  # 2D grid 64x64
    sigma_H = 1.0e-6
    sigma = 6e-6
    T_peak = 1000.0  # eV
    T_min = 1e-3     # eV, small temperature floor
    center = 5e-6
    T_eV = T_min + (T_peak - T_min) * np.exp(-((X - center)**2 + (Y - center)**2) / (2 * sigma**2))
 
    # Species keys
    with open('../sim_folder/build/species.net', 'r') as f:
        species_keys = re.findall(r'\n\s.*\s([A-Z][a-z]*\d)', f.read())

    # Populations array
    populations = np.zeros((X.shape[0], X.shape[1], len(species_keys)))
    populations[:, :, species_keys.index('H1')] = 1.0

    # Save file
    save_to_openpmd({'x': [r.min(), r.max()], 'y': [r.min(), r.max()]},
                populations, T_eV, '2d_sedov_taylor.h5', species_keys)
    print("Starting simulation...")
    time_s = time.time()
    run_castro_simulation_cluster("problem.initial_conditions_file=2d_sedov_taylor.h5")
    time_e = time.time()
    print(f"Simulation completed in {time_e - time_s:.2f} seconds.")

    if False:
        # Run the code
        print("Starting simulation...")
        time_s = time.time()
        run_castro_simulation("problem.initial_conditions_file=1d_sedov_taylor.h5")
        time_e = time.time()
        print(f"Simulation completed in {time_e - time_s:.2f} seconds.")
        # Physical tests #
        print("Running physical tests...\n")
        phys_test = physical_test_2d('.', init_param = (5.0 / 3.0, 1205.9, 1.67e-6))
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

        test_energy, val_energy = phys_test.test_energy(tol = 10)
        if test_energy is True :
            print(f"\t Test energy conservation : PASSED (Avg. Deviation = {val_energy:.1f} % < 10% tol.)")
        else :
            print(f"\t Test energy conservation : FAILED (Avg. Deviation = {val_energy:.1f} % > 10% tol.)")

    #clean_cluster_sim()
    # Evaluate checksum
    #evaluate_checksum("1d_sedov_taylor", "plt_1d_*")

    # Remove generated plotfiles and checkpoints
    #cleanup_outputs('1d_sedov_taylor.h5')

def test_2d_desy_benchmark():
    """
    Test the code in the scenario that benchmarked with DESY team
    (close - but not identical - to the one from Mewes et al., PRR 5, 033112, 2023)
    """
    # Generate openPMD inital conditions according to the agreed-upon benchmark
    sigma1 = 38e-6  # in m
    sigma2 = 35e-6  # in m
    Te_max = 27 # in eV
    Ta = 0.03 # in eV
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
    print("\n Starting 2D tests... \n")
    test_2d_sedov_taylor()
    print("\n 2D Sedov-Taylor test completed. \n")

    #test_2d_desy_benchmark()
