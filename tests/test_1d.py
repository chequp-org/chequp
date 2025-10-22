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
sys.path.append("../initial_condition")
from ionization_routines import save_to_openpmd
sys.path.append('../sim_folder/analysis/')
from analysis_tool import CastroSimulation
from checksum.checksumAPI import evaluate_checksum
from scipy.constants import m_p, k

def cleanup_outputs(extra_file = ""):
    # Remove previously generated plotfiles and checkpoints
    os.system(f"rm -rf plt_* chk* amr_diag.out species_diag.out grid_diag.out " + extra_file)

class physical_test_1d:
    def __init__(self, folder_name):
        self.folder_name = folder_name
        self.file_start = 'plt_1d_'
        self.cs = CastroSimulation(folder_name, self.file_start)
        self.t_arr, self.r_arr, self.rmax_arr, self.q_arr = self.open_rt('density')
        
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

    def compute_energies(self):
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

        E_total_arr = np.array(E_th_K_arr) +  np.array(E_pot_arr)
        var_E = np.var(E_total_arr)
        # Relative deviation in percent
        dev_percent = 100 * (E_total_arr - np.mean(E_total_arr)) / np.mean(E_total_arr)

        # RMS deviation in percent
        rms_percent = np.sqrt(np.mean(dev_percent**2))

        # Maximum deviation in percent
        max_percent = np.max(np.abs(dev_percent))
        if max_percent < 1.0:
            return True
        else :
            return [max_percent, rms_percent]

    def test_rho_r(self):
        L_error = []
        for i, rand_idx in enumerate(np.arange() * len(self.r_arr)):
            r, rho_sim, t = self.r_arr[int(rand_idx)], self.q_arr[int(rand_idx)], self.t_arr[int(rand_idx)]
            rho_analytical = self.sol.evaluate( 'density', r, t)
            min_idx = min(np.argmax(rho_analytical), np.argmax(rho_sim))
            error = np.linalg.norm(rho_analytical[:min_idx]-rho_sim[:min_idx])/np.linalg.norm(rho_sim[:min_idx])
            L_error.append(error)
        glob_error = np.mean(np.array(L_error))
        if glob_error < 0.15 :# error under 15%
            True
        else :
            False

    def test_r_t(self):
        return True

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
    T_eV = np.ones_like(r) * 1000 * np.exp(-r**2/sigma**2)
    # Parse the species names for which Castro has been compiled
    with open('../sim_folder/build/species.net', 'r') as f:
        species_keys = re.findall(r'\n\s.*\s([A-Z][a-z]*\d)', f.read())
    populations = np.zeros((len(r), len(species_keys)))
    # Set fraction to 1 for H+
    populations[:, species_keys.index('H1')] = 1
    # Save file
    save_to_openpmd( {'r': [r.min(), r.max()]}, populations,
        T_eV, '1d_sedov_taylor.h5', species_keys)

    # Run the code
    print("Running simulation...")
    run_castro_simulation("problem.initial_conditions_file=1d_sedov_taylor.h5")

    # Physical tests #
    print("Running physical tests...")
    phys_test = physical_test_1d('.')
    test_rho = phys_test.test_rho_r()
    if test_rho :
        print("Test density profile : PASS")
    # Evaluate checksum
    #evaluate_checksum("1d_sedov_taylor", "plt_1d_*")

    # Remove generated plotfiles and checkpoints
    #cleanup_outputs('1d_sedov_taylor.h5')

def test_1d_desy_benchmark():
    """
    Test the code in the scenario that benchmarked with DESY team
    (close - but not identical - to the one from Mewes et al., PRR 5, 033112, 2023)
    """
    # Generate openPMD inital conditions according to the agreed-upon benchmark
    sigma1 = 38e-6  # in m
    sigma2 = 35e-6  # in m
    Te_max = 27 # in eV
    Ta = 0.03 # in eV
    print('Generating initial conditions...')
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
    print("Running simulation...")
    run_castro_simulation("problem.initial_conditions_file=1d_desy_benchmark.h5")
    # Physical tests #
    print("Running physical tests...")
    # Evaluate checksum
    #evaluate_checksum("1d_desy_benchmark", "plt_1d_*")

    # Remove generated plotfiles and checkpoints
    #cleanup_outputs('1d_desy_benchmark.h5')



if __name__ == "__main__":
    test_1d_sedov_taylor()
    #test_1d_desy_benchmark()