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

    os.system("rm -rf plt_1d_* chk* amr_diag.out species_diag.out grid_diag.out Backtrace.0" + extra_file)

def load_sim():
    cs = CastroSimulation('.', 'plt_1d_')
    """Extract rmax for each output time."""
    r_arr, rmax_arr, q_arr, E_tot_arr = [], [], [], []
    t_arr = np.array(cs.output_times)
    for t0 in t_arr:
        r, q, t = cs.extract_data(t0, 'density', level=3)
        rmax = r[np.argmax(q)]
        rmax_arr.append(rmax)
        q_arr.append(q)
        r_arr.append(r)
        E_tot_arr.append(cs.get_energy(t, level=3)[0])
    return {'time': np.array(t_arr), 'r': np.array(r_arr), 'rmax': np.array(rmax_arr), 'q': np.array(q_arr), 'E_tot': np.array(E_tot_arr)}

def check_energy_conservation(sim_data, tol: float = 1.0):
    rel_err = np.abs(sim_data['E_tot'] - sim_data['E_tot'][0]) / sim_data['E_tot'][0] * 100.0
    test = np.all(rel_err < tol)
    value = np.max(rel_err)
    assert test, f"Energy conservation test failed: Avg. Deviation = {value:.1e} % > {tol}% tol."

def check_r_t_ST(sim_data, sol, tol: int = 10):
    popt, _ = curve_fit(lambda t, a: a * np.sqrt(t), sim_data['time'][1:], sim_data['rmax'][1:])
    r_fit = popt[0] * np.sqrt(sim_data['time'])
    r_analytical = np.array(sol.blast_radius(sim_data['time']))
    rel_error = np.linalg.norm(r_fit*1e4 - r_analytical*1e4) / np.linalg.norm(r_analytical*1e4) * 100.
    assert rel_error < tol, f"Shock radius comparison to Sedov Taylor theory failed: rel. err. = {rel_error:.1f} % > {tol} % tol."

def check_rho_r_ST(sim_data, sol, tol: int = 15):
    indices = np.arange(0.7, 0.99, 0.05) * len(sim_data['time'])
    errors = []
    for idx in np.unique(indices):
        idx = int(idx)
        r = sim_data['r'][idx]
        rho_sim = sim_data['q'][idx]
        t = sim_data['time'][idx]
    rho_analytical = sol.evaluate('density', r, t)

    # compare up to the first peak present in both profiles
    peak_idx = min(np.argmax(rho_analytical), np.argmax(rho_sim))
    if r[peak_idx] > 1e-2: # dont compared for low blast radius
        denom = np.linalg.norm(rho_sim[:peak_idx])
        err = np.linalg.norm(rho_analytical[:peak_idx] - rho_sim[:peak_idx]) / denom
        errors.append(err)

    mean_rel_error = np.mean(np.array(errors)) * 100.
    assert mean_rel_error < tol, f"Density profile comparison to Sedov Taylor theory failed: rel. err. = {mean_rel_error:.1f} % > {tol} % tol."

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
    """
    print("Generating initial conditions...")
    # Generate openPMD inital conditions for a small-radius plasma
    # Gaussian temperature profile with sigma=4 microns, peak T=1000 eV
    r = np.linspace(0, 10e-6, 1024)
    sigma = 4e-6
    T0_eV = 1000
    T_eV = np.ones_like(r) * T0_eV * np.exp(-r**2/sigma**2) # Gaussian profile to fasten convergence
    # Parse the species names for which Castro has been compiled
    with open('../sim_folder/build/species.net', 'r') as f:
        species_keys = re.findall(r'\n\s.*\s([A-Z][a-z]*\d)', f.read())
    populations = np.zeros((len(r), len(species_keys)))
    # Set fraction to 1 for H+
    populations[:, species_keys.index('H1')] = 1 - 1e-3
    populations[:, species_keys.index('H0')] = 1e-3
    # Save file
    save_to_openpmd( {'r': [r.min(), r.max()]}, populations,
        T_eV, '1d_sedov_taylor.h5', species_keys)

    # Run the code
    print("Starting simulation...")
    time_s = time.time()
    run_castro_simulation("amr.n_cell=128 castro.add_ext_src=0 castro.diffuse_temp=0 problem.initial_conditions_file=1d_sedov_taylor.h5")
    time_e = time.time()
    print(f"Simulation completed in {time_e - time_s:.2f} seconds.")
    # Physical tests #
    print("Running physical tests...\n")
    sim_data = load_sim()

    # Comparison with Sedov Taylor theory
    rho_initial = 1.67e-6  # in g.cm^-3
    mp_g = m_p*1e3 # in g
    sigma_cm = sigma*1e2 # in cm
    deposited_energy = 3*np.pi/2 * T0_eV*e * sigma_cm**2 * rho_initial/mp_g * 1e7 # in erg/cm (computed by integrating initial conditions)
    analytical_data = SedovTalorProblem(5.0 / 3.0, deposited_energy, rho_initial)

    check_energy_conservation(sim_data, tol=1.0)
    check_r_t_ST(sim_data, analytical_data, tol=10)
    check_rho_r_ST(sim_data, analytical_data, tol=22)

    print("Physical tests passed.\n")
    # Evaluate checksum
    evaluate_checksum("1d_sedov_taylor", "plt_1d_*", rtol=4.e-7)

    # Remove generated plotfiles and checkpoints
    cleanup_outputs('1d_sedov_taylor.h5')


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
    evaluate_checksum("1d_desy_benchmark", "plt_1d_*", rtol=4.e-7)

    # Remove generated plotfiles and checkpoints
    cleanup_outputs('1d_desy_benchmark.h5')

if __name__ == "__main__":

    test_1d_sedov_taylor()
        
    #test_1d_desy_benchmark()