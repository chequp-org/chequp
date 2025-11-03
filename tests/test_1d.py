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

def cleanup_outputs(extra_file = ""):
    # Remove previously generated plotfiles and checkpoints
    os.system(f"rm -rf plt_* chk* amr_diag.out species_diag.out grid_diag.out Backtrace.0" + extra_file)

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
    # Generate openPMD inital conditions for a small-radius plasma
    # 1000eV plasma in the first 5 microns, low-temperature plasma in the rest
    r = np.linspace(0, 10e-6, 1024)
    T_eV = np.ones_like(r) * 1000
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
    run_castro_simulation("castro.add_ext_src=0 castro.diffuse_temp=0 problem.initial_conditions_file=1d_sedov_taylor.h5")

    # Check the results
    # TODO: Compare the results with Sedov-Taylor theory
    # Evaluate checksum
    evaluate_checksum("1d_sedov_taylor", "plt_1d_*")

    # Remove generated plotfiles and checkpoints
    cleanup_outputs('1d_sedov_taylor.h5')

def load_comsol_data():
        all_data = {}
        try:
            for _ in ['Te', 'na']: # only uselful quantities
                filename = "Exp_"+str(_)+".txt"
                r, z, t0, t1, t2, t5, t8, t10 = np.loadtxt(
                    filename, skiprows=9, unpack=True)
                data = {
                    'r': r,
                    'z': z,
                    't0': t0,
                    't1': t1,
                    't2': t2,
                    't5': t5,
                    't8': t8,
                    't10': t10
                }
                all_data[_] = data
            return all_data
        except Exception as e:
            print(f"Error loading COMSOL data: {e}")
            return {}

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

def check_r_t_CM(sim_data, tol: int = 10):
    """
    Compare radial density profiles at several output times to the COMSOL solution.
    Returns True if the mean relative L2 error is below tol%.
    """
    comsol_data = load_comsol_data()
    r_comsol = comsol_data['na']['r']  # in meters
    t_comsol = np.array([0, 1, 2, 5, 8, 10]) * 1e-9  # in seconds
    r_comsol_vals = []
    for t in [0, 1, 2, 5, 8, 10]:
        na_t = comsol_data['na'][f't{t}']
        rmax = r_comsol[np.argmax(na_t)]
        r_comsol_vals.append(rmax)

    # Simulation data
    t_sim, r_sim = sim_data['time'], sim_data['rmax']
    r_comsol_interp = np.interp(t_sim[1:], t_comsol, r_comsol_vals)
    rel_error = np.linalg.norm(r_sim[1:]*1e4 - r_comsol_interp*1e6) / np.linalg.norm(r_comsol_interp*1e6) * 100.
    assert rel_error < tol, f"Shock radius comparison to COMSOL failed: rel. err. = {rel_error:.1f} % > {tol} % tol."

def check_rho_r_CM(sim_data, tol: int = 50):
    """
    Compare radial density profiles at several output times to the COMSOL solution.
    Returns True if the mean relative L2 error is below tol%.
    """
    # Comsol data
    comsol_data = load_comsol_data()
    comsol_r = comsol_data['na']['r']
    comsol_rho = comsol_data['na'] # Normalize
    # Compute errors for different times
    times = [2e-9, 5e-9, 8e-9]
    diffs = []
    for i, t in enumerate(times):
            idx = np.argmin(np.abs(sim_data['time'] - t))
            r, q = sim_data['r'][idx], sim_data['q'][idx]
            na_comsol_interp = np.interp(r, comsol_r, comsol_rho[f't{int(t*1e9)}']/1e24)
            diff = np.linalg.norm(q/1.67e-6 - na_comsol_interp) / np.linalg.norm(na_comsol_interp)
            diffs.append(diff)
    mean_rel_error = np.mean(diffs) * 100.
    assert mean_rel_error < tol, f"Shock radius comparison to COMSOL failed: rel. err. = {mean_rel_error:.1f} % > {tol} % tol."

def test_1d_desy_benchmark():
    """
    Test the code in the scenario that benchmarked with DESY team
    (close - but not identical - to the one from Mewes et al., PRR 5, 033112, 2023)
    """
    print("Generating initial conditions...")
    # Generate openPMD inital conditions according to the agreed-upon benchmark
    sigma1 = 38e-6  # in m
    sigma2 = 35e-6  # in m
    Te_max = 27 # in eV
    kb = 8.617333262145e-5  # eV/K
    Ta = 2000 * kb  # Background temperature in eV (constrain from COMSOL simulation)
    # Create r array from 0 to 6e-4 with 1e-6 increment
    r = np.arange(0, 6e-4 + 1e-6, 1e-6)
    # Calculate ionization fraction, with minimal ionization fraction of 1e-3
    # (the minimal fraction is needed for the electron temperature to be defined everywhere)
    ioniz_fraction = (1. - 1.e-3)*np.exp(-np.power(r*r/(2*sigma1*sigma1), 12)) + 1.e-3
    # Calculate electron temperature profile such that it match with the COMSOL one
    T_eV = Te_max * np.exp(-np.power(r*r/(2*sigma2*sigma2), 3)) + Ta
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
    print("Starting simulation...")
    time_s = time.time()
    run_castro_simulation("castro.add_ext_src=1 castro.diffuse_temp=1 problem.initial_conditions_file=1d_desy_benchmark.h5")
    time_e = time.time()
    print(f"Simulation completed in {time_e - time_s:.2f} seconds.")
    # Physical tests #
    print("Running physical tests...\n")
    sim_data = load_sim()

    check_energy_conservation(sim_data, tol = 1.0)
    check_r_t_CM(sim_data, tol = 12)
    check_rho_r_CM(sim_data, tol = 50)

    # Evaluate checksum
    evaluate_checksum("1d_desy_benchmark", "plt_1d_*")

    # Remove generated plotfiles and checkpoints
    cleanup_outputs('1d_desy_benchmark.h5')

if __name__ == "__main__":
    #test_1d_sedov_taylor()
    test_1d_desy_benchmark()
